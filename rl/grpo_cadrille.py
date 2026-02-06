import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from trl import GRPOConfig, TrlParser
sys.path.append("/workspace-SR008.nfs2/users/barannikov/trl_cadevolve")

from grpo_video_trainer import VideoTopSampleGRPOTrainer
from pathlib import Path

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, GenerationConfig
from dataclasses import dataclass
import torch
from torch import optim
from metrics_async import init_pool, close_pool, get_metrics_from_texts

import math
import numpy as np

MODEL_PATH = "/workspace-SR008.nfs2/users/barannikov/cadrille/models/cadrille"
SEED = 16



@dataclass
class RewardArgs:
    failure_reward: float = -10.0
    iou_coef: float = 10.0
    cd_coef: float = 0.0
    auc_coef: float = 0.0
    get_nc: bool = False
    # how many points to sample from surface
    nc_n_points: int = 16384
    # what percentage of overall mesh extents to look for neighbors in
    nc_tol: int = 5
    gen_sample_steps: int = 25 // 3
    pool_size: int = 16

@dataclass
class ModelConfig:
    sft_path: str = "/workspace-SR008.nfs2/users/zhemchuzhnikov/iterative_generation/train/work_dirs/qwen2vl_image2code_stls_v2_aug_updated/final_model"
    group_port: int = 51216

@dataclass
class TrainingArgs:
    adv_multiplier: int = 1
    clip_cov: bool = False
    top_samples: int = 4


parser = TrlParser((GRPOConfig, RewardArgs, ModelConfig, TrainingArgs))
grpo, rargs, margs, targs = parser.parse_args_and_config()
grpo.output_dir = "models/" + grpo.output_dir

init_pool(rargs.pool_size)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


from cadrille_plugin.cadrille import Cadrille
sys.path.append("/workspace-SR008.nfs2/users/barannikov/trl_cadevolve/vllm_cadrille/cadrille_module")  # parent of cadrille_plugin
import cadrille_plugin


model = Cadrille.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",)
model.gradient_checkpointing_disable()
model.enable_input_require_grads() 
model.config.gradient_checkpointing = False
model.freeze_pc()


processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",
                                            padding_side="left",
                                            use_fast=False
                                            )
processor.video_processor.do_pad = False
processor.video_processor.default_to_square = False
processor.video_processor.do_center_crop = False
processor.video_processor.size_divisor = 1
processor.video_processor.min_frames = 1
processor.video_processor.max_pixels = 280*280
processor.video_processor.min_pixels = 280*280


from patch_vllm import patch_vllm_proccessor, patch_vllm_for_videos
patch_vllm_proccessor(processor)

patch_vllm_for_videos()

from datasets import load_from_disk
ds = load_from_disk(".data/abc_rendered_all")

def make_conversation(ex):
    conversation = [
        {"role": "user",
         "content": [
             {"type": "video", 'fps': 1.0},
             {"type": "text", "text": ex["instruction"]}
         ]}
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    return {"prompt": prompt}


ds = ds.map(
    make_conversation,
    num_proc=156,
    remove_columns=[],)

train_dataset = ds

# -------------- coefficients for the geometric reward -------------
def reward_from_metrics(cd: float, iou: float, auc: float = 0, mode: str = "default") -> float:
    #if math.isnan(iou): iou = 0.0
    if cd is None or math.isnan(cd) or cd <= 0: cd = 1.0
    if mode == "10_iou":
        r = 10.0 * float(iou)
    elif mode == "cd_to_reward":
        ln = math.log(max(cd, 1e-8))
        denom = (ln - 1.0)
        if abs(denom) < 1e-4: denom = 1e-4 if denom >= 0 else -1e-4
        r = 10.0 * (1.0 + 1.0 / denom)
    elif mode == "iou":
        r = float(iou)
    elif mode == "10_normal_auc":
        r  = 10.0 * auc
    else:
        r = 10.0 * float(iou)
    return float(np.clip(r, -10.0, 10.0))

def update_step_tol(step):
    if step >= 150:
        tol = 2
    elif step >= 75:
        tol = 3
    else:
        tol = nc_params["tol"]
    #print(f"[tol] step {step}: tol={tol}")
    return tol


def reward_from_auc(auc, step, scale=10.0, eps=1e-8, tau=3e-3):
    # get smoother growth with normal consistency reward to regularize growth to be more linear 
    auc = np.asarray(auc, dtype=float)
    z = np.clip((auc - 0.5) / 0.5, eps, 1 - eps)

    base = scale * (0.5 + 0.5 * z)
    # gate ensures rewards > 9.5 occur only when AUC > 0.975
    gate = 1.0 / (1.0 + np.exp(-(auc - 0.975) / tau))

    below = auc <= 0.975
    capped = np.where(below, np.minimum(base, 9.5), base)
    rewarded = np.where(below, capped, 9.5 + (base - 9.5) * gate)
    return rewarded


def get_reward_function(failure_reward, iou_coef=10, cd_coef=0, auc_coef=0, nc_params=None):
    def combined_reward(completions, mesh_path, trainer_state=None, **kwargs):
        # Get individual rewards
        rewards = []
        '''
        if nc_params.get("get_nc") == True:
            updt_tol = update_step_tol(step=getattr(trainer_state, "global_step", 0))
            nc_params["tol"] = updt_tol'''
        pred_metrics = get_metrics_from_texts(
            completions, mesh_path, nc_params, var_name="r")
        # print("MESHES", pred_meshes, flush=True)
        for m in pred_metrics:
            reward = 0
            iou = m["iou"] if m is not None else None
            cd =  m["cd"] if m is not None else None
            auc =  m["auc"] if m is not None else None
            if iou is None:
                reward = failure_reward
            else:
                reward = reward_from_metrics(cd, iou, auc=auc, mode="10_iou")
                #reward = reward_from_auc(auc=auc, step=trainer_state.global_step)
            rewards.append(float(reward))
        print(f"Rewards : {rewards}\n\n\n")

        # ---- print one sample every 50 steps ----
        top_idx = rewards.index(max(rewards))
        top_generation = completions[top_idx]
        top_mesh_path = mesh_path[top_idx]
        _maybe_print_sample(top_generation, top_mesh_path, step=trainer_state.global_step)

        return rewards
    return combined_reward

nc_params = {
    "get_nc": rargs.get_nc,
    "n_points" : rargs.nc_n_points, 
    "tol" : rargs.nc_tol,
}


reward_fn = get_reward_function(failure_reward=rargs.failure_reward, iou_coef=rargs.iou_coef, cd_coef=rargs.cd_coef, auc_coef=rargs.auc_coef, nc_params=nc_params)


optimizer = optim.AdamW(model.parameters(), lr=grpo.learning_rate)
#lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
lr_scheduler = None


#----------- callback for outputting samples
def _maybe_print_sample(completion, mesh_path, step, every=rargs.gen_sample_steps):
    if step == 0 or step % every != 0:
        return
    print(f"\n[SAMPLE @ step {step}]\n Mesh path : {mesh_path} \n {completion}\n", flush=True)

# those parameters will be passed to vllm generation trainer
bad_words = ["<|image_pad|>", "<|vision_pad|>", "<|vision_start|>", "<|vision_end|>", "<|video_pad|>"]
grpo.generation_kwargs = {
    "bad_words": bad_words,

}

# override parameters automatically set by TRL trainer 
# to generate big batches while training on smaller ones, as we select the top K answers
grpo.steps_per_generation = 1
grpo.max_steps = grpo.num_train_epochs * len(train_dataset) * grpo.num_iterations // (grpo.per_device_train_batch_size * int(os.environ["WORLD_SIZE"]))
print(f"Total training steps : {grpo.max_steps} ")
grpo.num_train_epochs = 0

trainer = VideoTopSampleGRPOTrainer(
    clip_cov=targs.clip_cov,
    top_samples=targs.top_samples,
    model=model,
    processing_class=processor,
    reward_funcs=[reward_fn],
    train_dataset=train_dataset,
    args=grpo,
    optimizers=(optimizer, lr_scheduler),
)

trainer.train(
    #resume_from_checkpoint="/workspace-SR008.nfs2/users/barannikov/trl_cadevolve/models/09_17_cadrille_sequence_samp/checkpoint-1300"
    )
close_pool()