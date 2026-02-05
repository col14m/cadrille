# CAD recode imports
import os
from dataclasses import asdict, dataclass
from datetime import timedelta
from functools import partial

import pyrallis
import wandb
from cad_recode_model_mm import Cadrille
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoProcessor

from dataset_utils import RealDatasetMM
from grpo_mm import train_with_grpo_mm
from utils import extract_mesh_from_texts, \
    evaluate_model_mm

os.environ["PYGLET_HEADLESS"] = "True"
os.environ["TOKENIZERS_PARALLELISM"] = "True"

import torch.distributed as dist
import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from qwen_vl_utils import process_vision_info


@dataclass
class TrainConfig:
    sft_path: str

    project: str = "CAD_test"
    group: str = "Dr-CCPO"
    name: str = "dr-ccpo"
    save_path: str = "./models"

    failure_reward: float = -10
    train_size: int = None
    train_mode: str = "img"
    train_file: str = "train_small.pkl"
    freeze_pc: bool = False
    # GRPO params
    train_epochs: int = 20
    batch_size: int = 16
    num_generations: int = 16
    top_samples: int = 4
    max_completion_length: int = 400
    learning_rate: float = 3e-5
    batch_updates: int = 3
    epsilon_high: float = 0.1
    epsilon_low: float = 0.1

    use_gpg: bool = False
    use_buffer: bool = False


def collate_img_pc_v1(batch, processor, n_points, eval=False):
    messages = []
    is_pc = [0] * len(batch)
    is_img = [0] * len(batch)

    for i, m in enumerate(batch):
        if 'video' in m.keys():
            is_img[i] = 1
            message = [{
                'role': 'user',
                'content': [
                    {'type': 'video', 'video': m['video'], 'fps': 1.0},
                    {'type': 'text', 'text': m['description']}
                ]
            }]
        else:
            if 'point_cloud' in m.keys():
                is_pc[i] = 1
            message = [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': m['description']}
                ]
            }]
        messages.append(message)
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
             for msg in messages]

    points_inputs = ''.join(n_points * [processor.tokenizer.pad_token])

    for i in range(len(texts)):
        if is_pc[i]:
            texts[i] = points_inputs + texts[i]

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt")

    inputs['point_clouds'] = torch.stack([torch.tensor(m['point_cloud']) if is_pc[i]
                                          else torch.zeros(n_points, 3) for i, m in enumerate(batch)])
    inputs['is_pc'] = torch.tensor(is_pc, dtype=torch.bool)
    inputs['is_img'] = torch.tensor(is_img, dtype=torch.bool)

    if 'pixel_values_videos' in inputs.keys():
        pixel_values_videos = inputs['pixel_values_videos'].new_zeros(
            (len(batch), torch.prod(inputs['video_grid_thw'][0]),
             inputs['pixel_values_videos'].shape[1]))
        pixel_values_videos[inputs['is_img']] = torch.stack(torch.chunk(inputs['pixel_values_videos'],
                                                                        chunks=sum(inputs['is_img'])))
        inputs['pixel_values_videos'] = pixel_values_videos

        video_grid_thw = inputs['video_grid_thw'].new_zeros((len(batch), 3))
        video_grid_thw[inputs['is_img']] = inputs['video_grid_thw']
        inputs['video_grid_thw'] = video_grid_thw

    inputs['mesh_path'] = [m['mesh_path'] for m in batch]
    inputs['mesh'] = [m['mesh'] for m in batch]
    inputs['idx'] = [m['idx'] for m in batch]
    return inputs


def get_reward_function(failure_reward):
    def combined_reward(completions, answer):
        # Get individual rewards
        rewards = []
        # excepts = []
        pred_ious = extract_mesh_from_texts(completions, answer)
        # print("MESHES", pred_meshes, flush=True)
        for i, pred_iou in enumerate(pred_ious):
            reward = 0
            if pred_iou is None:
                reward = failure_reward
            elif pred_iou < 0:
                reward = 0
            else:
                reward = pred_iou * 10
            rewards.append(reward)
        return rewards
    return combined_reward

def optimize_model_memory(model):
    """
    Optimizes the model to use less memory during training.
    """
    model.train()
    model.config.use_cache = False

    # First ensure inputs will require gradients
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Then enable gradient checkpointing
    model.gradient_checkpointing_enable()

    return model


def setup(world_size):
    """ Initialize the process group for distributed training """
    dist.init_process_group(backend="nccl", world_size=world_size, timeout=timedelta(hours=5))


def cleanup():
    """ Destroy the process group """
    dist.destroy_process_group()

@record
@pyrallis.wrap()
def main(config: TrainConfig):
    world_size = int(os.getenv("WORLD_SIZE"))
    setup(world_size)

    rank = dist.get_rank()
    rank = rank % torch.cuda.device_count()

    torch.cuda.set_device(rank)
    print("RANK, WS:", rank, world_size, flush=True)
    attn_implementation = 'flash_attention_2' if torch.cuda.is_available() else None

    model = Cadrille.from_pretrained(
        config.sft_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=rank).train().to(rank)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",
                                              min_pixels=256 * 28 * 28,
                                              max_pixels=1280 * 28 * 28,
                                              padding_side="left")

    eval_data_deepcad = RealDatasetMM(path=f'./data/deepcad_test', file_name='test.pkl', n_points=256, size=1000)
    eval_data_fusion = RealDatasetMM(path=f'./data/fusion360_test', file_name='test.pkl', n_points=256, size=1000)
    train_data = RealDatasetMM(path=f'./data/deepcad_fusion_train', file_name=config.train_file, n_points=256, mode=config.train_mode, noise_scale_pc=0.01, size=config.train_size)

    # text_train_dataset = Text2CADDataset(path=f'/home/jovyan/tarasov/data/deepcad_fusion_train', file_name='text_train.pkl', idx_offset=len(train_data))
    # text_eval_dataset = Text2CADDataset(path=f'/home/jovyan/tarasov/data/deepcad_test', file_name='text_test.pkl')
    # train_data = ConcatDataset([train_data, text_train_dataset])

    # Main execution
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")

    model = optimize_model_memory(model)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    if config.freeze_pc:
        model.freeze_pc()

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    dist.barrier()
    part_collate = partial(collate_img_pc_v1, processor=processor, n_points=256)


    if rank == 0:
        print("\nInitial model evaluation before finetuning and after filtering:")
        eval_data_deepcad.mode = 'pc'
        ious, cds, incorrect, failed_intersect = evaluate_model_mm(model.module, processor, eval_data_deepcad, rank, part_collate, batch_size=200)
        eval_data_deepcad.mode = 'img'
        ious_im, cds_im, incorrect_im, failed_intersect_im = evaluate_model_mm(model.module, processor, eval_data_deepcad, rank, part_collate, batch_size=200)
        eval_data_fusion.mode = 'pc'
        ious_f, cds_f, incorrect_f, failed_intersect_f = evaluate_model_mm(model.module, processor, eval_data_fusion, rank, part_collate, batch_size=200)
        eval_data_fusion.mode = 'img'
        ious_f_im, cds_f_im, incorrect_f_im, failed_intersect_f_im = evaluate_model_mm(model.module, processor, eval_data_fusion, rank, part_collate, batch_size=200)

        # ious_txt, cds_txt, incorrect_txt, failed_intersect_txt = evaluate_model_mm(model.module, processor, text_eval_dataset, rank, part_collate, batch_size=50)
    dist.barrier()

    print("\nStarting RL fine-tuning using GRPO...")
    training_config = {
        'train_epochs': config.train_epochs,
        'batch_size': config.batch_size,  # reduce if you have fewer GPUs
        'num_generations': config.num_generations,  # reduce if you have GPUs with less VRAM
        'top_samples': config.top_samples,  # reduce if you have GPUs with less VRAM
        'max_completion_length': config.max_completion_length,  # reduce if you have GPUs with less VRAM
        'learning_rate': config.learning_rate,
        'batch_updates': config.batch_updates,
        'epsilon_high': config.epsilon_high,
        'epsilon_low': config.epsilon_low,
    }
    sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    # Initialize Weights & Biases
    run_id = None
    if rank == 0:
        dict_config = asdict(config)
        wandb.init(project=config.project, group=config.group, name=config.name, reinit=True, config=dict_config)
        print("Weights & Biases initialized.")
        run_id = wandb.run.id

        wandb.log({
            "eval/pc/DeepCAD test/IoU mean": np.mean(ious),
            "eval/pc/DeepCAD test/CD mean": np.mean(cds),
            "eval/pc/DeepCAD test/IoU median": np.median(ious),
            "eval/pc/DeepCAD test/CD median": np.median(cds),
            "eval/pc/DeepCAD test/Failures fraction": incorrect,
            "eval/pc/Fusion360 test/IoU mean": np.mean(ious_f),
            "eval/pc/Fusion360 test/CD mean": np.mean(cds_f),
            "eval/pc/Fusion360 test/IoU median": np.median(ious_f),
            "eval/pc/Fusion360 test/CD median": np.median(cds_f),
            "eval/pc/Fusion360 test/Failures fraction": incorrect_f,

            "eval/img/DeepCAD test/IoU mean": np.mean(ious_im),
            "eval/img/DeepCAD test/CD mean": np.mean(cds_im),
            "eval/img/DeepCAD test/IoU median": np.median(ious_im),
            "eval/img/DeepCAD test/CD median": np.median(cds_im),
            "eval/img/DeepCAD test/Failures fraction": incorrect_im,
            "eval/img/Fusion360 test/IoU mean": np.mean(ious_f_im),
            "eval/img/Fusion360 test/CD mean": np.mean(cds_f_im),
            "eval/img/Fusion360 test/IoU median": np.median(ious_f_im),
            "eval/img/Fusion360 test/CD median": np.median(cds_f_im),
            "eval/img/Fusion360 test/Failures fraction": incorrect_f_im,

            # "eval/txt/DeepCAD test/IoU mean": np.mean(ious_txt),
            # "eval/txt/DeepCAD test/CD mean": np.mean(cds_txt),
            # "eval/txt/DeepCAD test/IoU median": np.median(ious_txt),
            # "eval/txt/DeepCAD test/CD median": np.median(cds_txt),
            # "eval/txt/DeepCAD test/Failures fraction": incorrect_txt + failed_intersect_txt,
        })


    model = train_with_grpo_mm(
        model=model,
        processor=processor,
        train_data=train_data,
        eval_data_deepcad=eval_data_deepcad,
        eval_data_fusion=eval_data_fusion,
        # eval_data_text=text_eval_dataset,
        eval_data_text=None,
        sampler=sampler,
        reward_function=get_reward_function(config.failure_reward),
        collate_fn=part_collate,
        run_id=run_id,
        gpg=config.use_gpg,
        use_buffer=config.use_buffer,
        save_path=config.save_path,
        **training_config
    )
    if rank == 0:

        wandb.finish()
        print("Training completed and wandb run finished.")

        print("\nSaving GRPO fine-tuned model...")

        model.save_pretrained(f"{config.save_path}/{run_id}")
        processor.save_pretrained(f"{config.save_path}/{run_id}")
    cleanup()


if __name__ == "__main__":
    main()
