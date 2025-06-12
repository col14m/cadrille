import os

import numpy as np
import wandb

os.environ["PYGLET_HEADLESS"] = "True"

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn.functional as F

from utils import evaluate_model_mm
from dataset_utils import IndexBuffer


def selective_log_softmax(logits, input_ids):
    """
    Computes log probabilities for specific tokens in the vocabulary.
    """
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


def compute_log_probs(model, batch, logits_to_keep):
    """
    Computes the log probabilities for a batch of tokens.
    """
    input_ids, attention_mask, point_cloud, is_pc, is_img, pixel_values_videos, video_grid_thw = batch
    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.to(model.device)
    if video_grid_thw is not None:
        video_grid_thw = video_grid_thw.to(model.device)
    logits = model(
        input_ids=input_ids.clone(),
        attention_mask=attention_mask.clone(),
        point_clouds=point_cloud.clone(),
        is_pc=is_pc.to(model.device),
        is_img=is_img.to(model.device),
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw).logits[:, :-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return selective_log_softmax(logits, input_ids)


def create_completion_mask(completion_ids, eos_token_id):
    """
    Creates a mask for completion tokens that excludes tokens after the EOS token.
    """
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (sequence_indices <= eos_idx.unsqueeze(1)).int()


def generate_completions(model, processor, inputs, num_generations=4, max_completion_length=32):
    """
    Generates multiple completions for each prompt.
    """
    device = model.device
    prompt_ids = inputs["input_ids"].clone().detach().to(device)
    prompt_mask = inputs["attention_mask"].clone().detach().to(device)
    point_cloud = inputs["point_clouds"].clone().detach().to(device)
    is_pc = inputs["is_pc"].clone().detach().to(device)
    is_img = inputs["is_img"].clone().detach().to(device)
    pixel_values_videos = inputs['pixel_values_videos'].clone().detach().to(device) if inputs.get('pixel_values_videos',
                                                                                               None) is not None else None
    video_grid_thw = inputs['video_grid_thw'].clone().detach().to(device) if inputs.get('video_grid_thw',
                                                                                     None) is not None else None
    prompt_length = prompt_ids.size(1)
    batch_size = prompt_ids.size(0)
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    point_cloud = point_cloud.repeat_interleave(num_generations, dim=0)
    is_pc = is_pc.repeat_interleave(num_generations, dim=0)
    is_img = is_img.repeat_interleave(num_generations, dim=0)
    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.repeat_interleave(num_generations, dim=0)
    if video_grid_thw is not None:
        video_grid_thw = video_grid_thw.repeat_interleave(num_generations, dim=0)
    outputs = model.generate(input_ids=prompt_ids.clone(),
                             attention_mask=prompt_mask.clone(),
                             point_clouds=point_cloud.clone(),
                             is_pc=is_pc.clone(),
                             is_img=is_img.clone(),
                             pixel_values_videos=pixel_values_videos.clone() if pixel_values_videos is not None else None,
                             video_grid_thw=video_grid_thw.clone() if video_grid_thw is not None else None,
                             max_new_tokens=max_completion_length,
                             do_sample=True,
                             temperature=1.0,
                             top_p=1.0,
                             top_k=50,
                             early_stopping=False,
                             bad_words_ids=[[model.config.video_token_id]],
                             )
    completion_ids = outputs[:, prompt_length:]
    completion_mask = create_completion_mask(completion_ids, processor.tokenizer.eos_token_id)
    return point_cloud, prompt_ids, torch.tensor(inputs["attention_mask"]).clone().to(device).repeat_interleave(
        num_generations, dim=0), is_pc, is_img, pixel_values_videos, video_grid_thw, completion_ids, completion_mask


def generate_rollout_data(model, reward_function,
                          processor, batch_samples, num_generations, max_completion_length, top_samples=None,
                          gpg=False, buffer=None):
    """
    Generates data for GRPO rollouts including completions and log probabilities.
    """
    prompts = batch_samples
    with torch.no_grad():
        point_cloud, prompt_ids, prompt_mask, is_pc, is_img, pixel_values_videos, video_grid_thw, completion_ids, completion_mask = generate_completions(
            model, processor, prompts, num_generations, max_completion_length
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        formatted_completions = [processor.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for ids in completion_ids]
        repeated_answers = [a for a in batch_samples['mesh_path'] for _ in range(num_generations)]

        rewards = torch.tensor(
            reward_function(completions=formatted_completions, answer=repeated_answers),
            dtype=torch.float32,
            device=model.device
        )
        # print("Rewards", rewards, flush=True)

        batch_size = len(prompts['input_ids'])
        num_generations = num_generations
        if top_samples is None:
            top_samples = num_generations
        rewards = rewards.view(batch_size, num_generations)
        avg_reward = rewards.mean().item()
        print("Average Reward:", avg_reward)
        mean_rewards = rewards.mean(dim=1).repeat_interleave(num_generations)

        # Expand buffer
        buffer_expand_size = batch_size // 2
        std_rewards = rewards.std(dim=1).view(-1)
        std_vals, std_indices = torch.topk(std_rewards, buffer_expand_size)
        dataset_indices = [batch_samples['idx'][int(i)] for i in std_indices]
        buffer.add_many(dataset_indices)

        abs_adv = torch.abs(rewards - mean_rewards.view(batch_size, num_generations))
        _, top_indices = torch.topk(abs_adv, top_samples, dim=1)

        row_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, top_samples).to(model.device)
        flattened_indices = row_indices * num_generations + top_indices

        advantages = (rewards.view(-1) - mean_rewards)[flattened_indices].reshape(batch_size * top_samples,
                                                                                  -1)  # .unsqueeze(1)

        input_ids = input_ids[flattened_indices].reshape(batch_size * top_samples, *input_ids.shape[1:])
        attention_mask = attention_mask[flattened_indices].reshape(batch_size * top_samples, *attention_mask.shape[1:])
        point_cloud = point_cloud[flattened_indices].reshape(batch_size * top_samples, *point_cloud.shape[1:])
        completion_mask = completion_mask[flattened_indices].reshape(batch_size * top_samples,
                                                                     *completion_mask.shape[1:])
        is_pc = is_pc[flattened_indices].reshape(batch_size * top_samples, *is_pc.shape[1:])
        is_img = is_img[flattened_indices].reshape(batch_size * top_samples, *is_img.shape[1:])
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos[flattened_indices].reshape(batch_size * top_samples,
                                                                       *pixel_values_videos.shape[1:])
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw[flattened_indices].reshape(batch_size * top_samples, *video_grid_thw.shape[1:])
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "formatted_completions": formatted_completions,
            "repeated_answers": repeated_answers,
            "logits_to_keep": logits_to_keep,
            "batch_size": len(prompts['input_ids']),
            "num_generations": num_generations,
            "point_cloud": point_cloud,
            "advantages": advantages,
            "is_pc": is_pc,
            "is_img": is_img,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }
        if not gpg:
            old_log_probs = compute_log_probs(model, (input_ids.clone(), attention_mask.clone(), point_cloud.clone(), is_pc.clone(), is_img.clone(), pixel_values_videos.clone() if pixel_values_videos is not None else None, video_grid_thw.clone() if video_grid_thw is not None else None),
                                              logits_to_keep)
            result["old_log_probs"] = old_log_probs.detach()
    return result, avg_reward


def grpo_loss(model, rollout_data, processor, reward_function, epsilon_high=0.2, epsilon_low=0.2, top_samples=None):
    """
    Computes the GRPO loss for updating the policy model.
    """
    device = model.device
    input_ids = rollout_data["input_ids"]
    point_cloud = rollout_data["point_cloud"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    advantages = rollout_data["advantages"]
    is_pc = rollout_data["is_pc"]
    is_img = rollout_data["is_img"]
    pixel_values_videos = rollout_data["pixel_values_videos"]
    video_grid_thw = rollout_data["video_grid_thw"]
    token_log_probs = compute_log_probs(model, (input_ids.clone(), attention_mask.clone(), point_cloud.clone(), is_pc.clone(), is_img.clone(), pixel_values_videos, video_grid_thw), logits_to_keep)
    ratio = torch.exp(token_log_probs - old_log_probs)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high) * advantages
    surrogate_loss = torch.min(surr1, surr2)
    per_token_loss = surrogate_loss
    loss = -torch.clamp(torch.nan_to_num(((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)), 0, 0,
                             0), min=-10, max=10).mean()
    return loss


def gpg_loss(model, rollout_data, tokenizer, reward_function, epsilon_high=0.2, epsilon_low=0.2, top_samples=None):
    device = model.device
    input_ids = rollout_data["input_ids"]
    point_cloud = rollout_data["point_cloud"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    advantages = rollout_data["advantages"]
    is_pc = rollout_data["is_pc"]
    is_img = rollout_data["is_img"]
    pixel_values_videos = rollout_data["pixel_values_videos"]
    video_grid_thw = rollout_data["video_grid_thw"]
    token_log_probs = compute_log_probs(model, (input_ids.clone(), attention_mask.clone(), point_cloud.clone(), is_pc.clone(), is_img.clone(), pixel_values_videos, video_grid_thw), logits_to_keep)
    per_token_loss = token_log_probs * advantages
    loss = -torch.nan_to_num(((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)), 0, 0,
                             0).mean()
    return loss


def merge_collated_batches(batch1, batch2, padding_value):
    merged = {}
    bs1 = batch1['input_ids'].shape[0]
    bs2 = batch2['input_ids'].shape[0]
    for key in batch1:
        if key not in batch2:
            batch2[key] = torch.zeros(bs2, *batch1[key].shape[1:], dtype=batch1[key].dtype)
        if key == 'input_ids':
            max_dim = max(batch1[key].shape[1], batch2[key].shape[1])
            pad1 = [max_dim - batch1[key].shape[1], 0]
            pad2 = [max_dim - batch2[key].shape[1], 0]
            batch1[key] = F.pad(batch1[key], pad1, value=padding_value)
            batch2[key] = F.pad(batch2[key], pad2, value=padding_value)
        elif key == 'attention_mask':
            max_dim = max(batch1[key].shape[1], batch2[key].shape[1])
            pad1 = [max_dim - batch1[key].shape[1], 0]
            pad2 = [max_dim - batch2[key].shape[1], 0]
            batch1[key] = F.pad(batch1[key], pad1, value=0)
            batch2[key] = F.pad(batch2[key], pad2, value=0)
        if isinstance(batch1[key], torch.Tensor):
            merged[key] = torch.cat([batch1[key], batch2[key]], dim=0)
        elif isinstance(batch1[key], list):
            merged[key] = batch1[key] + batch2[key]
        else:
            raise TypeError(f"Unsupported type for merging: {key}, {type(batch1[key])}")

    for key in batch2:
        if key not in merged:
            batch1[key] = torch.zeros(bs1, *batch2[key].shape[1:], dtype=batch2[key].dtype)
            merged[key] = torch.cat([batch1[key], batch2[key]], dim=0)
    return merged


def train_with_grpo_mm(model, processor, train_data, eval_data_deepcad, eval_data_fusion, eval_data_text, sampler, batch_size=4,
                    num_generations=4, top_samples=None, max_completion_length=128,
                    learning_rate=5e-6, batch_updates=3, epsilon_high=0.2, epsilon_low=0.2, train_epochs=1,
                    reward_function=None, collate_fn=None, run_id=None, gpg=False, use_buffer=False, save_path="./models"):
    """
    This function is your original working code (train_with_grpo_static)
    with an added outer loop for iterative GRPO updates per the pseudocode.
    """
    rank = dist.get_rank()
    rank = rank % torch.cuda.device_count()

    if top_samples is None:
        top_samples = num_generations

    step = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    loss_fn = grpo_loss if not gpg else gpg_loss
    for epoch in range(train_epochs):
        train_data.swap()
        dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler,
                                num_workers=30)
        buffer = IndexBuffer()
        print(f"\nEpoch {epoch + 1}/{train_epochs}")
        # Inner loop: your original training steps.
        for batch_samples in dataloader:
            if use_buffer and len(buffer) > 0:
                indices = buffer.sample(min(batch_size, len(buffer)))
                samples = [train_data[i] for i in indices]
                buffer_batch = collate_fn(samples)
                batch_samples = merge_collated_batches(batch_samples, buffer_batch, padding_value=processor.tokenizer.pad_token_id)
            rollout_data, avg_reward = generate_rollout_data(
                model.module,
                reward_function,
                processor,
                batch_samples,
                num_generations,
                max_completion_length,
                top_samples=top_samples,
                gpg=gpg,
                buffer=buffer,
            )
            for grpo_iter in range(batch_updates):
                loss = loss_fn(
                    model,
                    rollout_data,
                    processor,
                    reward_function,
                    epsilon_high=epsilon_high,
                    epsilon_low=epsilon_low,
                    top_samples=top_samples
                )


                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                # Log to wandb
                if rank == 0:
                    wandb.log({
                        "loss": loss.item(),
                        "average_reward": avg_reward,
                        "step": step + 1,
                        "grpo_iter": grpo_iter + 1,
                        "iter epoch": epoch + 1,
                    })
                    print(f"Epoch {epoch + 1}/{train_epochs}, Step {step + 1}/{len(dataloader)}, "
                          f"GRPO iter {grpo_iter + 1}/{batch_updates}, loss: {loss.item():.4f}")
            step += 1

        if rank == 0:
            eval_data_deepcad.mode = 'pc'
            eval_data_fusion.mode = 'pc'
            ious, cds, incorrect, failed_intersect = evaluate_model_mm(model.module, processor, eval_data_deepcad, 0,
                                                                    collate_fn,
                                                                    batch_size=200)
            ious_f, cds_f, incorrect_f, failed_intersect_f = evaluate_model_mm(model.module, processor, eval_data_fusion,
                                                                            0, collate_fn,
                                                                            batch_size=200)
            eval_data_deepcad.mode = 'img'
            eval_data_fusion.mode = 'img'
            ious_im, cds_im, incorrect_im, failed_intersect_im = evaluate_model_mm(model.module, processor, eval_data_deepcad, 0,
                                                                       collate_fn,
                                                                       batch_size=200)
            ious_f_im, cds_f_im, incorrect_f_im, failed_intersect_f_im = evaluate_model_mm(model.module, processor,
                                                                               eval_data_fusion,
                                                                               0, collate_fn,
                                                                               batch_size=200)

            # ious_txt, cds_txt, incorrect_txt, failed_intersect_txt = evaluate_model_mm(model.module, processor,
            #                                                                        eval_data_text, 0,
            #                                                                        collate_fn,
            #                                                                        batch_size=50)
            model.module.save_pretrained(f"{save_path}/{run_id}_{epoch}")
            processor.save_pretrained(f"{save_path}/{run_id}_{epoch}")

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
        dist.barrier()
    return model.module
