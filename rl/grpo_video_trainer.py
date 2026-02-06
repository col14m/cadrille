import copy
import inspect
import os
import re
import textwrap
from collections import defaultdict, deque
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union

from copy import deepcopy
import datasets
import torch
import torch.utils.data
import transformers
from accelerate import logging
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from trl.extras.vllm_client import VLLMClient

from trl.trainer.utils import (
    disable_dropout_in_model,
    entropy_from_logits,
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)

from trl.trainer.grpo_trainer import nanmin, nanmax, nanstd, RepeatSampler, shuffle_sequence_dict, split_tensor_dict
from transformers import Trainer
from trl import GRPOTrainer, GRPOConfig, TrlParser
from trl.extras.profiling import profiling_context, profiling_decorator

from grpo_loss import cppo_compute_loss, adv_select_top_samples

def unsplit_pixel_values_by_grid(batch):
    """
    Opposite of `split_pixel_values_by_grid`. Merges a list of tensors in `batch["pixel_values"]` back into a single
    tensor along the first dimension.
    """
    video = "pixel_values_videos" in batch

    values_key = "pixel_values_videos" if video else "pixel_values"
    values = batch.get(values_key)
    if isinstance(values, list):
        merged = torch.cat(values, dim=0)
        return {**batch, values_key: merged}
    else:
        return batch


def split_pixel_values_by_grid(batch):
    video = "video_grid_thw" in batch
    values_key = "pixel_values_videos" if video else "pixel_values"
    grid_key = "video_grid_thw" if video else "image_grid_thw"
    if grid_key not in batch or values_key not in batch:
        return batch

    lengths = batch[grid_key].prod(dim=1).tolist()  # [batch_size]
    pixel_values = batch[values_key]  # [total, feature_dim]

    if sum(lengths) != pixel_values.size(0):
        raise ValueError(f"Mismatch: sum(lengths) = {sum(lengths)} != pixel_values.size(0) = {pixel_values.size(0)}")

    split_values = list(torch.split(pixel_values, lengths, dim=0))
    return {**batch, values_key: split_values}



class VideoTopSampleGRPOTrainer(GRPOTrainer):
    """
    additional modifications :
        - text inputs must be passed pre processed up until tokenization. that includes applying chat template, adding multimodal tokens
        - dr CPPO loss, only train on top samples based on advantages
        - 
    """
    def __init__(self, top_samples, clip_cov=False, **kwargs):
        #kwargs[""]
        super().__init__(**kwargs)
        self.top_samples = top_samples
        self.clip_cov = clip_cov

    # custom loss
    def compute_loss(self, model, inputs, return_outputs=False,  **kwargs):
        self.model.gradient_checkpointing_enable()
        return cppo_compute_loss(self, model, inputs, return_outputs=return_outputs, clip_cov=self.clip_cov)
    
    def select_top_samples(self, inputs, num_generations: int, top_samples):
        return adv_select_top_samples(self, inputs, num_generations, top_samples)

    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
        point_clouds=None,
        is_pc=None,
        is_img=None
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values

        if video_grid_thw is not None and pixel_values_videos is not None:
            model_inputs["video_grid_thw"] = video_grid_thw
        if pixel_values_videos is not None:
            model_inputs["pixel_values_videos"] = pixel_values_videos

        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes
    
        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state


    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
        video_grid_thw=None,
        pixel_values_videos=None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Compute log-probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}

            if image_grid_thw is not None and pixel_values is not None:
                model_inputs["image_grid_thw"] = image_grid_thw[start : start + batch_size]
                start_pixel_idx = image_grid_thw[:start].prod(-1).sum().item()
                end_pixel_idx = image_grid_thw[: start + batch_size].prod(-1).sum().item()
                model_inputs["pixel_values"] = pixel_values[start_pixel_idx:end_pixel_idx]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]

            # videos
            if video_grid_thw is not None and pixel_values_videos is not None:
                model_inputs["video_grid_thw"] = video_grid_thw[start : start + batch_size]
                start_pixel_idx = video_grid_thw[:start].prod(-1).sum().item()
                end_pixel_idx = video_grid_thw[: start + batch_size].prod(-1).sum().item()
                model_inputs["pixel_values_videos"] = pixel_values_videos[start_pixel_idx:end_pixel_idx]
            elif pixel_values_videos is not None:
                model_inputs["pixel_values_videos"] = pixel_values_videos[start : start + batch_size]
            
            #  add modalities for Cadrille
            if pixel_values_videos is not None:
                model_inputs["point_clouds"] = None
                model_inputs["is_pc"] = torch.zeros(batch_size, dtype=torch.bool, device=model.device)
                model_inputs["is_img"] = torch.ones(batch_size,  dtype=torch.bool, device=model.device)

            
            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            logits = model(**model_inputs).logits
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies


    def _prepare_inputs(self, generation_batch):
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            # 1 is self.args.steps_per_generation
            generate_every = 1 * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, 1)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % 1]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs


    
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        repeated_inputs = self.repeat_inputs(inputs)
        inputs = repeated_inputs
        prompts_text = [x["prompt"] for x in inputs]
        

        # We don't yet support visual reward models/function, so we keep a copy of the original text-only prompts for
        # later use in the reward computation. If images are present, we insert {"type": "image"} as required by the
        # VLM chat template.
        original_prompts = copy.deepcopy(prompts_text)

        kwargs = {}
        has_images = "image" in inputs[0]
        if has_images:
            images = [example.get("image") for example in inputs]
            kwargs = {"images": [[img] for img in images]}

        has_videos = "video" in inputs[0]
        if has_videos:
            videos = [example.get("video") for example in inputs]
            # ADD TWO FRAMES AS IT IS DONE BY PROCESS VISION INFO OF QWEN UTILS
            kwargs = {"videos": [[vid[0], vid[0]] for vid in videos]}

        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs,
        )
        # add modalities for Cadrille
        if has_videos:
            prompt_inputs["is_pc"] = torch.zeros_like(prompt_inputs["pixel_values_videos"], dtype=torch.bool, device=device)
            prompt_inputs["is_img"] = torch.ones_like(prompt_inputs["pixel_values_videos"], dtype=torch.bool, device=device)
            prompt_inputs["point_clouds"] = None

        # sends inputs to device
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if has_images:
                    all_images = gather_object(images)
                if has_videos:
                    all_videos = gather_object(videos)

                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]

                    if has_images:
                        ordered_set_of_images = all_images[:: self.num_generations]
                    else:
                        ordered_set_of_images = None

                    if has_videos:
                        ordered_set_of_videos = all_videos[:: self.num_generations]
                    else:
                        ordered_set_of_videos = None
                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            images=ordered_set_of_images,
                            videos=ordered_set_of_videos,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                # Broadcast the completions from the main process to all processes, ensuring each process receives its
                # corresponding slice.
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts_text),
                    (self.accelerator.process_index + 1) * len(prompts_text),
                )
                completion_ids = completion_ids[process_slice]
            
            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                nullcontext(),
            ):
                unwrapped_model.gradient_checkpointing_disable()

                prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = prompt_ids, prompt_mask
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs, generation_config=self.generation_config, disable_compile=True
                )
            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
        
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]
        ### remove special tokens not in base vocabulary
        if (completion_ids > 151643).any():
            completion_ids = completion_ids.masked_fill(((completion_ids > 151643).cumsum(dim=1) > 0), self.eos_token_id)

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)
        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = 1 * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                    pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                    image_sizes=prompt_inputs.get("image_sizes"),
                    pixel_values_videos=prompt_inputs.get("pixel_values_videos"),
                    video_grid_thw=prompt_inputs.get("video_grid_thw"),
                )
            else:
                old_per_token_logps = None

            # Compute the per-token log probabilities for the reference model
            ref_per_token_logps = None
        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions = completions_text
        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, original_prompts, completions, completion_ids_list)
        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts_text),
            (self.accelerator.process_index + 1) * len(prompts_text),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if has_images:
            self._logs["image"].extend(gather_object(images))

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in prompt_inputs:
            output["pixel_values"] = prompt_inputs["pixel_values"]
        if "image_grid_thw" in prompt_inputs:
            output["image_grid_thw"] = prompt_inputs["image_grid_thw"]
        if "pixel_attention_mask" in prompt_inputs:
            output["pixel_attention_mask"] = prompt_inputs["pixel_attention_mask"]
        if "image_sizes" in prompt_inputs:
            output["image_sizes"] = prompt_inputs["image_sizes"]
        if "pixel_values_videos" in prompt_inputs:
            output["pixel_values_videos"] = prompt_inputs["pixel_values_videos"]
        if "video_grid_thw" in prompt_inputs:
            output["video_grid_thw"] = prompt_inputs["video_grid_thw"]
        
        return output
    
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME, check_torch_load_is_safe
        from transformers.trainer import logger
        from transformers.configuration_utils import PretrainedConfig
        from transformers import __version__
        import safetensors.torch

        if model is None:
            model = self.model

        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)

        if not (
            any(
                os.path.isfile(f)
                for f in [
                    weights_file,
                    safe_weights_file,
                    weights_index_file,
                    safe_weights_index_file
                ]
            )
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(config_file):
            config = PretrainedConfig.from_json_file(config_file)
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file):
            # We load the model state dict on the CPU to avoid an OOM error.
            if self.args.save_safetensors and os.path.isfile(safe_weights_file):
                state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")
            else:
                check_torch_load_is_safe()
                state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)

            state_dict = _remap_qwen2vl_keys(state_dict)
            # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
            # which takes *args instead of **kwargs
            load_result = model.load_state_dict(state_dict, False)
            if set(load_result.missing_keys) == {"lm_head.weight"}: 
                model.tie_weights()
            # release memory
            del state_dict
            self._issue_warnings_after_load(load_result)
        else:
            raise NotImplementedError("Can't load sharded model for now")

    def _get_train_sampler(self, dataset=None):
        return RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=1, # override trainer RepeatSampler to manually repeat
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def repeat_inputs(self, inputs):
    # Manually repeat inputs on every device
        return [deepcopy(d) for d in inputs for _ in range(self.num_generations)]


        
def _remap_qwen2vl_keys(sd):
    import re
    f = lambda k: re.sub(r"^model(?!\.(language_model|visual))\.", "model.language_model.",
                 re.sub(r"^visual\.", "model.visual.",
                    (k[7:] if k.startswith("module.") else k).replace("_orig_mod.", "")))
    return {f(k): v for k, v in sd.items()}