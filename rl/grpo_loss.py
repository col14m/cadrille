import torch

from trl.trainer.grpo_trainer import nanmin, nanmax


def adv_select_top_samples(self, inputs, num_generations: int, top_samples:int):
    completion_ids = inputs["completion_ids"]
    print(f"completion_ids shape {completion_ids.shape} ")
    G = self.num_generations
    B = completion_ids.size(0) # G * batch size per device / num_iterations --- 16 * 4 = 64
    assert B % G == 0, f"inputs must be B * num_generations, got B : {B} G : {G}"
    num_prompts = B // G
    device = completion_ids.device

    adv = inputs["advantages"].view(B, -1).squeeze(-1)        # (B,)
    abs_adv = adv.abs().view(num_prompts, G)              # (N, G)
    print(f"top samples {top_samples}")
    _, top_indices = torch.topk(abs_adv, top_samples, dim=1)
    row_indices = torch.arange(num_prompts, device=device).unsqueeze(1).expand(-1, top_samples)
    flat = (row_indices * G + top_indices).reshape(-1)
    def take(x):
        return x[flat] if x is not None else None

    new_inputs = {}
    for k, v in inputs.items():
        if v is None:
            new_inputs[k] = None
            continue
        # pixel_values might be packed, handle separately
        if k == "pixel_values" or k == "pixel_values_videos":
            P = v.size(0) // B
            D = v.size(1)
            new_inputs[k] = v.view(B, -1, D)[flat].reshape(-1, D)
        else:
            new_inputs[k] = take(v)

    return new_inputs


def cppo_compute_loss(self, model, inputs, return_outputs=False, clip_cov=False, has_videos = False):
        completion_ids_full = inputs["completion_ids"]
        logits_to_keep = completion_ids_full.size(1)  # we only need to compute the logits for the completion tokens
        
        sel_inputs = self.select_top_samples(inputs, num_generations=self.num_generations, top_samples=self.top_samples)

        prompt_ids, prompt_mask = sel_inputs["prompt_ids"], sel_inputs["prompt_mask"]
        completion_ids, completion_mask = sel_inputs["completion_ids"], sel_inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        print(f"new completion_ids shape {completion_ids.shape}")
        # ----------------------------------------

        # Compute the per_token_logps and the entropy at each position in the completion
        
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=sel_inputs.get("pixel_values"),
            image_grid_thw=sel_inputs.get("image_grid_thw"),
            pixel_attention_mask=sel_inputs.get("pixel_attention_mask"),
            image_sizes=sel_inputs.get("image_sizes"),
            video_grid_thw=sel_inputs.get("video_grid_thw"),
            pixel_values_videos=sel_inputs.get("pixel_values_videos"),
        )


        # Compute the loss
        advantages = sel_inputs["advantages"]
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.

        old_per_token_logps = sel_inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )
        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)


        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        if clip_cov:
            cov_lb = 1
            cov_hb = 5
            select_ratio = 2e-4
            covs = (per_token_logps - per_token_logps.mean()) * (advantages.unsqueeze(1) - advantages.mean())
            mask = (covs > cov_lb) & (covs < cov_hb)
            all_idx = torch.nonzero(mask).reshape(-1)
            select_num = int(select_ratio * per_token_logps.numel())

            if all_idx.numel() >= select_num > 0:
                perm= torch.randperm(all_idx.numel(), device=all_idx.device)
                clip_idx = all_idx[perm[:select_num]]
                # remove gradients from high entropy covariance tokens
                per_token_loss1[clip_idx] = per_token_loss1[clip_idx].detach()
                per_token_loss2[clip_idx] = per_token_loss2[clip_idx].detach()

        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        
        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss


