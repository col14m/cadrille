# PyTorch and related libraries for deep learning

# Hugging Face libraries for transformer models

# CAD recode imports

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import Qwen2VLForConditionalGeneration
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast


class FourierEmbedder(nn.Module):
    def __init__(self,
                 num_freqs=6,
                 logspace=True,
                 include_input=True,
                 include_pi=True):
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32)

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer('frequencies', frequencies, persistent=False)
        self.include_input = include_input

    def forward(self, x):
        embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
        if self.include_input:
            return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class FourierPointEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fourier_embedder = FourierEmbedder(num_freqs=8, include_pi=False)
        self.projection = nn.Linear(51, hidden_size)

    def forward(self, points):
        x = self.fourier_embedder(points[..., :3])
        x = self.projection(x)
        #ps = sum([p.sum() for p in self.projection.parameters()])
        return x #+ ps * 0


class Cadrille(Qwen2VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        torch.set_default_dtype(torch.float32)
        self.point_encoder = FourierPointEncoder(config.hidden_size)
        torch.set_default_dtype(torch.bfloat16)

    def freeze_pc(self):
        for param in self.point_encoder.parameters():
            param.requires_grad = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
            rope_deltas=None,
            cache_position=None,
            point_clouds=None,
            is_pc=None,
            is_img=None,
            **kwargs,
            ):

        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        
        prefill = past_key_values is None or past_key_values.get_seq_length() == 0

        # If no point clouds are used, defer to the stock v4.56 flow.
        use_pc = is_pc is not None and is_pc.sum() > 0
        if not use_pc and inputs_embeds is None:
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
                **kwargs,
            )
        else:
            if inputs_embeds is None:
                inputs_embeds = self.model.get_input_embeddings()(input_ids)  # text embeddings

            if pixel_values is not None:
                image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(f"Image tokens ({n_image_tokens}) and features ({n_image_features}) mismatch.")
                image_mask, _ = self.model.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if is_img.sum() > 0 and pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                _, video_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if position_ids is None:
                if self.model.rope_deltas is None or cache_position is None or cache_position[0] == 0:
                    position_ids, rope_deltas = self.model.get_rope_index(
                        input_ids, image_grid_thw, video_grid_thw, attention_mask
                    )
                    self.model.rope_deltas = rope_deltas

                # then use the prev pre-calculated rope-deltas to get the correct position ids
                else:
                    batch_size, seq_length, _ = inputs_embeds.shape
                    position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                    position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                    if cache_position is not None:
                        delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
                    else:
                        delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    position_ids += delta.to(position_ids.device)

            if prefill and is_pc.sum() > 0:
                pc_emb = self.point_encoder(point_clouds.float()).to(inputs_embeds.dtype).to(inputs_embeds.device)  # [B,Tpc,H]
                start = attention_mask.shape[1] - attention_mask.sum(dim=1)  # right-pad assumption
                Tpc = pc_emb.shape[1]
                for i, s in enumerate(start.tolist()):
                    if is_pc[i]:
                        inputs_embeds[i, s:s+Tpc, :] = pc_emb[i]

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

            pixel_values, pixel_values_videos = None, None
            input_ids = None  # because we’re passing inputs_embeds explicitly

            lm_outputs = self.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
                **kwargs,
            )

            outputs = Qwen2VLModelOutputWithPast(
                last_hidden_state=lm_outputs.last_hidden_state,
                past_key_values=lm_outputs.past_key_values,
                hidden_states=lm_outputs.hidden_states,
                attentions=lm_outputs.attentions,
                rope_deltas=self.rope_deltas,
            )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )


    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        point_clouds=None,
        is_pc=None,
        is_img=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        # v4.56: position ids + rope_deltas are managed here by the base class we just called.

        # Ensure custom fields are propagated to decoding steps
        model_inputs["is_pc"] = is_pc
        model_inputs["is_img"] = is_img
        model_inputs["point_clouds"] = point_clouds

        if model_inputs["cache_position"][0] != 0:
            model_inputs['point_clouds'] = None

        return model_inputs



class CADRecodeMM(Qwen2VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        torch.set_default_dtype(torch.float32)
        self.point_encoder = FourierPointEncoder(config.hidden_size)
        torch.set_default_dtype(torch.bfloat16)

    def freeze_pc(self):
        for param in self.point_encoder.parameters():
            param.requires_grad = False
            
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
            rope_deltas=None,
            cache_position=None,
            point_clouds=None,
            is_pc=None,
            is_img=None):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if is_img.sum() > 0 and pixel_values_videos is not None:
                #pixel_values_videos = pixel_values_videos[is_img]
                pixel_values_videos = pixel_values_videos.view(-1, pixel_values_videos.shape[-1])
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                #video_grid_thw = video_grid_thw[is_img]
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # add point cloud embeddings, we add only next 6 lines
            if is_pc.sum() > 0 and (past_key_values is None or past_key_values.get_seq_length() == 0):
                point_embeds = self.point_encoder(point_clouds.float()).bfloat16()
                start_idxs = attention_mask.shape[1] - attention_mask.sum(axis=1)
                for i, start_idx in enumerate(start_idxs):
                    if is_pc[i]:
                        inputs_embeds[i, start_idx:start_idx + point_embeds.shape[1], :] = point_embeds[i]

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                    (cache_position is not None and cache_position[0] == 0)
                    or self.rope_deltas is None
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output


        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        model_inputs['point_clouds'] = kwargs['point_clouds']
        model_inputs['is_pc'] = kwargs['is_pc']
        model_inputs['is_img'] = kwargs['is_img']
        return model_inputs
