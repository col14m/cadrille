import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast


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


def collate(batch, processor, n_points, eval=False):
    messages = []
    is_pc = [0] * len(batch)
    is_img = [0] * len(batch)
    if not eval:
        for i, m in enumerate(batch):
            if 'video' in m.keys():
                is_img[i] = 1
                message = [{
                        'role': 'user',
                        'content': [
                            {'type': 'video', 'video': m['video'], 'fps': 1.0},
                            {'type': 'text', 'text': m['description']}
                        ]
                    },
                    {
                        'role': 'assistant',
                        'content': [
                            {'type': 'text', 'text': m['answer']}
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
                    },
                    {
                        'role': 'assistant',
                        'content': [
                            {'type': 'text', 'text': m['answer']}
                        ]
                    }]
            messages.append(message)
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) 
            for msg in messages] 
    else:
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
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
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
        return_tensors='pt')

    inputs['point_clouds'] = torch.stack([
        torch.tensor(m['point_cloud']) if is_pc[i] else torch.zeros(n_points, 3)
        for i, m in enumerate(batch)])
    inputs['is_pc'] = torch.tensor(is_pc, dtype=torch.bool)
    inputs['is_img'] = torch.tensor(is_img, dtype=torch.bool)

    if 'pixel_values_videos' in inputs.keys():
        pixel_values_videos = inputs['pixel_values_videos'].new_zeros((
            len(batch), torch.prod(inputs['video_grid_thw'][0]),
            inputs['pixel_values_videos'].shape[1]))
        pixel_values_videos[inputs['is_img']] = torch.stack(
            torch.chunk(inputs['pixel_values_videos'], 
            chunks=sum(inputs['is_img'])))
        inputs['pixel_values_videos'] = pixel_values_videos
    
        video_grid_thw = inputs['video_grid_thw'].new_zeros((len(batch), 3))
        video_grid_thw[inputs['is_img']] = inputs['video_grid_thw']
        inputs['video_grid_thw'] = video_grid_thw

    if not eval:
        input_ids_lists = inputs['input_ids'].tolist()
        assert len(messages) == len(input_ids_lists)

        labels_list = []
        for ids_list in input_ids_lists:
            label_ids = [-100] * len(ids_list)
            for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
                label_ids[begin_end_indexs[0] + 2: begin_end_indexs[1] + 1] = \
                    ids_list[begin_end_indexs[0] + 2: begin_end_indexs[1] + 1]
            labels_list.append(label_ids)
        labels_ids = torch.tensor(labels_list, dtype=torch.int64)
        inputs['labels'] = labels_ids
    else:
        if 'file_name' in batch[0]:
            inputs['file_name'] = [m['file_name'] for m in batch]
        else:
            inputs['mesh_path'] = [m['mesh_path'] for m in batch]
            inputs['mesh'] = [m['mesh'] for m in batch]
            inputs['idx'] = [m['idx'] for m in batch]
    return inputs


def find_assistant_content_sublist_indexes(l):
    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 1):
        # Check if the current and next element form the start sequence
        if l[i] == 151644 and l[i + 1] == 77091:
            start_indexes.append(i)
            # Now look for the first 151645 after the start
            for j in range(i + 2, len(l)):
                if l[j] == 151645:
                    end_indexes.append(j)
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))


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
        return x
    

class Cadrille(Qwen2VLForConditionalGeneration):    
    def __init__(self, config):
        super().__init__(config)
     
        torch.set_default_dtype(torch.float32)
        self.point_encoder = FourierPointEncoder(config.hidden_size)
        torch.set_default_dtype(torch.bfloat16)

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
                pixel_values_videos = pixel_values_videos[is_img]
                pixel_values_videos = pixel_values_videos.view(-1, pixel_values_videos.shape[-1])
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_grid_thw = video_grid_thw[is_img]
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
