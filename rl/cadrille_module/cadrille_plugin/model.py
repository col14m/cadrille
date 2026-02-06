from vllm.model_executor.models.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLMultiModalProcessor, Qwen2VLProcessingInfo, Qwen2VLDummyInputsBuilder
from vllm.multimodal import MULTIMODAL_REGISTRY
from .cadrille import Cadrille, FourierPointEncoder
from vllm import ModelRegistry
from transformers import AutoProcessor, Qwen2VLImageProcessor
import torch
import logging
import types
from vllm.transformers_utils.config import uses_mrope
logger = logging.getLogger("vllm.cadrille")

print(f"Registering model")


@MULTIMODAL_REGISTRY.register_processor(Qwen2VLMultiModalProcessor,
                                        info=Qwen2VLProcessingInfo,
                                        dummy_inputs=Qwen2VLDummyInputsBuilder)
class CadrilleForCausalLM(Qwen2VLForConditionalGeneration):
    print(f"instantiating class")
    @classmethod
    def from_engine_args(cls, engine_args, **kwargs):
        model = Cadrille.from_pretrained(
            engine_args.model,  # path to safetensors + config.json
            torch_dtype=engine_args.dtype,
            device_map=None,  # vLLM handles placement
        )
        return cls(model.config, model=model)

    def __init__(self, *, vllm_config, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # create your extra module here so it’s part of this nn.Module
        self.point_encoder = FourierPointEncoder(config.hidden_size)
        self.visual.get_dtype = types.MethodType(lambda m: m.dtype, self.visual)
    """
    @property
    def model(self):
        return self.language_model.model


    @property
    def lm_head(self):
        return self.language_model.lm_head"""

    def _forward_hf(self, **hf_kwargs):
        debug_states(**hf_kwargs)
        return self.model(**hf_kwargs)


    def forward(self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors = None,
            inputs_embeds = None, **kwargs):   
            
        if intermediate_tensors is not None:
            inputs_embeds = None

        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            video_input = self._parse_and_validate_video_input(**kwargs)

            if image_input is None and video_input is None:
                inputs_embeds = None
            else:
                if uses_mrope(self.config):
                    assert positions.ndim == 2 and positions.size(0) == 3, (
                        "multimodal section rotary embedding requires "
                        f"(3, seq_len) positions, but got {positions.size()}")
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                    video_input=video_input)
                input_ids = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states



    def load_weights(self, weights):
        pc_w, pc_b = None, None
        def filtered():
            nonlocal pc_w, pc_b  
            for name, w in weights:
                if name.endswith("point_encoder.projection.weight"):
                    pc_w = w
                    continue
                elif name.endswith("point_encoder.projection.bias"):
                    pc_b = w
                    continue
                yield name, w
        
        super().load_weights(filtered())

        if pc_b is not None:
            p = self.point_encoder.projection.bias
            t = pc_b.to_torch(dtype=p.dtype, device=p.device) if hasattr(pc_b, "to_torch") else pc_b.to(p.dtype).to(p.device)
            with torch.no_grad():
                p.copy_(t, non_blocking=True)

        if pc_w is not None:
            p = self.point_encoder.projection.weight
            t = pc_w.to_torch(dtype=p.dtype, device=p.device) if hasattr(pc_w, "to_torch") else pc_w.to(p.dtype).to(p.device)
            with torch.no_grad():
                p.copy_(t, non_blocking=True)


