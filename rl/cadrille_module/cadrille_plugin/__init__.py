def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model("Cadrille", "cadrille_plugin.model:CadrilleForCausalLM")

    from transformers import AutoProcessor
    from vllm.model_executor.models.qwen2_vl import Qwen2VLProcessingInfo

    processor = AutoProcessor.from_pretrained("/workspace-SR008.nfs2/users/barannikov/cadrille/models/cadrille", padding='longest', use_fast=False)    
    processor.tokenizer.padding_side = "left"

    processor.image_processor.size.update({
        "shortest_edge": 3136,
        "longest_edge": 12845056,
    })

    def _patched_get_hf_processor(self, *, min_pixels=None, max_pixels=None):
        return processor

    def _patched_get_image_processor(self, *, min_pixels=None, max_pixels=None):
        proc = _patched_get_hf_processor(self, min_pixels=min_pixels, max_pixels=max_pixels)
        image_processor = proc.image_processor
        return image_processor

    # Bind methods to the class (affects all instances)
    Qwen2VLProcessingInfo.get_hf_processor   = _patched_get_hf_processor
    Qwen2VLProcessingInfo.get_image_processor= _patched_get_image_processor