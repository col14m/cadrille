def patch_vllm_proccessor(processor):
    from types import MethodType
    from transformers.models.qwen2_vl import Qwen2VLImageProcessor, Qwen2VLProcessor
    from vllm.model_executor.models.qwen2_vl import Qwen2VLProcessingInfo

    def _patched_get_hf_processor(self, *, min_pixels=None, max_pixels=None) -> Qwen2VLProcessor:
        print(("getting hf base processor"))
        return processor

    def _patched_get_image_processor(self, *, min_pixels=None, max_pixels=None):
        print("getting image processor")
        proc = _patched_get_hf_processor(self, min_pixels=min_pixels, max_pixels=max_pixels)
        image_processor = proc.image_processor
        assert isinstance(image_processor, Qwen2VLImageProcessor)
        return image_processor

    # Bind methods to the class (affects all instances)
    Qwen2VLProcessingInfo.get_hf_processor   = MethodType(_patched_get_hf_processor, Qwen2VLProcessingInfo)
    Qwen2VLProcessingInfo.get_image_processor= MethodType(_patched_get_image_processor, Qwen2VLProcessingInfo)

def patch_vllm_group_port(group_port):
    from trl.extras.vllm_client import VLLMClient
    _orig_init = VLLMClient.__init__

    def _init(self, *args, **kwargs):
        kwargs["group_port"] = group_port  # force a non-default port
        return _orig_init(self, *args, **kwargs)

    VLLMClient.__init__ = _init

def patch_vllm_for_videos():
    """
    add videos to class GenerateRequest(BaseModel) in vllm_serve
    add videos to "multi_modal_data" argument in request
    add videos to generate() of vllm client

    """
    from trl.extras.vllm_client import VLLMClient
    from typing import Optional
    from io import BytesIO
    import base64
    
    def video_generate(
        self,
        prompts: list[str],
        images: Optional[list] = None,
        videos: Optional[list[list]] = None,
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
        generation_kwargs: Optional[dict] = None,
    ) -> list[list[int]]:
        """
        Generates model completions for the provided prompts.

        Args:
            prompts (`list[str]`):
                List of text prompts for which the model will generate completions.
            images (`list[PIL.Image]` or `None`, *optional*, defaults to `None`):
                List of PIL Images to send along with the prompts.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each prompt.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.`1.0` means no truncation.
            top_k (`int`, *optional*, defaults to `-1`):
                Top-k sampling parameter. `-1` means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each prompt.
            guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
                Regular expression to guide the decoding process.
            generation_kwargs (`dict` or `None`, *optional*, defaults to `None`):
                Additional generation parameters to pass to the vLLM `SamplingParams`. This can include parameters like
                `seed`, `frequency_penalty`, etc. If it contains keys that conflict with the other parameters, they
                will override them.

        Returns:
            `list[list[int]]`:
                List of lists of token IDs representing the model-generated completions for each prompt.
        """
        url = f"{self.base_url}/generate/"

        def pil_to_base64(image):
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            return base64.b64encode(img_bytes).decode("utf-8")

        # Convert PIL images to base64 strings
        images = [pil_to_base64(img) for img in images] if images else None
        videos = [[pil_to_base64(frame) for frame in frames] for frames in videos] if videos else None
        response = self.session.post(
            url,
            json={
                "prompts": prompts,
                "images": images,
                "videos": videos,
                "n": n,
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_tokens": max_tokens,
                "guided_decoding_regex": guided_decoding_regex,
                "generation_kwargs": generation_kwargs or {},
            },
        )
        if response.status_code == 200:
            return response.json()["completion_ids"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    
    VLLMClient.generate = video_generate
