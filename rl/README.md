In order to train with TRL, we need to have a HuggingFace format dataset made of prompt / image pairs.
If we do not have one, we first convert it. 
then train with grpo_cadrille
two options 
- with vLLM : set use_vllm=true
also requires installing cadrille_module with pip to register Cadrille architecture to vLLM registry
this is faster, but requires launching a separate generation server with 
CUDA_VISIBLE_DEVICES=0 python rl/vllm_video_server.py --model /workspace-SR008.nfs2/users/barannikov/cadrille/models/cadrille --port 8000 --max_model_len 700
- without : set use_vllm=false

 