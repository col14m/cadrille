# Data

DeepCAD and Fusion360 test files must be placed into corresponding directories in `data`.

DeepCAD and Fusion360 train files must be placed into `data/deepcad_fusion_train`.

# Running training
Launch with
```
torchrun --standalone --nnodes=1 --nproc-per-node=8 train_cadrille_grpo.py --sft_path=INIT_MODEL_PATH [--other_args]
```
