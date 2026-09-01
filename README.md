## `cadrille`: Multi-modal CAD Reconstruction with Online Reinforcement Learning

**News**:
 * :fire: May, 2025. `cadrille` is state-of-the-art in three CAD reconstruction benchmarks: DeepCAD, Fusion360, CC3D.
 
This repository contains an implementation of `cadrille`, a multi-modal (point clouds / images / text) 3D CAD reconstruction method introduced in our paper:

> **cadrille: Multi-modal CAD Reconstruction with Online Reinforcement Learning**<br>
> [Maksim Kolodiazhnyi](https://github.com/col14m),
> [Denis Tarasov](https://dt6a.github.io),
> [Dmitrii Zhemchuzhnikov](https://github.com/zhemdi),
> [Alexander Nikulin](https://howuhh.github.io),
> [Ilya Zisman](https://zis.mn),
> [Anna Vorontsova](https://highrut.github.io),
> [Anton Konushin](https://scholar.google.com/citations?user=ZT_k-wMAAAAJ),
> [Vladislav Kurenkov](https://dunnolab.ai),
> [Danila Rukhovich](https://github.com/filaPro) <br>
> https://arxiv.org/abs/2505.22914

### Installation

Install Python packages according to our [Dockerfile](Dockerfile). We support DeepCAD (test), Fusion360 (test), Text2CAD (train / val / test), and CAD-Recode (train, val) datasets. Follow our [instruction](data/README.md) to download and preprocess data.

### Train

To start training run *train.py* script:
```shell
python train.py --mode pc_img --use-text
```
To disable some of the modalities set *--mode* to *img* or *pc*, or disable *--use-text*. We don't provide RL fine-tuning code for now. Alternatively both [SFT](https://huggingface.co/maksimko123/cadrille) and [RL](https://huggingface.co/maksimko123/cadrille-rl) models can be downloaded from :hugs: HuggningFace.

### Inference

To predict CadQuery codes run *test.py* script:
```shell
python test.py --split deepcad_test_mesh --mode pc
```
To run on other datasets and modalities use *--split fusion360_test_mesh* or set *--mode* to *img* or *text*.

### Evaluation

To evaluate IoU, invalidity ratio, and chamfer distance run *evaluate.py* script:
```shell
python evaluate.py
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/8b811b14-e646-48d6-9a0c-06a9655bdbaf" alt="cadrille scheme"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/d6ae21f5-6c3c-4b7b-a2e9-ff0a310caa3d" alt="cadrille predictions"/>
</p>

### Citation

If you find this work useful for your research, please cite our paper:

```
@article{kolodiazhnyi2025cadrille,
  title={cadrille: Multi-modal CAD Reconstruction with Online Reinforcement Learning},
  author={Maksim Kolodiazhnyi, Denis Tarasov, Dmitrii Zhemchuzhnikov, Alexander Nikulin, Ilya Zisman, Anna Vorontsova, Anton Konushin, Vladislav Kurenkov, Danila Rukhovich},
  journal={arXiv preprint arXiv:2505.22914},
  year={2025}
}
```
