
# XNet v2: Fewer limitations, Better Results and Greater Universality

This is the official code of [XNet v2: Fewer limitations, Better Results and Greater Universality](https://).

## Overview

<p align="center">
<img src="https://i.postimg.cc/Mptz9DBJ/figure-1.png#pic_center" width="100%" ></img>
<center>Architecture of XNet v2</center>
</p>

## Quantitative Comparison
Comparison with semi-supervised state-of-the-art models on GlaS, CREMI and ISIC-2017 test set. All models are trained with 20% labeled images and 80% unlabeled images, which is the common semi-supervised experimental partition. <font color="Red">**Red**</font> and **bold** indicate the best and second best performance.
<p align="center">
<img src="https://i.postimg.cc/zG4hpKR7/2D.png#pic_center" width="100%" >
</p>

Comparison with semi-supervised state-of-the-art models on P-CT and LiTS test set. All models are trained with 20% labeled images and 80% unlabeled images. Due to GPU memory limitations, some semi-supervised models using smaller architectures, ✝ indicates models are based on lightweight 3D UNet (half of channels). - indicates training failed. <font color="Red">**Red**</font> and **bold** indicate the best and second best performance.
<p align="center">
<img src="https://i.postimg.cc/zG4hpKR7/2D.png#pic_center" width="100%" >
</p>

Comparison with fully-supervised XNet on GlaS, CREMI, ISIC-2017, P-CT and LiTS test set.
<p align="center">
<img src="https://i.postimg.cc/zG4hpKR7/2D.png#pic_center" width="100%" >
</p>

## Requirements
```
albumentations==1.2.1
MedPy==0.4.0
numpy==1.21.5
opencv_python_headless==4.5.4.60
Pillow==9.4.0
PyWavelets==1.3.0
scikit_learn==1.2.1
scipy==1.7.3
SimpleITK==2.2.1
torch==1.8.0+cu111
torchio==0.18.84
torchvision==0.9.0+cu111
visdom==0.1.8.9
```

## Usage
**Data preparation**
Your datasets directory tree should be look like this:
```
dataset
├── train_sup_100
    ├── image
        ├── 1.tif
        ├── 2.tif
        └── ...
    └── mask
        ├── 1.tif
        ├── 2.tif
        └── ...
├── train_sup_20
    ├── image
    └── mask
├── train_unsup_80
    ├── image
└── val
    ├── image
    └── mask
```

**Supervised training**
```
python -m torch.distributed.launch --nproc_per_node=4 train_sup_XNetv2.py
```
**Semi-supervised training**
```
python -m torch.distributed.launch --nproc_per_node=4 train_semi_XNetv2.py
```
**Testing**
```
python -m torch.distributed.launch --nproc_per_node=4 test.py
```

## Citation
If our work is useful for your research, please cite our paper:
```
```

