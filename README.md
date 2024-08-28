
# XNet v2: Fewer limitations, Better Results and Greater Universality

This is the official code of [XNet v2: Fewer limitations, Better Results and Greater Universality](https://) (BIBM 2024).

The corresponding oral video demonstration is [here](https://).

## Limitations of XNet

- **Performance Degradation with Hardly HF Information**
	>XNet emphasizes high-frequency (HF) information. When images hardly have HF information, XNet performance is negatively impacted.
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/XNetv2/blob/main/figure/Performance%20Degradation%20with%20Hardly%20HF%20Information_1.png" width="60%" >
</p>
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/XNetv2/blob/main/figure/Performance%20Degradation%20with%20Hardly%20HF%20Information_2.png" width="60%" >
</p>

- **Underutilization of Raw Image Information**
	>XNet only uses low-frequency (LF) and HF images as input. Raw images are not involved in training. Although LF and HF information can be fused into complete information in fusion module, the raw image may still contain useful but unappreciated information.
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/XNetv2/blob/main/figure/Underutilization%20of%20Raw%20Image%20Information.png" width="60%" >
</p>

- **Insufficient Fusion**
	>XNet only uses deep features for fusion. Shallow feature fusion and image-level fusion are also necessary.


## XNet v2
- **Overview**
  <p align="center">
  <img src="https://github.com/Yanfeng-Zhou/XNetv2/blob/main/figure/Overview.png" width="100%" >
  </p>
 
  >
  >$$L_{total}=L_{sup}+\lambda L_{unsup}$$
  >
  >$$L_{sup}=L_{unsup}^M(p_{i}^{M}, y_i)+L_{unsup}^L(p_{i}^{L}, y_i)+L_{unsup}^H(p_{i}^{H}, y_i)$$
  >
  >$$L_{unsup} = L_{unsup}^{M,L}(p_{i}^M, p_{i}^{L})+L_{unsup}^{M,H}(p_{i}^M, p_{i}^{H})$$

- **Image-level Fusion**
	>Different from XNet, after using wavelet transform to generate 	  	LF image $I_L$ and HF image $I_H$, we fuse them in different ratios to generate complementary image $x_L$ and $x_H$. $x_L$ and $x_H$ are defined as:
	>$$x_L=I_L+\alpha I_H,$$
	>
	>$$x_H=I_H+\beta I_L.$$
	>
	>The input of XNet is a special case when $α=β=0$, but our definition is a more general expression.
	>This strategy achieves image-level information fusion. More importantly, it solves the limitation of XNet not working with less HF information. To be specific, when hardly have HF information, i.e., $I_H \approx 0$:
	>$$x_L=I_L+\alpha I_H \approx I_L,$$
	>
	>$$x_H=I_H+\beta I_L \approx \beta I_L \approx \beta x^L.$$
	>
	>$x^H$ degenerates into a perturbation form of $x^L$, which can be regarded as consistent learning of two different LF perturbations. It effectively overcomes the failure to extract features when HF information is scarce.
	<p align="center">
	<img src="https://github.com/Yanfeng-Zhou/XNetv2/blob/main/figure/Image-level%20Fusion.png" width="100%" >
	</p>

- **Feature-Level Fusion**
	<p align="center">
	<img src="https://github.com/Yanfeng-Zhou/XNetv2/blob/main/figure/Feature-Level%20Fusion.png" width="70%" >
	</p>

## Quantitative Comparison
- **Semi-Supervision**
	<p align="center">
	<img src="https://github.com/Yanfeng-Zhou/XNetv2/blob/main/figure/Semi-Supervision_1.png" width="100%" >
	</p>

	<p align="center">
	<img src="https://github.com/Yanfeng-Zhou/XNetv2/blob/main/figure/Semi-Supervision_2.png" width="100%" >
	</p>
	
- **Fully-Supervision**
	<p align="center">
	<img src="https://github.com/Yanfeng-Zhou/XNetv2/blob/main/figure/Fully-Supervision.png" width="50%" >
	</p>

## Qualitative Comparison
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/XNetv2/blob/main/figure/Qualitative%20Comparison.png" width="100%" >
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
Build your own dataset and its directory tree should be look like this:
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

**Configure dataset parameters**
>Add configuration in [/config/dataset_config/dataset_config.py](https://github.com/Yanfeng-Zhou/XNetv2/tree/main/config/dataset_config/dataset_config.py)
>The configuration should be as follows：
>
```
# 2D Dataset
'CREMI':
	{
		'PATH_DATASET': '.../XNetv2/dataset/CREMI',
		'PATH_TRAINED_MODEL': '.../XNetv2/checkpoints',
		'PATH_SEG_RESULT': '.../XNetv2/seg_pred',
		'IN_CHANNELS': 1,
		'NUM_CLASSES': 2,
		'MEAN': [0.503902],
		'STD': [0.110739],
		'INPUT_SIZE': (128, 128),
		'PALETTE': list(np.array([
			[255, 255, 255],
			[0, 0, 0],
		]).flatten())
	},

# 3D Dataset
'LiTS':
	{
		'PATH_DATASET': '.../XNetv2/dataset/LiTS',
		'PATH_TRAINED_MODEL': '.../XNetv2/checkpoints',
		'PATH_SEG_RESULT': '.../XNetv2/seg_pred',
		'IN_CHANNELS': 1,
		'NUM_CLASSES': 3,
		'NORMALIZE': tio.ZNormalization.mean,
		'PATCH_SIZE': (112, 112, 32),
		'PATCH_OVERLAP': (56, 56, 16),
		'NUM_SAMPLE_TRAIN': 8,
		'NUM_SAMPLE_VAL': 12,
		'QUEUE_LENGTH': 48
	},
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

