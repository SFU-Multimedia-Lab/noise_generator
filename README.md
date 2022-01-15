# Practical Noise Generator

A practical noise generator is provided in this repository.

## Noise Generator:
The Poissonian-Gaussian model is used to the model the noise. The Poissonian-Gaussian model 
model has two parameters: 𝑎 and 𝑏, where 𝑎 is related to the variance of Poissonian component
and 𝑏 the variance of the Gaussian component. The Smartphone Image Denoising Dataset (SIDD) is
used to estimate 𝑎 and 𝑏 for each of the color components (R,G,B).

The following example generates N number of noisy image for the given $IMG_DIR using the provided noise generator
```python
python ./noise_sampling.py --img_dir $IMG_DIR --n_obs $N
```

The code first obtains (𝑎, 𝑏) for each color component. Then generates noisy image according to the sampled parameters.
img_syn_noisy_q variable is the quantized noisy image that can be stored or used in the training dataloader. 

### How to set up

1) create conda environment: ``conda create -n test_generator python=3.6.9``
2) activate environment: ``conda activate test_generator``
3) install dependencies: ``pip install -r requirements.txt``


<!--
References


----------
```BibTex
[1] @ARTICLE{foi,
  author={Foi, Alessandro and Trimeche, Mejdi and Katkovnik, Vladimir and Egiazarian, Karen},
  journal={IEEE Transactions on Image Processing}, 
  title={Practical Poissonian-Gaussian Noise Modeling and Fitting for Single-Image Raw-Data}, 
  year={2008},
  volume={17},
  number={10},
  pages={1737-1754},
  doi={10.1109/TIP.2008.2001399}}
[2] @InProceedings{SIDD_2018_CVPR,
author = {Abdelhamed, Abdelrahman and Lin, Stephen and Brown, Michael S.},
title = {A High-Quality Denoising Dataset for Smartphone Cameras},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}

```
—>
