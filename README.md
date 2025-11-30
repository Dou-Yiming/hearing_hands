# Hearing Hands: Generating Sounds from Physical Interactions in 3D Scenes

### [Dataset]() · [Checkpoints](https://www.dropbox.com/scl/fi/c9zn9sfzqk3n20vcto3te/checkpoints.tar.gz?rlkey=3mqnw3t0uc21ne5cdw1ceq8hd&st=fc1benyd&dl=0) · [Website](https://www.yimingdou.com/hearing_hands/) · [Paper](https://arxiv.org/abs/2506.09989)

##### [Yiming Dou](https://www.yimingdou.com/), [Wonseok Oh](https://prbs5kong.github.io/), [Yuqing Luo](https://www.linkedin.com/in/yuqing-luo-452715249/), [Antonio Loquercio](https://antonilo.github.io/), [Andrew Owens](https://andrewowens.com/)

## Installation
#### 1. Clone this repo
`git clone --branch main --single-branch https://github.com/Dou-Yiming/hearing_hands.git`

#### 2. Create Conda environment

```sh
cd hearing_hands
conda env create -f environment.yml
```

## Prepare data and pretrained models

#### 1. Download dataset

Download the dataset from [this]() link, then extract them:

` tar -xvf dataset.tar.gz video2audio/data/dataset`

#### 2. Download pretrained models

Download the pretrained checkpoints from [this](https://www.dropbox.com/scl/fi/c9zn9sfzqk3n20vcto3te/checkpoints.tar.gz?rlkey=3mqnw3t0uc21ne5cdw1ceq8hd&st=fc1benyd&dl=0) link, then extract them:

```sh
tar -xvf checkpoints.tar.gz ./
mkdir video2audio/logs; mv checkpoints/sarf_full video2audio/logs
mv checkpoints/adm_checkpoints video2audio/ldm/adm/checkpoints
mv checkpoints/vocoder_checkpoints/* video2audio/ldm/vocoder/bigvgan/checkpoints
rm checkpoints.tar.gz
```

## Run Inference and Training

#### 1. Run inference with the pretrained model

`sh scripts/inference.sh`

#### 2. Train video-to-audio model

`sh scripts/train.sh`

### Bibtex

If you find our work useful, please consider citing:

```
@inproceedings{dou2025hearing,
  title={Hearing Hands: Generating Sounds from Physical Interactions in 3D Scenes},
  author={Dou, Yiming and Oh, Wonseok and Luo, Yuqing and Loquercio, Antonio and Owens, Andrew},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={1795--1804},
  year={2025}
}
```