# MM811: Pedestrian Detection For Low-light Vision

# Acknowledgements
This project is based on the LLVIP paper:
- arXiv: https://arxiv.org/abs/2108.10831
- github: https://github.com/bupt-ai-cz/LLVIP

Original authors: Xinyu Jia, Chuang Zhu*, Minzhen Li, Wenqi Tang, Shengjie Liu, Wenli Zhou

This is a course project of MM811 Fall 2023 at the University of Alberta.

Authors: [Dulong Sang](dulong@ualberta.ca), [Mingwei Lu](mlu1@ualberta.ca), [Haoyu Qiu](hqiu3@ualberta.ca)


# Requirements
- Python 3.9 or above
- CPU or NVIDIA GPU + CUDA CuDNN

## Install Dependencies
```bash
pip install -r requirements.txt
```
Note: you may install each dependency later in the following steps

# Image Fusion

Baselines
   - [FusionGAN](https://github.com/jiayi-ma/FusionGAN)
   - [Densefuse](https://github.com/hli1221/imagefusion_densefuse)

## DenseFuse

### Preparation
- Install requirements
  ```bash
  git clone https://github.com/bupt-ai-cz/LLVIP
  cd LLVIP/imagefusion_densefuse
  
  # Create your virtual environment using anaconda
  conda create -n Densefuse python=3.7
  conda activate Densefuse
  
  conda install scikit-image scipy==1.2.1 tensorflow-gpu==1.14.0
  ```
- File structure
  ```
  imagefusion_densefuse
  ├── ...
  ├──datasets
  |  ├──010001_ir.jpg
  |  ├──010001_vi.jpg
  |  └── ...
  ├──test
  |  ├──190001_ir.jpg
  |  ├──190001_vi.jpg
  |  └── ...
  └──LLVIP
     ├── infrared
     |   ├──train
     |   |  ├── 010001.jpg
     |   |  ├── 010002.jpg
     |   |  └── ...
     |   └──test
     |      ├── 190001.jpg
     |      ├── 190002.jpg
     |      └── ...
     └── visible
         ├──train
         |   ├── 010001.jpg
         |   ├── 010002.jpg
         |   └── ...
         └── test
             ├── 190001.jpg
             ├── 190002.jpg
             └── ...
  ```
  
### Train & Test
  ```bash
  python main.py 
  ```
Check and modify training/testing options in `main.py`. Before training/testing, you need to rename the images in LLVIP dataset and put them in the designated folder. We have provided a script named `rename.py` to rename the images and save them in the `datasets` or `test` folder. Checkpoints are saved in `./models/densefuse_gray/`. To acquire complete LLVIP dataset, please visit https://bupt-ai-cz.github.io/LLVIP/.

## FusionGAN
### Preparation
- Install requirements
  ```bash
  git clone https://github.com/bupt-ai-cz/LLVIP.git
  cd LLVIP/FusionGAN
  # Create your virtual environment using anaconda
  conda create -n FusionGAN python=3.7
  conda activate FusionGAN
  
  conda install matplotlib scipy==1.2.1 tensorflow-gpu==1.14.0 
  pip install opencv-python
  sudo apt install libgl1-mesa-glx
  ```
- File structure
  ```
  FusionGAN
  ├── ...
  ├── Test_LLVIP_ir
  |   ├── 190001.jpg
  |   ├── 190002.jpg
  |   └── ...
  ├── Test_LLVIP_vi
  |   ├── 190001.jpg
  |   ├── 190002.jpg
  |   └── ...
  ├── Train_LLVIP_ir
  |   ├── 010001.jpg
  |   ├── 010002.jpg
  |   └── ...
  └── Train_LLVIP_vi
      ├── 010001.jpg
      ├── 010002.jpg
      └── ...
  ```
### Train
  ```bash
  python main.py --epoch 10 --batch_size 32
  ```
See more training options in `main.py`.
### Test
  ```bash
  python test_one_image.py
  ```
Remember to put pretrained model in your `checkpoint` folder and change corresponding model name in `test_one_image.py`.
To acquire complete LLVIP dataset, please visit https://bupt-ai-cz.github.io/LLVIP/.

# YOLOv8 Object Detection

## Install YOLOv8

```bash
pip install ultralytics
```

## Set up directory structure

```
root
├── datasets 
    └── LLVIP
        ├── images
            ├── train
            └── val
        └── labels
            ├── train
            └── val
├── yolov8
    ├── train.py
    └── val.py
├── utils
    └── xml2yolov8_txt.py
└── LLVIP.yaml
```

## Pre-process LLVIP Annotations

```bash
python utils/xml2yolov8_txt.py --xml-anno-dir datasets/LLVIP/Annotations --images-dir datasets/LLVIP/images/train
```

## Train / Fine-tune YOLOv8 Weights

```bash
python yolov8/train.py -h # check available options
```

You can check the available options for training using the above command, and choose the appropriate training settings.

## Run YOLOv8 Evaluation

```bash
python yolov8/val.py --data="LLVIP.yaml"
```

# InfraGAN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- Install required python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).
```bash
pip install visdom
pip install dominate
```

### InfraGAN train/test on LLVIP dataset
- Download the LLVIP dataset from: https://bupt-ai-cz.github.io/LLVIP

- You should name the dataset folder LLVIP and put it into InfaGAN-LLVIP folder, it should be like:
```
root
└── InfraGAN-LLVIP
    └── LLVIP
        ├── Annotations
            └── 010001.xml
        ├── infrared
            ├── train
            └── val
        └── visible
            ├── train
            └── val
```

- Run visdom to view the training process, then click the link http://localhost:8097 with explorer
```bash
python -m visdom.server
```

- Download the pretrained model from the link https://drive.google.com/file/d/1eeuiUZcxJTRPuMP5wKP5AzhFf6o8dC73/view?usp=drive_link, unzip the file, and put the checkpoints folder in the InfaGAN-LLVIP folder, it should be like this:

```
root
├── InfraGAN-LLVIP
└── checkpoints
    └── infragan_LLVIP
        ├── web
        ├── history.pth
        ├── latest_net_D.pth
        ├── latest_net_G.pth
        ├── loss_log.txt
        └── opt.txt
```

- Train the Model with LLVIP dataset:
```bash
python train.py --dataset_mode LLVIP --dataroot LLVIP --name infragan_LLVIP --model infragan --which_model_netG unet_512 --which_model_netD unetdiscriminator --which_direction AtoB --input_nc 3 --output_nc 1 --lambda_A 100  --no_lsgan --norm batch --pool_size 0 --loadSize 512 --fineSize 512 --gpu_ids 0 --nThreads 8 --batchSize 4
```

- Each epoch the model will save, to train the pretrained model (replace `##The_latest_Epoch##` with the latest epoch number you trained):
```bash
python train.py --dataset_mode LLVIP --dataroot LLVIP --name infragan_LLVIP --model infragan --which_model_netG unet_512 --which_model_netD unetdiscriminator --which_direction AtoB --input_nc 3 --output_nc 1 --lambda_A 100  --no_lsgan --norm batch --pool_size 0 --loadSize 512 --fineSize 512 --gpu_ids 0 --nThreads 8 --batchSize 4 --which_epoch 'latest' --epoch_count ##The_latest_Epoch## --continue_train
```
- Test the model with the LLVIP dataset:
```bash
python test.py --dataset_mode LLVIP --dataroot LLVIP --name infragan_LLVIP --model infragan --which_model_netG unet_512 --which_direction AtoB --input_nc 3 --output_nc 1 --norm batch --fineSize 512 --gpu_ids 0 --how_many 4000
```

In the InfaGAN-LLVIP folder, the images will be in `'results\infragan_LLVIP\test_latest\images'`

# Windows 10/11 CUDA support installation

Guide: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

- Install CUDA Toolkit:
https://developer.nvidia.com/cuda-downloads

- Install CUDA compiled torch and torchvision:
https://pytorch.org/get-started/locally/
