# MM811: Pedestrian Detection For Low-light Vision

# Acknowledgements
This project is based on the LLVIP paper:
- arXiv: https://arxiv.org/abs/2108.10831
- github: https://github.com/bupt-ai-cz/LLVIP

Original authors: Xinyu Jia, Chuang Zhu*, Minzhen Li, Wenqi Tang, Shengjie Liu, Wenli Zhou

This is a course project of MM811 Fall 2023 at the University of Alberta.

Authors: [Dulong Sang](dulong@ualberta.ca), [Mingwei Lu](mlu1@ualberta.ca), [Haoyu Qiu](hqiu3@ualberta.ca)


# Install Dependencies

Required Python version: 3.9 or above

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
python3 utils/xml2yolov8_txt.py --xml-anno-dir datasets/LLVIP/Annotations --images-dir datasets/LLVIP/images/train
```

## Run YOLOv8 evaluation

```bash
python3 yolov8/val.py --data="LLVIP.yaml"
```

# Windows 10/11 CUDA support installation

Guide: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

- Install CUDA Toolkit:
https://developer.nvidia.com/cuda-downloads

- Install CUDA compiled torch and torchvision:
https://pytorch.org/get-started/locally/
