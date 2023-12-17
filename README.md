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
