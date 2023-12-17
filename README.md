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
