## Prerequisites
- Linux or Windows
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- Install required python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).
```bash
pip install visdom
pip install dominate
```


### InfraGAN train/test on VEDAI dataset
- Download the LLVIP dataset from:

```bash
https://bupt-ai-cz.github.io/LLVIP/
```
- You should name the dataset folder LLVIP and put it into InfaGAN-LLVIP folder, it should be like:

```
+InfraGAN-main

+----LLVIP

+---------Annotations
+--------------010001.xml
+--------------010002.xml
+--------------010003.xml

```
- 
```

+---------infrared

+--------------train
+-------------------010001.jpg
+-------------------010002.jpg
+-------------------010003.jpg
+-------------------010004.jpg
...

+--------------test
+-------------------190001.jpg
+-------------------190002.jpg
+-------------------190003.jpg
+-------------------190004.jpg
...

```
- 
```

+---------visible

+--------------train
+-------------------010001.jpg
+-------------------010002.jpg
+-------------------010003.jpg
+-------------------010004.jpg
...

+--------------test
+-------------------190001.jpg
+-------------------190002.jpg
+-------------------190003.jpg
+-------------------190004.jpg
...



```
- Run visdom to view the training process, then click the link http://localhost:8097 with explorer
```
python -m visdom.server

```


- Train the Model with LLVIP dataset:
```
python train.py --dataset_mode LLVIP --dataroot LLVIP --name infragan_LLVIP --model infragan --which_model_netG unet_512 --which_model_netD unetdiscriminator --which_direction AtoB --input_nc 3 --output_nc 1 --lambda_A 100  --no_lsgan --norm batch --pool_size 0 --loadSize 512 --fineSize 512 --gpu_ids 0 --nThreads 8 --batchSize 4
```

- Each epoch the model will save, to train the pretrained model (replace ##The_latest_Epoch## with the latest epoch number you trained):
```
python train.py --dataset_mode LLVIP --dataroot LLVIP --name infragan_LLVIP --model infragan --which_model_netG unet_512 --which_model_netD unetdiscriminator --which_direction AtoB --input_nc 3 --output_nc 1 --lambda_A 100  --no_lsgan --norm batch --pool_size 0 --loadSize 512 --fineSize 512 --gpu_ids 0 --nThreads 8 --batchSize 4 --which_epoch 'latest' --epoch_count ##The_latest_Epoch## --continue_train
```
- Test the model with the LLVIP dataset:
```
python test.py --dataset_mode LLVIP --dataroot LLVIP --name infragan_LLVIP --model infragan --which_model_netG unet_512  --which_direction AtoB --input_nc 3 --output_nc 1   --norm batch  --fineSize 512 --gpu_ids 0 --how_many 4000
```

In the InfaGAN-LLVIP folder, the images will be in 'results\infragan_LLVIP\test_latest\images'


