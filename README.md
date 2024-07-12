# UMHE: Unsupervised Multispectral Homography Estimation (IEEE Sensors Journal)


 >**UMHE: Unsupervised Multispectral Homography Estimation
 >
 >[[Paper_pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10494213)]]


image
<p align="center">
  <img src="assets/movie.gif" alt="Robustness" width="600" />
</p>

If you find our work useful in your research, kindly consider citing our paper:

```
@InProceedings{IEEE Sensors Journal,
    author    = {Jeongmin, Shin and Jiwon, Kim and Seokjun, Kwon and Namil, Kim and Soonmin, Hwang and Yukyung, Choi},
    title     = {UMHE: Unsupervised Multispectral Homography Estimation},
    booktitle = {IEEE Sensors Journal},
    month     = {April},
    year      = {2024},
}
```


## Getting Started

### Git Clone

```
git clone https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection.git
cd MLPD-Multi-Label-Pedestrian-Detection
```

### Docker

- Prerequisite
  - [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

```
cd docker
make docker-make
```

#### Make Contianer

```
cd ..
nvidia-docker run -it --name umhe -v $PWD:/workspace -p 8888:8888 -e NVIDIA_VISIBLE_DEVICES=all --shm-size=8G umhe /bin/bash
```

## Datasets

For multispectral homography estimation, we train and test the proposed model on the FLIR dataset, you should first download the dataset. 
We provide the dataset comprising original multispectral image pairs from the FLIR dataset, pseudo multispectral pairs generated using our style augmentation, and GT corresponding points for evaluation. ([download url](https://drive.google.com/file/d/1rl_Z2R2ScYj69-RuSb7B3FPM_obO7Pst/view?usp=sharing))
Download and place them in the directory 'dataset/'

``` 
<DATA_PATH>
+-- config
+-- docker
+-- scripts
+-- src
+-- dataset
|   +-- flir_aligned
|   |   +-- train
|   |   |   +-- PreviewData
|   |   |   |   +-- FLIR_00002.npy
|   |   |   +-- PreviewData_StyleAug
|   |   |   |   +-- FLIR_00002.npy
|   |   |   +-- RGB
|   |   |   |   +-- FLIR_00002.npy
|   |   |   +-- RGB_StyleAug
|   |   |   |   +-- FLIR_00002.npy
|   |   |   +-- align_train.txt
|   |   +-- validation
|   |   |   +-- Coordinates
|   |   |   |   +-- match_08864.json
|   |   |   +-- PreviewData
|   |   |   |   +-- FLIR_08864.npy
|   |   |   +-- RGB
|   |   |   |   +-- FLIR_08864.npy
|   |   |   +-- align_validation_day.txt
|   |   |   +-- align_validation_night.txt
|   |   |   +-- align_validation.txt


```
### Training
The models can be trained on the FLIR dataset by running: 
```
python train.py --config_file config/flir/umhe.yaml
```
or
```
sh scripts/train.sh
```
The hyperparameters are defined in the config file (i.e., "config/flir/umhe.yaml")

## Evaluation
We provide the evaluation for the FLIR Corresponding dataset.
```
sh scripts/test.sh
```

## Pretrained Models

We provide the [pretrained weights]() for our network. 


## References

* [biHomE](https://github.com/NeurAI-Lab/biHomE)

