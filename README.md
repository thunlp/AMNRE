# AMNRE

## Introduction

Source code and dataset for COLING2018 paper "Adversarial Multi-lingual Neural Relation Extraction".

## Requirements

- pytorch==0.3.1
- scikit-learn==0.19.1
- numpy==1.14.1
- matplotlib==2.1.2

## Data

We use the same dataset and pre-trained word embedding as the [MNRE](https://github.com/thunlp/MNRE). You can download the raw data in [this page](https://cloud.tsinghua.edu.cn/f/7dbd74f9e9bc45ffb5f9/?dl=1). You need to download it to the `Data/` path and use init.py in `CNN/src/` to preprocess it.

We also provide the preprocessed .npy format data in [this page](https://cloud.tsinghua.edu.cn/f/193ba7015c4047d6868a/?dl=1). Download it to the `Data/` path and unpack it, then you can run the code.

## Run

Run `python train.py` in corresponding directory to train the model. It will output the average precision on test set to `AUC.txt` and the prediction result as `.npy` every epoch.

Run `python draw.py <label file's name> <prediction result file's name> <output image's name>` to get the precision-recall curve for one specific prediction result.

If you want to tune the hyper parameters, see the `src/constant.py` and change the parameters defined in the file.

## Cite

If the codes help you, please cite the following paper:

[Xiaozhi Wang, Xu Han, Yankai Lin, Zhiyuan Liu, Maosong Sun. Adversarial Multi-lingual Neural Relation Extraction (COlING 2018)](http://aclweb.org/anthology/C18-1099)
