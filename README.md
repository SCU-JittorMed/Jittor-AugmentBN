# AugmentBN

This repo holds the pytorch implementation of AugmentBN:<br />

**Regularizing Deep Neural Networks for Medical Image Analysis with Augmented Batch Normalization.**

## Requirements
jittor==1.3.9.14<br />

## Usage
### 0. Installation
* Clone this repo
```
git clone https://github.com/SCU-JittorMed/Jittor-AugmentBN.git
```


### 1. Training and Evaluation
* For image classification on CIFAR-10, modify `net_name` in `train_CIFAR_NABN.py`, and run
```
python train_CIFAR_NABN.py
```

### Citation

```bibtex
@article{zhu2024regularizing,
  title={Regularizing deep neural networks for medical image analysis with augmented batch normalization},
  author={Zhu, Shengqian and Yu, Chengrong and Hu, Junjie},
  journal={Applied Soft Computing},
  publisher={Elsevier}
}
```













