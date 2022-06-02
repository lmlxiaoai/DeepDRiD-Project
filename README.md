# DeepDRiD-Project



## Overview

./baseline.py Resnet50+ABM主程序

./baseline3.py 使用其他预训练模型主程序

./pre_processing.py  基于Ben's preprocessing的预处理，因结果相差不大没用在最后的实验里

./dataset.py 数据类

./models/resnet50.py  在CANet[^1]基础上修改的Resnet50+BAM模型（包含单输入Resnet50+BAM，双输入Resnet50+BAM+融合两幅图像的结果，双输入+BAM+CBAM），但排列组合试了多种网络结构/数据增强/loss/学习率调整方法，都和只使用Resnet50相差不大



## Preparation

### prerequisites

- Python 3.9.12
- Pytorch 1.8.1
- CUDA 10.1

### Data Preparation

- DeepDRiD：原始regular-fundus-training.csv文件中部分数据image_id中的左右和其对应的标签对应有误，需手动进行处理

### Pretrained Model

- ResNet: [ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth), [ResNet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)
- EfficientNet: [EfficientNet-b1](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth), [EfficientNet-b3](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth), [EfficientNet-b4](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth)
- [DenseNet121](https://download.pytorch.org/models/densenet121-a639ec97.pth)
- [InceptionV3](https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth)



## Result

Use all model  pre-trained on ImageNet


| Method          | Image_size | Loss   | Kappa  |
| --------------- | :--------- | ------ | :----- |
| ResNet101       | 256 × 256  | CE+SL1 | 0.8095 |
| EfficientNet-b1 | 256 × 256  | CE+SL1 | 0.8140 |
| EfficientNet-b3 | 256 × 256  | CE+SL1 | 0.7969 |
| EfficientNet-b4 | 256 × 256  | CE+SL1 | 0.7990 |
| DenseNet121     | 256 × 256  | CE+SL1 | 0.5650 |
| InceptionV3     | 512 × 512  | CE+SL1 | 0.8249 |




| Method         | Image_size | Loss   | Kappa  |
| -------------- | :--------- | ------ | :----- |
| ResNet50       | 256 × 256  | CE     | 0.7866 |
| ResNet50       | 256 × 256  | CE+SL1 | 0.7866 |
| ResNet50 + BAM | 256 × 256  | CE     | 0.8147 |
| ResNet50 + BAM | 256 × 256  | CE+SL1 | 0.8193 |




[^1]: X. Li, X. Hu, L. Yu, L. Zhu, C. -W. Fu and P. -A. Heng, "CANet: Cross-Disease Attention Network for Joint Diabetic Retinopathy and Diabetic Macular Edema Grading," in IEEE Transactions on Medical Imaging, vol. 39, no. 5, pp. 1483-1493, May 2020, doi: 10.1109/TMI.2019.2951844.

