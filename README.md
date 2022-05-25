# DeepDRiD-Project



## Overview

./baseline.py Resnet50+ABM主程序

./baseline3.py 使用其他预训练模型主程序

./pre_processing.py  基于Ben's preprocessing的预处理，因结果相差不大没用在最后的实验里

./dataset.py 数据类

./models/resnet50.py  直接在CANet[^1]代码基础上改的Resnet50+BAM模型（包含单输入Resnet50+BAM，双输入Resnet50+BAM+融合两幅图像的结果，双输入+BAM+CBAM），但排列组合试了多种网络结构/数据增强/loss/学习率调整方法，都和只使用Resnet50相差不大



## Result

Use all model  pre-trained on ImageNet


| Method          | Image_size | Loss   | Kappa  |
| --------------- | :--------- | ------ | :----- |
| Resnet101       | 256 × 256  | CE+SL1 | 0.8095 |
| EfficientNet-b1 | 256 × 256  | CE+SL1 | 0.8140 |
| EfficientNet-b3 | 256 × 256  | CE+SL1 | 0.7969 |
| EfficientNet-b4 | 256 × 256  | CE+SL1 | 0.7990 |
| DenseNet121     | 256 × 256  | CE+SL1 | 0.5650 |
| InceptionV3     | 512 × 512  | CE+SL1 | 0.8249 |




| Method         | Image_size | Loss   | Kappa  |
| -------------- | :--------- | ------ | :----- |
| Resnet50       | 256 × 256  | CE     | 0.7866 |
| ResNet50       | 256 × 256  | CE+SL1 | 0.7866 |
| ResNet50 + BAM | 256 × 256  | CE     | 0.8147 |
| ResNet50 + BAM | 256 × 256  | CE+SL1 | 0.8193 |



![image-20220526000559736](D:\jupyter\deepdrid_project\DeepDRiD-Project\README.assets\image-20220526000559736.png)

[^1]: X. Li, X. Hu, L. Yu, L. Zhu, C. -W. Fu and P. -A. Heng, "CANet: Cross-Disease Attention Network for Joint Diabetic Retinopathy and Diabetic Macular Edema Grading," in IEEE Transactions on Medical Imaging, vol. 39, no. 5, pp. 1483-1493, May 2020, doi: 10.1109/TMI.2019.2951844.

