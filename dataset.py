import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader,Dataset
import cv2
import gzip
import os
import os.path
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter


def read_csv(csv_path):
    data = []

    with open(csv_path) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)

        for i, row in enumerate(csvreader):
            # 164 lose left image
            if (row[0] == '164'):
                continue

            path = row[2]

            path = path.split('\\')
            new_path = path[-2] + '/' + path[-1]

            if i % 2 == 0:
                file_info = {}
                file_info['eye_index'] = int(i / 2)
                file_info['img_path1'] = new_path

                if row[1].split('_')[-1][0] == 'l':
                    file_info['eye_level1'] = int(row[4])
                else:
                    file_info['eye_level1'] = int(row[5])
            else:
                file_info['img_path2'] = new_path

                if row[1].split('_')[-1][0] == 'l':
                    file_info['eye_level2'] = int(row[4])
                else:
                    file_info['eye_level2'] = int(row[5])

                file_info['patient_level'] = int(row[6])

                data.append(file_info)

    return data


class dataset(Dataset):
    def __init__(self, root, csv_path, transform=None, args=None):
        self.root = root
        self.transform = transform
        self.csv_path = csv_path
        self.data = read_csv(csv_path=self.csv_path)

    def __getitem__(self, index):
        data = self.data[index]

        img1 = Image.open(os.path.join(self.root, data['img_path1']))
        img1 = img1.convert('RGB')
        img1 = self.transform(img1)

        img2 = Image.open(os.path.join(self.root, data['img_path2']))
        img2 = img2.convert('RGB')
        img2 = self.transform(img2)

        label1 = data['eye_level1']
        label2 = data['eye_level2']
        label = max(label1, label2)
        patient_label = data['patient_level']

        label1 = torch.tensor(label1, dtype=torch.int64)
        label2 = torch.tensor(label2, dtype=torch.int64)
        label = torch.tensor(label, dtype=torch.int64)
        patient_label = torch.tensor(patient_label, dtype=torch.int64)

        return [img1, img2], [label1, label2], label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    # traindataset_1 = dataset(csv_path='/home1/yujiali/DR/dataset/regular_fundus_images/regular-fundus-training/regular-fundus-training.csv',
    #                               root='/home1/yujiali/DR/dataset/regular_fundus_images/regular-fundus-training/Images')
    traindataset_1 = dataset(csv_path='./dataset/regular_fundus_images/regular-fundus-training/regular-fundus-training.csv',
                             root='./dataset/regular_fundus_images/regular-fundus-training/Images')

    
    





