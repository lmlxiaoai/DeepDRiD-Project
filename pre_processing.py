from PIL import Image
import os
import os.path
import sys
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import csv
import pdb
import cv2

def get_image_file_list(img_file):
    imgs_lists = []
    if os.path.isfile(img_file):
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path):
                imgs_lists.append(file_path)
    return imgs_lists

def crop_image1(img, tol=7):
    # img is image data
    # tol is tolerance

    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    # If we have a normal RGB images
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
            return img


def ben_color(image, sigmaX=20):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (299, 299))
    height, width, depth = image.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    image = cv2.bitwise_and(image, image, mask=circle_img)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX=20), -4, 128)
    return image

if __name__ == '__main__':
    path = './dataset/regular_fundus_images/regular-fundus-validation/Images'
    path2 = './dataset/regular_fundus_images/regular-fundus-validation2/Images'
    for single_file in os.listdir(path):
        file_path = os.path.join(path, single_file)
        file_path = os.path.normpath(file_path)
        for file in os.listdir(file_path):
            img_path = os.path.join(file_path, file)
            img_path = os.path.normpath(img_path)
            print(img_path)
            save_dir = os.path.join(path2, single_file)
            save_dir = os.path.normpath(save_dir)
            save_path = os.path.join(path2, single_file,file)
            save_path = os.path.normpath(save_path)
            print(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            img = cv2.imread(img_path)
            img2 = ben_color(img)
            cv2.imwrite(save_path, img2)