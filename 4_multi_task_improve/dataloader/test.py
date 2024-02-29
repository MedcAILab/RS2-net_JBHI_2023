import os
import numpy as np
import random
from torch.utils.data import Dataset
import h5py
from sklearn import preprocessing
import torch
import time
from skimage import transform
import cv2
import matplotlib.pyplot as plt
from resnest.torch import resnest269

net = resnest269(pretrained=True)
1

def collect_file(file_path, match='.h5'):
    file_list = []
    for root, dir, files in os.walk(file_path):
        for f in files:
            if match.lower() in f.lower():
                file_list.append(os.path.join(root, f))
    return file_list

def get_patient_ID(path, fold, cv=10):
    ID_list = []
    ID = []
    path_list = os.listdir(path)
    random.seed(1234)
    random.shuffle(path_list)
    r = round(len(path_list)/cv)
    for i in range(0, len(path_list), r):
        ID_list.append(path_list[i:i+r])
    del ID_list[fold-1]
    for f in ID_list:
        ID.extend(f)
    ID.sort()
    return ID


def get_data_path(path, fold, cv=10):
    data_path = collect_file(path)
    ID = get_patient_ID(path, fold, cv)
    path_list = []
    for f in data_path:
        for id in ID:
            if id  in f:
                path_list.append(f)
    # path_list.sort()
    # ti = 1000*time.time()
    # random.seed(int(ti))
    # random.shuffle(path_list)
    return path_list

# 定义平移
def translate(image, x=0, y=0):
    # 定义平移矩阵
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # 返回转换后的图像
    return shifted

# 定义旋转rotate函数
def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated

# 定义缩放函数
def scale(image, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为缩放中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行缩放
    M = cv2.getRotationMatrix2D(center, 0, scale)
    scaled = cv2.warpAffine(image, M, (w, h))

    # 返回缩放后的图像
    return scaled

# 定义翻转函数
def flip(image, flipOK=1):  # 横向翻转图像
    if flipOK == 1:
        flipped = cv2.flip(image, 1)  # 翻转
    else:
        flipped = image  # 不翻转

    return flipped

def data_aug(img, mask):
    random.seed(int(10e7 * time.time()))
    flip_num = random.choice([0, 1])
    x = random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
    y = random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
    angl = random.choice([-90, 0, 90, 180])
    data_s = flip(img, flip_num)
    mask_s = flip(mask, flip_num)
    data_s = translate(data_s, x, y)
    mask_s = translate(mask_s, x, y)
    data_s = rotate(data_s, angl, None, 1.0)
    mask_s = rotate(mask_s, angl, None, 1.0)
    return data_s, mask_s



