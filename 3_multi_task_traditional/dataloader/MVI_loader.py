# CJ made in 2020.07.08

import os
import numpy as np
import random
from torch.utils.data import Dataset
import h5py
from sklearn import preprocessing
import torch
import time
from skimage import transform
from utils.toolfuc import *


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

    if fold == cv:
        del ID_list[0]
    else:
        del ID_list[fold-1]

    for f in ID_list:
        ID.extend(f)
    ID.sort()
    return ID

def get_patient_ID_vaild(path, fold, cv=10):
    ID_list = []
    ID = []
    path_list = os.listdir(path)
    random.seed(1234)
    random.shuffle(path_list)
    r = round(len(path_list)/cv)
    for i in range(0, len(path_list), r):
        ID_list.append(path_list[i:i+r])

    return ID_list[fold-1]

def get_patient_ID_test(path, fold, cv=10):
    ID_list = []
    ID = []
    path_list = os.listdir(path)
    random.seed(1234)
    random.shuffle(path_list)
    r = round(len(path_list)/cv)
    for i in range(0, len(path_list), r):
        ID_list.append(path_list[i:i+r])
    # del ID_list[fold-1]
    for f in ID_list:
        ID.extend(f)
    ID.sort()
    return ID_list[fold-1]

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

def get_data_path_vaild(path, fold, cv=10):
    data_path = collect_file(path)
    ID = get_patient_ID_vaild(path, fold, cv)
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

def get_data_path_test(path, ID):
    data_path = collect_file(path)
    # ID = get_patient_ID_test(path, fold, cv)
    path_list = []
    for f in data_path:
        if ID in f:
            path_list.append(f)
    return path_list




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


class DatasetGenerator(Dataset):
    def __init__(self, path, Aug=True, n_class=2, set_name='train', fold=1, cv=10):
        self.Aug = Aug
        self.n_class = n_class
        self.set_name = set_name
        if set_name == 'train':
            self.data_path = get_data_path(path, fold, cv)
        # if set_name == 'vaild':
        #     self.data_path = get_data_path_vaild(path, fold, cv)


    def __getitem__(self, index):
        data_path = self.data_path[index]
        n_class = self.n_class
        data = h5py.File(data_path)
        image = data['img'].value
        mask = data['gt'].value
        label = data['label'].value
        lb = preprocessing.LabelBinarizer()
        lb.fit(np.array(range(n_class+1)))
        label = lb.transform([int(label)])
        label = label[:, :n_class]
        if self.Aug is True:
            imager, mask = data_aug(image, mask)
            imager = np.expand_dims(imager, axis=0)
            imageg, imageb = imager, imager
            img_rgb = np.concatenate([imager, imageg], axis=0)
            img_rgb = np.concatenate([img_rgb, imageb], axis=0)
        else:
            imager = np.expand_dims(image, axis=0)
            imageg, imageb = imager, imager
            img_rgb = np.concatenate([imager, imageg], axis=0)
            img_rgb = np.concatenate([img_rgb, imageb], axis=0)


        img_rgb = transform.resize(img_rgb, [3, 224, 224])
        mask = transform.resize(mask, [224, 224])
        img_rgb = np.asarray(img_rgb).astype('float32')
        mask = np.asarray(mask).astype('float32')

        if self.set_name=='train':
            return torch.FloatTensor(img_rgb), torch.FloatTensor(mask), torch.FloatTensor(label)
        elif self.set_name=='vaild':
            return torch.FloatTensor(img_rgb), torch.FloatTensor(mask), torch.FloatTensor(label)
        else:
            return torch.FloatTensor(img_rgb), torch.FloatTensor(mask)

    def __len__(self):
        return len(self.data_path)

class DatasetGenerator_test(Dataset):
    def __init__(self, path, ID, Aug=True, n_class=2, set_name='test'):
        self.data_path = get_data_path_test(path, ID)
        self.Aug = Aug
        self.n_class = n_class
        self.set_name = set_name

    def __getitem__(self, index):
        data_path = self.data_path[index]
        n_class = self.n_class
        data = h5py.File(data_path)
        image = data['img'].value
        mask = data['gt'].value
        label = data['label'].value
        lb = preprocessing.LabelBinarizer()
        lb.fit(np.array(range(n_class+1)))
        label = lb.transform([int(label)])
        label = label[:, :n_class]
        if self.Aug is True:
            imager, mask = data_aug(image, mask)
            imager = np.expand_dims(imager, axis=0)
            imageg, imageb = imager, imager
            img_rgb = np.concatenate([imager, imageg], axis=0)
            img_rgb = np.concatenate([img_rgb, imageb], axis=0)
        else:
            imager = np.expand_dims(image, axis=0)
            imageg, imageb = imager, imager
            img_rgb = np.concatenate([imager, imageg], axis=0)
            img_rgb = np.concatenate([img_rgb, imageb], axis=0)

        img_rgb = transform.resize(img_rgb, [3, 224, 224])
        mask = transform.resize(mask, [224, 224])
        img_rgb = np.asarray(img_rgb).astype('float32')
        mask = np.asarray(mask).astype('float32')

        if self.set_name=='train':
            return torch.FloatTensor(img_rgb), torch.FloatTensor(mask), torch.FloatTensor(label)
        else:
            return torch.FloatTensor(img_rgb), torch.FloatTensor(mask), torch.FloatTensor(label)

    def __len__(self):
        return len(self.data_path)


# test_data_root = '/media/root/32686b5b-6d88-429d-b7cd-35208a8181c2/Database/Preprocessing/data/bbox_data'
# a = collect_file(test_data_root)
# b = get_data_path(test_data_root, 10)
# c = get_data_path_vaild(test_data_root, 1)
# 1
