# CJ made in 2020.07.08

from utils.toolfuc import *
from utils.cvtransform import *
import numpy as np
import random
from torch.utils.data import Dataset
import h5py
from sklearn import preprocessing
import torch
import time
from skimage import transform
import imgaug.augmenters as iaa
import imgaug as ia
from utils.toolfuc import setup_seed


def get_patient_ID_random(path, fold, cv=10, mode='train'):
    ID_list = []
    ID = []
    path_list = os.listdir(path)
    random.seed(1234)
    random.shuffle(path_list)
    r = round(len(path_list)/cv)
    if r*cv < len(path_list):
        for i in range(0, len(path_list), r):
            ID_list.append(path_list[i:i + r])
        ID_list[cv-1] = ID_list[cv-1] + ID_list[cv]
        del ID_list[cv]
    else:
        for i in range(0, len(path_list), r):
            ID_list.append(path_list[i:i + r])
    if mode == 'train':
        if fold == cv:
            del ID_list[fold-1]
            del ID_list[0]
        else:
            del ID_list[fold-1]
            del ID_list[fold-1]
        for f in ID_list:
            ID.extend(f)
        ID.sort()
    else:
        ID = ID_list[fold-1]
        ID.sort()
    return ID

def get_patient_ID_validandtest(path, fold, cv=10, mode='train'):
    ID_list = [['0105', '0104', '0058', '0095', '0029', '0114', '0087'],
               ['0003', '0014', '0073', '0092', '0083', '0119', '0030'],
               ['0023', '0110', '0036', '0052', '0082', '0117'],
               ['0106', '0018', '0112', '0040', '0062', '0037', '0074'],
               ['0025', '0011', '0115', '0091', '0056', '0096', '0116'],
               ['0020', '0006', '0053', '0064', '0045', '0075'],
               ['0107', '0024', '0049', '0067', '0054', '0071', '0113'],
               ['0016', '0022', '0088', '0118', '0057', '0121'],
               ['0108', '0009', '0069', '0111', '0090', '0089', '0097'],
               ['0015', '0008', '0051', '0078', '0060', '0120']]
    ID = []
    if mode == 'train':
        if fold == cv:
            del ID_list[fold - 1]
            del ID_list[0]
        else:
            del ID_list[fold - 1]
            del ID_list[fold - 1]
        for f in ID_list:
            ID.extend(f)
        ID.sort()
    else:
        ID = ID_list[fold - 1]
        ID.sort()

    return ID

def get_patient_ID(path, fold, cv=10, mode='train'):

    # ID_list = [['0105', '0104', '0058', '0095', '0029', '0114', '0087'],
    #            ['0003', '0014', '0073', '0092', '0083', '0119', '0030'],
    #            ['0023', '0110', '0036', '0052', '0082', '0117'],
    #            ['0106', '0018', '0112', '0040', '0062', '0037', '0074'],
    #            ['0025', '0011', '0115', '0091', '0056', '0096', '0116'],
    #            ['0020', '0006', '0053', '0064', '0045', '0075'],
    #            ['0107', '0024', '0049', '0067', '0054', '0071', '0113'],
    #            ['0016', '0022', '0088', '0118', '0057', '0121'],
    #            ['0108', '0009', '0069', '0111', '0090', '0089', '0097'],
    #            ['0015', '0008', '0051', '0078', '0060', '0120']]
    # ID_list = [['0003', '0035', '0058', '0094', '0062', '0001', '0083', '0010', '0086', '0016', '0037', '0011', '0002', '0096', '0079', '0084', '0013', '0059', '0076', '0064'],
    #            ['0004', '0006', '0024', '0061', '0034', '0005', '0020', '0074', '0038', '0019', '0015', '0032', '0025', '0066', '0022', '0043', '0031', '0071', '0029', '0068'],
    #            ['0089', '0092', '0054', '0051', '0072', '0008', '0039', '0093', '0044', '0028', '0042', '0053', '0047', '0030', '0018', '0017', '0048', '0080', '0070', '0027'],
    #            ['0065', '0045', '0067', '0060', '0040', '0009', '0046', '0063', '0014', '0088', '0033', '0056', '0075', '0049', '0026', '0041', '0052', '0082', '0090', '0012']]
    # ID_list = [['0020'],
    #            ['0021'],
    #            ['0022'],
    #            ['0023']]
    ID_list = [['0003', '0035', '0058', '0094', '0062', '0001', '0083', '0010', '0086', '0016', '0037', '0011', '0002',
                '0012'],
               ['0004', '0006', '0024', '0061', '0034', '0005', '0020', '0074', '0038', '0019', '0015', '0032', '0025'],
               ['0089', '0092', '0054', '0051', '0072', '0008', '0039', '0093', '0044', '0028', '0042', '0053', '0047'],
               ['0065', '0045', '0067', '0060', '0040', '0009', '0046', '0063', '0014', '0088', '0033', '0056', '0075'],
               ['0096', '0079', '0084', '0013', '0076', '0059', '0066', '0022', '0043', '0031', '0029', '0071', '0064',
                '0068'],
               ['0030', '0018', '0017', '0048', '0070', '0080', '0027', '0049', '0026', '0041', '0052', '0082', '0090']]
    ID = []
    if mode == 'train':
        # if fold == cv:
        #     del ID_list[fold - 1]
        #     del ID_list[0]
        # else:
        #     del ID_list[fold - 1]
        #     del ID_list[fold - 1]
        del ID_list[fold - 1]
        for f in ID_list:
            ID.extend(f)
        ID.sort()
    else:
        ID = ID_list[fold - 1]
        ID.sort()

    return ID

def get_patient_ID_ExternalData():

    # ID_list = ['0001', '0002', '0003', '0004', '0005', '0006', '0007',
    #            '0008', '0009', '0010', '0011', '0012', '0013', '0014',
    #            '0015', '0016', '0017', '0018', '0019', '0020', '0021',
    #            '0022', '0023', '0024', '0025', '0026', '0027', '0028',
    #            '0029', '0031', '0032', '0034', '0035', '0036', '0037',
    #            '0038', '0039', '0040']
    ID_list = ['0098', '0099', '0100', '0101', '0102', '0103', '0104',
               '0105', '0106', '0107', '0108', '0109', '0111', '0112',
               '0113', '0114', '0116', '0117']

    return ID_list


def get_data_path(path, fold, cv=10, mode='train'):
    data_path = collect_file(path)
    ID = get_patient_ID(path, fold, cv, mode)
    path_list = []

    for f in data_path:
        for id in ID:
            if id in f:
                path_list.append(f)
    return path_list

def get_data_path_test(path, ID):
    data_path = collect_file(path) # 把所有患者文件夹中的h5文件路径全部读取进来
    path_list = []
    for f in data_path:  # 遍历整个list,找到测试集患者的那些h5文件路径
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
    def __init__(self, path, ID='0001', Aug=False, n_class=2, set_name='train', fold=1, cv=10):
        self.Aug = Aug
        self.n_class = n_class
        self.set_name = set_name
        if set_name == 'train':
            self.data_path = get_data_path(path, fold, cv, set_name)
        if set_name == 'vaild':
            self.data_path = get_data_path(path, fold, cv, set_name)
        if set_name == 'test':
            self.data_path = get_data_path_test(path, ID)

        sometimes = lambda aug: iaa.Sometimes(0.9, aug)  # 设定随机函数,90%几率扩增,or
        self.seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),  # 50%图像进行水平翻转
                iaa.Flipud(0.5),  # 50%图像做垂直翻转
                sometimes(iaa.Crop(percent=(0, 0.1))),  # 对随机的一部分图像做crop操作 crop的幅度为0到10%
                sometimes(iaa.Affine(  # 对一部分图像做仿射变换
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},  # 图像缩放为80%到120%之间
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # 平移±20%之间
                    rotate=(-20, 20),  # 旋转±20度之间
                    shear=(-10, 10),  # 剪切变换±16度，（矩形变平行四边形）
                    order=[0, 1],  # 使用最邻近差值或者双线性差值
                    # mode=ia.ALL,  # 边缘填充
                )),
                # 使用下面的0个到3个之间的方法去增强图像
                # iaa.SomeOf((0, 3), [
                #     iaa.Sharpen(alpha=(0, 1.0), lightness=(0.8, 1.2)),  # 锐化处理
                #     iaa.contrast.LinearContrast((0.8, 1.2), per_channel=0.5),  # 改变对比度
                #     iaa.OneOf([
                #         iaa.GaussianBlur((0, 1.5)),
                #         iaa.AverageBlur(k=(2, 5)),  # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
                #     ]),  # 用高斯模糊，均值模糊，中值模糊中的一种增强
                # ],
                #            random_order=True  # 随机的顺序把这些操作用在图像上
                #            )
            ],
            random_order=True  # 随机的顺序把这些操作用在图像上
        )

    def __getitem__(self, index):
        data_path = self.data_path[index]
        n_class = self.n_class
        data = h5py.File(data_path)
        '''画框后的原图的h5文件'''
        image = data['ct'].value
        mask = data['seg_gt'].value
        label = data['grade'].value
        '''分割结果对应的原图的h5文件'''
        # image = data['seg_ct'].value
        # mask = data['seg_gt'].value
        # label = data['grade'].value
        lb = preprocessing.LabelBinarizer()
        lb.fit(np.array(range(n_class+1)))
        label = lb.transform([int(label)])
        label = label[:, :n_class]

        if self.Aug is True:
            image = np.expand_dims(image, axis=(0, 3))
            mask = np.expand_dims(mask, axis=(0, 3)).astype(np.int32)

            imager, masker = self.seq(images=image, segmentation_maps=mask)
            # imager = image
            imageg, imageb = imager, imager
            img_rg = np.concatenate([imager, imageg], axis=0)
            img_rgb = np.concatenate([img_rg, imageb], axis=0)
            img_rgb = img_rgb[:, :, :, 0]
            mask_rgb = masker[0, :, :, 0]

        else:
            imager = np.expand_dims(image, axis=0)
            masker = mask.astype(np.int32)
            imageg, imageb = imager, imager
            img_rg = np.concatenate([imager, imageg], axis=0)
            img_rgb = np.concatenate([img_rg, imageb], axis=0)
            mask_rgb = masker

        # img_rgb = transform.resize(img_rgb, [3, 224, 224])
        # mask = transform.resize(mask, [224, 224])
        img_rgb = np.asarray(img_rgb).astype('float32')
        mask_rgb = np.asarray(mask_rgb).astype('float32')

        return torch.FloatTensor(img_rgb), torch.FloatTensor(mask_rgb), torch.FloatTensor(label)

    def __len__(self):
        return len(self.data_path)