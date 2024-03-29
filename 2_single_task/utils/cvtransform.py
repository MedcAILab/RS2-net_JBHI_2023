import cv2
import numpy as np


#  定义平移
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