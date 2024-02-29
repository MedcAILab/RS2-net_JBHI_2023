import torch.nn as nn
import torch
from networks.unet.unet_model import UNet
from torch_struct.segmentation_models_pytorch_w.resnest_encoder.resnest import resnest50
from torch_struct.segmentation_models_pytorch_w.resnest_encoder.resnet import GlobalAvgPool2d
from networks.resnest_chia.resnest import ResNet_cla_seg, Bottleneck
import torch.nn.functional as F
import os


class mergeAndseg(nn.Module):
    def __init__(self, in_channel, out_channel=1):
        super(mergeAndseg, self).__init__()
        middel_channel = 512
        self.conv1 = nn.Conv2d(in_channel, middel_channel, 3,1,1)
        self.conv2 = nn.Conv2d(middel_channel, out_channel, 3,1,1)
        self.normalization = nn.GroupNorm(16, middel_channel)
    def update_resize(self, feature_list, aimshape = [512, 512]):
        size_list = [i.shape for i in feature_list]
        middel_size_level = 1
        self.downsample = nn.AdaptiveAvgPool2d((size_list[middel_size_level][2],
                                                size_list[middel_size_level][3]))   # 自适应核大小和步长。 二元自适应均值下采样层
        self.upsample = nn.UpsamplingBilinear2d((size_list[middel_size_level][2],
                                                size_list[middel_size_level][3]))   # 上采样双线性插值
        self.final_reshape = nn.UpsamplingBilinear2d(aimshape)  # 上采样双线性插值
    def forward(self, input, aimshape):
        self.update_resize(input, aimshape)
        cat_feature = torch.cat(
            [self.downsample(input[0]),
             self.downsample(input[1]),
             self.upsample(input[2]),
             self.upsample(input[3]),
             self.upsample(input[4])
             ], axis=1)
        middel_f = F.relu(self.normalization(self.conv1(cat_feature)))
        segout = self.conv2(self.final_reshape(middel_f))
        return segout

class cascade_net(nn.Module):
    def __init__(self, class_num = 2, seg_num =1):
        super(cascade_net, self).__init__()
        self. multi_model = ResNet_cla_seg(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False)
    def forward(self, x):
        cls1, seg1 = self.multi_model(x)
        cls2, seg2 = self.multi_model(0.5*x + 0.5*torch.cat([seg1, seg1, seg1], axis=1))
        return [cls2, seg1, seg2]


if __name__ == '__main__':
    input = torch.ones([2,3,256,256])
    model = cascade_net(class_num=3, seg_num=1)
    output = model(input)
    _ = [print(i.shape) for i in output]




