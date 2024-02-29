# encoding: utf-8
import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F
# import pretrainedmodels
from torchvision import models
from .efficientnet_pytorch import EfficientNet


def load_weights(model, weight_path):
    pretrained_weights = torch.load(weight_path)

    model_weights = model.state_dict()

    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}
    model_weights.update(load_weights)
    model.load_state_dict(model_weights)

    return model

class EfficientNet_b0(nn.Module):
    def __init__(self, n_class, pre_train = True):
        super(EfficientNet_b0, self).__init__()

        self.model = EfficientNet.from_name('efficientnet-b0')
        # self.model = EfficientNet.from_name('efficientnet-b1')
        # self.model = EfficientNet.from_name('efficientnet-b2')
        # self.model = EfficientNet.from_name('efficientnet-b3')
        # self.model = EfficientNet.from_name('efficientnet-b4')
        # self.model = EfficientNet.from_name('efficientnet-b5')
        # self.model = EfficientNet.from_name('efficientnet-b6')
        # self.model = EfficientNet.from_name('efficientnet-b7')

        if pre_train:
            pretrained_dict = torch.load('/media/root/32686b5b-6d88-429d-b7cd-35208a8181c2/Graduation_P/CK19/networks/efficientnet/model/efficientnet-b0-355c32eb.pth')
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            # self.model = load_weights(self.model, './networks/efficient_model/efficientnet-b0-355c32eb.pth')
            # self.model = load_weights(self.model, './networks/efficient_model/efficientnet-b1-f1951068.pth')
            # self.model = load_weights(self.model, './networks/efficient_model/efficientnet-b2-8bb594d6.pth')
            # self.model = load_weights(self.model, './networks/efficient_model/efficientnet-b3-5fb5a3c3.pth')
            # self.model = load_weights(self.model, './networks/efficient_model/efficientnet-b4-6ed6700e.pth')
            # self.model = load_weights(self.model, './networks/efficient_model/efficientnet-b5-b6417697.pth')
            # self.model = load_weights(self.model, './networks/efficient_model/efficientnet-b6-c76e70fd.pth')
            # self.model = load_weights(self.model, './networks/efficient_model/efficientnet-b7-dcc49843.pth')

        self.features = self.model.extract_features
        self.num_ftrs = 1280
        # self.num_ftrs = 1280
        # self.num_ftrs = 1408
        # self.num_ftrs = 1536
        # self.num_ftrs = 1792
        # self.num_ftrs = 2048
        # self.num_ftrs = 2304
        # self.num_ftrs = 2560

        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier1 = nn.Sequential(nn.Linear(self.num_ftrs, n_class), nn.Sigmoid())
        #self.classifier2 = nn.Linear(self.num_ftrs, n_class)

    def forward(self, x):
        features = self.features(x)
        f = self.GlobalAvgPool(features)
        f = f.view(f.size(0), -1)
        output1 = self.classifier1(f)
        return output1

