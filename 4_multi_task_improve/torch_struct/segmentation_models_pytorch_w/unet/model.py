from typing import Optional, Union, List
from .decoder import UnetDecoder
from ..encoders import get_encoder
from ..base import SegmentationModel
from ..base import SegmentationHead, ClassificationHead
from ..resnest_encoder.resnest import resnest200,resnest269,resnest50,resnest101
import torch
import torch.nn as nn







class Unet(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_depth (int): number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature tensor will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
            One of [True, False, 'inplace']
        decoder_attention_type: attention module used in decoder of the model
            One of [``None``, ``scse``]
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function to apply after final convolution;
            One of [``sigmoid``, ``softmax``, ``logsoftmax``, ``identity``, callable, None]
        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)
        dual_seg_path:是否采用双路径输出(用于多任务,比如分割and还原任务一起做)

    Returns:
        ``torch.nn.Module``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        dual_seg_path = False,
        weight_for_resnest = None,
    ):
        super().__init__()

        # encoder 部分
        # 注意,resnest输入必须是3通道的
        if encoder_name == 'resnest50':
            self.encoder = resnest50(pretrained=False,encoder_depth = encoder_depth)
            if weight_for_resnest is not None:
                self.encoder.load_state_dict(torch.load(weight_for_resnest))
                print('load encoder weight from ',weight_for_resnest)
        elif encoder_name == 'resnest101':
            self.encoder = resnest101(pretrained=False,encoder_depth = encoder_depth)
            if weight_for_resnest is not None:
                self.encoder.load_state_dict(torch.load(weight_for_resnest))
                print('load encoder weight from ', weight_for_resnest)
        elif encoder_name == 'resnest200':
            self.encoder = resnest200(pretrained=False,encoder_depth = encoder_depth)
            if weight_for_resnest is not None:
                self.encoder.load_state_dict(torch.load(weight_for_resnest))
                print('load encoder weight from ', weight_for_resnest)
        elif encoder_name == 'resnest269':
            self.encoder = resnest269(pretrained=False,encoder_depth = encoder_depth)
            if weight_for_resnest is not None:
                self.encoder.load_state_dict(torch.load(weight_for_resnest))
                print('load encoder weight from ', weight_for_resnest)
        else:
            self.encoder = get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=encoder_weights,
            )
        print('encoder out_channels :',self.encoder.out_channels)

        # decoder 部分
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        # old
        # self.segmentation_head = SegmentationHead(
        #     in_channels=decoder_channels[-1],
        #     out_channels=classes,
        #     activation=activation,
        #     kernel_size=3,
        # )


        # ==code by wang=====================================================
        # seghead 部分
        # new:multi head for deep supervised
        # list中head的通道由高到底,第-1个head是最高分辨率的输出,如[256, 128, 64, 32, 16]
        self.dual_seg_path = dual_seg_path
        pathh = 1
        if self.dual_seg_path:
            pathh = 2
        print('seg model with ',encoder_depth,' heads and', pathh,' seg paths')
        self.segmentation_head_list = []
        for _decoder_channel in decoder_channels:
            self.segmentation_head_list.append(SegmentationHead(
            in_channels=_decoder_channel,
            out_channels=classes,
            activation=activation,
            kernel_size=3,))

        #必须加下面这一行,不然就废了,权重是无法被torch识别的,只能在cpu跑
        self.segmentation_head_list = nn.ModuleList(self.segmentation_head_list)

        if self.dual_seg_path: # 如果双路径,则添加辅助head
            self.aux_segmentation_head_list = []
            for _decoder_channel in decoder_channels:
                self.aux_segmentation_head_list.append(SegmentationHead(
                    in_channels=_decoder_channel,
                    out_channels=classes,
                    activation=activation,
                    kernel_size=3, ))
            self.aux_segmentation_head_list = nn.ModuleList(self.aux_segmentation_head_list)
        # =====================================================================



        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()
