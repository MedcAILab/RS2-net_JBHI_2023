import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        for segmentation_head in self.segmentation_head_list:
            init.initialize_head(segmentation_head)
        if self.dual_seg_path:
            for aux_segmentation_head in self.aux_segmentation_head_list:
                init.initialize_head(aux_segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        # get multiscope feature maps from encoder
        features = self.encoder(x)

        # get multiscope feature maps from decoder
        x_list = self.decoder(*features)

        # feed feature maps into each heads
        masks_list = []
        for indexx, segmentation_head in enumerate(self.segmentation_head_list):
            masks_list.append(segmentation_head(x_list[indexx]))

        if self.dual_seg_path:
            aux_masks_list = []
            for indexx, aux_segmentation_head in enumerate(self.aux_segmentation_head_list):
                aux_masks_list.append(aux_segmentation_head(x_list[indexx]))


        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            if self.dual_seg_path:
                return masks_list, aux_masks_list, labels
            else:
                return masks_list, labels

        if self.dual_seg_path:
            return masks_list, aux_masks_list
        else:
            return masks_list

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x
