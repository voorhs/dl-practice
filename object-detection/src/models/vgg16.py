from torchvision.models import VGG16_Weights
from torchvision import models
import torch.nn as nn
import torch
from typing import Tuple, List
import torch.nn.functional as F
from ..train_utils import LightningCkptLoadable, HParamsPuller


class L2Norm(nn.Module):
    """pointwise normalization then multiply each channel on a trained scale"""
    def __init__(self, n_channels, scale):
        """
        Input tensor is supposed to have shape `(batch, n_channels, height, width)`. `scale` is the initial constant value for the trained parameters.
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.gamma = scale
        
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1, eps=1e-10)
        x = self.weight[None, :, None, None] * x
        return x


class SSD_VGG16(nn.Module, LightningCkptLoadable, HParamsPuller):
    def __init__(self, n_priors_list: List, n_classes=3):
        super().__init__()

        self.num_priors = n_priors_list
        self.num_labels = n_classes

        self.layers_for_prediction = [33, 37, 41, 45, 49]
        self.norm_layer = L2Norm(512, 20)

        base_layers = self._base_layers()
        base_layers.requires_grad_(False)
        extra_layers = self._extra_layers()
        self.total_layers = base_layers + extra_layers

        self.clf_layers, self.loc_layers = self._predictor_heads(self.num_priors, self.layers_for_prediction)

    def _base_layers(self) -> nn.ModuleList:
        """load VGG16 and apply modifications introduced in SSD paper to use it as feature extractor"""
        backbone_model = models.vgg16(weights=VGG16_Weights.DEFAULT)  # False

        base_layers = nn.ModuleList(list(backbone_model.features)[:-1])
        base_layers[16].ceil_mode = True

        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        relu6 = nn.ReLU(inplace=True)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        relu7 = nn.ReLU(inplace=True)   #

        nn.init.xavier_uniform_(conv6.weight)
        nn.init.zeros_(conv6.bias)
        nn.init.xavier_uniform_(conv7.weight)
        nn.init.zeros_(conv7.bias)

        base_layers.extend([pool5, conv6, relu6, conv7, relu7])

        return base_layers

    def _extra_layers(self) -> nn.ModuleList:
        conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
        relu8_1 = nn.ReLU(inplace=True)
        conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        relu8_2 = nn.ReLU(inplace=True)     #
        conv9_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        relu9_1 = nn.ReLU(inplace=True)
        conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        relu9_2 = nn.ReLU(inplace=True)     #
        conv10_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        relu10_1 = nn.ReLU(inplace=True)
        conv10_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        relu10_2 = nn.ReLU(inplace=True)    #
        conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        relu11_1 = nn.ReLU(inplace=True)
        conv11_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        relu11_2 = nn.ReLU(inplace=True)    #

        nn.init.xavier_uniform_(conv8_1.weight)
        nn.init.zeros_(conv8_1.bias)
        nn.init.xavier_uniform_(conv8_2.weight)
        nn.init.zeros_(conv8_2.bias)
        nn.init.xavier_uniform_(conv9_1.weight)
        nn.init.zeros_(conv9_1.bias)
        nn.init.xavier_uniform_(conv9_2.weight)
        nn.init.zeros_(conv9_2.bias)
        nn.init.xavier_uniform_(conv10_1.weight)
        nn.init.zeros_(conv10_1.bias)
        nn.init.xavier_uniform_(conv10_2.weight)
        nn.init.zeros_(conv10_2.bias)
        nn.init.xavier_uniform_(conv11_1.weight)
        nn.init.zeros_(conv11_1.bias)
        nn.init.xavier_uniform_(conv11_2.weight)
        nn.init.zeros_(conv11_2.bias)

        return nn.ModuleList([
            conv8_1,
            relu8_1,
            conv8_2,
            relu8_2,
            conv9_1,
            relu9_1,
            conv9_2,
            relu9_2,
            conv10_1,
            relu10_1,
            conv10_2,
            relu10_2,
            conv11_1,
            relu11_1,
            conv11_2,
            relu11_2,
        ])

    def _predictor_heads(self, num_priors, layers_for_prediction) -> Tuple[nn.ModuleList, nn.ModuleList]:
        """for each layer in `layers_for_prediction"""

        clf_layers, loc_layers = [], []
        for n_boxes, layer_id in zip(num_priors, layers_for_prediction):
            clf_layer = nn.Conv2d(
                self.total_layers[layer_id].out_channels,
                n_boxes * self.num_labels,
                kernel_size=3,
                padding=1,
            )
            loc_layer = nn.Conv2d(
                self.total_layers[layer_id].out_channels,
                n_boxes * 4,
                kernel_size=3,
                padding=1,
            )

            nn.init.xavier_uniform_(clf_layer.weight)
            nn.init.zeros_(clf_layer.bias)
            nn.init.xavier_uniform_(loc_layer.weight)
            nn.init.zeros_(loc_layer.bias)

            clf_layers += [clf_layer]
            loc_layers += [loc_layer]

        clf_layers = nn.ModuleList(clf_layers)
        loc_layers = nn.ModuleList(loc_layers)

        return clf_layers, loc_layers

    def forward(self, x):
        """
        Predict classes and anchor (bbox) adjustments. Number of bboxes is height * width summated for each of layers for prediction.

        Arguments
        ---
        `x`: `(B, C, H, W)`

        Returns
        ---
        - `loc`: `(batch, n_boxes, 4)`
        - `clf`: `(batch, n_boxes, n_classes)`
        """

        # extract features and save some outputs
        outputs_for_prediction = []
        for i_layer, layer in enumerate(self.total_layers, -1):
            x = layer(x)
            if i_layer in self.layers_for_prediction:
                outputs_for_prediction.append(x if i_layer != 21 else self.norm_layer(x))
                # layer 21 is the earliest one in `layers_for_prediction`,
                # so its output can be unstable, thus normalization is required
        
        # predict location and class for each saved output
        outputs_clf = []
        outputs_loc = []
        for logits, loc, clf in zip(outputs_for_prediction, self.loc_layers, self.clf_layers):
            outputs_clf.append(clf(logits).permute(0, 2, 3, 1).contiguous())    # (batch, height, width, classes logits)
            outputs_loc.append(loc(logits).permute(0, 2, 3, 1).contiguous())    # (batch, height, width, bbox params)
        
        # resize to desired shape
        clf = torch.cat([o.view(o.size(0), -1) for o in outputs_clf], dim=1)
        loc = torch.cat([o.view(o.size(0), -1) for o in outputs_loc], dim=1)

        clf = clf.view(clf.size(0), -1, self.num_labels)
        loc = loc.view(loc.size(0), -1, 4)

        return loc, clf
