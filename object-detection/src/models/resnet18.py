from timm import create_model
import torch.nn as nn
from ..train_utils import LightningCkptLoadable, HParamsPuller
from typing import Tuple, List
import torch
from .utils import Conv


class MyIdentity(nn.Module):
    def forward(self, x):
        return x


def get_resnet18():
    model = create_model('resnet18', pretrained=True)
    model.global_pool = MyIdentity()
    model.fc = MyIdentity()
    return model


class SSD_Resnet18(nn.Module, LightningCkptLoadable, HParamsPuller):
    def __init__(self, n_priors_list: List, n_classes=3):
        super().__init__()

        self.n_priors_list = n_priors_list
        self.n_classes = n_classes

        self.backbone = get_resnet18()
        # self.backbone.requires_grad_(False)
        self.extra_layers = self._extra_layers()

        self.clf_layers, self.loc_layers = self._predictor_heads(n_priors_list)     

    def _extra_layers(self) -> nn.ModuleList:
        layers = nn.ModuleList()
        layers.append(Conv(in_channels=512, out_channels=256, kernel_size=1, stride=1))
        layers.append(Conv(in_channels=256, out_channels=512, kernel_size=3, stride=2))  #
        layers.append(Conv(in_channels=512, out_channels=128, kernel_size=1, stride=1))
        layers.append(Conv(in_channels=128, out_channels=256, kernel_size=3, stride=2, pool=True))  #
        layers.append(Conv(in_channels=256, out_channels=128, kernel_size=1, stride=1))
        layers.append(Conv(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, pool=True)) #
        layers.append(Conv(in_channels=256, out_channels=128, kernel_size=1, stride=1))
        layers.append(Conv(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)) #
        return layers

    def _predictor_heads(self, n_priors_list) -> Tuple[nn.ModuleList, nn.ModuleList]:
        """for each layer in `layers_for_prediction"""

        prediction_fmaps_channels = [512, 512, 256, 256, 256]

        clf_layers, loc_layers = [], []
        for n_boxes, n_channels in zip(n_priors_list, prediction_fmaps_channels):
            clf_layer = nn.Conv2d(
                n_channels,
                n_boxes * self.n_classes,
                kernel_size=3,
                padding=1,
            )
            loc_layer = nn.Conv2d(
                n_channels,
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
        
        x = self.backbone(x)
        outputs_for_prediction.append(x)

        for i, layer in enumerate(self.extra_layers):
            x = layer(x)
            if i % 2 == 1:
                outputs_for_prediction.append(x)
        
        # for x in outputs_for_prediction:
        #     print(x.shape)

        # predict location and class for each saved output
        outputs_clf = []
        outputs_loc = []
        for logits, loc, clf in zip(outputs_for_prediction, self.loc_layers, self.clf_layers):
            outputs_clf.append(clf(logits).permute(0, 2, 3, 1).contiguous())    # (batch, height, width, classes logits)
            outputs_loc.append(loc(logits).permute(0, 2, 3, 1).contiguous())    # (batch, height, width, bbox params)
        
        # resize to desired shape
        clf = torch.cat([o.view(o.size(0), -1) for o in outputs_clf], dim=1)
        loc = torch.cat([o.view(o.size(0), -1) for o in outputs_loc], dim=1)

        clf = clf.view(clf.size(0), -1, self.n_classes)
        loc = loc.view(loc.size(0), -1, 4)

        return loc, clf
