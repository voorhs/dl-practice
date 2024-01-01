from timm import create_model
from torch import nn


class MyIdentity(nn.Module):
    def forward(self, x):
        return x


def get_efficient_net():
    model = create_model('efficientnetv2_rw_s', pretrained=True)
    model.conv_stem = nn.Conv2d(3, 24, kernel_size=(3, 3), stride=(2, 1), padding=(1,1), bias=False)
    model.global_pool = MyIdentity()
    model.classifier = MyIdentity()
    return model
