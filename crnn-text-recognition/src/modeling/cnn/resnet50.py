from timm import create_model
from torch import nn


class MyIdentity(nn.Module):
    def forward(self, x):
        return x


def get_resnet50():
    model = create_model('resnet50', pretrained=True)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 1), padding=(3,3), bias=False)
    model.maxpool = nn.MaxPool2d(kernel_size=(2,1))
    model.global_pool = MyIdentity()
    model.fc = MyIdentity()
    return model
