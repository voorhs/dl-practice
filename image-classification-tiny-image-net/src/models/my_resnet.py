import torch.nn as nn
from ..train_utils import HParamsPuller, LightningCkptLoadable


def conv_block(in_channels, x_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, x_channels, kernel_size=3, padding=1, bias=True), 
        nn.BatchNorm2d(x_channels), 
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))

    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(n_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(n_channels), 
        )
        self.act_fn = nn.ReLU(inplace=True)

        self._init()
    
    def _init(self):
        nn.init.zeros_(self.layers[-1].weight.data)
    
    def forward(self, x):
        identity = x
        x = self.layers(x)
        x += identity
        x = self.act_fn(x)
        return x


class MyResNet(nn.Module, HParamsPuller, LightningCkptLoadable):
    def __init__(self, planes=32):
        super().__init__()
        self.planes = planes
        self.conv1 = conv_block(3, planes)
        self.conv2 = conv_block(planes, planes*2, pool=True) 
        planes = planes * 2
        self.res1 = ResBlock(planes)
        
        self.conv3 = conv_block(planes, planes*2, pool=True)
        planes = planes * 2
        self.conv4 = conv_block(planes, planes*2, pool=True) 
        planes = planes * 2
        self.res2 = ResBlock(planes)
        self.conv5 = conv_block(planes, planes*2, pool=True) 
        planes = planes * 2
        self.res3 = ResBlock(planes)
        
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(planes*4, 200)
        )

        self._init()
    
    def _init(self):
        for m in self.modules():
            if not isinstance(m, (nn.Linear, nn.Conv2d)):
                continue
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        x = self.conv5(x)
        x = self.res3(x) + x
        x = self.max_pool(x)
        x = self.fc(x)
        return x
