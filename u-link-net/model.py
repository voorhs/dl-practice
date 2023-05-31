import torch
import torch.nn.functional as F
from torchvision.models import vgg13, VGG13_Weights


# UNet

class VGG13Encoder(torch.nn.Module):
    def __init__(self, num_blocks, weights=VGG13_Weights.DEFAULT):
        super().__init__()
        self.num_blocks = num_blocks

        # Будем использовать предобученную VGG13 в качестве backbone
        feature_extractor = vgg13(weights=weights).features

        # Каждый блок энкодера U-Net — это блок VGG13 без MaxPool2d
        self.blocks = torch.nn.ModuleList()
        for idx in range(self.num_blocks):
            self.blocks.append(torch.nn.Sequential(
                feature_extractor[5*idx],
                feature_extractor[5*idx+1],
                feature_extractor[5*idx+2],
                feature_extractor[5*idx+3],
            ))

    def forward(self, x):
        acts = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            acts.append(x)
            if idx + 1 != len(self.blocks):
                x = F.max_pool2d(x, kernel_size=2)
        return acts


class DecoderBlock(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.upconv = torch.nn.Conv2d(
            in_channels=out_channels*2, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1
        )

        self.conv1 = torch.nn.Conv2d(
            in_channels=out_channels*2, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1
        )

        self.conv2 = torch.nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1
        )

    def forward(self, down, left):
        # upscale
        x = F.interpolate(down, scale_factor=2, mode='nearest')
        x = self.upconv(x)

        # skip connection
        x = torch.cat([left, x], dim=1)

        # feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x


class Decoder(torch.nn.Module):
    def __init__(self, num_filters, num_blocks):
        super().__init__()

        self.blocks = torch.nn.ModuleList()
        for idx in range(num_blocks):
            self.blocks.insert(0, DecoderBlock(num_filters * 2 ** idx))

    def forward(self, acts):
        up = acts[-1]
        for block, left in zip(self.blocks, acts[-2::-1]):
            up = block(up, left)
        return up


class UNet(torch.nn.Module):
    def __init__(self, num_classes=1, num_blocks=4):
        super().__init__()

        self.encoder = VGG13Encoder(num_blocks=num_blocks)
        self.decoder = Decoder(num_filters=64, num_blocks=num_blocks-1)
        self.final = torch.nn.Conv2d(
            in_channels=64, out_channels=num_classes,
            kernel_size=1, padding=0, dilation=1
        )

    def forward(self, x):
        acts = self.encoder(x)
        x = self.decoder(acts)
        x = self.final(x)
        return x

# LinkNet


class LinkNetDecoderBlock(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.upconv = torch.nn.Conv2d(
            in_channels=out_channels*2, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1
        )

        self.conv1 = torch.nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1
        )

        self.conv2 = torch.nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1
        )

    def forward(self, down, left):
        # upscale
        x = F.interpolate(down, scale_factor=2, mode='nearest')
        x = self.upconv(x)

        # skip connection
        x = left + x

        # feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x


class LinkNetDecoder(torch.nn.Module):
    def __init__(self, num_filters, num_blocks):
        super().__init__()

        self.blocks = torch.nn.ModuleList()
        for idx in range(num_blocks):
            self.blocks.insert(0, LinkNetDecoderBlock(num_filters * 2 ** idx))

    def forward(self, acts):
        up = acts[-1]
        for block, left in zip(self.blocks, acts[-2::-1]):
            up = block(up, left)
        return up


class LinkNet(torch.nn.Module):
    def __init__(self, num_classes=1, num_blocks=4):
        super().__init__()

        self.encoder = VGG13Encoder(num_blocks=num_blocks)
        self.decoder = LinkNetDecoder(num_filters=64, num_blocks=num_blocks-1)
        self.final = torch.nn.Conv2d(
            in_channels=64, out_channels=num_classes,
            kernel_size=1, padding=0, dilation=1
        )

    def forward(self, x):
        acts = self.encoder(x)
        x = self.decoder(acts)
        x = self.final(x)
        return x
