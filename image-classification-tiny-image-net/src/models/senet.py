from timm import create_model
from torch import nn

def get_seresnet():
    model = create_model('seresnet18', pretrained=False)
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False)
    model.fc = nn.Linear(in_features=512, out_features=200)
    return model

def get_seresnext():
    model = create_model('seresnext26d_32x4d', pretrained=False)
    model.fc = nn.Linear(in_features=2048, out_features=200)
    return model

def get_skresnet():
    model = create_model('skresnet18', pretrained=False)
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False)
    model.fc = nn.Linear(in_features=512, out_features=200)
    return model

def get_seresnet34():
    model = create_model('seresnet34', pretrained=False)
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False)
    model.fc = nn.Linear(in_features=512, out_features=200)
    return model

def get_resnet34():
    model = create_model('resnet34', pretrained=False)
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False)
    model.fc = nn.Linear(in_features=512, out_features=200)
    return model

def get_skresnet34():
    model = create_model('skresnet34', pretrained=False)
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False)
    model.fc = nn.Linear(in_features=512, out_features=200)
    return model