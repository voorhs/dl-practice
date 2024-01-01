from torch.utils.data import Dataset
import os
from typing import Literal
from PIL import Image
import torchvision.transforms.v2 as T
from .contrastive import get_byol_augs


class ImagenetContrastive(Dataset):
    def __init__(self, path, split: Literal['train', 'test', 'val']):
        self.path = os.path.join(path, split)
        self.split = split
        
        aug_1, aug_2 = get_byol_augs()
        self.aug_1 = aug_1
        self.aug_2 = aug_2

        self.image_names = [name for name in os.listdir(self.path)]

    def __getitem__(self, i):
        """returns (Tensor, Tensor)"""
        
        name = self.image_names[i]
        path = os.path.join(self.path, name)
        img = Image.open(path, 'r')

        aug_1 = self.aug_1(img)
        aug_2 = self.aug_2(img)
        
        return aug_1, aug_2

    def __len__(self):
        return len(self.image_names)
