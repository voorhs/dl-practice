from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2 as T
import os
import torch
import json
import numpy as np


class RecognitionDataset(Dataset):
    def __init__(self, path, split, alphabet, transforms=None):
        """
        Constructor for class.
        
        Args:
            - alphabet: String of chars required for predicting.
        """
        super().__init__()

        self.path = path
        self.split = split
        self.alphabet = alphabet
        
        if split in ['val', 'train', 'val-2', 'train-2']:
            config_path = os.path.join(path, f'{split}.json')
            self.image_files = json.load(open(config_path, 'r'))
        elif split == 'test':
            self.image_files = sorted(os.listdir(os.path.join(path, split)), key=lambda x: int(x.split('.')[0]))
        else:
            raise ValueError('split must be one of `train`, `val`, `test`')

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Resize(size=(32, 160), antialias=True)
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        image_file = self.image_files[i]
        folder = 'labeled' if self.split != 'test' else 'test'
        image = Image.open(os.path.join(self.path, folder, image_file))
        image = self.transforms(self.to_rgb(image))

        if self.split == 'test':
            return image

        text = image_file[10:-4]

        return dict(
            image=image,
            text=text,
        )

    def to_rgb(self, img):
        """Along with RGB there are RGBA and grayscale images in dataset. Convert all of them to RGB"""
        shape = np.array(img).shape
        if len(shape) == 2:
            res = Image.new("RGB", img.size)
            res.paste(img)
            return res
        elif shape[2] == 4:
            img.load() # required for .split()
            res = Image.new("RGB", img.size, (255, 255, 255))
            res.paste(img, mask=img.split()[3]) # 3 is the alpha channel
            return res
            
        return img
