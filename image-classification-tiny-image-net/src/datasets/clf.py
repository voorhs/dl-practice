from torch.utils.data import Dataset
import os
from typing import Literal
from PIL import Image

class TinyImagenetDataset(Dataset):
    def __init__(self, path, split: Literal['train', 'test', 'val'], transforms=None):
        self.path = os.path.join(path, split)
        self.split = split
        self.transforms = transforms
        
        self.class_directories = [name for name in os.listdir(self.path)]
        self.n_images_per_class = 500 if split == 'train' else 50
    
    def __len__(self):
        if self.split == 'train':
            return int(1e5)
        return int(1e4)
    
    def __getitem__(self, i):
        """returns (PIL image, int)"""
        if self.split == 'test':
            fname = f'test_{i}.JPEG'
            path = os.path.join(self.path, 'images', fname)
            img = Image.open(path, 'r')
            if self.transforms is not None:
                img = self.transforms(img)
            return img
        
        i_class = i // self.n_images_per_class
        idx_within_class = i % self.n_images_per_class
        class_codename = self.class_directories[i_class]
        class_dir_path = os.path.join(self.path, class_codename)
        if self.split == 'train':
            class_dir_path = os.path.join(class_dir_path, 'images')
        image_names = sorted([name for name in os.listdir(class_dir_path)])
        fname = image_names[idx_within_class]
        path = os.path.join(class_dir_path, fname)
        img = Image.open(path, 'r')

        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, i_class
