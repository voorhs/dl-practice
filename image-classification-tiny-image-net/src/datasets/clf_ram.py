from torch.utils.data import Dataset
import os
from typing import Literal
from PIL import Image

class TinyImagenetDatasetRAM(Dataset):
    def __init__(self, path, split: Literal['train', 'test', 'val'], transforms=None):
        self.path = os.path.join(path, split)
        self.split = split
        self.transforms = transforms
        
        self.class_directories = sorted([name for name in os.listdir(self.path)])
        self.n_images_per_class = 500 if split == 'train' else 50

        if self.split == 'test':
            self.images, self.image_names = self._read_test(self.path)
        else:
            self.images, self.image_names = self._read_train_val(self.path)

    def _read_train_val(self, path):
        res = []
        image_names = []
        for class_codename in self.class_directories:
            class_dir_path = os.path.join(path, class_codename)
            if self.split == 'train':
                class_dir_path = os.path.join(class_dir_path, 'images')
            image_names.append(sorted([name for name in os.listdir(class_dir_path)]))
            class_images = []
            for fname in image_names[-1]:
                img_path = os.path.join(class_dir_path, fname)
                img = Image.open(img_path, 'r')
                class_images.append(img)
            res.append(class_images)
        return res, image_names
    
    def _read_test(self, path):
        res = []
        path = os.path.join(path, 'images')
        image_names = sorted([name for name in os.listdir(path)])
        for fname in image_names:
            img_path = os.path.join(path, fname)
            img = Image.open(img_path, 'r')
            res.append(img)
        return res, image_names

    def __getitem__(self, i):
        """returns (PIL image, int)"""
        if self.split == 'test':
            img = self.images[i]
            if self.transforms is not None:
                img = self.transforms(img)
            return img
        
        i_class = i // self.n_images_per_class
        idx_within_class = i % self.n_images_per_class
        
        img = self.images[i_class][idx_within_class]

        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, i_class

    def __len__(self):
        if self.split == 'train':
            return 200 * 500
        return 200 * 50

