from torch.utils.data import Dataset
import os
from typing import Literal
from PIL import Image
import torchvision.transforms.v2 as T
import torch


class TinyImagenetDatasetContrastive(Dataset):
    def __init__(self, path, split: Literal['train', 'test', 'val']):
        self.path = os.path.join(path, split)
        self.split = split
        
        aug_1, aug_2 = get_byol_augs()
        self.aug_1 = aug_1
        self.aug_2 = aug_2
        
        self.class_directories = [name for name in os.listdir(self.path)]
        self.n_images_per_class = 500 if split == 'train' else 50

        if self.split == 'test':
            self.images = self._read_test(self.path)
        else:
            self.images = self._read_train_val(self.path)

    def _read_train_val(self, path):
        res = []
        for class_codename in self.class_directories:
            class_dir_path = os.path.join(path, class_codename)
            if self.split == 'train':
                class_dir_path = os.path.join(class_dir_path, 'images')
            image_names = [name for name in os.listdir(class_dir_path)]
            class_images = []
            for fname in image_names:
                img_path = os.path.join(class_dir_path, fname)
                img = Image.open(img_path, 'r')
                class_images.append(img)
            res.append(class_images)
        return res
    
    def _read_test(self, path):
        res = []
        path = os.path.join('images')
        image_names = [name for name in os.listdir(path)]
        for fname in image_names:
            img_path = os.path.join(path, fname)
            img = Image.open(img_path, 'r')
            res.append(img)
        return res

    def __getitem__(self, i):
        """returns (Tensor, Tensor)"""
        if self.split == 'test':
            img = self.images[i]
            aug_1 = self.aug_1(img)
            aug_2 = self.aug_2(img)
            return img
        
        i_class = i // self.n_images_per_class
        idx_within_class = i % self.n_images_per_class
        
        img = self.images[i_class][idx_within_class]

        aug_1 = self.aug_1(img)
        aug_2 = self.aug_2(img)
        
        return aug_1, aug_2

    def __len__(self):
        if self.split == 'train':
            return 200 * 500
        return 200 * 50


def get_byol_augs():
    aug_1 = T.Compose([
        T.RandomResizedCrop(size=(64, 64), scale=(0.3, 1), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=(.6,1.4),hue=0.2,saturation=(.6,1.4))]), p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply(torch.nn.ModuleList([T.GaussianBlur(kernel_size=5, sigma=[.1,2.])]), p=1),
        # T.RandomSolarize(),
        T.Normalize(mean=[0.4650, 0.4342, 0.3944], std=[0.2729, 0.2731, 0.2707]),
    ])
    aug_2 = T.Compose([
        T.RandomResizedCrop(size=(64, 64), scale=(0.3, 1), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=(.6,1.4),hue=0.2,saturation=(.6,1.4))]), p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply(torch.nn.ModuleList([T.GaussianBlur(kernel_size=5, sigma=[.1,2.])]), p=.1),
        T.RandomSolarize(threshold=0.5, p=0.2),
        T.Normalize(mean=[0.4650, 0.4342, 0.3944], std=[0.2729, 0.2731, 0.2707]),
    ])
    return aug_1, aug_2
