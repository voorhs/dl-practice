import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import v2 as T
import torch
from PIL import Image
import os
from typing import Literal


class VOCDetection(Dataset):
    CLASSES = ("background", "license_plate", "car")

    def __init__(self, path, split, transform=None):
        self.path = path
        self.split = split
        self.transform = transform

        samples_names_path = os.path.join(path, 'ImageSets', split + '.txt')
        ids = []
        for line in open(samples_names_path, "r").readlines():
            name = line.strip()
            annot_path = os.path.join(path, "Annotations", name + ".xml")
            img_path = os.path.join(path, "JPEGImages", name + ".jpg")
            ids.append((img_path, annot_path))
        
        self.ids = ids
    
    def __len__(self):
        return len(self.ids)

    def _parse(self, path):
        """
        Arguments
        ---------
        `path`: path to XML annotation of some image"""

        tree = ET.parse(path)
        boxes, labels = [], []
        for child in tree.getroot():
            if child.tag != "object":
                continue
            bndbox = child.find("bndbox")
            box = [float(bndbox.find(t).text) - 1 for t in ["xmin", "ymin", "xmax", "ymax"]]
            label = self.CLASSES.index(child.find("name").text)

            labels.append(label)
            boxes.append(box)

        return torch.tensor(boxes), torch.tensor(labels)

    def __getitem__(self, index):
        """
        - `image`: `(C, H, W)` tensor, preprocessed with `transform`
        - `boxes`: `(n, 4)` bounding boxes for each of `n` objects on image, in format (xmin, ymin, xmax, ymax)
        - `labels`: `(n,)` class index for each object, where `0`: background, `1`: license_plate, `2`: car
        """

        img_path, annot_path = self.ids[index]
        image = Image.open(img_path)
        boxes, labels = self._parse(annot_path)
        if self.transform is not None:
            image = self.transform(image)
            _, height, width = image.shape
        else:
            width, height = image.size   # for PIL image

        # map coords to [0,1]
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height

        return image, boxes, labels
