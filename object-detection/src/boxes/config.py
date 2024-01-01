from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class PriorBoxesConfig:
    feature_maps: List[Tuple[int, int]]
    min_sizes: List[float]
    max_sizes: List[float]
    aspect_ratios: List[List[int]]
    clip: bool
    n_classes: int


@dataclass
class VGG16PriorBoxesConfig:
    feature_maps=[(45,80), (23,40), (12,20), (10,18), (8,16)]
    min_sizes=[0.1, 0.20, 0.37, 0.54, 0.71]
    max_sizes=[0.2, 0.37, 0.54, 0.71, 1.00]
    aspect_ratios=[[], [], [], [], [], []]
    clip=True
    n_classes=3


@dataclass
class Resnet18PriorBoxesConfig:
    feature_maps=[(23,40), (11,19), (2,4), (1,2), (1,2)]
    min_sizes=[0.1, 0.20, 0.37, 0.54, 0.71]
    max_sizes=[0.2, 0.37, 0.54, 0.71, 1.00]
    aspect_ratios=[[], [], [], [], [], []]
    clip=True
    n_classes=3
