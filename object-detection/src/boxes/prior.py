import itertools as it
from math import sqrt
import torch
from .config import PriorBoxesConfig


def generate_prior_boxes(config: PriorBoxesConfig):
    """Generate a list of prior boxes in the center-offset format
    
    Return
    ---
    `boxes`: `(n_boxes, 4)`, format `(cx, cy, w, h)`
    """

    res = []
    for k, (height, width) in enumerate(config.feature_maps):
        for i, j in it.product(range(height), range(width)):
            # aspect_ratio: 1:1
            # scale: min_size
            cx = (j + 0.5) / width
            cy = (i + 0.5) / height
            scale = config.min_sizes[k]
            res += [cx, cy, scale, scale]
            
            # aspect_ratio: 1:1
            # scale: sqrt(s_k * s_(k+1)), i.e. geometric mean of two scales
            scale_extra = sqrt(config.min_sizes[k] * (config.max_sizes[k]))
            res += [cx, cy, scale_extra, scale_extra]
            
            # rest of aspect ratios
            for ar in config.aspect_ratios[k]:
                res += [cx, cy, scale * sqrt(ar), scale / sqrt(ar)]
                res += [cx, cy, scale / sqrt(ar), scale * sqrt(ar)]
    
    # back to torch land
    res = torch.Tensor(res).view(-1, 4)
    
    if config.clip:
        res.clamp_(max=1, min=0)

    return res
