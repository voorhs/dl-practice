from ..boxes import generate_prior_boxes, encode, decode
import torch
import numpy

custom_config = {
    "num_classes": 3,
    "feature_maps": [
        (45, 80),
        (23, 40),
        (12, 20),
        (6, 10),
        (3, 5),
        (2, 3),
    ],  # ResNet18
    "min_dim": 300,
    "min_sizes": [0.1, 0.20, 0.37, 0.54, 0.71, 1.00],
    "max_sizes": [0.2, 0.37, 0.54, 0.71, 1.00, 1.05],
    "aspect_ratios": [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    "variance": [0.1, 0.2],
    "clip": True,
}

threshold = 0.9
variance = [0.1, 0.2]

prior_box_s = generate_prior_boxes(custom_config)
prior_loc_s = torch.zeros(prior_box_s.shape)

result_box_s = decode(prior_loc_s, prior_box_s, variance)

result_loc_s = encode(result_box_s, prior_box_s, variance)

gt_label_s = torch.from_numpy(numpy.array([1, 2]))
gt_box_s = torch.from_numpy(
    numpy.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]])
)

num_gt_objects = 2
num_priors = prior_box_s.shape[0]

loc, conf = match_boxes(threshold, gt_box_s, prior_box_s, variance, gt_label_s)
