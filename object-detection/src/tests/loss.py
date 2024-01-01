from ..boxes import generate_prior_boxes
import torch, numpy
from ..loss import MultiBoxLoss


overlap_threshold = 0.5
num_classes       = 3
variance          = [0.1, 0.2] 
neg_pos_ratio     = 3
batch_size        = 1

custom_config = {
    'num_classes': 3,
    'feature_maps' : [(45,80), (23,40), (12,20), (6,10), (3,5), (2,3)], #ResNet18
    'min_dim'      : 300,
    'min_sizes'    : [0.1, 0.20, 0.37, 0.54, 0.71, 1.00],
    'max_sizes'    : [0.2, 0.37, 0.54, 0.71, 1.00, 1.05],
    
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance'     : [0.1, 0.2],
    'clip'         : True,
}

prior_box_s = generate_prior_boxes(custom_config)
num_priors  = prior_box_s.shape[0]

conf_s = torch.ones ( batch_size, num_priors, num_classes)
loc_s  = torch.zeros( batch_size, num_priors,           4)

gt_label_s = [ torch.from_numpy( numpy.array([1, 2]) ) ]
gt_box_s   = [ torch.from_numpy( numpy.array([[0.0, 0.0, 0.5, 0.5],[0.5, 0.5, 1.0, 1.0]]) ) ]

multi_box_loss = MultiBoxLoss( overlap_threshold, neg_pos_ratio)

print( multi_box_loss.forward((loc_s, conf_s, prior_box_s),(gt_label_s, gt_box_s)) )
