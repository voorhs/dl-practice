import torch
from .conversion import decode
import torchvision


def detect_objects(
    loc_data, clf_data, prior_data, num_classes, overlap_threshold, conf_threshold
):
    """
    Arguments
    ---
    - `loc_data`: `(batch, n_priors, 4)`
    - `clf_data`: `(batch, n_priors, n_classes)`
    - `prior_data`: `(num_priors, 4)` prior boxes in center-offset format
    
    Return
    ---
    - `box_data`: list of `(n_objects, 4)`
    - `label_data`: list of `(n_objects,)`
    - `clf_data`: list of `(n_objects,)`
    """
    device = loc_data.device
    variance = torch.tensor([0.1, 0.2], device=device)

    num = loc_data.size(0)

    res_boxes, res_conf_scores, res_labels = [], [], []
    clf_preds = torch.nn.functional.sigmoid(clf_data.transpose(2, 1))

    for i in range(num):
        decoded_boxes = decode(loc_data[i], prior_data, variance)
        
        conf_scores = clf_preds[i]

        cur_boxes, cur_conf_scores, cur_labels = [], [], []
        for i_class in range(1, num_classes):   # skip background class
            mask = conf_scores[i_class] >= conf_threshold
            scores = conf_scores[i_class][mask]
            if scores.size(0) == 0:
                continue

            boxes = decoded_boxes[mask].view(-1, 4)
            ids = torchvision.ops.nms(boxes, scores, overlap_threshold)
            count = ids.size(0)

            cur_boxes.append(boxes[ids[:count]])
            cur_conf_scores.append(scores[ids[:count]])
            cur_labels.append(torch.full((count,), i_class))

        res_boxes.append(
            torch.cat(cur_boxes, dim=0)
            if len(cur_boxes) > 0
            else torch.Tensor(0)
        )
        res_conf_scores.append(
            torch.cat(cur_conf_scores, dim=0)
            if len(cur_conf_scores) > 0
            else torch.Tensor(0)
        )
        res_labels.append(
            torch.cat(cur_labels, dim=0)
            if len(cur_labels) > 0
            else torch.Tensor(0)
        )

    return res_boxes, res_labels, res_conf_scores
