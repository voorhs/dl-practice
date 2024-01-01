from .match import match_boxes
import torch
import numpy as np
from bisect import bisect_right
from typing import Literal, Union
from torchmetrics.functional import precision_recall_curve
from sklearn.metrics import auc


def mAP(
        predictions, targets,
        threshold, n_classes,
        interpolation: Union[Literal['rectangle', 'trapezoid', 'maximum'], int],
        reduce=True
    ):
    """
    Arguments
    ---
    - `predictions` from `detect_objects()`
        - `loc`: list of `(n_objects, 4)` in corners from
        - `label`: list of `(n_objects,)`
        - `conf`: list of `(n_objects,)`
    - `targets` these lists have different length comparing to lists from `predictions`
        - `label`: list of `(n_objects,)`
        - `loc`: list of `(n_objects, 4)`
    """
    pred_loc, pred_label, pred_conf = predictions
    target_label_raw, target_loc_raw = targets

    # match prior and ground truth boxes
    target_loc = []
    target_labels = []
    batch_size = len(target_label_raw)
    for idx in range(batch_size):
        loc, labels = match_boxes(
            threshold,
            target_loc_raw[idx],
            pred_loc[idx],
            variances=None,
            labels=target_label_raw[idx],
            encode_to_regression=False,
            return_jaccard=False,
            priors_in_corners=True
        )
        target_loc.extend(torch.unbind(loc, dim=0))
        target_labels.extend(torch.unbind(labels, dim=0))

    target_loc = torch.stack(target_loc, dim=0)
    target_labels = torch.stack(target_labels, dim=0)
    
    # lists to np.array
    pred_loc = torch.concatenate(pred_loc, dim=0)
    pred_label = torch.concatenate(pred_label, dim=0)
    pred_conf = torch.concatenate(pred_conf, dim=0)

    classwise_metrics = []
    for i in range(1, n_classes):   # skip background
        mask = (pred_label == i)
        prec, recall, _ = precision_recall_curve(
            target=(target_labels[mask] == i),
            preds=pred_conf[mask],
            task='binary'
        )
        classwise_metrics.append(_mAP(interpolation, recall, prec))
    
    if reduce:
        return np.mean(classwise_metrics)
    return classwise_metrics


def _mAP(interpolation, recall, precision):
    recall = recall.cpu().tolist()
    precision = precision.cpu().tolist()
    if isinstance(interpolation, int):
        return n_points_interp(recall, precision, interpolation)
    elif interpolation == 'trapezoid':
        return trapezoid(recall, precision)
    elif interpolation == 'rectangle':
        return rectange(recall, precision)
    elif interpolation == 'maximum':
        return maximum(recall, precision)
    else:
        raise ValueError(f'unknown interpolation {interpolation}')

def trapezoid(recall, precision):
    return auc(recall, precision)

def _flip(x):
    return x[::-1]

def rectange(recall, precision):
    r, p = recall, precision
    r = _flip(r)
    p = _flip(p)
    r = [0] + r
    return sum((r[i+1] - r[i]) * p[i] for i in range(len(p)))

def maximum(recall, precision):
    r, p = recall, precision
    r = _flip(r)
    p = _flip(p)
    r = [0] + r
    # r = torch.nn.functional.pad(r, pad=(1, 0), value=0)
    return sum((r[i+1] - r[i]) * max(p[i:]) for i in range(len(p)))

def n_points_interp(recall, precision, n_points):
    r, p = recall, precision
    r_old = _flip(r)
    p_old = _flip(p)
    r_new = list(i / (n_points - 1) for i in range(n_points))
    p_new = [p_old[bisect_right(r_old, r)-1] for r in r_new]
    # p_new = [p_old[torch.searchsorted(r_old, r, right=True)-1] for r in r_new]
    return auc(r_new, p_new)
