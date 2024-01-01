import torch.nn.functional as F
import torch.nn as nn

from .boxes import match_boxes

import numpy
import torch


class MultiBoxLoss(nn.Module):
    """
    SSD Weighted Loss Function

    Compute Targets:
    1. Produce Confidence Target Indices by matching  ground truth boxes
        with (default) 'priorboxes' that have jaccard index > threshold parameter
        (default threshold: 0.5).
    2. Produce localization target by 'encoding' variance into offsets of ground
        truth boxes and their matched  'priorboxes'.
    3. Hard negative mining to filter the excessive number of negative examples
        that comes with using a large number of default bounding boxes.
        (default negative:positive ratio 3:1)

    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + alpha * Lloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, overlap_threshold, neg_pos_ratio, variance):
        super().__init__()
        self.threshold = overlap_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.variance = variance

    def forward(self, predictions, targets):
        """
        Multibox Loss

        Arguments
        ---
        - `predictions` (tuple)
            - `loc`: `(batch_size, n_priors, 4)`
            - `conf`: `(batch_size, n_priors, n_classes)`
            - `priors`: `(n_priors, 4)`
        - `targets` (tuple)
            - `labels`: list of `(n_objs)`
            - `loc`: list of `(n_objs, 4)`
        """

        pred_loc, pred_conf, priors = predictions
        target_labels_raw, target_boxes_raw = targets

        device = pred_loc.device

        # match prior and ground truth boxes
        target_loc = []
        target_labels = []
        batch_size = pred_loc.size(0)
        for idx in range(batch_size):
            loc, labels = match_boxes(
                self.threshold,
                target_boxes_raw[idx],
                priors,
                self.variance,
                target_labels_raw[idx]
            )
            target_loc.append(loc)
            target_labels.append(labels)

        target_loc = torch.stack(target_loc, dim=0).to(device)
        target_labels = torch.stack(target_labels, dim=0).to(device)

        # 0 is background
        pos = target_labels != 0

        # localization loss
        loss_l = F.smooth_l1_loss(
            pred_loc[pos].view(-1, 4),
            target_loc[pos].view(-1, 4),
            reduction="sum"
        )

        # classification loss
        n_classes = pred_conf.size(-1)
        n_priors = pred_loc.size(1)
        loss_c = F.cross_entropy(
            pred_conf.view(-1, n_classes),
            target_labels.view(-1),
            reduction="none"
        )
        loss_c = loss_c.view(batch_size, n_priors)
        loss_c_pos = loss_c[pos].sum()
        loss_c_neg, num_pos = self._negative_loss(pos, loss_c)

        # Finally we normalize the losses by the number of positives
        N = num_pos.sum()
        loss_l = loss_l / N
        loss_c = (loss_c_pos + loss_c_neg) / N

        return loss_l, loss_c

    def _negative_loss(self, pos, loss_c):
        """Hard negative mining, filter out the positive samples and pick the top negative losses."""
        num_pos = pos.sum(dim=1, keepdim=True)
        num_neg = torch.clamp(self.neg_pos_ratio * num_pos, max=pos.size(1) - 1).long()
        loss_c_neg = loss_c * ~pos
        loss_c_neg, _ = loss_c_neg.sort(dim=1, descending=True)
        neg_mask = torch.zeros_like(loss_c_neg)
        batch_size = loss_c.shape[0]
        neg_mask[torch.arange(batch_size), num_neg.view(-1)] = 1.0
        neg_mask = 1 - neg_mask.cumsum(dim=-1)
        loss_c_neg = (loss_c_neg * neg_mask).sum()
        return loss_c_neg, num_pos
