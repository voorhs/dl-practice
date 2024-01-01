import torch


def encode(matched, priors, variances):
    """
    Encode target bboxes to offset regression task.

    Arguments
    ---
    - `matched`: `(n_priors, 4)` coords of ground truth for each prior in corners form
    - `priors`: `(n_priors, 4)` prior boxes in center-offset form
    - `variances`: `(list[float])` variances of prior boxes

    Return
    ---
    encoded boxes: `(n_priors, 4)`
    """

    # delta between true and prior box centers scaled to [0,1]
    # (true center - prior center) / scale
    delta_centers = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    delta_centers /= priors[:, 2:] * variances[0]

    # logarithmic delta between true and prior box scales
    # log(true scale / prior scale)
    delta_scales = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    delta_scales = torch.log(delta_scales) / variances[1]

    # return target for smooth_l1_loss
    return torch.cat([delta_centers, delta_scales], dim=1)


def decode(loc, priors, variances):
    """
    Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.

    Arguments
    ---
    - `loc`: `(n_priors, 4)` location predictions for loc layers
    - `priors`: `(n_priors, 4)` prior boxes in center-offset form.
    - `variances`: `(list[float])` variances of prior boxes

    Return
    ---
    decoded bounding box predictions: `(n_priors, 4)` in a corners format
    """
    # center-offset format
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * priors[:, 2:] * variances[0],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
    ), dim=1)

    # corners format
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    
    return boxes


def to_corners_form(boxes):
    """
    Perform conversion from `(cx, cy, w, h)` to `(xmin, ymin, xmax, ymax)`.
    """
    return torch.cat([boxes[:, :2] - 0.5 * boxes[:, 2:], boxes[:, :2] + 0.5 * boxes[:, 2:]], dim=1)  
