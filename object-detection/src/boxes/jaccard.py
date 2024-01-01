import torch


def get_intersect(box_a, box_b):
    """
    Comptue intersection area of two sets of bboxes in corners form.

    Arguments
    ---
    - `box_a`: `(A, 4)`
    - `box_b`: `(B, 4)`
    
    Return
    ---
    intersection area: `(A, B)`
    """
    A = box_a.shape[0]
    B = box_b.shape[0]

    max_xy = torch.min(
        box_a[:, None, 2:].expand(A, B, 2),
        box_b[None, :, 2:].expand(A, B, 2),
    )
    min_xy = torch.max(
        box_a[:, None, :2].expand(A, B, 2),
        box_b[None, :, :2].expand(A, B, 2),
    )
    
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def get_jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes in corners form.

    Arguments
    ---
    - `box_a`: `(A, 4)`
    - `box_b`: `(B, 4)`

    Return
    ---
    jaccard overlap: `(A, B)`
    """

    inter = get_intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union
