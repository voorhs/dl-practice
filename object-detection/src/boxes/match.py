from .jaccard import get_jaccard
from .conversion import encode, to_corners_form


def match_boxes(threshold, truths, priors, variances, labels, encode_to_regression=True, return_jaccard=False, priors_in_corners=False):
    """
    For each prior bbox find best ground truth bbox and assign it as a target bbox.

    1. Match each prior box with the ground truth box of the highest jaccard overlap,
    2. Encode the bounding boxes
    3. Return the matched indices corresponding to both confidence and location preds.
    
    Arguments
    ---
    - `threshold`: `(float)` The overlap threshold used when mathing boxes
    - `truths`: `(n_obj, 4)` ground truth boxes
    - `priors`: `(n_priors, 4)` prior boxes
    - `variances`: `(n_priors, 4)` variances corresponding to each prior coord
    - `labels`: `(n_obj,)` all the class labels for the image
    
    Return
    ---
    The matched target boxes and labels corresponding to all prior boxes
    - location: `(n_priors, 4)`
    - classes: `(n_priors,)`
    """

    # jaccard index
    if not priors_in_corners:
        overlaps = get_jaccard(truths, to_corners_form(priors))
    else:
        overlaps = get_jaccard(truths, priors)

    # bipartite matching
    best_prior_idx = overlaps.argmax(dim=1)
    best_truth_overlap, best_truth_idx = overlaps.max(dim=0)
    best_truth_overlap.index_fill_(dim=0, index=best_prior_idx, value=2)  # ensure best prior, because max jaccard is 1
    
    # TODO refactor: index best_prior_idx with long tensor
    # ensure every ground truth matches with its prior of max overlap

    # two-sided vertice in a bipartite graph
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    
    target_boxes = truths[best_truth_idx]
    target_labels = labels[best_truth_idx]
    
    # label as background
    target_labels[best_truth_overlap < threshold] = 0

    if encode_to_regression:
        target_boxes = encode(target_boxes, priors, variances)
    
    res = (target_boxes, target_labels)
    if return_jaccard:
        res += (best_truth_overlap,)
    
    return res