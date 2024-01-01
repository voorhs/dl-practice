import torch


class DiceLoss(torch.nn.Module):
    def __init__(self, eps=1e-7, reduction=None, with_logits=True):
        """
        Arguments
        ---------
        eps : float
            eps in denominator
        reduction : Optional[str] (None, 'mean' or 'sum')
            specifies the reduction to apply to the output:

            None: no reduction will be applied
            'mean': the sum of the output will be divided by the number of elements in the batch
            'sum':  the output will be summed. 
        with_logits : bool
            If True, use additional sigmoid for inputs
        """
        super().__init__()
        self.eps = eps
        self.with_logits = with_logits
        self.reduction = reduction

    def forward(self, logits, true_labels):
        """
        Arguments
        ---------
        logits: torch.Tensor
            Unnormalized probability of true class. Shape: [B, ...]
        true_labels: torch.Tensor
            Mask of correct predictions. Shape: [B, ...]
        Returns
        -------
        torch.Tensor
            If reduction is 'mean' or 'sum' returns a tensor with a single element
            Otherwise, returns a tensor of shape [B]
        """
        true_labels = true_labels.long()

        if self.with_logits:
            logits = torch.sigmoid(logits)

        # we need to sum along the dimensions starting from 1
        d = len(logits.shape)
        dim = list(range(1, d))

        num = 2 * torch.sum(logits * true_labels, dim=dim)
        den = torch.sum(logits + true_labels + self.eps, dim=dim)
        losses = 1 - num / den

        loss_value = None
        if self.reduction == 'sum':
            loss_value = losses.sum()
        elif self.reduction == 'mean':
            loss_value = losses.mean()
        elif self.reduction is None:
            loss_value = losses

        return loss_value


class IoUScore(torch.nn.Module):
    def __init__(self, threshold=0, reduction=None, with_logits=True):
        """
        Arguments
        ---------
        threshold : float
            threshold for logits binarization
        reduction : Optional[str] (None, 'mean' or 'sum')
            specifies the reduction to apply to the output:

            None: no reduction will be applied
            'mean': the sum of the output will be divided by the number of elements in the batch
            'sum':  the output will be summed. 
        with_logits : bool
            If True, use additional sigmoid for inputs
        """
        super().__init__()

        self.threshold = threshold
        self.with_logits = with_logits
        self.reduction = reduction

    @torch.no_grad()
    def forward(self, logits, true_labels):
        """
        Arguments
        ---------
        logits: torch.Tensor
            Unnormalized probability of true class. Shape: [B, ...]
        true_labels: torch.Tensor[bool]
            Mask of correct predictions. Shape: [B, ...]
        Returns
        -------
        torch.Tensor
            If reduction is 'mean' or 'sum' returns a tensor with a single element
            Otherwise, returns a tensor of shape [B]
        """
        if self.with_logits:
            logits = logits >= self.threshold
        true_labels = true_labels.bool()

        assert logits.shape == true_labels.shape

        # we need to sum along the dimensions starting from 1
        d = len(logits.shape)
        dim = list(range(1, d))
        num = torch.sum(logits & true_labels, dim=dim)
        den = torch.sum(logits | true_labels, dim=dim)
        scores = num / den

        score = None
        if self.reduction == 'sum':
            score = scores.sum()
        elif self.reduction == 'mean':
            score = scores.mean()
        elif self.reduction is None:
            score = scores

        return score


class BCEDice(torch.nn.Module):
    def __init__(self, alpha, eps=1e-7, reduction=None, with_logits=True):
        super().__init__()
        self.eps = eps
        self.with_logits = with_logits
        self.reduction = reduction
        self.ce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, logits, true_labels):
        """
        Arguments
        ---------
        logits: torch.Tensor
            Unnormalized probability of true class. Shape: [B, ...]
        true_labels: torch.Tensor
            Mask of correct predictions. Shape: [B, ...]
        Returns
        -------
        torch.Tensor
            If reduction is 'mean' or 'sum' returns a tensor with a single element
            Otherwise, returns a tensor of shape [B]
        """
        true_labels = true_labels.long()

        if self.with_logits:
            logits = torch.sigmoid(logits)

        # we need to sum along the dimensions starting from 1
        d = len(logits.shape)
        dim = list(range(1, d))

        # dice
        num = 2 * torch.sum(logits * true_labels, dim=dim)
        den = torch.sum(logits + true_labels + self.eps, dim=dim)
        dice = 1 - num / den

        # ce
        ce = self.ce(logits, true_labels.float()).mean(dim=dim)

        # combime
        losses = self.alpha * dice + (1 - self.alpha) * ce

        loss_value = None
        if self.reduction == 'sum':
            loss_value = losses.sum()
        elif self.reduction == 'mean':
            loss_value = losses.mean()
        elif self.reduction is None:
            loss_value = losses

        return loss_value
