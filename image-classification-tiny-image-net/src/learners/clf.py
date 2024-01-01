from dataclasses import dataclass
from torchmetrics.functional.classification import accuracy
import torch.nn.functional as F
from .bases import BaseLearner, BaseLearnerConfig


@dataclass
class ClfLearnerConfig(BaseLearnerConfig):
    mixup_alpha: float = .2


class ClfLearner(BaseLearner):
    def __init__(self, model, config: ClfLearnerConfig, n_classes=200):
        super().__init__()
        
        self.model = model
        self.config = config
        self.n_classes = n_classes

    def forward(self, images):
        """images: (B, C, H, W)"""
        
        # (B, n_classes), where n_classes=200 for tiny imagenet
        logits = self.model(images)
        
        return logits
    
    def _loss_and_metric(self, batch):
        images, targets = batch
        logits = self.forward(images)
        
        loss = F.cross_entropy(logits, targets, reduction='mean', label_smoothing=0.01)
        metric = accuracy(logits, targets, task='multiclass', num_classes=self.n_classes)

        return loss, metric

    def training_step(self, batch, batch_idx):
        loss, metric = self._loss_and_metric(batch)
        self.log(
            name='train_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )
        self.log(
            name='train_metric',
            value=metric,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, metric = self._loss_and_metric(batch)
        self.log(
            name='val_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )
        self.log(
            name='val_metric',
            value=metric,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )
