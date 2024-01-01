from dataclasses import dataclass
import torch.nn.functional as F
import torch
import numpy as np
from .bases import BaseLearner, BaseLearnerConfig
from sklearn.neural_network import MLPClassifier
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR
import math


@dataclass
class ContrastiveLearnerConfig(BaseLearnerConfig):
    k: int = 5
    t: float = 0.05


class ContrastiveLearner(BaseLearner):
    def __init__(self, model, config: ContrastiveLearnerConfig, n_classes=200):
        super().__init__()
        
        self.model = model
        self.config = config
        self.n_classes = n_classes

        self.tiny_imagenet_train = []
        self.tiny_imagenet_val = []

    def forward(self, images):
        """images: (B, C, H, W)"""
        
        # (B, n_classes), where n_classes=200 for tiny imagenet
        logits = self.model(images)
        
        return logits
    
    def _loss_and_metric(self, batch):
        augs_1, augs_2 = batch

        logits_1 = self.forward(augs_1)
        logits_2 = self.forward(augs_2)

        scores = logits_1 @ logits_2.T
        batch_size = scores.shape[0]
        targets = torch.arange(batch_size, device=scores.device)
        loss_row = F.cross_entropy(scores, targets, reduction='mean')
        loss_col = F.cross_entropy(scores.T, targets, reduction='mean')
        loss = 0.5 * (loss_row + loss_col)

        topk_indicators = [i in top for i, top in enumerate(torch.topk(scores, k=self.config.k, dim=1).indices)]
        topk_accuracy = sum(topk_indicators) / batch_size

        return loss, topk_accuracy

    def training_step(self, batch, batch_idx):
        loss, metric = self._loss_and_metric(batch)
        self.log(
            name='train_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True
        )
        self.log(
            name='train_metric',
            value=metric,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        images, targets = batch
        embeddings = self.model(images).detach().cpu().numpy()
        res = list(zip(embeddings, targets.cpu().numpy()))

        if dataloader_idx == 0:
            self.tiny_imagenet_train.extend(res)
        elif dataloader_idx == 1:
            self.tiny_imagenet_val.extend(res)

    def on_validation_epoch_end(self) -> None:
        metric = get_clf_score(
            self.tiny_imagenet_train,
            self.tiny_imagenet_val
        )

        self.log(
            name='val_metric',
            value=metric,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            # batch_size=self.config.batch_size
        )

        self.tiny_imagenet_train.clear()
        self.tiny_imagenet_val.clear()
    
    def configure_optimizers(self):
        optim_groups = self.get_optim_groups()
        optimizer = AdamW(optim_groups, lr=self.config.lr, betas=self.config.betas)
        
        def lr_foo(step):
            warmup_steps = self.config.warmup_steps
            restarts = self.config.restarts
            
            if warmup_steps is None:
                return 1
            if restarts:
                return (step % warmup_steps + 1) / warmup_steps
            else:
                return (step + 1) / warmup_steps if step < warmup_steps else 1

        def cosine_annealing(step):
            if self.config.lr_last is None or self.config.max_steps is None:
                return 1
            lr_min = self.config.lr_last
            lr_max = self.config.lr
            scale = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(step / self.config.max_steps * math.pi))
            return scale

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=cosine_annealing
        )

        # scheduler = StepLR(
        #     optimizer,
        #     step_size=6,
        #     gamma=0.1
        # )

        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", 'frequency': 1}}



def get_clf_score(train_dataset, val_dataset, n_epochs=5):
    # configure model
    clf = MLPClassifier(
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=n_epochs
    )

    # configure data
    X_train = np.stack([emb for emb, _ in train_dataset], axis=0)
    y_train = np.stack([tar for _, tar in train_dataset], axis=0)
    X_val = np.stack([emb for emb, _ in val_dataset], axis=0)
    y_val = np.stack([tar for _, tar in val_dataset], axis=0)
    
    # train model
    clf.fit(X_train, y_train)

    # score model
    y_pred = clf.predict(X_val)
    score = np.mean(y_pred == y_val)
    
    return score
