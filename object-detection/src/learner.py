from dataclasses import dataclass, asdict
import lightning.pytorch as pl
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR
import torch
import math
from .loss import MultiBoxLoss
from .boxes import generate_prior_boxes, PriorBoxesConfig, mAP, detect_objects
from torchmetrics.detection.mean_ap import MeanAveragePrecision

@dataclass
class LearnerConfig:
    max_lr: float = 1e-3
    lr_div_factor: float = 1e3
    batch_size: int = 4
    warmup_pct: float = 0.1
    weight_decay = 1e-2
    betas : tuple = (0.9, 0.999)
    total_steps: int = None
    lr_decay: int = False

    detect_conf_threshold: float = .6
    match_overlap_threshold: float = 0.5
    nms_overlap_threshold: float = 0.1
    neg_pos_ratio: float = 3.
    variance: tuple = (.1, .2)


class Learner(pl.LightningModule):
    def __init__(self, model, config: LearnerConfig, prior_boxes_config: PriorBoxesConfig):
        super().__init__()
        
        self.model = model
        self.config = config
        self.prior_boxes_config = prior_boxes_config

        self.loss_fn = MultiBoxLoss(
            overlap_threshold=config.match_overlap_threshold,
            neg_pos_ratio=config.neg_pos_ratio,
            variance=config.variance
        )
        self.metric_fn = MeanAveragePrecision()

        self.priors = generate_prior_boxes(prior_boxes_config).cuda()
        self.priors.requires_grad_(False)
    
    @torch.no_grad()
    def _torchmetric(self, pred_loc, pred_conf, target_labels, target_boxes):
        detected_loc, detected_labels, detected_conf = detect_objects(
            pred_loc, pred_conf, self.priors,
            self.prior_boxes_config.n_classes,
            self.config.nms_overlap_threshold,
            self.config.detect_conf_threshold
        )
        preds = []
        for loc, lab, conf in zip(detected_loc, detected_labels, detected_conf):
            preds.append({
                'boxes': loc.cuda(),
                'scores': conf.cuda(),
                'labels': lab.cuda()
            })
        target = []
        for loc, lab in zip(target_boxes, target_labels):
            target.append({
                'boxes': loc.cuda(),
                'labels': lab.cuda()
            })
        res = self.metric_fn(preds=preds, target=target)
        del res['classes']
        return res

    def forward(self, batch, calc_metric):
        images, boxes, labels = batch
        loc, conf = self.model(images)
        loss_l, loss_c = self.loss_fn(
            predictions=(loc, conf, self.priors),
            targets=(labels, boxes)
        )
        if calc_metric:
            metric = self._torchmetric(loc, conf, labels, boxes)
            return loss_l, loss_c, metric
        return loss_l, loss_c
        
    def training_step(self, batch, batch_idx):
        """
        batch:
        - images: (B, C, H, W)
        - boxes: list of (n, 4), where n is number of object on a picture
        - labels: list of (n,)
        """

        loss_l, loss_c = self.forward(batch, calc_metric=False)
        self.log_dict(
            dictionary={'train_loss_loc': loss_l, 'train_loss_clf': loss_c},
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True
        )
        return (loss_l + loss_c) / 2

        # self.log_dict(
        #     dictionary={f'train_{name}': val for name, val in metrics.items()},
        #     prog_bar=False,
        #     logger=True,
        #     on_step=False,
        #     on_epoch=True
        # )
    
    def validation_step(self, batch, batch_idx):
        loss_l, loss_c, metric = self.forward(batch, calc_metric=True)
        dct = {'val_loss_loc': loss_l, 'val_loss_clf': loss_c}
        dct.update(metric)

        self.log_dict(
            dictionary=dct,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )
        
        # self.log_dict(
        #     dictionary={f'val_{name}': val for name, val in metrics.items()},
        #     prog_bar=False,
        #     logger=True,
        #     on_step=False,
        #     on_epoch=True
        # )

    def on_train_start(self):
        # hparams = self.model.get_hparams()
        hparams = asdict(self.config)
        self.logger.log_hyperparams(hparams)

    def get_optim_groups(self):
        """Separate out all parameters to those that will and won't experience regularizing weight decay. Taken from https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136"""
        decay = set()
        no_decay = set()
        from .models.vgg16 import L2Norm
        whitelist_weight_modules = (nn.Linear, nn.Conv2d, L2Norm)
        blacklist_weight_modules = (nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups

    def configure_optimizers(self):
        optim_groups = self.get_optim_groups()
        optimizer = AdamW(optim_groups, lr=self.config.max_lr, betas=self.config.betas)

        if self.config.lr_decay:
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.config.max_lr,
                div_factor=self.config.lr_div_factor,
                final_div_factor=self.config.lr_div_factor,
                pct_start=self.config.warmup_pct,
                total_steps=self.config.total_steps
            )
        else:
            def cosine_warmup_no_decay(step):
                lr_max = self.config.max_lr
                lr_min = lr_max / self.config.lr_div_factor

                warmup_pct = self.config.warmup_pct
                total_steps = self.config.total_steps
                warmup_steps = math.floor(warmup_pct * total_steps)
                
                if step < warmup_steps:
                    lr = lr_max - 0.5 * (lr_max - lr_min) * (1 + math.cos(step / warmup_steps * math.pi))
                    return lr / lr_max
                
                return 1
            
            scheduler = LambdaLR(
                optimizer,
                lr_lambda=cosine_warmup_no_decay
            )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", 'frequency': 1}}
