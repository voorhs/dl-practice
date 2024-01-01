from dataclasses import dataclass, asdict
import lightning.pytorch as pl
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR
import torch
import math
from .models import Projector
import torch.nn.functional as F


@dataclass
class LearnerConfig:
    max_lr: float = 1e-3
    lr_div_factor: float = 10
    batch_size: int = 64
    warmup_pct: float = 0.1
    weight_decay = 1e-2
    betas : tuple = (0.9, 0.999)
    total_steps: int = None
    lr_decay: int = True

    hidden_size: int = 768
    projection_size: int = 768
    k: int = 3
    clf: bool = True


class Learner(pl.LightningModule):
    def __init__(self, model, processor, config: LearnerConfig):
        super().__init__()
        
        self.model = model
        self.model.requires_grad_(False)
        self.processor = processor
        self.config = config
        
        self.qi_projector = Projector(self.config.hidden_size * 2, self.config.projection_size)
        
        if self.config.clf:
            self.clf = nn.Linear(self.config.projection_size, 461)
        else:
            self.a_projector = Projector(self.config.hidden_size, self.config.projection_size)

    def get_qi_embeddings(self, questions, images):
        questions = to(self.processor(text=questions, return_tensors='pt', padding=True))
        images = to(self.processor(images=images, return_tensors='pt', padding=True))

        q_embeddings = self.model.get_text_features(**questions)
        i_embeddings = self.model.get_image_features(**images)

        qi_embeddings = self.qi_projector(torch.cat([q_embeddings, i_embeddings], dim=1))
        qi_embeddings = F.normalize(qi_embeddings, dim=1)

        return qi_embeddings
    
    def get_a_embeddings(self, answers):
        answers = to(self.processor(text=answers, return_tensors='pt', padding=True))
        a_embeddings = self.model.get_text_features(**answers)
        a_embeddings = self.a_projector(a_embeddings)
        a_embeddings = F.normalize(a_embeddings, dim=1)
        return a_embeddings

    def contrastive_loss(self, qi_embeddings, a_embeddings):
        similarities = qi_embeddings @ a_embeddings.T

        labels = torch.arange(qi_embeddings.shape[0], device=qi_embeddings.device)
        loss_r = F.cross_entropy(similarities, labels, reduction='mean')
        loss_c = F.cross_entropy(similarities.T, labels, reduction='mean')

        loss = (loss_r + loss_c) / 2

        topk_indicators = [i in top for i, top in enumerate(torch.topk(similarities, k=self.config.k, dim=1).indices)]
        metric = sum(topk_indicators) / len(topk_indicators)

        return loss, metric

    def classification_loss(self, qi_embeddings, answers):
        logits = self.clf(qi_embeddings)
        target = torch.tensor(answers, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, target, reduction='mean')

        topk_indicators = [answers[i] in top for i, top in enumerate(torch.topk(logits, k=self.config.k, dim=1).indices)]
        metric = sum(topk_indicators) / len(topk_indicators)

        return loss, metric

    def forward(self, batch):
        questions, answers, images = batch
        qi_embeddings = self.get_qi_embeddings(questions, images)
        
        if self.config.clf:
            return self.classification_loss(qi_embeddings, answers)

        a_embeddings = self.get_a_embeddings(answers)
        return self.contrastive_loss(qi_embeddings, a_embeddings)

    def training_step(self, batch, batch_idx):
        loss, metric = self.forward(batch)
        
        self.log_dict(
            dictionary={'train_loss': loss, 'train_metric': metric},
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.config.batch_size
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, metric = self.forward(batch)
        
        self.log_dict(
            dictionary={'val_loss': loss, 'val_metric': metric},
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.config.batch_size
        )

    def on_train_start(self):
        # hparams = self.model.get_hparams()
        hparams = asdict(self.config)
        self.logger.log_hyperparams(hparams)

    def get_optim_groups(self):
        """Separate out all parameters to those that will and won't experience regularizing weight decay. Taken from https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136"""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear)
        blacklist_weight_modules = (nn.BatchNorm2d, nn.Embedding, nn.LayerNorm, nn.Conv2d)
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
        # assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    # % (str(param_dict.keys() - union_params), )

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

def to(dct):
    return {k: v.cuda() for k, v in dct.items()}
