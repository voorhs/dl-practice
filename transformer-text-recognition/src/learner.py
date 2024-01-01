from dataclasses import dataclass, asdict
import lightning.pytorch as pl
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch
import math
import torch.nn.functional as F
from torchmetrics.functional.text import edit_distance
from .models import TransformerOCR, Tokenizer


@dataclass
class LearnerConfig:
    max_lr: float = 2e-5
    lr_div_factor: float = 10
    batch_size: int = 16
    warmup_pct: float = 0.1
    weight_decay = 1e-2
    betas : tuple = (0.9, 0.999)
    total_steps: int = None
    lr_decay: bool = True


class Learner(pl.LightningModule):
    def __init__(self, model: TransformerOCR, config: LearnerConfig):
        super().__init__()
        
        self.model = model
        self.tokenizer = Tokenizer(model.transformer.config)
        self.config = config

    def forward(self, batch):
        images = batch['images'].to(self.device)
        tokenized = {k: v.to(self.device) for k, v in self.tokenizer(batch['texts']).items()}
        encoder_outputs, char_logits = self.model(images=images, **tokenized)  # (B,T,n_chars)
        return encoder_outputs, char_logits, tokenized

    def training_step(self, batch, batch_idx):
        _, logits, tokenized = self.forward(batch)
        loss = self.loss_fn(logits, tokenized)

        self.log(
            name='train_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.config.batch_size
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        encoder_outputs, char_logits, tokenized = self.forward(batch)
        loss = self.loss_fn(char_logits, tokenized)
        metric = self.metric_fn(encoder_outputs, batch['texts'])

        self.log_dict(
            dictionary={'val_loss': loss, 'val_edit_distance': metric},
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.config.batch_size
        )

    def on_train_start(self):
        hparams = asdict(self.config)
        hparams.update(asdict(self.model.config))
        self.logger.log_hyperparams(hparams)

    def get_optim_groups(self, module):
        """Separate out all parameters to those that will and won't experience regularizing weight decay. Taken from https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136"""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv2d)
        blacklist_weight_modules = (nn.BatchNorm2d, nn.LayerNorm, nn.Embedding)
        for mn, m in module.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if 'bias' in pn:
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif 'weight' in pn :#and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif 'weight' in pn:# and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in module.named_parameters()}
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
        optimizer = AdamW(self.get_optim_groups(self), amsgrad=True, betas=self.config.betas)

        def one_cycle_lr(step):
            warmup_pct = self.config.warmup_pct
            total_steps = self.config.total_steps
            warmup_steps = math.floor(warmup_pct * total_steps)
            
            if step < warmup_steps:
                return 1 - 0.5 * (1 - 1 / self.config.lr_div_factor) * (1 + math.cos(step / warmup_steps * math.pi))
            
            if self.config.lr_decay:
                return 1 / self.config.lr_div_factor + 0.5 * (1 - 1 / self.config.lr_div_factor) * (1 + math.cos((step - warmup_steps)/ (total_steps - warmup_steps) * math.pi))

            return 1
        
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=one_cycle_lr
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", 'frequency': 1}}

    def predict_step(self, images, batch_idx):
        if isinstance(images, dict):
            images = images['images'].to(self.device)
        return self.model.predict(images, self.tokenizer)

    def metric_fn(self, encoder_outputs, texts):
        return edit_distance(
            preds=self.model.decode(encoder_outputs, self.tokenizer),
            target=texts,
            reduction='mean'
        )
    
    def loss_fn(self, logits, tokenized):
        _, _, n_chars = logits.shape
        logits = logits[:, :-1].reshape(-1, n_chars)
        targets = tokenized['token_ids'][:, 1:].reshape(-1)
        mask = tokenized['attention_mask'][:, 1:].reshape(-1)

        loss = F.cross_entropy(logits, targets, reduction='none')
        loss = torch.sum(loss * mask) / torch.sum(mask)
        
        return loss
