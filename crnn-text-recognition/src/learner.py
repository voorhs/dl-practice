from dataclasses import dataclass, asdict
import lightning.pytorch as pl
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR
import torch
import math
import torch.nn.functional as F
from .modeling import CRNN, VariationalDropoutGRU
from ctc_decoder import beam_search
from torchmetrics.functional.text import edit_distance
from functools import partial
from tqdm.contrib.concurrent import process_map


@dataclass
class LearnerConfig:
    max_lr: float = 1e-3
    lr_div_factor: float = 10
    batch_size: int = 32
    warmup_pct: float = 0.1
    weight_decay = 1e-2
    betas : tuple = (0.9, 0.999)
    total_steps: int = None
    lr_decay: bool = True

    beam_width: int = 2


class Learner(pl.LightningModule):
    def __init__(self, model: CRNN, config: LearnerConfig):
        super().__init__()
        
        self.model = model
        self.config = config

    def forward(self, batch, compute_metric=False):
        images = batch['images'].to(self.device)
        targets_concated = batch['targets']
        target_lengths = batch['target_lengths']

        logits = self.model(images)     # (T,B,n_chars)
        log_probs = F.log_softmax(logits, dim=2)

        T, B, _ = log_probs.shape
        input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long)

        loss = F.ctc_loss(
            log_probs.cpu(),
            targets_concated,
            input_lengths,
            target_lengths,
            blank=len(self.model.config.alphabet)
        )

        if compute_metric:
            return loss, self.metric_fn(logits, batch['texts'])
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        
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
        loss, metric = self.forward(batch, compute_metric=True)
        
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
        blacklist_weight_modules = (nn.BatchNorm2d, nn.LayerNorm, nn.BatchNorm1d, nn.GRU, nn.GRUCell, VariationalDropoutGRU)
        for mn, m in module.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if 'bias' in pn:
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif 'weight' in pn and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif 'weight' in pn and isinstance(m, blacklist_weight_modules):
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
        # cnn = self.get_optim_groups(self.model.cnn)
        # rnn = self.get_optim_groups(self.model.rnn)
        # connector = self.get_optim_groups(self.model.connector)
        # predictor = self.get_optim_groups(self.model.predictor)
        
        # for i in range(2):
        #     cnn[i]['lr'] = self.config.max_lr / 1e1
        #     rnn[i]['lr'] = self.config.max_lr
        #     connector[i]['lr'] = self.config.max_lr
        #     predictor[i]['lr'] = self.config.max_lr

        #     cnn[i]['name'] = f'cnn{i}'
        #     rnn[i]['name'] = f'rnn{i}'
        #     connector[i]['name'] = f'connector{i}'
        #     predictor[i]['name'] = f'predictor{i}'

        # optim_groups = cnn + rnn + connector + predictor
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
        logits = self.model(images)  # (T,B,n_chars)
        return logits, self.decode(logits)

    def metric_fn(self, logits, target):
        return edit_distance(
            preds=self.decode(logits),
            target=target,
            reduction='mean'
        )

    def decode(self, logits):
        probs = F.softmax(logits, dim=2).cpu()    # (T,B,n_chars)
        probs = [x.numpy() for x in torch.unbind(probs, dim=1)]
        func = partial(beam_search, chars=self.model.config.alphabet, beam_width=self.config.beam_width)
        return process_map(func, probs, chunksize=1, max_workers=4, disable=True)
