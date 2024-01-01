import lightning.pytorch as pl
from dataclasses import dataclass, asdict
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math


@dataclass
class BaseLearnerConfig:
    lr: float = 1e-3
    batch_size: int = 32
    warmup_steps: int = None
    restarts: bool = False
    weight_decay = 1e-2
    betas = (0.9, 0.999)
    lr_last: float = 1e-6
    max_steps: int = None


class BaseLearner(pl.LightningModule):
    @staticmethod
    def get_default_config():
        raise NotImplementedError()

    def on_train_start(self):
        # hparams = self.model.get_hparams()
        hparams = asdict(self.config)
        self.logger.log_hyperparams(hparams)

    def get_optim_groups(self):
        """Taken from https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136"""
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv2d)
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

        def cosine_annealing_with_warmup(step):
            lr_max = self.config.lr

            warmup_steps = self.config.warmup_steps
            if warmup_steps is not None:
                if step < warmup_steps:
                    return 1 - 0.5 * (1 + math.cos(step / self.config.warmup_steps * math.pi))
                step -= warmup_steps
            
            if self.config.max_steps is None or self.config.lr_last is None:
                return 1
            
            lr_min = self.config.lr_last
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(step / self.config.max_steps * math.pi))
            scale = lr / lr_max
            return scale

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=cosine_annealing_with_warmup
        )

        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", 'frequency': 1}}
