import lightning as pl
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision.datasets import CIFAR100
import torchvision.transforms as T
from torch.utils.data import DataLoader

from net import MyResNet
import torch

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from datetime import datetime


# taken from https://huggingface.co/edadaltocg/resnet18_cifar100
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-3


class SupervisedLearner(pl.LightningModule):
    def __init__(self, net):
        super().__init__()
        self.learner = net

    def forward(self, images):
        return self.learner(images)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = F.cross_entropy(logits, labels)
        self.log(
            name='train_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = F.cross_entropy(logits, labels)
        self.log(
            name='val_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log(
            name='val_acc',
            value=acc,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            patience=5,
            threshold=0.005,
            mode='max',
            min_lr=1e-7,
            factor=0.5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_acc'
        }


# load data
def load_cifar100(train):
    if train:
        extra = [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
        ]
    else:
        extra = []

    transform = T.Compose(extra + [
        T.ToTensor(),
        T.Normalize(
            # taken from https://github.com/weiaicunzai/pytorch-cifar100/blob/master/conf/global_settings.py#L11
            (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        )
    ])

    res = CIFAR100(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )

    return res


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(0)
    
    train_cifar100 = load_cifar100(train=True)
    test_cifar100 = load_cifar100(train=False)

    train_cifar100_loader = DataLoader(
        dataset=train_cifar100,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=10
    )

    test_cifar100_loader = DataLoader(
        dataset=test_cifar100,
        batch_size=1024,
        shuffle=False,
        num_workers=10
    )

    # network for supervised learning
    net = MyResNet()

    # wrap into lightning
    model = SupervisedLearner(net)

    # checkpointing best model and last model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        save_last=True,
        save_top_k=3,
        mode='max',
        every_n_epochs=5
    )

    # log lr to tensorboard
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # training configs
    trainer = pl.Trainer(
        # max_epochs=100,
        max_time={'minutes': 30},

        # hardware settings
        accelerator='gpu',
        deterministic=True,  # for reproducibility
        precision="16-mixed",

        # logging and checkpointing
        logger=True,
        enable_progress_bar=False,
        log_every_n_steps=5,
        profiler=None,
        callbacks=[checkpoint_callback, lr_monitor],

        # check if model is implemented correctly
        overfit_batches=False,

        # check training_step and validation_step doesn't fails
        fast_dev_run=False,
        num_sanity_val_steps=False
    )

    print('Started at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    # do magic!
    trainer.fit(model, train_cifar100_loader, test_cifar100_loader)
