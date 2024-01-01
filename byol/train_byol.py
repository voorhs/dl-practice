import lightning as pl
from byol_pytorch import BYOL
from torch.optim import AdamW

from torchvision.datasets import CIFAR100
import torchvision.transforms as T
from torch.utils.data import DataLoader

from net import MyResNet
import torch

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from datetime import datetime


BATCH_SIZE = 256
LR = 5e-4
WEIGHT_DECAY = 0


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, batch, batch_idx):
        images, _ = batch
        loss = self.forward(images)
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
        images, _ = batch
        loss = self.forward(images)
        self.log(
            name='val_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        return AdamW(
            self.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()


# load data
def load_cifar100(train):
    res = CIFAR100(
        root='./data',
        train=train,
        download=True,
        transform=T.ToTensor()
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
        batch_size=256,
        shuffle=False,
        num_workers=10
    )

    # network for BYOL algorithm
    net = MyResNet()

    # wrap into lightning
    model = SelfSupervisedLearner(
        net,
        use_momentum=False,
        image_size=32,
        hidden_layer='max_pool',
        projection_size=128,
        projection_hidden_size=512,
        moving_average_decay=0.99
    )

    # checkpointing best model and last model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_last=True,
        save_top_k=3,
        mode='min',
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
