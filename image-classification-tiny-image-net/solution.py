# Don't erase the template code, except "Your code here" comments.

import subprocess
import sys

# List any extra packages you need here. Please, fix versions so reproduction of your results would be less painful.
PACKAGES_TO_INSTALL = ["gdown==4.4.0", "lightning==2.1.0", "wandb==0.15.12", 'timm==0.9.8']
subprocess.check_call([sys.executable, "-m", "pip", "install"] + PACKAGES_TO_INSTALL)


from torch.utils.data import DataLoader
from src.datasets import TinyImagenetDatasetRAM
import torchvision.transforms.v2 as T
from src.models import MyResNet, get_seresnet34, Ensemble
from src.learners import ClfLearner, ClfLearnerConfig
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor


def get_dataloader(path, kind, batch_size=32):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val' or 'test', the dataloader should be deterministic.
    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train', 'val' or 'test'

    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """
    ordinary_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.4824, 0.4495, 0.3981], std=[0.2765, 0.2691, 0.2825]),
    ])

    if kind == 'train':
        train_dataset = TinyImagenetDatasetRAM(path=path, split='train', transforms=ordinary_transforms)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )
        return train_loader
    
    dataset = TinyImagenetDatasetRAM(path=path, split=kind, transforms=ordinary_transforms)
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return loader


def get_model():
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.

    return:
    model:
        `torch.nn.Module`
    """
    
    return Ensemble(do_train=False).cuda()


def get_optimizer(model):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.

    return:
    optimizer:
        `torch.optim.Optimizer`
    """
    # my Learner class already has method `configure_optimizers()`
    return


def _get_trainer(
        logger='wb',
        logdir='./logs',
        name=None,
        minutes=3,
        interval=1,
        with_callbacks=True
    ):
    if with_callbacks:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_metric',
            save_last=False,
            save_top_k=1,
            mode='max',
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [checkpoint_callback, lr_monitor]
    else:
        callbacks = None

    if logger == 'tb':
        Logger = pl.loggers.TensorBoardLogger
    elif logger == 'wb':
        Logger = pl.loggers.WandbLogger
    elif logger == 'none':
        Logger = lambda **kwargs: False

    logger = Logger(
        save_dir=logdir,
        name=name
    )
    trainer = pl.Trainer(
        # budget
        max_time={'minutes': minutes},

        # hardware settings
        accelerator='gpu',
        precision="16-mixed",

        # logging and checkpointing
        check_val_every_n_epoch=interval,
        logger=logger,
        enable_progress_bar=False,
        callbacks=callbacks,
    )

    return trainer


def _get_learner(model):
    config = ClfLearnerConfig(
        batch_size=32,
        # temperature=0.3
    )
    learner = ClfLearner(model=model, config=config)
    return learner


def predict(model, batch):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).

    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """
    learner = _get_learner(model)
    images = batch.cuda()
    return learner.predict_step(images, None)
    

def validate(dataloader, model):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.

    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    learner = _get_learner(model)
    trainer = _get_trainer(logger='none', with_callbacks=False)
    res = trainer.validate(learner, dataloader)
    loss = res[0]['val_loss']
    metric = res[0]['val_metric']

    return metric, loss


def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.

    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """
    learner = _get_learner(model)
    trainer = _get_trainer()

    trainer.fit(learner, train_dataloader, val_dataloader)


def load_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.

    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    learner = ClfLearner.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        config=ClfLearnerConfig()
    )

    model.load_state_dict(learner.model.state_dict())


def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    md5_checksum = "518e24a6cc6ee1dfd3364d5d7b65c1be"
    google_drive_link = "https://drive.google.com/file/d/15ew43SJ8i7BQZk_vQmhN2DXZInlnZTLm/view?usp=sharing"

    return md5_checksum, google_drive_link
