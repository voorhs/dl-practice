from .learner import Learner, LearnerConfig

class LightningCkptLoadable:
    def load_checkpoint(self, path_to_ckpt, prior_boxes_config, map_location=None):
        model = Learner.load_from_checkpoint(
            path_to_ckpt,
            map_location=map_location,
            model=self,
            config=LearnerConfig(),
            prior_boxes_config=prior_boxes_config
        ).model
        
        self.load_state_dict(model.state_dict())


class HParamsPuller:
    def get_hparams(self):
        res = {}
        for attr, val in vars(self).items():
            if hasattr(val, 'get_hparams'):
                tmp = val.get_hparams()
                tmp = self.add_prefix(tmp, attr)
                res.update(tmp)
            elif isinstance(val, (int, float, str, bool)):
                res[attr] = val
        return res
    
    @staticmethod
    def add_prefix(dct, prefix):
        res = {}
        for key, val in dct.items():
            res[f'{prefix}.{key}'] = val
        return res


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


from dataclasses import dataclass
@dataclass
class TrainerConfig:
    name: str = None
    n_workers: int = 8
    n_epochs: int = 16
    seed: int = 0
    interval: int = 4
    logdir: str = './logs'
    logger: str  = 'tb'
    resume_from: str = None
    load_from: str = None


from .learner import LearnerConfig
def config_to_argparser(container_classes=[LearnerConfig, TrainerConfig]):
    from dataclasses import fields
    def add_arguments(container_class, parser):
        for field in fields(container_class):
            parser.add_argument(
                '--' + field.name.replace('_', '-'),
                dest=field.name,
                default=field.default,
                type=field.type
            )
    
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    for cls in container_classes:
        add_arguments(cls, parser)

    return parser


def retrieve_fields(namespace, contrainer_class):
    from dataclasses import fields
    res = {}
    for field in fields(contrainer_class):
        res[field.name] = getattr(namespace, field.name)
    return res


def train(learner, train_loader, val_loader, args: TrainerConfig):
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

    if args.logger != 'none':
        checkpoint_callback = ModelCheckpoint(
            monitor=f'val_edit_distance',
            save_last=True,
            save_top_k=1,
            # every_n_epochs=1,
            mode='min',
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [checkpoint_callback, lr_monitor]
    else:
        callbacks = None

    import lightning.pytorch as pl
    import os
    if args.logger == 'tb':
        Logger = pl.loggers.TensorBoardLogger
        logdir = os.path.join(args.logdir, 'tb')
    elif args.logger == 'wb':
        Logger = pl.loggers.WandbLogger
        logdir = os.path.join(args.logdir, 'wb')
    elif args.logger == 'none':
        Logger = lambda **kwargs: False
        logdir = None
    
    logger = Logger(
        save_dir=logdir,
        name=args.name
    )

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        # max_time={'minutes': 60},
        
        # max_time={'minutes': 10},

        # hardware settings
        accelerator='gpu',
        deterministic=False,
        # precision="16-mixed",

        # logging and checkpointing
        # val_check_interval=args.interval,
        check_val_every_n_epoch=args.interval,
        logger=logger,
        enable_progress_bar=False,
        profiler=None,
        callbacks=callbacks,
        # log_every_n_steps=5,

        # check if model is implemented correctly
        overfit_batches=False,

        # check training_step and validation_step doesn't fail
        fast_dev_run=False,
        num_sanity_val_steps=False
    )

    if args.resume_from is None:
        trainer.validate(learner, val_loader)

    from datetime import datetime
    print('Started at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    trainer.fit(
        learner, train_loader, val_loader,
        ckpt_path=args.resume_from
    )

    print('Finished at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    trainer.validate(learner, val_loader, ckpt_path='best')


def init_environment(seed):
    import torch
    torch.set_float32_matmul_precision('medium')
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    seed_everything(seed)


def validate(learner, val_loader, args: TrainerConfig):
    import lightning.pytorch as pl
    import os

    if args.logger == 'tb':
        Logger = pl.loggers.TensorBoardLogger
        logdir = os.path.join(args.logdir, 'tb')
    elif args.logger == 'wb':
        Logger = pl.loggers.WandbLogger
        logdir = os.path.join(args.logdir, 'wb')
    elif args.logger == 'none':
        Logger = lambda **kwargs: False
        logdir = None

    logger = Logger(
        save_dir=logdir,
        name=args.name
    )

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        # max_time={'minutes': 60},
        
        # max_time={'minutes': 10},

        # hardware settings
        accelerator='gpu',
        deterministic=False,
        # precision="16-mixed",

        # logging and checkpointing
        # val_check_interval=args.interval,
        check_val_every_n_epoch=args.interval,
        logger=logger,
        enable_progress_bar=True,
        profiler=None,
        # callbacks=callbacks,
        # log_every_n_steps=5,

        # check if model is implemented correctly
        overfit_batches=False,

        # check training_step and validation_step doesn't fail
        fast_dev_run=False,
        num_sanity_val_steps=False
    )

    if args.resume_from is None:
        raise ValueError('specify weights with `resume_from` argument')

    from datetime import datetime
    print('Started at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    trainer.validate(
        learner, val_loader,
        ckpt_path=args.resume_from
    )

    preds = trainer.predict(
        learner, val_loader,
        ckpt_path=args.resume_from
    )

    name = datetime.now().strftime("%H:%M:%S_%d-%m-%Y")
    print('Finished at', name)

    res_path = os.path.join('data', 'contest-data', 'predictions', name)
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    # save text predictions
    import itertools as it
    txt_preds = list(it.chain.from_iterable(preds))

    import json
    json.dump(txt_preds, open(os.path.join(res_path, 'txt.json'), 'w'))

    # save as csv submission
    import pandas as pd
    # df = pd.read_csv(os.path.join('data', 'contest-data', 'submission.csv'))
    df = pd.DataFrame({'pred': txt_preds, 'true': list(it.chain.from_iterable(map(lambda x: x['texts'], val_loader)))})
    df.to_csv(os.path.join(res_path, 'submission.csv'), index=False)


def predict(learner, test_loader, args: TrainerConfig):
    import lightning.pytorch as pl
    import os

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        # max_time={'minutes': 60},

        # hardware settings
        accelerator='gpu',
        deterministic=False,
        # precision="16-mixed",

        # logging and checkpointing
        # val_check_interval=args.interval,
        # check_val_every_n_epoch=args.interval,
        # logger=logger,
        enable_progress_bar=True,
        profiler=None,
        # callbacks=callbacks,
        # log_every_n_steps=5,

        # check if model is implemented correctly
        overfit_batches=False,

        # check training_step and validation_step doesn't fail
        fast_dev_run=False,
        num_sanity_val_steps=False
    )

    if args.resume_from is None:
        raise ValueError('specify weights with `resume_from` argument')

    from datetime import datetime
    print('Started at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    pred_texts = trainer.predict(
        learner, test_loader,
        ckpt_path=args.resume_from
    )

    name = datetime.now().strftime("%H:%M:%S_%d-%m-%Y")
    print('Finished at', name)

    res_path = os.path.join('data', 'contest-data', 'predictions', name)
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    # save text predictions
    import itertools as it
    txt_preds = list(it.chain.from_iterable(pred_texts))

    import json
    json.dump(txt_preds, open(os.path.join(res_path, 'txt.json'), 'w'))

    # save as csv submission
    import pandas as pd
    df = pd.read_csv(os.path.join('data', 'contest-data', 'submission.csv'))
    df['label'] = txt_preds
    df.to_csv(os.path.join(res_path, 'submission.csv'), index=False)
