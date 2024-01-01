if __name__ == "__main__":
    from src.train_utils import config_to_argparser, retrieve_fields, TrainerConfig
    from src.learner import LearnerConfig
    from src.modeling import CRNNConfig
    
    ap = config_to_argparser([LearnerConfig, TrainerConfig, CRNNConfig])
    ap.add_argument('--cnn', dest='cnn', choices=['resnet18', 'resnet34', 'resnet50', 'wide-resnet50', 'efficient-net'], default='efficient-net')
    ap.add_argument('--rnn', dest='rnn', choices=['gru', 'var-gru', 'all-outputs', 'transformer'], default='var-gru')
    ap.add_argument('--mode', dest='mode', choices=['train', 'val', 'test'], default='train')
    ap.add_argument('--more-data', dest='more_data', type=bool, default=False)
    args = ap.parse_args()
    
    learner_config = LearnerConfig(**retrieve_fields(args, LearnerConfig))
    trainer_config = TrainerConfig(**retrieve_fields(args, TrainerConfig))
    model_config = CRNNConfig(**retrieve_fields(args, CRNNConfig))

    from src.train_utils import init_environment
    init_environment(trainer_config.seed)

    # === data ===
    import torchvision.transforms.v2 as T
    from torch import nn
    import torch

    ordinary_transforms = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize(size=model_config.cnn_input_size, antialias=True),
        T.Normalize(mean=[0.4275, 0.4328, 0.4493], std=[0.0613, 0.0621, 0.0626]),
    ])

    aug_transforms = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize(size=model_config.cnn_input_size, antialias=True),
        T.RandomApply(nn.ModuleList([T.ColorJitter(brightness=(.6,1.4),hue=0.2,saturation=(.6,1.4))]), p=0.8),
        T.Normalize(mean=[0.4275, 0.4328, 0.4493], std=[0.0613, 0.0621, 0.0626]),
    ])

    from src.dataset import RecognitionDataset
    train_dataset = RecognitionDataset(
        path='data/contest-data',
        split='train-2' if args.more_data else 'train',
        alphabet=model_config.alphabet,
        transforms=aug_transforms
    )
    val_dataset = RecognitionDataset(
        path='data/contest-data',
        split='val-2' if args.more_data else 'val',
        alphabet=model_config.alphabet,
        transforms=ordinary_transforms
    )
    test_dataset = RecognitionDataset(
        path='data/contest-data',
        split='test',
        alphabet=model_config.alphabet,
        transforms=ordinary_transforms
    )

    learner_config.total_steps = len(train_dataset) * trainer_config.n_epochs // learner_config.batch_size

    from torch.utils.data import DataLoader

    import torch
    import itertools as it


    def collate_fn(batch):
        return {
            'images': torch.stack([item['image'] for item in batch]),
            'targets': torch.IntTensor(list(it.chain.from_iterable([item['seq'] for item in batch]))),
            'target_lengths': torch.IntTensor([item['seq_len'] for item in batch]),
            'texts': [item['text'] for item in batch]
        }

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=learner_config.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        # collate_fn=collate_fn
    )

    # === model and learner ===
    from src.modeling import *

    if args.cnn == 'resnet18':
        cnn = get_resnet18()
    elif args.cnn == 'resnet34':
        cnn = get_resnet34()
    elif args.cnn == 'resnet50':
        cnn = get_resnet50()
    elif args.cnn == 'wide-resnet50':
        cnn = get_wide_resnet50()
    elif args.cnn == 'efficient-net':
        cnn = get_efficient_net()
    else:
        raise ValueError(f'unknown cnn model {args.cnn}')
    
    if args.rnn == 'gru':
        rnn = get_gru(model_config)
    if args.rnn == 'var-gru':
        rnn = VariationalDropoutGRU(model_config)
    elif args.rnn == 'all-outputs':
        rnn = AllOutputsGRU(model_config)
    elif args.rnn == 'transformer':
        rnn = Transformer(model_config.cnn_output_features)
    else:
        raise ValueError(f'unknown rnn model {args.rnn}')
    
    model = CRNN(model_config, cnn, rnn)

    from src.learner import Learner

    learner = Learner(model, learner_config)

    # === trainer ===
    from src.train_utils import train, validate, predict

    if args.mode == 'train':
        train(learner, train_loader, val_loader, trainer_config)
    elif args.mode == 'val':
        validate(learner, val_loader, trainer_config)
    elif args.mode == 'test':
        predict(learner, test_loader, trainer_config)
    else:
        raise ValueError('mode must be one of `train`, `val`, `test`')
