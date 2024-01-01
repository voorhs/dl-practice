if __name__ == "__main__":
    N_CLASSES = 200

    from src.train_utils import get_argparser
    ap = get_argparser()
    ap.add_argument('--k', dest='k', default=5, type=int)
    ap.add_argument('--t', dest='t', default=0.05, type=float)
    args = ap.parse_args()

    from src.train_utils import init_environment
    init_environment(args)

    # === model and learner ===
    from src.models import MyResNet, se_resnet3, se_resnet5, se_resnet10
    if args.model == 'myresnet':
        model = MyResNet()
    elif args.model == 'senet3':
        model = se_resnet3()
    elif args.model == 'senet5':
        model = se_resnet5()
    elif args.model == 'senet10':
        model = se_resnet10()

    from src.learners import ContrastiveLearner, ContrastiveLearnerConfig
    config = ContrastiveLearnerConfig(
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        max_steps=args.max_steps,       # n_batches * n_epochs
        lr_last=args.lr_last,
        k=args.k,
        t=args.t
    )

    if args.weights_from is not None:
        learner = ContrastiveLearner.load_from_checkpoint(
            checkpoint_path=args.weights_from,
            model=model,
            config=config
        )
    else:
        learner = ContrastiveLearner(model, config)

    # === data ===
    import torchvision.transforms.v2 as T
    
    ordinary_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.4824, 0.4495, 0.3981], std=[0.2765, 0.2691, 0.2825]),
    ])

    import os
    imagenet_path = os.path.join('.', 'dataset' ,'imagenet')
    
    from src.datasets import ImagenetContrastive
    train_dataset = ImagenetContrastive(path=imagenet_path, split='train')
                                      
    from src.datasets import TinyImagenetDataset
    tiny_path = os.path.join('.', 'dataset', 'tiny-imagenet-200')
    tiny_train_dataset = TinyImagenetDataset(path=tiny_path, split='train', transforms=ordinary_transforms)
    tiny_val_dataset = TinyImagenetDataset(path=tiny_path, split='val', transforms=ordinary_transforms)
    
    from torch.utils.data import DataLoader
    imagenet_train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        drop_last=True,
    )
    tiny_train_loader = DataLoader(
        dataset=tiny_train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.n_workers
    )
    tiny_val_loader = DataLoader(
        dataset=tiny_val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.n_workers
    )
    
    # === trainer ===
    from src.train_utils import train

    train(learner, imagenet_train_loader, [tiny_train_loader, tiny_val_loader], args)
