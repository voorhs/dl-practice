if __name__ == "__main__":
    N_CLASSES = 200

    from src.train_utils import get_argparser
    ap = get_argparser()
    args = ap.parse_args()

    from src.train_utils import init_environment
    init_environment(args)

    # === model and learner ===
    from src.models import MyResNet, get_seresnext, get_seresnet, get_skresnet, get_seresnet34, get_skresnet34, get_resnet34, Ensemble
    if args.model == 'myresnet-small':
        model = MyResNet(planes=32)
    elif args.model == 'myresnet-base':
        model = MyResNet(planes=64)
    elif args.model == 'seresnet18':
        model = get_seresnet()
    elif args.model == 'seresnext':
        model = get_seresnext()
    elif args.model == 'skresnet':
        model = get_skresnet()
    elif args.model == 'seresnet34':
        model = get_seresnet34()
    elif args.model == 'skresnet34':
        model = get_skresnet34()
    elif args.model == 'resnet34':
        model = get_resnet34()
    elif args.model == 'ensemble':
        model = Ensemble()

    from src.learners import ClfLearner, ClfLearnerConfig
    config = ClfLearnerConfig(
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        max_steps=args.max_steps,       # n_batches * n_epochs
        lr_last=args.lr_last,
        mixup_alpha=args.mixup_alpha
    )

    learner = ClfLearner(model, config)

    # === data ===
    import torchvision.transforms.v2 as T
    from torch import nn
    
    ordinary_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.4824, 0.4495, 0.3981], std=[0.2765, 0.2691, 0.2825]),
    ])

    aug_transforms = T.Compose([
        T.ToTensor(),
        T.RandomResizedCrop(size=(64, 64), scale=(0.3, 1), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply(nn.ModuleList([T.ColorJitter(brightness=(.6,1.4),hue=0.2,saturation=(.6,1.4))]), p=0.8),
        T.Normalize(mean=[0.4650, 0.4342, 0.3944], std=[0.2729, 0.2731, 0.2707]),
    ])

    import os
    path = os.path.join('.', 'dataset', 'tiny-imagenet-200')
    
    from src.datasets import TinyImagenetDatasetRAM
    train_dataset = TinyImagenetDatasetRAM(path=path, split='train', transforms=aug_transforms)
    val_dataset = TinyImagenetDatasetRAM(path=path, split='val', transforms=ordinary_transforms)
    
    from torch.utils.data import DataLoader, default_collate
    from src.train_utils import mixup
    
    def collate_fn(batch):
        images, targets = default_collate(batch)
        return mixup(images, targets, config.mixup_alpha, N_CLASSES)
        
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        drop_last=True,
        collate_fn=collate_fn if config.mixup_alpha > 0 else default_collate
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.n_workers
    )
    
    # === trainer ===
    from src.train_utils import train

    train(learner, train_loader, val_loader, args)
