if __name__ == "__main__":
    N_CLASSES = 200

    from src.train_utils import get_argparser
    ap = get_argparser()
    args = ap.parse_args()

    from src.train_utils import init_environment
    init_environment(args)

    # === model and learner ===
    from src.models import Ensemble
    from src.learners import ClfLearner, ClfLearnerConfig
    model = Ensemble()

    config = ClfLearnerConfig(
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        max_steps=args.max_steps,       # n_batches * n_epochs
        lr_last=args.lr_last,
    )

    learner = ClfLearner(model, config)

    # === data ===
    import torchvision.transforms.v2 as T
    from torch import nn
    
    ordinary_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.4824, 0.4495, 0.3981], std=[0.2765, 0.2691, 0.2825]),
    ])

    import os
    path = os.path.join('.', 'dataset', 'tiny-imagenet-200')
    
    from src.datasets import TinyImagenetDatasetRAM
    train_dataset = TinyImagenetDatasetRAM(path=path, split='train', transforms=ordinary_transforms)
    val_dataset = TinyImagenetDatasetRAM(path=path, split='val', transforms=ordinary_transforms)
    
    from torch.utils.data import DataLoader
        
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        drop_last=True,
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
