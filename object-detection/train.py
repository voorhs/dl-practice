if __name__ == "__main__":
    from src.train_utils import config_to_argparser, retrieve_fields, TrainerConfig
    from src.learner import LearnerConfig
    
    ap = config_to_argparser([LearnerConfig, TrainerConfig])
    ap.add_argument('--model', dest='model', choices=['vgg16', 'resnet18'], default='vgg16')
    args = ap.parse_args()
    
    learner_config = LearnerConfig(**retrieve_fields(args, LearnerConfig))
    trainer_config = TrainerConfig(**retrieve_fields(args, TrainerConfig))

    from src.train_utils import init_environment
    init_environment(args.seed)

    # === data ===
    import torchvision.transforms.v2 as T
    from torch import nn
    ordinary_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.4824, 0.4495, 0.3981], std=[0.2765, 0.2691, 0.2825]),
    ])

    aug_transforms = T.Compose([
        T.ToTensor(),
        # T.RandomResizedCrop(size=(64, 64), scale=(0.3, 1), antialias=True),
        # T.RandomHorizontalFlip(p=0.5),
        T.RandomApply(nn.ModuleList([T.ColorJitter(brightness=(.6,1.4),hue=0.2,saturation=(.6,1.4))]), p=0.8),
        T.Normalize(mean=[0.4650, 0.4342, 0.3944], std=[0.2729, 0.2731, 0.2707]),
    ])

    from src.dataset import VOCDetection
    train_dataset = VOCDetection(
        path='data/dataset',
        split='my_splits/train',
        transform=aug_transforms
    )
    val_dataset = VOCDetection(
        path='data/dataset',
        split='my_splits/val',
        transform=ordinary_transforms
    )

    learner_config.total_steps = len(train_dataset) * trainer_config.n_epochs
    
    from torch.utils.data import DataLoader
    import torch

    def collate_fn(batch):
        label_ss, box_ss, image_s = [], [], []

        for img, box, lab in batch:
            image_s.append(img)
            box_ss.append(box)
            label_ss.append(lab)

        return torch.stack(image_s), box_ss, label_ss    

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

    # === model and learner ===
    from src import SSD_VGG16, SSD_Resnet18, Resnet18PriorBoxesConfig, VGG16PriorBoxesConfig

    if args.model == 'vgg16':
        model = SSD_VGG16(
            n_priors_list=[2, 2, 2, 2, 2],
            n_classes=3
        )
        boxes_config = VGG16PriorBoxesConfig()
    elif args.model == 'resnet18':
        model = SSD_Resnet18(
            n_priors_list=[2, 2, 2, 2, 2],
            n_classes=3
        )
        boxes_config = Resnet18PriorBoxesConfig()
    else:
        raise ValueError(f'unknown model {args.model}')

    from src import Learner, LearnerConfig, PriorBoxesConfig

    learner = Learner(model, learner_config, boxes_config)

    # === trainer ===
    from src.train_utils import train

    train(learner, train_loader, val_loader, trainer_config)
