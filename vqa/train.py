if __name__ == "__main__":
    from src.train_utils import config_to_argparser, retrieve_fields, TrainerConfig
    from src.learner import LearnerConfig
    
    ap = config_to_argparser([LearnerConfig, TrainerConfig])
    ap.add_argument('--model', dest='model', choices=['clip'], default='clip')
    args = ap.parse_args()
    
    learner_config = LearnerConfig(**retrieve_fields(args, LearnerConfig))
    trainer_config = TrainerConfig(**retrieve_fields(args, TrainerConfig))

    from src.train_utils import init_environment
    init_environment(trainer_config.seed)

    # === data ===
    from src.dataset import VQADataset
    train_dataset = VQADataset(path='data/my_splits/train.json')
    val_dataset = VQADataset(path='data/my_splits/val.json')

    learner_config.total_steps = len(train_dataset) * trainer_config.n_epochs // learner_config.batch_size
    
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        q_list, a_list, p_list = [], [], []
        for q, a, p in batch:
            q_list.append(q)
            a_list.append(a)
            p_list.append(p)
        return q_list, a_list, p_list

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
    from src.models import get_clip

    if args.model == 'clip':
        model, processor = get_clip()
    else:
        raise ValueError(f'unknown model {args.model}')

    from src.learner import Learner, LearnerConfig

    learner = Learner(model, processor, learner_config)

    # === trainer ===
    from src.train_utils import train

    train(learner, train_loader, val_loader, trainer_config)
