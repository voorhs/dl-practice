if __name__ == "__main__":
    N_CLASSES = 200

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', dest='model', choices=[
        'myresnet-base', 'myresnet-small', 'seresnet18', 'seresnext', 'skresnet', 'seresnet34'
    ], required=True)
    ap.add_argument('--name-out', dest='name_out', required=True)
    ap.add_argument('--seed', dest='seed', default=0, type=int)
    ap.add_argument('--n_workers', dest='n_workers', default=2, type=int)
    ap.add_argument('--batch_size', dest='batch_size', default=64, type=int)
    ap.add_argument('--weights-from', dest='weights_from', default=None)
    args = ap.parse_args()

    from src.train_utils import init_environment
    init_environment(args)

    # === model and learner ===
    from src.models import MyResNet, get_seresnext, get_seresnet, get_skresnet, get_seresnet34
    if args.model == 'myresnet-small':
        model_ = MyResNet(planes=32)
    elif args.model == 'myresnet-base':
        model_ = MyResNet(planes=64)
    elif args.model == 'seresnet18':
        model_ = get_seresnet()
    elif args.model == 'seresnext':
        model_ = get_seresnext()
    elif args.model == 'skresnet':
        model_ = get_skresnet()
    elif args.model == 'seresnet34':
        model_ = get_seresnet34()
    
    from src.learners import ClfLearner, ClfLearnerConfig
    config = ClfLearnerConfig(batch_size=args.batch_size)
    learner = ClfLearner(model_, config)


    # === data ===
    import torchvision.transforms.v2 as T
    
    ordinary_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.4824, 0.4495, 0.3981], std=[0.2765, 0.2691, 0.2825]),
    ])

    import os
    path = os.path.join('.', 'dataset', 'tiny-imagenet-200')
    
    from src.datasets import TinyImagenetDatasetRAM
    train_dataset = TinyImagenetDatasetRAM(path=path, split='train', transforms=ordinary_transforms)
    val_dataset = TinyImagenetDatasetRAM(path=path, split='val', transforms=ordinary_transforms)
    test_dataset = TinyImagenetDatasetRAM(path=path, split='test', transforms=ordinary_transforms)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers
    )

    # === trainer ===
    # import lightning.pytorch as pl

    # trainer = pl.Trainer(
    #     # max_epochs=args.max_epochs,
    #     # max_time={'hours': 10},
        
    #     # max_time={'minutes': 10},

    #     # hardware settings
    #     accelerator='gpu',
    #     deterministic=True,
    #     precision="16-mixed",

    #     # logging and checkpointing
    #     # val_check_interval=args.interval,
    #     # check_val_every_n_epoch=args.interval,
    #     logger=False,
    #     enable_progress_bar=True,
    #     profiler=None,
    #     # callbacks=callbacks,
    #     # log_every_n_steps=5,

    #     # check if model is implemented correctly
    #     overfit_batches=False,

    #     # check training_step and validation_step doesn't fail
    #     fast_dev_run=False,
    #     num_sanity_val_steps=False
    # )

    # trainer.validate(learner, train_loader, ckpt_path=args.weights_from)
    import torch
    torch.enable_grad(False)

    learner = ClfLearner.load_from_checkpoint(
        checkpoint_path=args.weights_from,
        model=model_,
        config=ClfLearnerConfig()
    ).eval()

    # predict on val
    import pandas as pd
    all_preds = []
    true = 0
    for batch in val_loader:
        images = batch[0].cuda()
        targets = batch[1]
        pred_labels = learner.predict_step(images).argmax(1).cpu()
        all_preds.append(pred_labels)
        true += torch.count_nonzero(targets == pred_labels).item()
    all_preds = torch.concat(all_preds)
    print('val accuracy:', true / len(all_preds))
    
    all_names = val_dataset.class_directories
    image_file_names = val_dataset.image_names
    val_df = pd.DataFrame({
        'id': [image_file_names[i//50][i%50] for i, i_class in enumerate(all_preds)],
        'pred': [all_names[i_class] for i_class in all_preds]
    }).set_index('id')

    # predict on test
    all_preds = []
    for batch in test_loader:
        images = batch.cuda()
        pred_labels = learner.predict_step(images).argmax(1).cpu()
        all_preds.append(pred_labels)
    all_preds = torch.concat(all_preds)

    image_file_names = test_dataset.image_names
    test_df = pd.DataFrame({
        'id': [image_file_names[i] for i, i_class in enumerate(all_preds)],
        'pred': [all_names[i_class] for i_class in all_preds]
    }).set_index('id')

    # merge test and val (make submission)
    subm = pd.concat([val_df, test_df], axis=0)
    path = os.path.join('submissions', f'{args.name_out}.csv')
    subm.to_csv(path)
