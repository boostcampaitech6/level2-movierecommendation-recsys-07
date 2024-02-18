import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts


def get_scheduler(optimizer: torch.optim.Optimizer, args):
    if args.scheduler.name.lower() == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=args.scheduler.patience,
            factor=args.scheduler.factor,
            mode="max",
            verbose=True,
        )
    elif args.scheduler.name.lower() == "cawr":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, args.scheduler.patience, verbose=True
        )
    return scheduler
