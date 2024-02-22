import torch
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
    StepLR,
)


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
    elif args.scheduler.name.lower() == "step":
        scheduler = StepLR(
            optimizer, args.scheduler.step_size, args.scheduler.gamma, verbose=True
        )
    return scheduler
