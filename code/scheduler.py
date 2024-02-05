import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts


def get_scheduler(optimizer: torch.optim.Optimizer, args):
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, mode="max", verbose=True
        )
    elif args.scheduler == "CAWR":
        scheduler = CosineAnnealingWarmRestarts(optimizer, args.patience, verbose=True)
    return scheduler
