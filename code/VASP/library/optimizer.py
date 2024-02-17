import torch
from torch.optim import Adam, AdamW


def get_optimizer(model: torch.nn.Module, args):
    if args.optimizer.name == "adam":
        optimizer = Adam(
            model.parameters(),
            lr=args.optimizer.lr,
            weight_decay=args.optimizer.weight_decay,
        )
    elif args.optimizer.name == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=args.optimizer.lr,
            weight_decay=args.optimizer.weight_decay,
        )
    # 모든 parameter들의 grad값을 0으로 초기화
    optimizer.zero_grad()
    return optimizer
