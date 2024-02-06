import os
from datetime import datetime

import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from library.utils import get_logger, set_seeds, logging_conf
from library.loader import MFDataset, get_loader
from library.trainer import get_model, run


logger = get_logger(logging_conf)


@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(args: DictConfig):
    # initialize
    OmegaConf.set_struct(args, False)
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # wandb initialize
    wandb.login()
    wandb.init(
        project=args.model.name,
        config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
    )

    # set directory to save a model
    os.makedirs(args.model_dir, exist_ok=True)
    args.model_dir = os.path.join(
        args.model_dir,
        args.model.name.lower(),
        datetime.utcfromtimestamp(wandb.run.start_time).strftime("%Y-%m-%d_%H:%M:%S")
        + wandb.run.name,
    )

    logger.info("Preparing data ...")
    train_dataset = MFDataset(args)
    _, idx_dict, seen = train_dataset.load_data(args, train=True, idx_dict=None)
    train_dataset.negative_sampling(args, n_neg=args.dataloader.n_neg)

    valid_dataset = MFDataset(args)
    valid_df, _, _ = valid_dataset.load_data(args, train=False, idx_dict=idx_dict)

    train_loader, valid_loader = get_loader(args, train_dataset, valid_dataset)

    logger.info("Building Model ...")
    model = get_model(args).to(args.device)

    logger.info("Start Training ...")
    run(args, model, train_loader, valid_loader, idx_dict, seen, valid_df)

    logger.info("Saving configuration")
    OmegaConf.save(config=args, f=os.path.join(args.model_dir, "default.yaml"))


if __name__ == "__main__":
    main()
