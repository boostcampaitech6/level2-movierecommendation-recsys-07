import os
from datetime import datetime

import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from library.utils import get_logger, set_seeds, logging_conf
from library.loader import MFDataset, FMDataset
from library.trainer import get_model, run

import split_valid


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
        entity="buzzer_beater",
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
    if args.model.name.lower() in ["mf", "lmf"]:
        train_dataset = MFDataset(args)
        _, idx_dict, seen = train_dataset.load_data(args, train=True, idx_dict=None)
        if args.dataloader.log_negative == True:
            train_dataset.log_negative_sampling(args)
        else:
            train_dataset.negative_sampling(args, n_neg=args.dataloader.n_neg)
        valid_dataset = MFDataset(args)
        valid_df, _, _ = valid_dataset.load_data(args, train=False, idx_dict=idx_dict)

    elif args.model.name.lower() in ["fm", "lfm", "cfm", "lcfm"]:
        train_dataset = FMDataset(args)
        _, idx_dict, seen = train_dataset.load_data(args, train=True, idx_dict=None)
        if args.dataloader.log_negative == True:
            train_dataset.log_negative_sampling(args)
        else:
            train_dataset.negative_sampling(args, n_neg=args.dataloader.n_neg)
        if args.dataloader.feature:
            train_dataset.load_side_information(args, train=True, idx_dict=idx_dict)
        else:
            args.feat_dim = []

        valid_dataset = FMDataset(args)
        valid_df, _, _ = valid_dataset.load_data(args, train=False, idx_dict=idx_dict)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        shuffle=True,
        batch_size=args.batch_size,
    )

    logger.info("Building Model ...")
    model = get_model(args).to(args.device)

    logger.info("Start Training ...")
    run(args, model, train_loader, seen, valid_df)

    logger.info("Saving configuration")
    OmegaConf.save(config=args, f=os.path.join(args.model_dir, "default.yaml"))


if __name__ == "__main__":
    split_valid.main()
    main()
