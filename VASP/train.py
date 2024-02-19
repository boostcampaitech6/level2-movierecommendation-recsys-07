import os
from datetime import datetime

import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from library.utils import get_logger, set_seeds, logging_conf
from library.loader import DataLoader
from library.trainer import get_model, run

logger = get_logger(logging_conf)


@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(args: DictConfig):
    # seed and device
    OmegaConf.set_struct(args, False)
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # wandb initialize
    wandb.login()
    wandb.init(
        project="vasp_dev",  # + args.model.name,
        config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
        entity="buzzer_beater",
    )

    # make and set directory to save a model
    os.makedirs(args.model_dir, exist_ok=True)
    args.model_dir = os.path.join(
        args.model_dir,
        args.model.name.lower(),
        datetime.utcfromtimestamp(wandb.run.start_time).strftime("%Y-%m-%d_%H:%M:%S")
        + wandb.run.name,
    )
    os.makedirs(args.model_dir, exist_ok=True)

    logger.info("Preparing data ...")
    loader = DataLoader(args)

    args.n_items = loader.load_n_items()
    train_data = loader.load_data("train")
    vad_data_tr, vad_data_te = loader.load_data("validation")
    test_data_tr, test_data_te = loader.load_data("test")
    data = [train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te]

    args.idxlist = list(range(args.N))

    logger.info("Building Model ...")
    model = get_model(args).to(args.device)

    logger.info("Start Training ...")
    run(args, model, data)

    logger.info("Saving configuration")
    OmegaConf.save(config=args, f=os.path.join(args.model_dir, "default.yaml"))


if __name__ == "__main__":
    main()
