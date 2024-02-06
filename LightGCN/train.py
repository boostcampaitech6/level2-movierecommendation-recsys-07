import os

import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from code.datasets import prepare_dataset
from code import trainer
from code.utils import get_logger, set_seeds, logging_conf


logger = get_logger(logging_conf)


@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    os.makedirs(name=args.model_dir, exist_ok=True)
    wandb.login()
    wandb.init(
        project="gcn",
        config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
    )
    set_seeds(args.seed)

    use_cuda: bool = torch.cuda.is_available() and args.use_cuda_if_available
    device = torch.device("cuda" if use_cuda else "cpu")

    logger.info("Preparing data ...")
    train_data, n_node, id2index = prepare_dataset(
        device=device, data_dir=args.data_dir
    )

    logger.info("Building Model ...")
    model = trainer.build(
        n_node=n_node,
        embedding_dim=args.hidden_dim,
        num_layers=args.n_layers,
        alpha=args.alpha,
    )
    model = model.to(device)

    logger.info("Start Training ...")
    trainer.run(
        model=model,
        train_data=train_data,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        model_dir=args.model_dir,
        id2index=id2index,
        args=args,
    )


if __name__ == "__main__":
    main()
