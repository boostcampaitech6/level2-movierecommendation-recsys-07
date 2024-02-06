import os

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from code.datasets import prepare_dataset
from code import trainer
from code.utils import get_logger, logging_conf, set_seeds


logger = get_logger(logging_conf)


@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    os.makedirs(name=args.model_dir, exist_ok=True)
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Preparing data ...")
    train_data, n_node, id2index = prepare_dataset(
        device=device, data_dir=args.data_dir
    )

    logger.info("Loading Model ...")
    weight: str = os.path.join(args.model_dir, args.model_name)
    model: torch.nn.Module = trainer.build(
        n_node=n_node,
        embedding_dim=args.hidden_dim,
        num_layers=args.n_layers,
        alpha=args.alpha,
        weight=weight,
    )
    model = model.to(device)

    logger.info("Make Predictions & Save Submission ...")
    trainer.inference(model=model, data=train_data, output_dir=args.output_dir, id2index=id2index)


if __name__ == "__main__":
    main()