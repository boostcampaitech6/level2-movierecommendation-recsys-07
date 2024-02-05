import os
import argparse

import torch
from utils import get_logger, set_seeds, logging_conf
from datetime import datetime
import wandb

from loader import MFDataset, get_loader
from trainer import get_model, run

logger = get_logger(logging_conf)


def main(args):
    os.makedirs(args.model_dir, exist_ok=True)
    wandb.login()
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project=f"{args.model}".lower(), config=args)

    args.model_dir = os.path.join(
        args.model_dir,
        args.model.lower(),
        datetime.utcfromtimestamp(wandb.run.start_time).strftime("%Y-%m-%d_%H:%M:%S")
        + wandb.run.name,
    )

    logger.info("Preparing data ...")
    train_dataset = MFDataset(args)
    _, idx_dict, seen = train_dataset.load_data(args, train=True, idx_dict=None)
    train_dataset.negative_sampling(args, n_neg=args.n_neg)

    valid_dataset = MFDataset(args)
    valid_df, _, _ = valid_dataset.load_data(
        args, train=False, idx_dict=idx_dict
    )  # recall 측정 위해 valid_df 저장.

    train_loader, valid_loader = get_loader(args, train_dataset, valid_dataset)

    logger.info("Building Model ...")
    model = get_model(args).to(args.device)

    logger.info("Start Training ...")
    run(args, model, train_loader, valid_loader, idx_dict, seen, valid_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--device", default="cuda", type=str, help="cpu or cuda")
    parser.add_argument(
        "--data_dir",
        default="../data/train/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--train_file_name",
        default="custom_train_ratings.csv",
        type=str,
        help="train file name",
    )
    parser.add_argument(
        "--model_dir", default="models/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="best_model.pt", type=str, help="model file name"
    )
    parser.add_argument(
        "--output_dir", default="outputs/", type=str, help="output directory"
    )
    parser.add_argument(
        "--valid_file_name",
        default="custom_valid_ratings.csv",
        type=str,
        help="valid file name",
    )
    parser.add_argument("--log_steps", default=50, type=int)

    parser.add_argument("--num_workers", default=4, type=int, help="number of workers")

    # 모델
    parser.add_argument(
        "--hidden_dim", default=64, type=int, help="hidden dimension size"
    )
    parser.add_argument(
        "--n_neg", default=400, type=int, help="the number of negative sample"
    )

    # 훈련
    parser.add_argument("--n_epochs", default=200, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=4096, type=int, help="batch size")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--patience", default=30, type=int, help="for early stopping")
    parser.add_argument("--weight_decay", default=0.0001, type=float)

    ### 중요 ###
    parser.add_argument("--model", default="mf", type=str, help="model type")
    parser.add_argument(
        "--optimizer",
        default="adam",
        type=str,
        help="optimizer type",
        choices=["adam", "adamW"],
    )
    parser.add_argument(
        "--scheduler",
        default="plateau",
        type=str,
        help="scheduler type",
        choices=["plateau"],
    )
    parser.add_argument(
        "--loss_function",
        default="roc_star",
        type=str,
        choices=["roc_star", "bce", "bpr"],
    )

    # Custom
    parser.add_argument("--gamma", default=0.3, type=float)

    args = parser.parse_args()
    main(args)
