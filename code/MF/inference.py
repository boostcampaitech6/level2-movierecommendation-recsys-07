import os
import argparse

import numpy as np
import torch
import pickle

from loader import MFDataset
import trainer
from utils import get_logger, logging_conf


logger = get_logger(logging_conf)


def main(args):
    os.makedirs(args.model_dir, exist_ok=True)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Preparing data ...")
    # load idx
    dict_path = os.path.join(args.model_dir, "idx.pickle")
    with open(dict_path, "rb") as pk:
        idx_dict = pickle.load(pk)
    train_dataset = MFDataset(args)
    train_data, _, seen = train_dataset.load_data(args, train=True, idx_dict=idx_dict)

    logger.info("Loading Model ...")
    model: torch.nn.Module = trainer.load_model(args=args).to(args.device)

    logger.info("Recommend & Save Submission ...")
    sub_df = trainer.recommend(args=args, seen=seen, model=model)
    sub_df["user"] = sub_df["user"].map(idx_dict["idx2user"])
    sub_df["item"] = sub_df["item"].map(idx_dict["idx2item"])
    output_file_name = f"{args.model}_submission.csv"
    write_path = os.path.join(args.output_dir, output_file_name)
    os.makedirs(name=args.output_dir, exist_ok=True)
    sub_df.to_csv(write_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--device", default="cuda", type=str, help="cpu or cuda")
    parser.add_argument(
        "--data_dir",
        default="../../data/train/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--train_file_name",
        default="train_ratings.csv",
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

    parser.add_argument("--num_workers", default=4, type=int, help="number of workers")

    # 모델
    parser.add_argument(
        "--hidden_dim", default=64, type=int, help="hidden dimension size"
    )
    parser.add_argument(
        "--n_neg", default=50, type=int, help="the number of negative sample"
    )

    # 훈련
    parser.add_argument("--n_epochs", default=20, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=1024, type=int, help="batch size")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--patience", default=30, type=int, help="for early stopping")
    parser.add_argument("--weight_decay", default=0.001, type=float)

    ### 중요 ###
    parser.add_argument("--model", default="mf", type=str, help="model type")
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type")
    parser.add_argument(
        "--scheduler", default="plateau", type=str, help="scheduler type"
    )
    parser.add_argument("--loss_function", default="roc_star", type=str)

    # Custom
    parser.add_argument("--gamma", default=0.3, type=float)

    args = parser.parse_args()
    main(args)
