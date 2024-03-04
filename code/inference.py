import os
import argparse

import numpy as np
import torch
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf

from library.loader import MFDataset, FMDataset
import library.trainer as trainer
from library.utils import get_logger, logging_conf

import split_valid

logger = get_logger(logging_conf)


@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(args):
    OmegaConf.set_struct(args, False)
    os.makedirs(args.model_dir, exist_ok=True)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.train_file_name = "train_ratings.csv"

    logger.info("Preparing data ...")
    # load idx
    dict_path = os.path.join(args.model_dir, "idx.pickle")
    with open(dict_path, "rb") as pk:
        idx_dict = pickle.load(pk)

    # get seen data for recommendation
    if args.model.name.lower() in ["mf", "lmf"]:
        train_dataset = MFDataset(args)
        _, _, seen = train_dataset.load_data(args, train=True, idx_dict=idx_dict)
    elif args.model.name.lower() in ["fm", "lfm", "cfm", "lcfm"]:
        train_dataset = FMDataset(args)
        _, _, seen = train_dataset.load_data(args, train=True, idx_dict=idx_dict)
        train_dataset.load_side_information(args, train=True, idx_dict=idx_dict)

    logger.info("Loading Model ...")
    model: torch.nn.Module = trainer.load_model(args=args).to(args.device)

    logger.info("Recommend & Save Submission ...")
    sub_df = trainer.recommend(args=args, seen=seen, model=model)
    sub_df["user"] = sub_df["user"].map(idx_dict["idx2user"])
    sub_df["item"] = sub_df["item"].map(idx_dict["idx2item"])
    output_file_name = f"{args.model.name}_submission.csv"
    write_path = os.path.join(args.output_dir, output_file_name)
    os.makedirs(name=args.output_dir, exist_ok=True)
    sub_df.to_csv(write_path, index=False)

    logger.info(f"Save Submission as {write_path}")


if __name__ == "__main__":
    split_valid.main()
    main()
