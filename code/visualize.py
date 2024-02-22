import pandas as pd
import numpy as np
import os

from typing import Union

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings(action="ignore")

from sklearn.manifold import TSNE

import numpy as np
import torch
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf

from library.loader import MFDataset, FMDataset
import library.trainer as trainer
from library.utils import get_logger, logging_conf

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
    model: torch.nn.Module = trainer.load_model(args=args)

    idx_dict = idx_dict["idx2item"]

    logger.info("Save Embedding")
    emb = np.array(
        [np.array(model.item_embedding.weight.data[idx]) for idx in idx_dict]
    )
    emb_df = pd.DataFrame(emb)
    emb_df["item"] = idx_dict.values()
    emb_df.to_csv(f"{args.model.name.lower()}_item_emb_df.csv", index=False)

    logger.info("TSNE")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_arr = tsne.fit_transform(emb_df.iloc[:, :-1])
    tsne_df = pd.DataFrame(tsne_arr)
    tsne_df["item"] = idx_dict.values()
    tsne_df.to_csv(f"{args.model.name.lower()}_item_tsne_df.csv", index=False)


if __name__ == "__main__":
    main()
