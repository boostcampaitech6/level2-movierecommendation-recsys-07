import os
import sys

sys.path.append("../")
from Word2Vec.word2vec import tsne, visualize

import argparse

import pandas as pd
import numpy as np
import pickle

import trainer
from utils import get_logger, logging_conf

logger = get_logger(logging_conf)


def main(args):
    # load idx
    dict_path = os.path.join(args.model_dir, "idx.pickle")
    with open(dict_path, "rb") as pk:
        idx_dict = pickle.load(pk)

    logger.info("Loading Model ...")
    model = trainer.load_model(args=args)
    emb = model.item_embedding.weight.detach().numpy()

    logger.info("Save Embedding ...")
    emb_df = pd.DataFrame(emb).reset_index()
    emb_df["item"] = emb_df.iloc[:, 0].map(idx_dict["idx2item"])
    emb_df = emb_df.iloc[:, 1:]
    emb_df.columns.astype(str)
    emb_df.to_csv(f"{args.model}_emb.csv", index=False)

    logger.info("TSNE ...")
    item_uniq = idx_dict["idx2item"].values()
    tsne_arr, _, _ = tsne(emb_df, f"{args.model}_tsne_df.csv", item_uniq)

    logger.info("Visualize ...")
    visualize(tsne_arr, item_uniq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 경로
    parser.add_argument(
        "--model_dir", default="models/", type=str, help="model directory"
    )

    # 모델
    parser.add_argument(
        "--model_name", default="best_model.pt", type=str, help="model file name"
    )
    parser.add_argument(
        "--hidden_dim", default=64, type=int, help="hidden dimension size"
    )
    parser.add_argument("--model", default="mf", type=str, help="model type")

    args = parser.parse_args()
    main(args)
