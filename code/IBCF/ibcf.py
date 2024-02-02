import pandas as pd
import numpy as np
import sys

sys.path.append("../")
from Word2Vec.word2vec import tsne, visualize


def main():
    print("Load df")
    df = pd.read_csv("../../data/train/custom_train_ratings.csv")
    df["label"] = [1] * len(df)

    print("Pivot")
    pivot = pd.pivot(df, index="user", columns="item", values="label")
    pivot = pivot.reset_index(drop=True).fillna(0).T
    pivot["item"] = pivot.index
    pivot = pivot.reset_index(drop=True)

    print("Save Emb")
    pivot.to_csv(f"IBCF_emb.csv", index=False)

    tsne_df_name = "IBCF_TSNE"
    tsne_arr, item_uniq, _ = tsne(pivot, tsne_df_name, pivot["item"].values)

    visualize(tsne_arr, item_uniq)


if __name__ == "__main__":
    main()
