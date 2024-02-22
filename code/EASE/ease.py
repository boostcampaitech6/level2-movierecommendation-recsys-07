import pandas as pd
import numpy as np
import torch

import sys

sys.path.append("../")
from library.recall import recall_at_10


def main():
    lambda_ = 500
    print(lambda_)

    train_df = pd.read_csv("../../data/train/custom_train_ratings.csv")
    valid_df = pd.read_csv("../../data/train/custom_valid_ratings.csv")

    user2idx = {v: i for i, v in enumerate(train_df["user"].unique())}
    item2idx = {v: i for i, v in enumerate(train_df["item"].unique())}
    idx2user = {i: v for i, v in enumerate(train_df["user"].unique())}
    idx2item = {i: v for i, v in enumerate(train_df["item"].unique())}

    train_df["user"] = train_df["user"].map(user2idx)
    train_df["item"] = train_df["item"].map(item2idx)

    train_df["label"] = [1] * len(train_df)
    pivot = train_df.pivot(index="user", columns="item", values="label").fillna(0)

    print("EASE")
    X = torch.tensor(pivot.values).to(dtype=torch.float)
    G = X.T @ X

    G += torch.eye(G.shape[0]) * lambda_

    P = G.inverse()

    B = P / (-1 * P.diag())
    for i in range(len(B)):
        B[i][i] = 0

    S = X @ B

    print("Mask")
    seen = train_df.groupby("user")["item"].unique()
    user_len = seen.apply(len)
    seen_item = np.concatenate(seen.values)
    seen_user = np.arange(len(user2idx)).repeat(user_len.values)

    S[seen_user, seen_item] = -99999

    print("Recommend")
    _, item = torch.topk(S, 10)
    user_arr = np.arange(len(user2idx)).repeat(10)
    rec_df = pd.DataFrame(
        zip(user_arr, item.reshape(-1).tolist()), columns=["user", "item"]
    )

    rec_df["user"] = rec_df["user"].map(idx2user)
    rec_df["item"] = rec_df["item"].map(idx2item)

    rec_df.to_csv("valid_ease_output.csv", index=False)
    print(recall_at_10(rec_df, valid_df))


if __name__ == "__main__":
    main()
