import pandas as pd
import numpy as np
import math


def main():
    np.random.seed(21)

    print("Load data")
    df = pd.read_csv("../data/train/train_ratings.csv")

    print("Sampling")
    group = df.groupby("user", group_keys=False)
    sampled_group = group.apply(
        lambda data: data.take(
            np.random.permutation(len(data))[: int(math.log(len(data), 4))]
        )
    )
    idx1 = sampled_group.index

    group = df.groupby("item", group_keys=False)
    sampled_group = group.apply(
        lambda data: data.take(
            np.random.permutation(len(data))[: int(math.log(len(data), 1.2))]
        )
    )
    idx2 = sampled_group.index
    idx = idx1.append(idx2).unique()
    idx = idx.sort_values()

    print("Split")
    valid_df = df.iloc[idx]
    train_df = df[~df.index.isin(idx)]

    print("Save")
    valid_df.to_csv("../data/train/custom_valid_ratings.csv", index=False)
    train_df.to_csv("../data/train/custom_train_ratings.csv", index=False)


if __name__ == "__main__":
    main()
