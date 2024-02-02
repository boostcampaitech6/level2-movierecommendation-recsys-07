import pandas as pd
import numpy as np


def main():
    np.random.seed(42)

    print("Load data")
    df = pd.read_csv("../data/train/train_ratings.csv")

    print("Sampling")
    group = df.groupby("user", group_keys=False)
    sampled_group = group.apply(
        lambda data: data.take(
            np.random.permutation(len(data))[: int(len(data) * 0.01)]
        )
    )
    idx = sampled_group.index
    idx = idx.sort_values()

    print("Split")
    valid_df = df.iloc[idx]
    train_df = df[~df.index.isin(idx)]

    print("Save")
    valid_df.to_csv("../data/train/custom_valid_ratings.csv", index=False)
    train_df.to_csv("../data/train/custom_train_ratings.csv", index=False)


if __name__ == "__main__":
    main()
