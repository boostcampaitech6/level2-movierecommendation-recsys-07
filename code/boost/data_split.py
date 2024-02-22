import pandas as pd
import numpy as np
import os
import math
from tqdm import tqdm


def negative_sampling(df, n_neg: int):
    np.random.seed(42)
    item_set = set(df["item"].unique())
    user_genre = dict(zip(df["user"], df["user_favotrite_genre"]))
    neg_item = np.array([], dtype=int)
    for user, u_items in tqdm(df.groupby("user")["item"]):
        u_set = set(u_items)
        user_neg_item = np.random.choice(list(item_set - u_set), n_neg, replace=False)
        neg_item = np.concatenate([neg_item, user_neg_item])

    users = df["user"].unique().repeat(n_neg)
    neg_df = pd.DataFrame(zip(users, neg_item), columns=["user", "item"])
    neg_df["user_favotrite_genre"] = neg_df["user"].map(user_genre)
    neg_df["label"] = [0] * len(neg_df)

    df = pd.concat([df, neg_df], axis=0)

    return df


def main():
    print("loading")
    data_path = "../../data/train"
    train_ratings = pd.read_csv(os.path.join(data_path, "train_ratings.csv"))
    genre_data = pd.read_csv(os.path.join(data_path, "genres.tsv"), sep="\t")

    print("adding feature")
    temp_df = pd.merge(train_ratings, genre_data, how="right", on="item")
    user_genre_counts = temp_df.groupby(["user", "genre"]).size().unstack(fill_value=0)
    most_viewed_genre = user_genre_counts.idxmax(axis=1)
    train_ratings["user_favotrite_genre"] = train_ratings["user"].map(most_viewed_genre)
    
    print("making negative_sampling")
    df = negative_sampling(train_ratings, 400).fillna(1).reset_index(drop=True)

    np.random.seed(21)
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
    valid_df.to_csv(os.path.join(data_path, "custom_valid_ratings.csv"), index=False)
    train_df.to_csv(os.path.join(data_path, "custom_train_ratings.csv"), index=False)



if __name__ == "__main__":
    main()