import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import pickle

from tqdm import tqdm


class MFDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.data = None
        self.args = args

    def load_data(
        self,
        args,
        train: bool,
        idx_dict: Optional[dict],
    ) -> Tuple[pd.DataFrame, dict, pd.Series]:
        if train:
            file_path = os.path.join(self.args.data_dir, self.args.train_file_name)
        else:
            file_path = os.path.join(self.args.data_dir, self.args.valid_file_name)

        self.data = pd.read_csv(file_path)[["user", "item"]]
        self.data["label"] = [1] * len(self.data)

        if idx_dict == None:
            user2idx = {v: i for i, v in enumerate(self.data["user"].unique())}
            idx2user = {i: v for i, v in enumerate(self.data["user"].unique())}
            item2idx = {v: i for i, v in enumerate(self.data["item"].unique())}
            idx2item = {i: v for i, v in enumerate(self.data["item"].unique())}

            idx = {
                "user2idx": user2idx,
                "idx2user": idx2user,
                "item2idx": item2idx,
                "idx2item": idx2item,
            }

            # save idx
            dict_path = os.path.join(args.model_dir, "idx.pickle")
            with open(dict_path, "wb") as pk:
                pickle.dump(idx, pk)
        else:
            idx = idx_dict

        self.data["user"] = self.data["user"].map(idx["user2idx"])
        self.data["item"] = self.data["item"].map(idx["item2idx"])

        if train:
            args.n_users = self.data["user"].nunique()
            args.n_items = self.data["item"].nunique()
            args.n_rows = len(self.data)

            seen = self.data.groupby("user")["item"].apply(list)
        else:
            seen = None

        return self.data, idx, seen

    def negative_sampling(self, args, n_neg: int):
        df = self.data
        item_set = set(df["item"].unique())
        neg_item = np.array([])
        for user, u_items in tqdm(df.groupby("user")["item"]):
            u_set = set(u_items)
            user_neg_item = np.random.choice(
                list(item_set - u_set), n_neg, replace=False
            )
            neg_item = np.concatenate([neg_item, user_neg_item])

        users = df["user"].unique().repeat(n_neg)
        neg_df = pd.DataFrame(zip(users, neg_item), columns=["user", "item"])
        neg_df["label"] = [0] * len(neg_df)

        df = pd.concat([df, neg_df], axis=0)

        self.data = df

    def __getitem__(self, index: int) -> np.ndarray:
        row = self.data.iloc[index].values
        return row

    def __len__(self) -> int:
        return len(self.data)


def get_loader(args, trainset, validset) -> Tuple[torch.utils.data.DataLoader]:
    train_loader = torch.utils.data.DataLoader(
        trainset, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size
    )
    valid_loader = torch.utils.data.DataLoader(
        validset,
        num_workers=args.num_workers,
        shuffle=False,
        batch_size=args.batch_size,
    )

    return train_loader, valid_loader
