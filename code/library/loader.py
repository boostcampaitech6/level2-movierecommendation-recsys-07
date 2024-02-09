import os
import pickle
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class MFDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.data = None
        self.args = args
        self.values = None

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
            os.makedirs(args.model_dir, exist_ok=True)
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

            seen = self.data.groupby("user")["item"].apply(np.array)
        else:
            seen = None

        return self.data, idx, seen

    def negative_sampling(self, args, n_neg: int):
        df = self.data
        item_set = set(df["item"].unique())
        neg_item = np.zeros((args.n_users, n_neg), dtype=int)
        for user, u_items in tqdm(df.groupby("user")["item"]):
            u_set = set(u_items)
            user_neg_item = np.random.choice(
                list(item_set - u_set), n_neg, replace=False
            )
            neg_item[user] = user_neg_item
        neg_item = np.concatenate(neg_item)

        users = df["user"].unique().repeat(n_neg)
        neg_df = pd.DataFrame(zip(users, neg_item), columns=["user", "item"])
        neg_df["label"] = [0] * len(neg_df)

        df = pd.concat([df, neg_df], axis=0)

        self.data = df

        self.values = self.data.values

    def __getitem__(self, index: int) -> np.ndarray:
        row = self.values[index]
        return row

    def __len__(self) -> int:
        return len(self.data)


class FMDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.data = None
        self.args = args
        self.values = None

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
            os.makedirs(args.model_dir, exist_ok=True)
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

            seen = self.data.groupby("user")["item"].apply(np.array)
        else:
            seen = None

        return self.data, idx, seen

    def negative_sampling(self, args, n_neg: int):
        df = self.data
        item_set = set(df["item"].unique())
        neg_item = np.zeros((args.n_users, n_neg), dtype=int)
        for user, u_items in tqdm(df.groupby("user")["item"]):
            u_set = set(u_items)
            user_neg_item = np.random.choice(
                list(item_set - u_set), n_neg, replace=False
            )
            neg_item[user] = user_neg_item
        neg_item = np.concatenate(neg_item)

        users = df["user"].unique().repeat(n_neg)
        neg_df = pd.DataFrame(zip(users, neg_item), columns=["user", "item"])
        neg_df["label"] = [0] * len(neg_df)

        df = pd.concat([df, neg_df], axis=0)

        self.data = df

        self.values = self.data.values

    def __getitem__(self, index: int) -> np.ndarray:
        row = self.values[index]
        return row

    def __len__(self) -> int:
        return len(self.data)

    def load_side_information(self, args, train: bool, idx_dict: dict):
        side_df = pd.DataFrame()
        args.feat_dim = []
        for feature in args.dataloader.feature:
            file_path = os.path.join(self.args.data_dir, f"{feature}.tsv")
            feature_df = pd.read_csv(file_path, sep="\t")

            # direcotrs.tsv의 column 이름은 director
            if feature in ["directors", "genres", "titles", "writers", "years"]:
                feature = feature[:-1]

            if train:
                # None 값을 위한 zero padding.
                feat2idx = {
                    v: i + 1 for i, v in enumerate(feature_df[feature].unique())
                }
                feat2idx["None"] = 0
                idx2feat = {
                    i + 1: v for i, v in enumerate(feature_df[feature].unique())
                }
                idx2feat[0] = "None"
                args.feat_dim.append(len(feat2idx))

                idx_dict[f"{feature}2idx"] = feat2idx
                idx_dict[f"idx2{feature}"] = idx2feat

                # save idx
                dict_path = os.path.join(args.model_dir, "idx.pickle")
                with open(dict_path, "wb") as pk:
                    pickle.dump(idx_dict, pk)
            else:
                # 이미 완성된 idx dict를 입력받음.
                pass

            # item: 0 / genre: 2^1 + 2^3 + 2^7
            # feature의 0은 None 값을 위한 padding
            feature_df[feature] = feature_df[feature].map(idx_dict[f"{feature}2idx"])
            feature_df = (
                feature_df.groupby("item")
                .apply(lambda r: sum([1 << i for i in r[f"{feature}"].unique()]))
                .reset_index()
                .rename(columns={0: f"{feature}"})
            )

            if side_df.empty:
                side_df = feature_df
            else:
                side_df = side_df.merge(feature_df, on="item", how="left")

        side_df["item"] = side_df["item"].map(idx_dict["item2idx"])
        side_df = side_df.sort_values("item").fillna(1).astype(dtype=int)
        # for recommend
        args.item2feat = []
        for feature in args.dataloader.feature:
            if feature in ["directors", "genres", "titles", "writers", "years"]:
                feature = feature[:-1]
            feat_list = side_df[f"{feature}"].tolist()
            args.item2feat.append(feat_list)
        # label이 마지막 컬럼이 되도록 정렬
        df = self.data.merge(side_df, on="item", how="left")
        col = df.columns.to_numpy()
        col = np.concatenate((col[:2], col[3:], col[2:3]))
        self.data = df[col]
        self.values = self.data.values
