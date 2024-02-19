import os

import numpy as np
import pandas as pd
import pickle
import math
from scipy import sparse


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, "item")
        tp = tp[tp["item"].isin(itemcount[itemcount["size"] >= min_sc]["item"])]

    if min_uc > 0:
        usercount = get_count(tp, "user")
        tp = tp[tp["user"].isin(usercount[usercount["size"] >= min_uc]["user"])]

    usercount, itemcount = get_count(tp, "user"), get_count(tp, "item")
    return tp, usercount, itemcount


def split_question_answer_proportion(data, proportion=0.2, type="valid"):
    data = data.sort_values("time")
    data_grouped_by_user = data.groupby("user")
    tr_list, te_list = list(), list()

    for _, group in data_grouped_by_user:
        n_items_u = len(group)

        if n_items_u >= 5:
            answer_size = (
                int(n_items_u * proportion)
                if type == "valid"
                else min(int(n_items_u * proportion), 10)
            )
            idx = np.zeros(n_items_u, dtype="bool")
            idx[
                np.random.choice(n_items_u, size=answer_size, replace=False).astype(
                    "int64"
                )
            ] = True
            idx[-1] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])

        else:
            tr_list.append(group)

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def split_for_data_augmentation(data, test_prop=0.5):
    data_grouped_by_user = data.groupby("uid")
    tr_list, te_list = list(), list()

    for _, group in data_grouped_by_user:
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype="bool")
            idx[
                np.random.choice(
                    n_items_u, size=int(test_prop * n_items_u), replace=False
                ).astype("int64")
            ] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])

        else:
            tr_list.append(group)

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def numerize(tp, profile2id, show2id):
    uid = tp["user"].apply(lambda x: profile2id[x])
    sid = tp["item"].apply(lambda x: show2id[x])
    return pd.DataFrame(data={"uid": uid, "sid": sid}, columns=["uid", "sid"])


class DataLoader:
    """
    Load Movielens dataset
    """

    def __init__(self, args):
        self.args = args
        DATA_DIR = args.data_dir
        raw_data = pd.read_csv(os.path.join(DATA_DIR, "train_ratings.csv"), header=0)

        raw_data, user_activity, item_popularity = filter_triplets(
            raw_data, min_uc=5, min_sc=0
        )

        unique_uid = user_activity["user"].unique()
        idx_perm = np.random.permutation(unique_uid.size)
        unique_uid = unique_uid[idx_perm]

        n_users = unique_uid.size  # 31360
        n_heldout_users = 3000

        tr_users = unique_uid[: (n_users - n_heldout_users * 2)]
        vd_users = unique_uid[
            (n_users - n_heldout_users * 2) : (n_users - n_heldout_users)
        ]
        te_users = unique_uid[(n_users - n_heldout_users) :]

        ##훈련 데이터에 해당하는 아이템들
        # Train에는 전체 데이터를 사용합니다.
        train_plays = raw_data.loc[raw_data["user"].isin(tr_users)]

        ##아이템 ID
        unique_sid = pd.unique(train_plays["item"])

        show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
        profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
        with open(os.path.join(args.model_dir, "item2idx.pickle"), "wb") as handle:
            pickle.dump(show2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(args.model_dir, "user2idx.pickle"), "wb") as handle:
            pickle.dump(profile2id, handle, protocol=pickle.HIGHEST_PROTOCOL)

        pro_dir = os.path.join(DATA_DIR, "pro_sg")

        if not os.path.exists(pro_dir):
            os.makedirs(pro_dir)

        with open(os.path.join(pro_dir, "unique_sid.txt"), "w") as f:
            for sid in unique_sid:
                f.write("%s\n" % sid)

        # Validation과 Test에는 input으로 사용될 tr 데이터와 정답을 확인하기 위한 te 데이터로 분리되었습니다.
        vad_plays = raw_data.loc[raw_data["user"].isin(vd_users)]
        vad_plays = vad_plays.loc[vad_plays["item"].isin(unique_sid)]
        vad_plays_tr, vad_plays_te = split_question_answer_proportion(
            vad_plays, type="valid"
        )

        test_plays = raw_data.loc[raw_data["user"].isin(te_users)]
        test_plays = test_plays.loc[test_plays["item"].isin(unique_sid)]
        test_plays_tr, test_plays_te = split_question_answer_proportion(
            test_plays, type="test"
        )

        train_data = numerize(train_plays, profile2id, show2id)
        train_data.to_csv(os.path.join(pro_dir, "train.csv"), index=False)

        vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
        vad_data_tr.to_csv(os.path.join(pro_dir, "validation_tr.csv"), index=False)

        vad_data_te = numerize(vad_plays_te, profile2id, show2id)
        vad_data_te.to_csv(os.path.join(pro_dir, "validation_te.csv"), index=False)

        test_data_tr = numerize(test_plays_tr, profile2id, show2id)
        test_data_tr.to_csv(os.path.join(pro_dir, "test_tr.csv"), index=False)

        test_data_te = numerize(test_plays_te, profile2id, show2id)
        test_data_te.to_csv(os.path.join(pro_dir, "test_te.csv"), index=False)

        print("Done!")

        self.pro_dir = os.path.join(args.data_dir, "pro_sg")
        assert os.path.exists(
            self.pro_dir
        ), "Preprocessed files do not exist. Run data.py"

        self.n_items = self.load_n_items()

    def load_data(self, datatype="train"):
        if datatype == "train":
            return self._load_train_data()
        elif datatype == "validation":
            return self._load_tr_te_data(datatype)
        elif datatype == "test":
            return self._load_tr_te_data(datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test]")

    def load_n_items(self):
        unique_sid = list()
        with open(os.path.join(self.pro_dir, "unique_sid.txt"), "r") as f:
            for line in f:
                unique_sid.append(line.strip())
        n_items = len(unique_sid)
        return n_items

    def _load_train_data(self):
        path = os.path.join(self.pro_dir, "train.csv")

        tp = pd.read_csv(path)
        n_users = tp["uid"].max() + 1

        rows, cols = tp["uid"], tp["sid"]
        data = sparse.csr_matrix(
            (np.ones_like(rows), (rows, cols)),
            dtype="float64",
            shape=(n_users, self.n_items),
        )
        self.args.N = data.shape[0]
        if self.args.data_augmentation:
            data1, data2 = split_for_data_augmentation(tp, 0.5)
            rows, cols = data1["uid"], data1["sid"]
            data1 = sparse.csr_matrix(
                (np.ones_like(rows), (rows, cols)),
                dtype="float64",
                shape=(n_users, self.n_items),
            )
            rows, cols = data2["uid"], data2["sid"]
            data2 = sparse.csr_matrix(
                (np.ones_like(rows), (rows, cols)),
                dtype="float64",
                shape=(n_users, self.n_items),
            )
            data = [data1, data2]
        return data

    def _load_tr_te_data(self, datatype="test"):
        tr_path = os.path.join(self.pro_dir, "{}_tr.csv".format(datatype))
        te_path = os.path.join(self.pro_dir, "{}_te.csv".format(datatype))

        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)

        start_idx = min(tp_tr["uid"].min(), tp_te["uid"].min())
        end_idx = max(tp_tr["uid"].max(), tp_te["uid"].max())

        rows_tr, cols_tr = tp_tr["uid"] - start_idx, tp_tr["sid"]
        rows_te, cols_te = tp_te["uid"] - start_idx, tp_te["sid"]

        data_tr = sparse.csr_matrix(
            (np.ones_like(rows_tr), (rows_tr, cols_tr)),
            dtype="float64",
            shape=(end_idx - start_idx + 1, self.n_items),
        )
        data_te = sparse.csr_matrix(
            (np.ones_like(rows_te), (rows_te, cols_te)),
            dtype="float64",
            shape=(end_idx - start_idx + 1, self.n_items),
        )
        return data_tr, data_te
