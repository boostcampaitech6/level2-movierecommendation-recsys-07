import os
from typing import Tuple

import pandas as pd
import numpy as np
import torch

from tqdm import tqdm

from code.utils import get_logger, logging_conf
import pickle

def save_preprocessed_data(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_preprocessed_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

logger = get_logger(logging_conf)

def prepare_dataset(
    device: str, data_dir: str
) -> Tuple[dict, int, dict]:
    
    file_name = f"LGCN_preprocessed_data_{device}.pkl"
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        logger.info("Loading preprocessed data...")
        return load_preprocessed_data(file_path)
    
    # 데이터 전처리
    data = load_data(data_dir=data_dir)
    id2index: dict = indexing_data(data=data)
    train_data_proc = process_data(data=data, id2index=id2index, device=device)
    print_data_stat(data, "Train")
    processed_data = (train_data_proc, len(id2index), id2index)

    # 데이터 저장
    save_preprocessed_data(processed_data, file_path)
    return processed_data


def load_data(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "train_ratings.csv")
    data = pd.read_csv(path)
    # User id Max 138493
    # User 와 Item id가 겹치지 않도록 Item id + 200000
    data["item"] += 200000
    data.drop_duplicates(
        subset=["user", "item"], keep="last", inplace=True
    )
    return data


#def separate_data(data: pd.DataFrame) -> Tuple[pd.DataFrame]:
#    train_data = data[data.answerCode >= 0]
#    test_data = data[data.answerCode < 0]
#    return train_data, test_data


def indexing_data(data: pd.DataFrame) -> dict:
    userid, itemid = (
        sorted(list(set(data.user))),
        sorted(list(set(data.item))),
    )
    n_user, n_item = len(userid), len(itemid)
    userid2index = {str(v): i + 1 for i, v in enumerate(userid)}
    
    itemid2index = {str(v): i + n_user + 1 for i, v in enumerate(itemid)}
    
    id2index = dict(userid2index, **itemid2index)
    id2index["unknown"] = 0
    return id2index


def process_data(data: pd.DataFrame, id2index: dict, device: str, neg_sample_ratio: int = 1) -> dict:
    edge, label = [], []
    # 사용자별 아이템 목록 생성
    user_items = data.groupby('user')['item'].apply(set)

    # 모든 아이템 목록
    all_items = set(data['item'].unique())
    
    for user, pos_items_set in tqdm(user_items.items()):
        neg_items = list(all_items - pos_items_set)
        pos_items = list(pos_items_set)
        
        # 긍정적인 레이블 추가
        for item in pos_items:
            uid, iid = id2index[str(user)], id2index[str(item)]
            edge.append([uid, iid])
            label.append(1)

        # 부정적인 레이블 추가
        neg_samples = np.random.choice(neg_items, len(pos_items) * neg_sample_ratio, replace=False)
        for item in neg_samples:
            uid, iid = id2index[str(user)], id2index[str(item)]
            edge.append([uid, iid])
            label.append(0)

    edge = torch.LongTensor(edge).T
    label = torch.LongTensor(label)
    return dict(edge=edge.to(device), label=label.to(device))


def print_data_stat(data: pd.DataFrame, name: str) -> None:
    userid, itemid = list(set(data.user)), list(set(data.item))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")

def create_ground_truth(data_dir: str) -> dict:
    data = load_data(data_dir=data_dir)

    # 사용자별로 상호작용한 아이템 목록 생성
    user_items = data.groupby('user')['item'].apply(set)
    user_list = data["user"].unique()
    return user_items.to_dict(), user_list