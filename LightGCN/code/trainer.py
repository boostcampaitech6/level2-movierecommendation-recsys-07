import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
import torch
from torch import nn
from torch_geometric.nn.models import LightGCN
import wandb
from collections import defaultdict
from .datasets import create_ground_truth
from tqdm import tqdm
from .utils import get_logger, logging_conf


logger = get_logger(logger_conf=logging_conf)


def build(n_node: int, weight: str = None, **kwargs):
    model = LightGCN(num_nodes=n_node, **kwargs)
    if weight:
        if not os.path.isfile(path=weight):
            logger.fatal("Model Weight File Not Exist")
        logger.info("Load model")
        state = torch.load(f=weight)["model"]
        model.load_state_dict(state)
        return model
    else:
        logger.info("No load model")
        return model


def run(
    model: nn.Module,
    train_data: dict,
    valid_data: dict = None,
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    model_dir: str = None,
    id2index: dict = "",
    args="",
):
    model.train()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    os.makedirs(name=model_dir, exist_ok=True)

    if valid_data is None:
        eids = np.arange(len(train_data["label"]))
        valid_len = int(len(train_data["label"]) * 0.3)

        eids_list = np.random.RandomState(seed=args.seed).permutation(eids)

        eids = eids_list[:valid_len]
        not_eids = eids_list[valid_len:]

        edge, label = train_data["edge"], train_data["label"]
        valid_label = label.to("cpu").detach().numpy()
        valid_data = dict(edge=edge[:, eids], label=valid_label[eids])
        train_data = dict(edge=edge[:, not_eids], label=label[not_eids])

    logger.info(f"Training Started : n_epochs={n_epochs}")
    best_recall, best_epoch = 0, -1
    index2id = {v: k for k, v in id2index.items()}

    user_tensor = train_data["edge"][:, train_data["label"] == 1][0, :]
    item_tensor = train_data["edge"][:, train_data["label"] == 1][1, :]
    train_item_dict = create_user_item_interaction_dict(
        user_tensor, item_tensor, index2id=index2id
    )

    for e in range(n_epochs):
        logger.info("Epoch: %s", e)
        # TRAIN
        train_auc, train_acc, train_loss = train(
            train_data=train_data,
            model=model,
            optimizer=optimizer,
            index2id=index2id,
            args=args,
        )

        # VALID
        auc, acc, recall = validate(
            valid_data=valid_data,
            model=model,
            train_data=train_data,
            args=args,
            index2id=index2id,
            train_item_dict=train_item_dict,
        )
        wandb.log(
            dict(
                train_loss_epoch=train_loss,
                train_auc_epoch=train_auc,
                train_acc_epoch=train_acc,
                valid_auc_epoch=auc,
                valid_acc_epoch=acc,
                valid_reacll=recall,
                valid_recall_best=max(recall, best_recall),
            )
        )
        if recall > best_recall:
            logger.info(
                "Best model updated Recall@K from %.4f to %.4f", best_recall, recall
            )
            best_recall, best_epoch = recall, e
            torch.save(
                obj={"model": model.state_dict(), "epoch": e + 1},
                f=os.path.join(model_dir, f"best_model.pt"),
            )
        torch.save(
            obj={"model": model.state_dict(), "epoch": e + 1},
            f=os.path.join(model_dir, f"last_model.pt"),
        )

    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")


def train(
    model: nn.Module,
    train_data: dict,
    optimizer: torch.optim.Optimizer,
    index2id: dict,
    args,
):
    
    pred = model(train_data["edge"])

    loss = model.link_pred_loss(pred=pred, edge_label=train_data["label"])

    prob = model.predict_link(edge_index=train_data["edge"], prob=True)
    prob = prob.detach().cpu().numpy()

    label = train_data["label"].cpu().numpy()
    acc = accuracy_score(y_true=label, y_pred=prob > 0.5)
    auc = roc_auc_score(y_true=label, y_score=prob)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    logger.info(
        "TRAIN AUC : %.4f ACC : %.4f LOSS : %.4f",
        auc,
        acc,
        loss.item(),
    )
    return auc, acc, loss


def validate(
    valid_data: dict,
    model: nn.Module,
    train_data: dict,
    index2id="",
    args="",
    train_item_dict="",
):
    with torch.no_grad():
        prob = model.predict_link(edge_index=valid_data["edge"], prob=True)
        prob = prob.detach().cpu().numpy()

        user_list = sorted(train_item_dict.keys())
        topk = np.zeros(shape=313600)
        n = 0
        breakpoint()
        for i in user_list:
            no_seen = list(set(list(range(31361, 38168))) - train_item_dict[i])
            topk[n * 10 : n * 10 + 10] = model.recommend(
                edge_index=train_data["edge"],
                src_index=torch.LongTensor(i),
                dst_index=torch.LongTensor(no_seen),
                k=10,
            )
            n += 1

        topk = topk.detach().cpu().numpy()
        avg_recall, avg_nor_recall = calculate_normalized_recall_at_k(
            topk=topk, k=10, args=args, index2id=index2id
        )
        label = valid_data["label"]

        acc = accuracy_score(y_true=label, y_pred=prob > 0.5)
        auc = roc_auc_score(y_true=label, y_score=prob)
    logger.info(
        "VALID AUC : %.4f ACC : %.4f normalized Recall@K : %.4f",
        auc,
        acc,
        avg_nor_recall,
    )
    return auc, acc, avg_nor_recall


def inference(model: nn.Module, data: dict, output_dir: str, id2index):
    model.eval()
    with torch.no_grad():
        pred = model.recommend(
            edge_index=data["edge"],
            src_index=torch.LongTensor(list(range(1, 31361))),
            dst_index=torch.LongTensor(list(range(31361, 38168))),
            k=10,
        )

    logger.info("Saving Result ...")
    pred = pred.detach().cpu().numpy()
    os.makedirs(name=output_dir, exist_ok=True)
    write_path = os.path.join(output_dir, "submission.csv")

    # id2index 딕셔너리의 역매핑 생성
    index2id = {v: k for k, v in id2index.items()}

    sb_df = pd.read_csv(
        "/data/ephemeral/level2-movierecommendation-recsys-07/input/data/eval/sample_submission.csv"
    )
    # 1차원 array 변환
    flat_pred = np.array(pred).flatten()
    id_list = [int(index2id.get(index)) - 200000 for index in flat_pred]
    sb_df["item"] = id_list
    sb_df.to_csv(path_or_buf=write_path, index=False)
    logger.info("Successfully saved submission as %s", write_path)


def calculate_normalized_recall_at_k(topk, k=10, args="", index2id=""):
    total_recall = 0
    total_normalized_recall = 0
    num_users = len(topk)
    ground_truth, user_list = create_ground_truth(args.data_dir)

    for user_idx, recommended_items in enumerate(topk):

        # 실제 아이템 id 반환
        recommended_items_id = [int(index2id.get(item)) for item in recommended_items]

        # 사용자가 실제 상호작용한 아이템 목록
        actual_items = ground_truth[user_list[user_idx]]

        num_actual_items = len(actual_items)

        # Recall@K 계산
        hit = len(set(recommended_items_id).intersection(actual_items))
        recall_at_k = hit / num_actual_items if num_actual_items > 0 else 0

        # 최대 가능 Recall 계산
        max_possible_recall = (
            min(num_actual_items, k) / num_actual_items if num_actual_items > 0 else 0
        )

        # Normalized Recall@K
        normalized_recall_at_k = (
            recall_at_k / max_possible_recall if max_possible_recall > 0 else 0
        )

        total_recall += recall_at_k
        total_normalized_recall += normalized_recall_at_k

    # 전체 사용자에 대한 평균 Recall@K와 Normalized Recall@K
    avg_recall_at_k = total_recall / num_users
    avg_normalized_recall_at_k = total_normalized_recall / num_users

    return avg_recall_at_k, avg_normalized_recall_at_k


def create_user_item_interaction_dict(user_tensor, item_tensor, index2id=""):
    # Tensor를 CPU로 옮기고 NumPy 배열로 변환
    user_inds = user_tensor.clone()
    item_inds = item_tensor.clone()
    user_inds = user_inds.cpu().numpy()
    item_inds = item_inds.cpu().numpy()
    user_item_dict = {}
    for user_ind, item_ind in zip(user_inds, item_inds):
        # 사용자 INDEX를 키로 하여 아이템 INDEX를 추가
        if user_ind in user_item_dict:
            user_item_dict[user_ind].add(item_ind)
        else:
            user_item_dict[user_ind] = {item_ind}

    return user_item_dict
