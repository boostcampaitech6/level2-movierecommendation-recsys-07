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

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # breakpoint()
    os.makedirs(name=model_dir, exist_ok=True)
    if args.data_prep == "user":
        if valid_data is None:
            eids = np.arange(len(train_data["pos_edge"][0]))
            valid_len = int(len(train_data["pos_edge"][0]) * 0.1)

            eids_list = np.random.RandomState(seed=args.seed).permutation(eids)

            eids = eids_list[:valid_len]
            not_eids = eids_list[valid_len:]

            pos_edge, pos_label = train_data["pos_edge"], torch.LongTensor(
                [1] * len(train_data["pos_edge"][0])
            )
            neg_edge, neg_label = train_data["neg_edge"], torch.LongTensor(
                [0] * len(train_data["pos_edge"][0])
            )

            valid_pos_label = pos_label.to("cpu").detach().numpy()
            valid_neg_label = neg_label.to("cpu").detach().numpy()

            valid_data = dict(
                pos_edge=pos_edge[:, eids],
                neg_edge=neg_edge[:, eids],
                pos_label=valid_pos_label[eids],
                neg_label=valid_neg_label[eids],
            )
            train_data = dict(
                pos_edge=pos_edge[:, not_eids],
                neg_edge=neg_edge[:, not_eids],
                pos_label=pos_label[not_eids],
                neg_label=neg_label[not_eids],
            )

    else:
        if valid_data is None:
            eids = np.arange(len(train_data["label"]))
            valid_len = int(len(train_data["label"]) * 0.1)

            eids_list = np.random.RandomState(seed=args.seed).permutation(eids)

            eids = eids_list[:valid_len]
            not_eids = eids_list[valid_len:]

            edge, label = train_data["edge"], train_data["label"]
            valid_label = label.to("cpu").detach().numpy()
            valid_data = dict(edge=edge[:, eids], label=valid_label[eids])
            train_data = dict(edge=edge[:, not_eids], label=label[not_eids])

    logger.info(f"Training Started : n_epochs={n_epochs}")

    best_recall, best_epoch = 0, -1
    best_mrr, best_mrr_epoch = 0, -1

    index2id = {v: k for k, v in id2index.items()}
    # breakpoint()
    # user_tensor = train_data["edge"][:, train_data["label"] == 1][0, :]
    # item_tensor = train_data["edge"][:, train_data["label"] == 1][1, :]

    user_tensor = train_data["pos_edge"][0, :]
    item_tensor = train_data["pos_edge"][1, :]

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
        auc, acc, recall, mrr = validate(
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
                valid_mrr=mrr,
                valid_mrr_best=max(mrr, best_mrr),
            )
        )
        if recall > best_recall:
            logger.info(
                "Best model updated Recall@K from %.4f to %.4f", best_recall, recall
            )
            best_recall, best_epoch = recall, e
            torch.save(
                obj={"model": model.state_dict(), "epoch": e + 1},
                f=os.path.join(model_dir, f"best_recall_model.pt"),
            )
        if mrr > best_mrr:
            logger.info("Best model updated MRR from %.4f to %.4f", best_mrr, mrr)
            best_mrr, best_mrr_epoch = mrr, e
            torch.save(
                obj={"model": model.state_dict(), "epoch": e + 1},
                f=os.path.join(model_dir, f"best_mrr_model.pt"),
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
    # breakpoint()
    train_edge = torch.cat([train_data["pos_edge"], train_data["neg_edge"]], dim=1)
    train_label = torch.cat([train_data["pos_label"], train_data["neg_label"]], dim=0)
    pred_pos = model(train_data["pos_edge"])
    pred_neg = model(train_data["neg_edge"])
    # pos_rank = torch.sigmoid(pred_pos)
    # neg_rank = torch.sigmoid(pred_neg)
    # breakpoint()
    # loss = model.link_pred_loss(pred=pred, edge_label=train_data["label"])
    loss = model.recommendation_loss(pred_pos, pred_neg, lambda_reg=args.lambda_reg)

    prob = model.predict_link(edge_index=train_edge, prob=True)
    prob = prob.detach().cpu().numpy()

    label = train_label.cpu().numpy()
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
        valid_edge = torch.cat([valid_data["pos_edge"], valid_data["neg_edge"]], dim=1)
        valid_label = np.concatenate((valid_data["pos_label"], valid_data["neg_label"]))
        train_edge = torch.cat([train_data["pos_edge"], train_data["neg_edge"]], dim=1)

        prob = model.predict_link(edge_index=valid_edge, prob=True)
        prob = prob.detach().cpu().numpy()

        # user_list = sorted(train_item_dict.keys())
        user_indices = torch.LongTensor(list(range(1, 31361)))
        all_items = torch.LongTensor(list(range(31361, 38168)))
        topk_list = model.recommend(
            edge_index=train_edge,
            src_index=user_indices,
            dst_index=all_items,
            k=600,
        )

        topk = topk_list.detach().cpu().numpy()

        # breakpoint()

        # 미리 할당된 배열 생성
        final_topk = np.zeros((len(user_indices), 10), dtype=int)
        for i, user_index in tqdm(enumerate(user_indices)):
            # user_id = index2id.get(user_index.item())
            seen_items = train_item_dict[user_index.item()]
            seen_items.add(0)

            recommendations = [item for item in topk[i] if item not in seen_items]
            final_topk[i, : len(recommendations[:10])] = recommendations[:10]

        avg_recall, avg_nor_recall, avg_mrr = calculate_normalized_recall_at_k(
            topk=final_topk, k=10, args=args, index2id=index2id
        )
        # breakpoint()
        label = valid_label

        acc = accuracy_score(y_true=label, y_pred=prob > 0.5)
        auc = roc_auc_score(y_true=label, y_score=prob)
    logger.info(
        "VALID AUC : %.4f ACC : %.4f Recall@K : %.4f normalized Recall@K : %.4f MRR : %.4f",
        auc,
        acc,
        avg_recall,
        avg_nor_recall,
        avg_mrr,
    )
    return auc, acc, avg_nor_recall, avg_mrr


def inference(model: nn.Module, data: dict, output_dir: str, index2id):
    model.eval()
    with torch.no_grad():
        user_tensor = data["edge"][:, data["label"] == 1][0, :]
        item_tensor = data["edge"][:, data["label"] == 1][1, :]

        train_item_dict = create_user_item_interaction_dict(
            user_tensor, item_tensor, index2id=index2id
        )

        user_indices = torch.LongTensor(list(range(1, 31361)))
        all_items = torch.LongTensor(list(range(31361, 38168)))
        topk_list = model.recommend(
            edge_index=data["edge"],
            src_index=user_indices,
            dst_index=all_items,
            k=1000,
        )

        topk = topk_list.detach().cpu().numpy()

        final_topk = np.zeros((len(user_indices), 10), dtype=int)

        for i, user_index in tqdm(enumerate(user_indices)):
            # user_id = index2id.get(user_index.item())
            seen_items = train_item_dict[user_index.item()]
            # unkwon item
            seen_items.add(0)

            recommendations = [item for item in topk[i] if item not in seen_items]
            final_topk[i, : len(recommendations[:10])] = recommendations[:10]

    logger.info("Saving Result ...")
    # pred = pred.detach().cpu().numpy()
    os.makedirs(name=output_dir, exist_ok=True)
    write_path = os.path.join(output_dir, "submission.csv")

    # id2index 딕셔너리의 역매핑 생성
    # index2id = {v: k for k, v in id2index.items()}

    sb_df = pd.read_csv(
        "/data/ephemeral/level2-movierecommendation-recsys-07/data/eval/sample_submission.csv"
    )
    # 1차원 array 변환
    flat_pred = np.array(final_topk).flatten()
    # breakpoint()
    id_list = [int(index2id.get(index)) - 200000 for index in flat_pred]
    sb_df["item"] = id_list
    sb_df.to_csv(path_or_buf=write_path, index=False)
    logger.info("Successfully saved submission as %s", write_path)


def calculate_normalized_recall_at_k(topk, k=10, args="", index2id=""):
    total_recall = 0
    total_normalized_recall = 0
    total_mrr = 0

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
        # MRR 계산
        first_relevant_rank = next(
            (
                i + 1
                for i, item in enumerate(recommended_items_id)
                if item in actual_items
            ),
            None,
        )
        mrr = 1 / first_relevant_rank if first_relevant_rank else 0
        total_mrr += mrr

        total_recall += recall_at_k
        total_normalized_recall += normalized_recall_at_k
        avg_mrr = total_mrr / num_users  # 평균 MRR 계산
    # 전체 사용자에 대한 평균 Recall@K와 Normalized Recall@K
    avg_recall_at_k = total_recall / num_users
    avg_normalized_recall_at_k = total_normalized_recall / num_users

    return avg_recall_at_k, avg_normalized_recall_at_k, avg_mrr


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
