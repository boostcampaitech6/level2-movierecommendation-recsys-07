import os
import sys

sys.path.append("../")
from recall import recall_at_10

import pandas as pd
import numpy as np
import heapq
import torch
import torch.nn as nn

import wandb
from tqdm import tqdm

from model import MF

from utils import get_logger, logging_conf
from optimizer import get_optimizer
from scheduler import get_scheduler
from criterion import get_criterion

from sklearn.metrics import accuracy_score, roc_auc_score

logger = get_logger(logger_conf=logging_conf)


def get_model(args) -> nn.Module:
    try:
        model_name = args.model.lower()
        model = {
            "mf": MF,
        }.get(
            model_name
        )(args)
    except KeyError:
        logger.warn("No model name %s found", model_name)
    except Exception as e:
        logger.warn("Error while loading %s with args: %s", model_name, args)
        raise e
    return model


def run(
    args,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    idx_dict: dict,
    seen: pd.Series,
    valid_df: pd.DataFrame,
):

    optimizer = get_optimizer(model=model, args=args)
    scheduler = get_scheduler(optimizer=optimizer, args=args)

    best_recall = -1
    early_stopping_counter = 0

    for epoch in range(args.n_epochs):
        logger.info("Start Training: Epoch %s", epoch + 1)

        # Train
        train_auc, train_acc, train_loss = train(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            args=args,
        )

        # VALID label이 모두 1이라 auc 측정 x. 논의 후 삭제
        """valid_auc, valid_acc, valid_loss = validate(
            valid_loader=valid_loader, model=model, args=args
        )"""

        recommend_df = recommend(model, seen, args)
        recall = recall_at_10(recommend_df, valid_df)

        logger.info("Training epoch: %s Recall@10: %.4f", epoch + 1, recall)

        wandb.log(
            dict(
                epoch=epoch,
                train_loss_epoch=train_loss,
                train_auc_epoch=train_auc,
                train_acc_epoch=train_acc,
                recall_at_10=recall,
                best_recall=max(best_recall, recall),
            )
        )

        if recall > best_recall:
            best_recall = recall
            # nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, "module") else model
            model_filename = f"{args.model}_{args.model_name}"
            save_checkpoint(
                state={"epoch": epoch + 1, "state_dict": model_to_save.state_dict()},
                model_dir=args.model_dir,
                model_filename=model_filename,
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                logger.info(
                    "EarlyStopping counter: %s out of %s",
                    early_stopping_counter,
                    args.patience,
                )
                break

        # scheduler
        if args.scheduler == "plateau":
            scheduler.step(best_recall)


def train(
    train_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args,
):
    model.train()

    total_preds = []
    total_targets = []
    losses = []

    for step, batch in tqdm(
        enumerate(train_loader), total=len(train_loader), desc="Train"
    ):
        if args.model.lower() in ["mf", "lmf"]:
            batch = batch.to(args.device)
            input = batch[:, :-1]
            preds = model(input)
            targets = batch[:, -1]
        else:
            raise NotImplementedError

        loss = get_criterion(preds, targets.float(), args=args)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())
        losses.append(loss)

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc = roc_auc_score(total_targets, total_preds)
    acc = accuracy_score(total_targets, np.where(total_preds >= 0.5, 1, 0))
    loss_avg = sum(losses) / len(losses)
    logger.info("TRAIN AUC : %.4f ACC : %.4f", auc, acc)
    return auc, acc, loss_avg


def validate(valid_loader: nn.Module, model: nn.Module, args):
    model.eval()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(valid_loader):
        if args.model.lower() in ["mf", "lmf"]:
            batch = batch.to(args.device)
            input = batch[:, :-1]
            preds = model(input)
            targets = batch[:, -1]
        else:
            raise NotImplementedError

        loss = get_criterion(preds, targets.float(), args=args)

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())
        losses.append(loss)

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc = roc_auc_score(total_targets, total_preds)
    acc = accuracy_score(total_targets, np.where(total_preds >= 0.5, 1, 0))
    loss_avg = sum(losses) / len(losses)
    logger.info("VALID AUC : %.4f ACC : %.4f loss: %.4f", auc, acc, loss_avg)
    return auc, acc, loss_avg


def recommend(model: nn.Module, seen: pd.Series, args) -> pd.DataFrame:
    """recommend top 10 item for each user"""
    if args.model.lower() in ["mf", "lmf"]:
        rec = torch.tensor([]).to(args.device)
        for user in tqdm(range(args.n_users), desc="recommendation"):
            mask = torch.zeros(args.n_items).to(args.device)
            for idx in seen[user]:
                mask[idx] -= 999
            user_tensor = torch.tensor(user).repeat(args.n_items).reshape(-1, 1)
            item_tensor = torch.arange(args.n_items).reshape(-1, 1)
            input = torch.concat((user_tensor, item_tensor), dim=1).to(args.device)
            pred = model(input) + mask

            _, item = torch.topk(pred, 10)
            rec = torch.concat((rec, item))
            """heap = [(-v, i) for i, v in enumerate(pred)]
            heapq.heapify(heap)
            cnt = 0
            while cnt < 10:
                idx = heapq.heappop(heap)[1]
                if not idx in user_seen:
                    rec.append(idx)
                    cnt += 1"""
        user_arr = np.arange(args.n_users).repeat(10)

        df = pd.DataFrame(zip(user_arr, rec.tolist()), columns=["user", "item"])
        return df

    else:
        raise NotImplementedError(f"Not implemented for model {args.model}")


def save_checkpoint(state: dict, model_dir: str, model_filename: str) -> None:
    """Saves checkpoint to a given directory."""
    save_path = os.path.join(model_dir, model_filename)
    logger.info("saving model as %s...", save_path)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(state, save_path)


def load_model(args):
    model_name = f"{args.model}_{args.model_name}"
    model_path = os.path.join(args.model_dir, model_name)
    logger.info("Loading Model from: %s", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True)
    logger.info("Successfully loaded model state from: %s", model_path)
    return model
