import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
import wandb
from tqdm import tqdm

from .model import MF, LMF, FM, LFM, CFM, LCFM
from .utils import get_logger, logging_conf
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion
from .recall import recall_at_10


logger = get_logger(logger_conf=logging_conf)


def get_model(args) -> nn.Module:
    try:
        model_name = args.model.name.lower()
        model = {
            "mf": MF,
            "lmf": LMF,
            "fm": FM,
            "lfm": LFM,
            "cfm": CFM,
            "lcfm": LCFM,
        }.get(model_name)(args)
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
    seen: pd.Series,
    valid_df: pd.DataFrame,
):
    torch.backends.cudnn.benchmark = True

    optimizer = get_optimizer(model=model, args=args)
    scheduler = get_scheduler(optimizer=optimizer, args=args)

    best_recall = -1
    early_stopping_counter = 0

    for epoch in range(args.n_epochs):
        logger.info("Start Training: Epoch %s", epoch)

        # Train
        train_auc, train_acc, train_loss = train(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            args=args,
        )

        logger.info("Recommending ...")
        recommend_df = recommend(model, seen, args)
        logger.info("Calculating Recall@10 ...")
        recall = recall_at_10(recommend_df, valid_df)

        logger.info("Training epoch: %s / Recall@10: %.4f", epoch, recall)

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
            model_filename = f"{args.model.name}_{args.model_file_name}"
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
    losses = 0
    loss_cnt = 0

    for step, batch in enumerate(train_loader):
        if args.model.name.lower() in [
            "mf",
            "lmf",
            "fm",
            "lfm",
            "cfm",
            "lcfm",
        ]:  # To do change this -> loader-model interaction
            batch = batch.to(args.device)
            input = batch[:, :-1]
            preds = model(input)
            targets = batch[:, -1]
            del batch
        else:
            raise NotImplementedError

        loss = get_criterion(preds, targets.float(), args=args)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % args.log_steps == 0:
            logger.info(
                "Training steps: %s / %s, Loss: %.4f",
                step,
                len(train_loader),
                loss.item(),
            )

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())
        del preds
        del targets
        losses += loss.item()
        loss_cnt += 1

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc = roc_auc_score(total_targets, total_preds)
    acc = accuracy_score(total_targets, np.where(total_preds >= 0.5, 1, 0))
    loss_avg = losses / loss_cnt
    logger.info("TRAIN AUC : %.4f ACC : %.4f", auc, acc)
    return auc, acc, loss_avg


@torch.no_grad()
def recommend(model: nn.Module, seen: pd.Series, args) -> pd.DataFrame:
    """recommend top 10 item for each user"""
    if args.model.name.lower() in ["mf", "lmf"]:
        rec = torch.tensor([]).to(args.device)
        item_tensor = torch.arange(args.n_items).reshape(-1, 1)
        for user in tqdm(range(args.n_users)):
            user_tensor = torch.tensor(user).repeat(args.n_items).reshape(-1, 1)
            input = torch.concat((user_tensor, item_tensor), dim=1).to(args.device)
            pred = model(input)
            pred[seen[user]] = -999  # masking

            _, item = torch.topk(pred, 10)
            rec = torch.concat((rec, item))
        user_arr = np.arange(args.n_users).repeat(10)

        df = pd.DataFrame(zip(user_arr, rec.tolist()), columns=["user", "item"])
        return df
    elif args.model.name.lower() in ["fm", "lfm", "cfm", "lcfm"]:
        rec = torch.zeros(10 * args.n_users).to(args.device)
        user_repeat = np.arange(args.n_users).repeat(args.n_items)
        user_tensor = torch.tensor(user_repeat).reshape(-1, 1)
        item_tensor = torch.arange(args.n_items).reshape(-1, 1).repeat(args.n_users, 1)

        full_tensor = torch.concat((user_tensor, item_tensor), dim=1)
        if args.item2feat:
            item_feat_tensor = torch.tensor(args.item2feat).T.repeat(args.n_users, 1)
            full_tensor = torch.concat((full_tensor, item_feat_tensor), dim=1)
        if args.user2feat:
            user_feat_tensor = (
                torch.tensor(args.user2feat)
                .repeat(args.n_items, 1)
                .T.reshape(-1, len(args.user2feat))
            )
            full_tensor = torch.concat((full_tensor, user_feat_tensor), dim=1)

        # n_users가 div의 배수라는 전제가 필요함.
        # div가 작을수록 추천 속도 빨라지지만 OOM의 위험이 커짐.
        div = args.div
        user_len = seen.apply(len)
        for batch in tqdm(range(div)):
            offset_size = args.n_items * (args.n_users // div)
            offset = batch * offset_size
            input = full_tensor[offset : offset + offset_size, :].to(args.device)
            pred = model(input).reshape(-1, args.n_items)
            del input

            user_left, user_right = batch * len(pred), (batch + 1) * len(pred)

            seen_item = np.concatenate(seen[user_left:user_right].values)
            seen_user = np.arange(user_left, user_right).repeat(
                user_len[user_left:user_right].values
            )

            pred[seen_user - user_left, seen_item] = -999
            _, item = torch.topk(pred, 10)
            rec[user_left * 10 : user_right * 10] = item.reshape(-1)
        user_arr = np.arange(args.n_users).repeat(10)

        df = pd.DataFrame(zip(user_arr, rec.tolist()), columns=["user", "item"])
        return df
    else:
        raise NotImplementedError(f"Not implemented for model {args.model.name}")


def save_checkpoint(state: dict, model_dir: str, model_filename: str) -> None:
    """Saves checkpoint to a given directory."""
    save_path = os.path.join(model_dir, model_filename)
    logger.info("saving model as %s...", save_path)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(state, save_path)


def load_model(args):
    model_file_name = f"{args.model.name}_{args.model_file_name}"
    model_path = os.path.join(args.model_dir, model_file_name)
    logger.info("Loading Model from: %s", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=False)
    logger.info("Successfully loaded model state from: %s", model_path)
    return model
