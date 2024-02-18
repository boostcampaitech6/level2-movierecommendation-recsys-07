import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import bottleneck as bn
import wandb
import time
from tqdm import tqdm

from .model import NEASE, MultiDAE, MultiVAE
from .utils import get_logger, logging_conf
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion


logger = get_logger(logger_conf=logging_conf)


def sparse2torch_sparse(data):
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i: row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(
        indices, torch.from_numpy(values).float(), [samples, features]
    )
    return t


def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    """
    Normalized Discounted Cumulative Gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    """
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1.0 / np.log2(np.arange(2, k + 2))

    DCG = (
        heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].toarray() * tp
    ).sum(axis=1)
    IDCG = np.array([(tp[: min(n, k)]).sum() for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


def get_model(args) -> nn.Module:
    try:
        model_name = args.model.name.lower()
        model = {"nease": NEASE, "dae": MultiDAE, "vae": MultiVAE}.get(model_name)(args)
    except KeyError:
        logger.warn("No model name %s found", model_name)
    except Exception as e:
        logger.warn("Error while loading %s with args: %s", model_name, args)
        raise e
    return model


def run(
    args,
    model: nn.Module,
    data,
):
    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)

    best_n100 = -np.inf
    args.update_count = 0

    train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te = (
        data[0],
        data[1],
        data[2],
        data[3],
        data[4],
    )

    for epoch in range(1, args.n_epochs + 1):
        args.epoch = epoch
        epoch_start_time = time.time()
        train(model, criterion, optimizer, train_data, args)
        val_loss, n100, r20, r50 = evaluate(
            model, criterion, vad_data_tr, vad_data_te, args
        )
        print("-" * 89)
        print(
            "| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | "
            "n100 {:5.3f} | r20 {:5.3f} | r50 {:5.3f}".format(
                epoch, time.time() - epoch_start_time, val_loss, n100, r20, r50
            )
        )
        print("-" * 89)
        n_iter = epoch * len(range(0, args.N, args.batch_size))
        # Save the model if the n100 is the best we've seen so far.
        if n100 > best_n100:
            with open(os.path.join(args.model_dir, args.model_file_name), "wb") as f:
                torch.save(model, f)
            best_n100 = n100
    # Load the best saved model.
    with open(os.path.join(args.model_dir, args.model_file_name), "rb") as f:
        model = torch.load(f)
    # Run on test data.
    test_loss, n100, r20, r50 = evaluate(
        model, criterion, test_data_tr, test_data_te, args
    )
    print("=" * 89)
    print(
        "| End of training | test loss {:4.2f} | n100 {:4.2f} | r20 {:4.2f} | "
        "r50 {:4.2f}".format(test_loss, n100, r20, r50)
    )
    print("=" * 89)


def train(model, criterion, optimizer, train_data, args):
    # Turn on training mode
    model.train()
    train_loss = 0.0
    start_time = time.time()

    np.random.shuffle(args.idxlist)

    for batch_idx, start_idx in enumerate(range(0, args.N, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, args.N)
        data = train_data[args.idxlist[start_idx:end_idx]]
        data = naive_sparse2tensor(data).to(args.device)
        optimizer.zero_grad()

        if args.model.name.lower()[-3:] == "vae" or args.model.name.lower() == "vasp":
            if args.model.total_anneal_steps > 0:
                anneal = min(
                    args.model.anneal_cap,
                    1.0 * args.update_count / args.model.total_anneal_steps,
                )
            else:
                anneal = args.model.anneal_cap

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)

            loss = criterion(recon_batch, data, mu, logvar, anneal, args)
        else:
            recon_batch = model(data)
            loss = criterion(recon_batch, data, args)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # scheduler.step(loss)
        if args.model.name == "multieVAE":
            model.update += 1  # todo

        if batch_idx % args.log_steps == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | "
                "loss {:4.2f}".format(
                    args.epoch,
                    batch_idx,
                    len(range(0, args.N, args.batch_size)),
                    elapsed * 1000 / args.log_steps,
                    train_loss / args.log_steps,
                )
            )

            start_time = time.time()
            train_loss = 0.0


def evaluate(model, criterion, data_tr, data_te, args):
    # Turn on evaluation mode
    model.eval()
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]
    total_val_loss_list = []
    n100_list = []
    r20_list = []
    r50_list = []

    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, args.N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]
            data_tensor = naive_sparse2tensor(data).to(
                args.device
            )  # todo. why? data_tensor should be distingshed from data

            if (
                args.model.name.lower()[-3:] == "vae"
                or args.model.name.lower() == "vasp"
            ):
                if args.model.total_anneal_steps > 0:
                    anneal = min(
                        args.model.anneal_cap,
                        1.0 * args.update_count / args.model.total_anneal_steps,
                    )
                else:
                    anneal = args.model.anneal_cap
                recon_batch, mu, logvar = model(data_tensor)
                loss = criterion(recon_batch, data_tensor, mu, logvar, anneal, args)
            else:
                recon_batch = model(data_tensor)
                loss = criterion(recon_batch, data_tensor, args)

            total_val_loss_list.append(loss.item())

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf

            n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
            r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)

            n100_list.append(n100)
            r20_list.append(r20)
            r50_list.append(r50)

    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    return (
        np.nanmean(total_val_loss_list),
        np.nanmean(n100_list),
        np.nanmean(r20_list),
        np.nanmean(r50_list),
    )


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
