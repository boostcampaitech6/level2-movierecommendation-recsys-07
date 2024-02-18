import torch
import numpy as np
import pandas as pd
import os
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(args):
    OmegaConf.set_struct(args, False)
    os.makedirs(args.model_dir, exist_ok=True)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = pd.read_csv(os.path.join(args.data_dir, args.train_file_name))

    file_path = os.path.join(args.model_dir, "item2idx.pickle")
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        print("File exists and is not empty")
    else:
        print("File does not exist or is empty")

    with open(os.path.join(args.model_dir, "item2idx.pickle"), "rb") as handle:
        item2idx = pickle.load(handle)
    with open(os.path.join(args.model_dir, "user2idx.pickle"), "rb") as handle:
        user2idx = pickle.load(handle)
    idx2user = {v: i for i, v in user2idx.items()}
    idx2item = {v: i for i, v in item2idx.items()}

    train_df["user"] = train_df["user"].map(user2idx)
    train_df["item"] = train_df["item"].map(item2idx)

    train_df["label"] = [1] * len(train_df)
    pivot = train_df.pivot(index="user", columns="item", values="label").fillna(0)

    input_data = torch.tensor(pivot.values).to(dtype=torch.float)

    # Load the best saved model.
    with open(os.path.join(args.model_dir, args.model_file_name), "rb") as f:
        model = torch.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_data = input_data.to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_data)

    if args.model.name.lower() in ["vasp", "multivae"]:
        output = output[0]

    print("Mask")
    seen = train_df.groupby("user")["item"].unique()
    user_len = seen.apply(len)
    seen_item = np.concatenate(seen.values)
    seen_user = np.arange(len(user2idx)).repeat(user_len.values)

    output[seen_user, seen_item] = -99999

    print("Recommend")
    _, item = torch.topk(output, 10)
    user_arr = np.arange(len(user2idx)).repeat(10)
    rec_df = pd.DataFrame(
        zip(user_arr, item.reshape(-1).tolist()), columns=["user", "item"]
    )

    # rec_df의 'user'와 'item' 열을 역매핑
    rec_df["user"] = rec_df["user"].map(idx2user)
    rec_df["item"] = rec_df["item"].map(idx2item)

    # 데이터 저장
    dataframe = pd.DataFrame(rec_df, columns=["user", "item"])
    dataframe.sort_values(by="user", inplace=True)
    dataframe.to_csv(os.path.join(args.model_dir, "submission.csv"), index=False)

    print("inference done!")


if __name__ == "__main__":
    main()
