import argparse
import torch
import numpy as np
import pandas as pd
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        default="/data/ephemeral/level2-movierecommendation-recsys-07/VASP/models/vae/2024-02-18_09:02:25beaming-dumpling-48/best_model.pt",
        help="name of models",
    )
    parser.add_argument("--vae", type=bool, default=True)
    # python run_inference.py --model_path=/data/ephemeral/level2-movierecommendation-recsys-07/VASP/models/vae/2024-02-18_09:02:25beaming-dumpling-48/best_model.pt 로 실행
    args = parser.parse_args()

    train_df = pd.read_csv("../data/train/train_ratings.csv")

    # inference랑 train 단계의 mapping 맞추기
    user2idx = {v: i for i, v in enumerate(train_df["user"].unique())}
    item2idx = {v: i for i, v in enumerate(train_df["item"].unique())}

    train_df["user"] = train_df["user"].map(user2idx)
    train_df["item"] = train_df["item"].map(item2idx)

    train_df["label"] = [1] * len(train_df)
    pivot = train_df.pivot(index="user", columns="item", values="label").fillna(0)

    input_data = torch.tensor(pivot.values).to(dtype=torch.float)

    # Load the best saved model.
    with open(args.model_path, "rb") as f:
        model = torch.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_data = input_data.to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_data)

    if args.vae:
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

    # user2idx와 item2idx의 역매핑 생성
    idx2user = {i: v for v, i in user2idx.items()}
    idx2item = {i: v for v, i in item2idx.items()}

    # rec_df의 'user'와 'item' 열을 역매핑
    rec_df["user"] = rec_df["user"].map(idx2user)
    rec_df["item"] = rec_df["item"].map(idx2item)

    # 폴더 이름 설정
    folder_name = "output"

    # 폴더가 있는지 확인하고, 없으면 만들기
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 데이터 저장
    dataframe = pd.DataFrame(rec_df, columns=["user", "item"])
    dataframe.sort_values(by="user", inplace=True)
    dataframe.to_csv(folder_name + "/submission.csv", index=False)

    print("inference done!")
