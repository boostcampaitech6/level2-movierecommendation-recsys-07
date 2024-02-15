import pandas as pd
import os
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder


def making_embedding_df(df, colname, embedding_dim):
    le = LabelEncoder()
    encoding = le.fit_transform(df[colname])
    encoded_tensor = torch.LongTensor(encoding)
    embedding = nn.Embedding(num_embeddings=len(encoding), embedding_dim=embedding_dim)
    embedding_tensor = embedding(encoded_tensor)
    temp_embeddings = embedding_tensor.detach().cpu().numpy()
    embeddings_df = pd.DataFrame(
        temp_embeddings, columns=[colname + "_" + str(i) for i in range(10)]
    )
    custom_data = pd.concat([df["item"], embeddings_df], axis=1)
    return custom_data.groupby("item").mean()


print("loading")
data_path = "../../data/train"
train_ratings = pd.read_csv(os.path.join(data_path, "train_ratings.csv"))
year_data = pd.read_csv(os.path.join(data_path, "years.tsv"), sep="\t")
writer_data = pd.read_csv(os.path.join(data_path, "writers.tsv"), sep="\t")
title_data = pd.read_csv(os.path.join(data_path, "titles.tsv"), sep="\t")
genre_data = pd.read_csv(os.path.join(data_path, "genres.tsv"), sep="\t")
director_data = pd.read_csv(os.path.join(data_path, "directors.tsv"), sep="\t")

print("adding features")
year_data["decade"] = year_data["year"] // 10 * 10

print("loading data")
valid_df = pd.read_csv(
    os.path.join(data_path, "custom_valid_ratings.csv"),
    dtype={
        "user": int,
        "item": int,
        "time": float,
        "user_favotrite_genre": str,
        "label": float,
    },
)
train_df = pd.read_csv(
    os.path.join(data_path, "custom_train_ratings.csv"),
    dtype={
        "user": int,
        "item": int,
        "time": float,
        "user_favotrite_genre": str,
        "label": float,
    },
)

print("making_embedding_df")
embedding_genre = making_embedding_df(genre_data, "genre", 10)
embedding_writer = making_embedding_df(writer_data, "writer", 10)
embedding_direc = making_embedding_df(director_data, "director", 10)

print("merging")
train_df = pd.merge(train_df, year_data, how="left", on="item")
train_df = pd.merge(train_df, embedding_genre, how="left", on="item")
train_df = pd.merge(train_df, embedding_writer, how="left", on="item")
train_df = pd.merge(train_df, embedding_direc, how="left", on="item")

valid_df = pd.merge(valid_df, year_data, how="left", on="item")
valid_df = pd.merge(valid_df, embedding_genre, how="left", on="item")
valid_df = pd.merge(valid_df, embedding_writer, how="left", on="item")
valid_df = pd.merge(valid_df, embedding_direc, how="left", on="item")

train_df[["user", "item", "year", "decade"]] = train_df[
    ["user", "item", "year", "decade"]
].astype("str")
valid_df[["user", "item", "year", "decade"]] = valid_df[
    ["user", "item", "year", "decade"]
].astype("str")

# 사용할 Feature 설정
FEATS = [
    col
    for col in train_df.columns
    if col not in ["time", "label", "user_favotrite_genre"]
]

# X, y 값 분리
y_train = train_df["label"]
train = train_df.drop(["label"], axis=1)

y_valid = valid_df["label"]
valid = valid_df.drop(["label"], axis=1)


def objective(trial):
    # Optuna를 사용한 하이퍼파라미터 설정
    param = {
        "iterations": trial.suggest_int("iterations", 10, 100),
        "od_wait": trial.suggest_int("od_wait", 20, 50),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 0.1),
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.1),
        "subsample": trial.suggest_float("subsample", 0.4, 1),
        "random_strength": trial.suggest_float("random_strength", 1, 10),
        "depth": trial.suggest_int("depth", 3, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
        "leaf_estimation_iterations": trial.suggest_int(
            "leaf_estimation_iterations", 1, 10
        ),
        "bagging_temperature": trial.suggest_float(
            "bagging_temperature", 0.01, 10.00, log=True
        ),
    }

    # 모델 학습
    model = CatBoostClassifier(**param, eval_metric="AUC", verbose=10, random_seed=42)

    cat_features = ["user", "item", "year", "decade"]
    model.fit(
        train[FEATS],
        y_train,
        cat_features=cat_features,
        eval_set=(valid[FEATS], y_valid),
    )
    # 모델 평가
    preds = model.predict_proba(valid[FEATS])[:, 1]
    auc = roc_auc_score(y_valid, preds)
    return auc


# Optuna 스터디 생성 및 최적화 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=300)

# 최적의 파라미터 출력
print(f"Best parameters: {study.best_trial.params}")
print(f"Best AUC: {study.best_value}")

best_params = study.best_trial.params
model = CatBoostClassifier(**best_params, eval_metric="AUC", verbose=10, random_seed=42)

cat_features = ["user", "item", "year", "decade"]
model.fit(
    train[FEATS], y_train, cat_features=cat_features, eval_set=(valid[FEATS], y_valid)
)
# 모델 평가
preds = model.predict_proba(valid[FEATS])[:, 1]
acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
auc = roc_auc_score(y_valid, preds)
print(f"VALID AUC : {auc} ACC : {acc}\n")

feature_importances = model.get_feature_importance()

# INSTALL MATPLOTLIB IN ADVANCE
plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importances)), feature_importances)
plt.xticks(range(len(feature_importances)), FEATS, rotation="vertical")
plt.title("Feature Importances")
plt.savefig(f"./feature_importance/emb_catboost_feature_importance.png")

print("inference")
all_items = set(train_ratings["item"].unique())
final_df = []

for user, u_items in train_ratings.groupby("user")["item"]:
    u_set = set(u_items)
    neg_items = list(all_items - u_set)

    # Create a DataFrame with negative items
    df = pd.DataFrame({"user": user, "item": neg_items}, columns=["user", "item"])

    # Merge all necessary data into the DataFrame
    df = df.merge(year_data, on="item")
    df = df.merge(embedding_genre, on="item")
    df = df.merge(embedding_writer, on="item")
    df = df.merge(embedding_direc, on="item")

    df = df.reset_index(drop=True)
    df[["user", "item", "year", "decade"]] = df[
        ["user", "item", "year", "decade"]
    ].astype("str")
    total_preds = model.predict_proba(df[FEATS])[:, 1]
    top_k_indices = np.argsort(-total_preds)[:10]
    top_k_items = df["item"][top_k_indices]
    final_df.extend([(user, item) for item in top_k_items])

final_df = pd.DataFrame(final_df, columns=["user", "item"])
final_df.to_csv("./catboost_emb_submission.csv", index=False)
