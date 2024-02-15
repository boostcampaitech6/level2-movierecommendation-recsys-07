import pandas as pd
import os
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
import math
import matplotlib.pyplot as plt

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

item_group_sizes = train_ratings.groupby("item").size()
title_data["movie_popularity"] = title_data["item"].map(item_group_sizes)

temp_df = pd.merge(train_ratings, genre_data, how="right", on="item")
user_genre_counts = temp_df.groupby(["user", "genre"]).size().unstack(fill_value=0)
most_viewed_genre = user_genre_counts.idxmax(axis=1)
train_ratings["user_favotrite_genre"] = train_ratings["user"].map(most_viewed_genre)
genre_data["genre_items"] = genre_data.groupby("genre")["item"].transform("count")
genre_group_sizes = temp_df.groupby("genre").size()
genre_data["genre_popularity"] = genre_data["genre"].map(genre_group_sizes)

temp_df = pd.merge(train_ratings, writer_data, how="right", on="item")
writer_group_sizes = temp_df.groupby("writer").size()
writer_data["writer_popularity"] = writer_data["writer"].map(writer_group_sizes)
writer_data["writer_items"] = writer_data.groupby("writer")["item"].transform("count")

temp_df = pd.merge(train_ratings, director_data, how="right", on="item")
director_group_sizes = temp_df.groupby("director").size()
director_data["director_popularity"] = director_data["director"].map(
    director_group_sizes
)
director_data["director_items"] = director_data.groupby("director")["item"].transform(
    "count"
)

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

print("merging")
train_df = pd.merge(train_df, year_data, how="left", on="item")
train_df = pd.merge(
    train_df, title_data.loc[:, ["item", "movie_popularity"]], how="left", on="item"
)
train_df = pd.merge(train_df, genre_data, how="left", on="item").drop_duplicates(
    subset=["user", "item", "time"]
)
train_df = pd.merge(train_df, writer_data, how="left", on="item").drop_duplicates(
    subset=["user", "item", "time"]
)
train_df = pd.merge(train_df, director_data, how="left", on="item").drop_duplicates(
    subset=["user", "item", "time"]
)

valid_df = pd.merge(valid_df, year_data, how="left", on="item")
valid_df = pd.merge(
    valid_df, title_data.loc[:, ["item", "movie_popularity"]], how="left", on="item"
)
valid_df = pd.merge(valid_df, genre_data, how="left", on="item").drop_duplicates(
    subset=["user", "item", "time"]
)
valid_df = pd.merge(valid_df, writer_data, how="left", on="item").drop_duplicates(
    subset=["user", "item", "time"]
)
valid_df = pd.merge(valid_df, director_data, how="left", on="item").drop_duplicates(
    subset=["user", "item", "time"]
)

train_df[
    [
        "user",
        "item",
        "year",
        "decade",
        "genre",
        "user_favotrite_genre",
        "writer",
        "director",
    ]
] = train_df[
    [
        "user",
        "item",
        "year",
        "decade",
        "genre",
        "user_favotrite_genre",
        "writer",
        "director",
    ]
].astype(
    "str"
)
valid_df[
    [
        "user",
        "item",
        "year",
        "decade",
        "genre",
        "user_favotrite_genre",
        "writer",
        "director",
    ]
] = valid_df[
    [
        "user",
        "item",
        "year",
        "decade",
        "genre",
        "user_favotrite_genre",
        "writer",
        "director",
    ]
].astype(
    "str"
)

# 사용할 Feature 설정
FEATS = [col for col in train_df.columns if col not in ["time", "label"]]

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

    cat_features = [
        "user",
        "item",
        "year",
        "decade",
        "genre",
        "user_favotrite_genre",
        "writer",
        "director",
    ]
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

cat_features = [
    "user",
    "item",
    "year",
    "decade",
    "genre",
    "user_favotrite_genre",
    "writer",
    "director",
]
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
plt.savefig(f"./feature_importance/full_catboost_feature_importance.png")

print("inference")
all_items = set(train_ratings["item"].unique())
user_genre = dict(zip(train_ratings["user"], train_ratings["user_favotrite_genre"]))
final_df = []

for user, u_items in train_ratings.groupby("user")["item"]:
    u_set = set(u_items)
    neg_items = list(all_items - u_set)

    df = pd.DataFrame(
        {"user": user, "item": neg_items, "user_favotrite_genre": user_genre[user]},
        columns=["user", "item", "user_favotrite_genre"],
    )

    df = pd.merge(df, year_data, how="left", on="item")
    df = pd.merge(
        df, title_data.loc[:, ["item", "movie_popularity"]], how="left", on="item"
    )
    df = pd.merge(df, genre_data, how="left", on="item").drop_duplicates(
        subset=["user", "item"]
    )
    df = pd.merge(df, writer_data, how="left", on="item").drop_duplicates(
        subset=["user", "item"]
    )
    df = pd.merge(df, director_data, how="left", on="item").drop_duplicates(
        subset=["user", "item"]
    )

    df = df.reset_index(drop=True)
    df[
        [
            "user",
            "item",
            "year",
            "decade",
            "genre",
            "user_favotrite_genre",
            "writer",
            "director",
        ]
    ] = df[
        [
            "user",
            "item",
            "year",
            "decade",
            "genre",
            "user_favotrite_genre",
            "writer",
            "director",
        ]
    ].astype(
        "str"
    )
    total_preds = model.predict_proba(df[FEATS])[:, 1]
    top_k_indices = np.argsort(-total_preds)[:10]
    top_k_items = df["item"][top_k_indices]
    final_df.extend([(user, item) for item in top_k_items])

final_df = pd.DataFrame(final_df, columns=["user", "item"])
final_df.to_csv("./catboost_full_submission.csv", index=False)
