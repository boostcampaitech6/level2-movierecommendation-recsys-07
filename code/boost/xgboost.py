import pandas as pd
import argparse
import os
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
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

def merge_df(df_dict, base_df):
    cluster = df_dict['cluster']
    user_most_genre = df_dict['user_most_genre']
    year_data = df_dict['year_data']
    title_data = df_dict['title_data']
    genre_data = df_dict['genre_data']
    writer_data = df_dict['writer_data']
    writer_most_genre = df_dict['writer_most_genre']
    writer_most_viewed_genre = df_dict['writer_most_viewed_genre']
    director_data = df_dict['director_data']
    director_most_genre = df_dict['director_most_genre']
    director_most_viewed_genre = df_dict['director_most_viewed_genre']
    item_cluster = df_dict['item_cluster']

    temp_df = pd.merge(base_df, cluster, how="left", on="user")
    temp_df = pd.merge(temp_df, user_most_genre, how="left", on="user")
    temp_df = pd.merge(temp_df, year_data, how="left", on="item")
    temp_df = pd.merge(
        temp_df, title_data.loc[:, ["item", "movie_popularity"]], how="left", on="item"
    )
    temp_df = pd.merge(temp_df, genre_data, how="left", on="item").drop_duplicates(
        subset=["user", "item"]
    )
    temp_df = pd.merge(temp_df, writer_data, how="left", on="item").drop_duplicates(
        subset=["user", "item"]
    )
    temp_df = pd.merge(temp_df, writer_most_genre, how="left", on="writer")
    temp_df = pd.merge(temp_df, writer_most_viewed_genre, how="left", on="writer")
    temp_df = pd.merge(temp_df, director_data, how="left", on="item").drop_duplicates(
        subset=["user", "item"]
    )
    temp_df = pd.merge(temp_df, director_most_genre, how="left", on="director")
    temp_df = pd.merge(temp_df, director_most_viewed_genre, how="left", on="director")
    temp_df = pd.merge(temp_df, item_cluster, how="left", on="item")

    category_f = [
        "user",
        "item",
        "year",
        "decade",
        "genre",
        "user_favotrite_genre",
        "writer",
        "director",
        "director_most_genre",
        "director_most_viewed_genre",
        "user_most_genre",
        "writer_most_genre",
        "writer_most_viewed_genre",
        "cluster",
        "item_cluster"
    ]
    temp_df[category_f] = temp_df[category_f].astype("category")

    return temp_df

def merge_embedding(df_dict, base_df):
    
    year_data = df_dict['year_data']
    genre_data = df_dict['genre_data']
    writer_data = df_dict['writer_data']
    director_data = df_dict['director_data']
    
    temp = pd.merge(base_df, year_data, how="left", on="item")
    temp = pd.merge(temp, genre_data, how="left", on="item")
    temp = pd.merge(temp, writer_data, how="left", on="item")
    temp = pd.merge(temp, director_data, how="left", on="item")

    temp[["user", "item", "year"]] = temp[
        ["user", "item", "year"]
    ].astype("str")

    return temp
    
def dataload():
    print("loading")
    data_path = "../../data/train"
    train_ratings = pd.read_csv(os.path.join(data_path, "train_ratings.csv"))
    year_data = pd.read_csv(os.path.join(data_path, "years.tsv"), sep="\t")
    writer_data = pd.read_csv(os.path.join(data_path, "writers.tsv"), sep="\t")
    title_data = pd.read_csv(os.path.join(data_path, "titles.tsv"), sep="\t")
    genre_data = pd.read_csv(os.path.join(data_path, "genres.tsv"), sep="\t")
    director_data = pd.read_csv(os.path.join(data_path, "directors.tsv"), sep="\t")
    director_most_genre = pd.read_csv(os.path.join(data_path, "director_most_genre.tsv"), sep="\t")
    director_most_viewed_genre = pd.read_csv(os.path.join(data_path, "director_most_viewed_genre.tsv"), sep="\t")
    user_most_genre = pd.read_csv(os.path.join(data_path, "user_most_genre.tsv"), sep="\t")
    writer_most_genre = pd.read_csv(os.path.join(data_path, "writer_most_genre.tsv"), sep="\t")
    writer_most_viewed_genre = pd.read_csv(os.path.join(data_path, "writer_most_viewed_genre.tsv"), sep="\t")
    cluster = pd.read_csv(os.path.join(data_path, "cluster.tsv"), sep="\t")
    item_cluster = pd.read_csv(os.path.join(data_path, "item_cluster.tsv"), sep="\t")

    df_dict = {
        'train_ratings': train_ratings,
        'year_data': year_data,
        'writer_data': writer_data,
        'title_data': title_data,
        'genre_data': genre_data,
        'director_data': director_data,
        'director_most_genre': director_most_genre,
        'director_most_viewed_genre': director_most_viewed_genre,
        'user_most_genre': user_most_genre,
        'writer_most_genre': writer_most_genre,
        'writer_most_viewed_genre': writer_most_viewed_genre,
        'cluster': cluster,
        'item_cluster': item_cluster
    }

    return df_dict
    
def embdding_data_preparation():
    df_dict = dataload()
    data_path = "../../data/train"
    # negative sampling 후 split한 데이터
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
    df_dict['genre_data'] = making_embedding_df(df_dict['genre_data'], "genre", 10)
    df_dict['writer_data'] = making_embedding_df(df_dict['writer_data'], "writer", 10)
    df_dict['director_data'] = making_embedding_df(df_dict['director_data'], "director", 10)

    print("merging")
    train_df=merge_embedding(df_dict, train_df)
    valid_df=merge_embedding(df_dict, valid_df)

    df_dict['train_df']=train_df
    df_dict['valid_df']=valid_df

    return df_dict

def data_preparation():
    df_dict = dataload()
    data_path = "../../data/train"

    # negative sampling 후 split한 데이터
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


    # decade 계산
    df_dict['year_data']["decade"] = df_dict['year_data']["year"] // 10 * 10

    # 영화 인기도 계산
    item_group_sizes = df_dict['train_ratings'].groupby("item").size()
    df_dict['title_data']["movie_popularity"] = df_dict['title_data']["item"].map(item_group_sizes)

    # 사용자별 가장 많이 본 장르
    temp_df = pd.merge(df_dict['train_ratings'], df_dict['genre_data'], how="right", on="item")
    user_genre_counts = temp_df.groupby(["user", "genre"]).size().unstack(fill_value=0)
    most_viewed_genre = user_genre_counts.idxmax(axis=1)
    df_dict['train_ratings']["user_favotrite_genre"] = df_dict['train_ratings']["user"].map(most_viewed_genre)

    # 장르 인기도 및 아이템 수
    df_dict['genre_data']["genre_items"] = df_dict['genre_data'].groupby("genre")["item"].transform("count")
    genre_group_sizes = temp_df.groupby("genre").size()
    df_dict['genre_data']["genre_popularity"] = df_dict['genre_data']["genre"].map(genre_group_sizes)

    # 작가별 인기도 및 아이템 수
    temp_df = pd.merge(df_dict['train_ratings'], df_dict['writer_data'], how="right", on="item")
    writer_group_sizes = temp_df.groupby("writer").size()
    df_dict['writer_data']["writer_popularity"] = df_dict['writer_data']["writer"].map(writer_group_sizes)
    df_dict['writer_data']["writer_items"] = df_dict['writer_data'].groupby("writer")["item"].transform("count")

    # 감독별 인기도 및 아이템 수
    temp_df = pd.merge(df_dict['train_ratings'], df_dict['director_data'], how="right", on="item")
    director_group_sizes = temp_df.groupby("director").size()
    df_dict['director_data']["director_popularity"] = df_dict['director_data']["director"].map(director_group_sizes)
    df_dict['director_data']["director_items"] = df_dict['director_data'].groupby("director")["item"].transform("count")

    
    print("merging")
    train_df=merge_df(df_dict, train_df)
    valid_df=merge_df(df_dict, valid_df)

    df_dict['train_df']=train_df
    df_dict['valid_df']=valid_df

    return df_dict

def inference(args, model, dfs_dict, FEATS):
    train_ratings = dfs_dict['train_ratings']

    all_items = set(train_ratings["item"].unique())
    user_genre = dict(zip(train_ratings["user"], train_ratings["user_favotrite_genre"]))
    final_df = []

    for user, u_items in tqdm(train_ratings.groupby("user")["item"]):
        u_set = set(u_items)
        neg_items = list(all_items - u_set)

        df = pd.DataFrame(
            {"user": user, "item": neg_items, "user_favotrite_genre": user_genre[user]},
            columns=["user", "item", "user_favotrite_genre"],
        )
        if args.embedding:
            df=merge_embedding(dfs_dict, df)
        else: 
            df=merge_df(dfs_dict, df)

        df = df.reset_index(drop=True)
        dtest = xgb.DMatrix(df[FEATS], enable_categorical=True)
        total_preds = model.predict(dtest)
        top_k_indices = np.argsort(-total_preds)[:10]
        top_k_items = df["item"][top_k_indices]
        final_df.extend([(user, item) for item in top_k_items])

    final_df = pd.DataFrame(final_df, columns=["user", "item"])
    final_df.to_csv("./xgboost_full_submission.csv", index=False)

def main(args):
    if args.embedding:
        df_dict = embdding_data_preparation()
    else:
        df_dict = data_preparation()

    not_use = ["time", "label", "user_most_genre", "movie_popularity","writer", "director", "director_most_genre","director_most_viewed_genre", "writer_most_genre","writer_most_viewed_genre",'genre_items','genre_popularity','writer_popularity','writer_items','director_popularity','director_items',"year"]
    FEATS = [col for col in df_dict['train_df'].columns if col not in not_use]
    # X, y 값 분리
    y_train = df_dict['train_df']["label"]
    train = df_dict['train_df'].drop(["label"], axis=1)

    y_valid = df_dict['valid_df']["label"]
    valid = df_dict['valid_df'].drop(["label"], axis=1)

    print("making dataset")
    dtrain = xgb.DMatrix(train[FEATS], label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(valid[FEATS], label=y_valid, enable_categorical=True)


    def objective(trial):
        # Optuna를 사용한 하이퍼파라미터 설정
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "lambda": trial.suggest_float("lambda", 1e-3, 0.1),
            "alpha": trial.suggest_float("alpha", 1e-3, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1),
            "subsample": trial.suggest_float("subsample", 0.4, 1),
            "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.1),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 50),
        }

        num_rounds = 300

        # 모델 학습
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_rounds,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=10,
        )
        # 모델 평가
        preds = model.predict(dvalid)
        auc = roc_auc_score(y_valid, preds)
        return auc


    if args.optuna:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        # 최적의 파라미터 출력
        print(f"Best parameters: {study.best_trial.params}")
        print(f"Best AUC: {study.best_value}")
        best_params = study.best_trial.params
        params = study.best_trial.params
        additional_params = {"objective": "binary:logistic", "eval_metric": "auc"}
        params.update(additional_params)

    else:
        best_params = {'iterations': 100, 'od_wait': 25, 'reg_lambda': 0.043043716721463586, 'learning_rate': 0.09811034307288567, 'subsample': 0.5981070567147279, 'random_strength': 3.544474967800164, 'depth': 10, 'min_data_in_leaf': 9, 'leaf_estimation_iterations': 3, 'bagging_temperature': 0.05087691411379589}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=20,
    )

    preds = model.predict(dvalid)
    acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_valid, preds)
    print(f"VALID AUC : {auc} ACC : {acc}\n")

    fig, ax = plt.subplots(figsize=(10, 15))
    _ = xgb.plot_importance(model, ax=ax)
    fig.savefig(f"./feature_importance/full_xgboost_feature_importance.png")

    inference(args, model, df_dict, FEATS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to demonstrate argparse usage with optuna.')
    parser.add_argument('--optuna', action='store_true', help='Runs optuna')
    parser.add_argument('--embedding', action='store_true', help='Runs embedding version')
    args = parser.parse_args()

    # main 함수 호출
    main(args)