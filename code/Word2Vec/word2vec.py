import pandas as pd
import numpy as np
import os

from typing import Union

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings(action="ignore")

from sklearn.manifold import TSNE


class callback(CallbackAny2Vec):
    """Callback to print loss after each epoch."""

    def __init__(self, shuffle):
        self.epoch = 1
        self.loss_to_be_subed = 0
        self.min_loss = 987654321
        self.shuffle = shuffle

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        print(f"Loss after epoch {self.epoch}: {loss_now}, Total_loss : {loss}")

        self.loss_to_be_subed = loss
        if loss_now < self.min_loss:
            self.min_loss = loss_now
            if self.shuffle:
                model.save("Word2Vec.model")
            else:
                model.save("Word2Vec_not_shuffle.model")
            print("Model 저장")
        self.epoch += 1


def main() -> Union[Word2Vec, str, np.ndarray, bool]:
    """
    output: (model: Word2Vec 모델, emb_df_name: 임베딩이 저장될 df 파일명, item_uniq: item 목록, shuffle: sequence의 셔플 여부)\n
    Word2Vec 모델을 훈련시키고, 최적의 모델을 저장.\n
    각 item을 하나의 단어로, 각 user의 관람 이력을 하나의 sequence로 간주함.\n
    """
    shuffle = False
    print("Load df")
    df = pd.read_csv("../../data/train/train_ratings.csv")
    group_df = df.groupby("user")

    print("Make Sentences")
    sentences = [group["item"].tolist() for user, group in group_df]

    if shuffle:
        np.random.seed(42)
        for group in sentences:
            np.random.shuffle(group)

    print("Training")
    model = Word2Vec(
        sentences=sentences,
        seed=42,
        epochs=50,
        min_count=1,
        vector_size=150,
        sg=1,
        negative=15,
        ns_exponent=0.75,
        window=50,
        hs=0,
        sample=0.00001,
        compute_loss=True,
        callbacks=[callback(shuffle)],
    )

    print("Model Load")
    model = (
        Word2Vec.load("Word2Vec.model")
        if shuffle
        else Word2Vec.load("Word2Vec_not_shuffle.model")
    )

    emb_df_name = "Word2Vec_emb_df" if shuffle else "Word2Vec_not_shuffle_emb_df"

    item_uniq = df["item"].unique()

    return model, emb_df_name, item_uniq, shuffle


def save_embedding(
    model: Word2Vec, df_name: str, item_uniq: np.ndarray, shuffle: bool = False
) -> Union[pd.DataFrame, str, np.ndarray, bool]:
    """
    output: (emb_df: 임베딩이 저장된 df, tsne_df_name: tsne 결과를 저장할 df 파일명, item_uniq: item 목록, shuffle: sequence의 셔플 여부)\n
    각 item에 Word2Vec 모델을 적용해 임베딩으로 변환 후 저장.\n
    """
    print("Save Embedding")
    emb = np.array([model.wv[item] for item in item_uniq])
    emb_df = pd.DataFrame(emb)
    emb_df["item"] = item_uniq
    emb_df.to_csv(f"{df_name}.csv", index=False)

    tsne_df_name = "TSNE_df" if shuffle else "TSNE_not_shuffle_df"

    return emb_df, tsne_df_name, item_uniq, shuffle


def tsne(
    emb_df: pd.DataFrame,
    tsne_df_name: str,
    item_uniq: np.ndarray,
    shuffle: bool = False,
) -> Union[np.ndarray, np.ndarray, bool]:
    """
    output: (tsne_arr: tsne로 2차원으로 축소한 array, item_uniq: item 목록, shuffle: sequence의 셔플 여부)\n
    emb_df를 읽고 TSNE를 이용해 2차원으로 차원 축소 후 df로 저장.\n
    2차원으로 축소된 np.ndarray 반환\n
    """
    print("TSNE")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_arr = tsne.fit_transform(emb_df)
    tsne_df = pd.DataFrame(tsne_arr)
    tsne_df["item"] = item_uniq
    tsne_df.to_csv(f"{tsne_df_name}.csv", index=False)

    return tsne_arr, item_uniq, shuffle


def visualize(
    tsne_arr: np.ndarray, item_uniq: np.ndarray, shuffle: bool = False
) -> None:
    """
    tsne로 2차원에 축소된 array를 scatter plot으로 표현 후 저장.\n
    한 plot에 모든 점을 표현한 것과 여러 plot에 나누어 표현한 2가지 figure를 저장.  \n
    장르에 따라 시각화를 하도록 구현되어 있음.
    """
    print("Visualize")
    # 시각화를 위해 indexing
    item2idx = {}
    for idx, item in enumerate(item_uniq):
        item2idx[item] = idx

    x_min, x_max = min(tsne_arr[:, 0]), max(tsne_arr[:, 0])
    y_min, y_max = min(tsne_arr[:, 1]), max(tsne_arr[:, 1])

    fig1 = plt.figure(1, figsize=(12, 12))  # 한 plot에 모든 점 표현
    fig2 = plt.figure(2, figsize=(40, 32))  # 여러 plot에 각각 점 표현
    ax1 = fig1.add_subplot(111)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    i = 0

    genre_df = pd.read_csv("../../data/train/genres.tsv", sep="\t")
    colors = plt.cm.jet(np.linspace(0, 1, genre_df["genre"].nunique()))

    for genre, g_df in genre_df.groupby("genre"):
        i += 1
        item_list = g_df["item"].tolist()
        idx_list = [item2idx[item] for item in item_list]

        ax1.scatter(
            tsne_arr[idx_list, 0],
            tsne_arr[idx_list, 1],
            alpha=0.5,
            s=10,
            label=f"{genre}",
            color=colors[i - 1],
        )
        ax = fig2.add_subplot(4, 5, i)
        ax.scatter(tsne_arr[idx_list, 0], tsne_arr[idx_list, 1], s=10)
        ax.set_title(f"{genre}", fontsize=40)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    ax1.set_title("Genre", fontsize=60)
    ax1.legend()

    if shuffle:
        fig1.savefig("2d_full_tsne_genre.png")
        plt.close(fig1)

        fig2.savefig("2d_tsne_genre.png")
        plt.close(fig2)
    else:
        fig1.savefig("2d_full_tsne_genre_not_shuffle.png")
        plt.close(fig1)

        fig2.savefig("2d_tsne_genre_not_shuffle.png")
        plt.close(fig2)


if __name__ == "__main__":
    model, emb_df_name, item_uniq, shuffle = main()
    emb_df, tsne_df_name, item_uniq, shuffle = save_embedding(
        model, emb_df_name, item_uniq, shuffle
    )
    tsne_arr, item_uniq, shuffle = tsne(emb_df, tsne_df_name, item_uniq, shuffle)
    visualize(tsne_arr, item_uniq, shuffle)
