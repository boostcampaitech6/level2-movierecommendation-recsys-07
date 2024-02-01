import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings(action="ignore")


def clustering(
    emb_df_name: str,
    tsne_df_name: str,
    output_df_name: str,
    png_name: str,
    n_clusters: int,
) -> None:
    """
    inertia.py의 결과에 따라 KMeans를 이용해 embedding df를 n개로 clustering.
    각 item이 어느 cluster에 속하는지 {output_df_name}.csv에 저장.
    2차원 TSNE 결과를 plot으로 표현하되, cluster를 기준으로 분리하여 표현.
    """
    df = pd.read_csv(f"{emb_df_name}.csv")

    model = KMeans(n_clusters=n_clusters)
    model.fit(df)
    pred = model.predict(df)  # n_items 길이의 cluster array. [3, 1, 1, 2, ...]

    tsne_df = pd.read_csv(f"{tsne_df_name}.csv")
    tsne_arr = np.array(tsne_df)

    raw_df = pd.read_csv("../../data/train/train_ratings.csv")

    item2idx = {}
    for idx, item in enumerate(raw_df["item"].unique()):
        item2idx[item] = idx

    raw_df["cluster"] = raw_df["item"].apply(lambda x: pred[item2idx[x]])
    output_df = raw_df.drop_duplicates("item", keep="first")[["item", "cluster"]]
    output_df.to_csv(f"{output_df_name}.csv", index=False)

    x_min, x_max = min(tsne_arr[:, 0]), max(tsne_arr[:, 0])
    y_min, y_max = min(tsne_arr[:, 1]), max(tsne_arr[:, 1])

    fig1 = plt.figure(1, figsize=(24, 24))
    fig2 = plt.figure(2, figsize=(40, 32))
    ax1 = fig1.add_subplot(111)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    i = 0

    colors = plt.cm.jet(np.linspace(0, 1, n_clusters))
    for cluster, g_df in raw_df.groupby("cluster"):
        i += 1
        item_list = g_df["item"].tolist()
        idx_list = [item2idx[item] for item in item_list]

        ax1.scatter(
            tsne_arr[idx_list, 0],
            tsne_arr[idx_list, 1],
            alpha=1,
            s=30,
            label=f"{cluster}",
            color=colors[i - 1],
        )
        ax = fig2.add_subplot(4, 5, i)
        ax.scatter(tsne_arr[idx_list, 0], tsne_arr[idx_list, 1], s=10)
        ax.set_title(f"{cluster}", fontsize=40)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    ax1.set_title("Cluster", fontsize=60)
    ax1.legend(fontsize=30, markerscale=5.0)

    fig1.savefig(f"2d_full_tsne_{png_name}.png")
    plt.close(fig1)

    fig2.savefig(f"2d_tsne_{png_name}.png")
    plt.close(fig2)


if __name__ == "__main__":
    clustering(
        "Word2Vec_not_shuffle_emb_df",
        "TSNE_not_shuffle_df",
        "Cluster_df",
        "cluster_not_shuffle",
        4,
    )
