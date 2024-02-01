import pandas as pd

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from tqdm import tqdm

import warnings

warnings.filterwarnings(action="ignore")


def inertia(df_name: str) -> None:
    """
    KMeans clustering을 할 때 적절한 k값을 찾기 위함.
    결과 그래프에서 급격히 꺾이며 완만해지는 구간이 적절한 k.
    """
    df = pd.read_csv(f"{df_name}.csv")
    ks = range(1, 30)
    inertias = []

    for k in tqdm(ks):
        model = KMeans(n_clusters=k)
        model.fit(df)
        inertias.append(model.inertia_)

    # Plot ks vs inertias
    plt.figure(figsize=(12, 12))

    plt.plot(ks, inertias, "-o")
    plt.xlabel("number of clusters, k")
    plt.ylabel("inertia")
    plt.xticks(ks)
    plt.savefig(f"{df_name}_Intertias.png")


if __name__ == "__main__":
    inertia("Word2Vec_not_shuffle_emb_df")
