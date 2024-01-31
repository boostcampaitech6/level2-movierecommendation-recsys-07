import pandas as pd

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from tqdm import tqdm

def main():
    df_name = "Word2Vec_not_shuffle_emb_df"
    df = pd.read_csv(f'{df_name}.csv')
    ks = range(1,20)
    inertias = []

    for k in tqdm(ks):
        model = KMeans(n_clusters=k)
        model.fit(df)
        inertias.append(model.inertia_)

    # Plot ks vs inertias
    plt.figure(figsize=(4, 4))

    plt.plot(ks, inertias, '-o')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.savefig(f'{df_name}_Intertias.png')

if __name__ == '__main__':
    main()