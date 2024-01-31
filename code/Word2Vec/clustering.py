import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def main():
    n_clusters = 6
    df_name = "Word2Vec_emb_df"
    df = pd.read_csv(f'{df_name}.csv')

    model = KMeans(n_clusters=n_clusters)
    model.fit(df)
    pred = model.predict(df)

    tsne_df = pd.read_csv('TSNE_df.csv')
    tsne_arr = np.array(tsne_df)
    
    raw_df = pd.read_csv('../../data/train/train_ratings.csv')

    item2idx = {}
    for idx, item in enumerate(raw_df['item'].unique()):
        item2idx[item] = idx

    raw_df['cluster'] = raw_df['item'].apply(lambda x: pred[item2idx[x]])
    raw_df[['item','cluster']].to_csv('Cluster_df.csv', index=False)
    
    x_min, x_max = min(tsne_arr[:,0]), max(tsne_arr[:,0])
    y_min, y_max = min(tsne_arr[:,1]), max(tsne_arr[:,1])

    fig1 = plt.figure(1, figsize= (24, 24))
    fig2 = plt.figure(2, figsize= (40, 32))
    ax1 = fig1.add_subplot(111)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    i = 0

    colors = plt.cm.jet(np.linspace(0,1,n_clusters))
    for cluster, g_df in raw_df.groupby('cluster'):
        i+=1
        item_list = g_df['item'].tolist()
        idx_list = [item2idx[item] for item in item_list]
    
        ax1.scatter(tsne_arr[idx_list, 0], tsne_arr[idx_list, 1], alpha = 1, s = 30, label = f'{cluster}', color=colors[i-1])
        ax = fig2.add_subplot(4, 5, i)
        ax.scatter(tsne_arr[idx_list, 0], tsne_arr[idx_list, 1], s = 10)
        ax.set_title(f'{cluster}', fontsize = 40)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    ax1.set_title("Cluster", fontsize = 60)
    ax1.legend(fontsize = 30, markerscale = 5.0)
    
    fig1.savefig("2d_full_tsne_cluster.png")
    plt.close(fig1)

    fig2.savefig("2d_tsne_cluster.png")
    plt.close(fig2)

if __name__ == '__main__':
    main()