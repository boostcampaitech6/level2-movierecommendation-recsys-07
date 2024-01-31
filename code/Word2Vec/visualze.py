import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def main():
    df = pd.read_csv('../../data/train/train_ratings.csv')

    item2idx = {}
    for idx, item in enumerate(df['item'].unique()):
        item2idx[item] = idx

    tsne_df = pd.read_csv('TSNE_df.csv')
    tsne_arr = np.array(tsne_df)
    
    #genre_df = pd.read_csv('../data/train/genres.tsv', sep='\t')
    year_df = pd.read_csv('../../data/train/years.tsv', sep='\t')
    year_df['decade'] = year_df['year'].apply(lambda x: (x//10)*10 if x < 2000 else (x//4)*4)
    
    x_min, x_max = min(tsne_arr[:,0]), max(tsne_arr[:,0])
    y_min, y_max = min(tsne_arr[:,1]), max(tsne_arr[:,1])

    fig1 = plt.figure(1, figsize= (24, 24))
    fig2 = plt.figure(2, figsize= (40, 32))
    ax1 = fig1.add_subplot(111)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    i = 0

    colors = plt.cm.jet(np.linspace(0,1,year_df['decade'].nunique()))
    for decade, g_df in year_df.groupby('decade'):
        i+=1
        item_list = g_df['item'].tolist()
        idx_list = [item2idx[item] for item in item_list]
    
        ax1.scatter(tsne_arr[idx_list, 0], tsne_arr[idx_list, 1], alpha = 1, s = 30, label = f'{decade}', color=colors[i-1])
        ax = fig2.add_subplot(4, 5, i)
        ax.scatter(tsne_arr[idx_list, 0], tsne_arr[idx_list, 1], s = 10)
        ax.set_title(f'{decade}', fontsize = 40)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    ax1.set_title("Decade", fontsize = 60)
    ax1.legend(fontsize = 30, markerscale = 5.0)
    
    fig1.savefig("2d_full_tsne_decade.png")
    plt.close(fig1)

    fig2.savefig("2d_tsne_decade.png")
    plt.close(fig2)

if __name__ == '__main__':
    main()