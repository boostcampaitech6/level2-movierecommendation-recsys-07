import pandas as pd
import numpy as np
import os

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore')

from sklearn.manifold import TSNE

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self, shuffle):
        self.epoch = 1
        self.loss_to_be_subed = 0
        self.loss_now = 987654321
        self.shuffle = shuffle

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        print(f'Loss after epoch {self.epoch}: {loss_now}, Total_loss : {loss}')

        self.loss_to_be_subed = loss
        if loss_now < self.loss_now:
            self.loss_now = loss_now
            if self.shuffle:
                model.save('Word2Vec.model')
            else:
                model.save('Word2Vec_not_shuffle.model')
            print('Model 저장')
        self.epoch += 1

def main():
    shuffle = True
    print('Load df')
    df = pd.read_csv('../../data/train/train_ratings.csv')
    group_df = df.groupby('user')

    print('Make Sentences')
    sentences = [group['item'].tolist() for user, group in group_df]

    if shuffle:
        np.random.seed(42)
        for group in sentences:
            np.random.shuffle(group)

    print('Training')
    model = Word2Vec(
                sentences = sentences,
                seed = 42,
                epochs = 50,
                min_count = 1,
                vector_size = 150,
                sg = 1,
                negative = 15,
                ns_exponent = 0.75,
                window = 50,
                hs = 0,
                sample = 0.00001,
                compute_loss = True, 
                callbacks=[callback(shuffle)],
                 )
    
    print('Model Load')
    if shuffle:
        model = Word2Vec.load('Word2Vec.model')
    else:
        model = Word2Vec.load('Word2Vec_not_shuffle.model')

    print('Save Embedding')
    arr = np.array([model.wv[item] for item in df['item'].unique()])
    arr_df = pd.DataFrame(arr)
    arr_df['item'] = df['item'].unique()
    if shuffle:
        arr_df.to_csv('Word2Vec_emb_df.csv', index=False)
    else:
        arr_df.to_csv('Word2Vec_not_shuffle_emb_df.csv', index=False)
    
    item2idx = {}
    for idx, item in enumerate(df['item'].unique()):
        item2idx[item] = idx

    print('TSNE')
    tsne = TSNE(n_components = 2, random_state = 42)
    tsne_arr = tsne.fit_transform(arr_df)
    if shuffle:
        pd.DataFrame(tsne_arr).to_csv('TSNE_df.csv', index=False)
    else:
        pd.DataFrame(tsne_arr).to_csv('TSNE_not_shuffle_df.csv', index=False)

    print('Visualize')
    genre_df = pd.read_csv('../../data/train/genres.tsv', sep='\t')
    
    x_min, x_max = min(tsne_arr[:,0]), max(tsne_arr[:,0])
    y_min, y_max = min(tsne_arr[:,1]), max(tsne_arr[:,1])

    fig1 = plt.figure(1, figsize= (12, 12))
    fig2 = plt.figure(2, figsize= (40, 32))
    ax1 = fig1.add_subplot(111)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    i = 0

    colors = plt.cm.jet(np.linspace(0,1,genre_df['genre'].nunique()))
    for genre, g_df in genre_df.groupby('genre'):
        i+=1
        item_list = g_df['item'].tolist()
        idx_list = [item2idx[item] for item in item_list]
    
        ax1.scatter(tsne_arr[idx_list, 0], tsne_arr[idx_list, 1], alpha = 0.5, s = 10, label = f'{genre}', color=colors[i-1])
        ax = fig2.add_subplot(4, 5, i)
        ax.scatter(tsne_arr[idx_list, 0], tsne_arr[idx_list, 1], s = 10)
        ax.set_title(f'{genre}', fontsize = 40)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    ax1.set_title("Genre", fontsize = 60)
    ax1.legend()
    
    if shuffle:
        fig1.savefig("2d_full_tsne.png")
        plt.close(fig1)

        fig2.savefig("2d_tsne.png")
        plt.close(fig2)
    else:
        fig1.savefig("2d_full_tsne_not_shuffle.png")
        plt.close(fig1)

        fig2.savefig("2d_tsne_not_shuffle.png")
        plt.close(fig2)

if __name__ == '__main__':
    main()