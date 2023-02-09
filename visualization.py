import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd


def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y


def plot_embeddings(embeddings, dataset):
    X, Y = read_node_label(f'./dataset/{dataset}/labels.txt')
    embeddings = dict(zip(X, np.array(embeddings)))
    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_set = "IMDB"
    data = pd.read_csv(f'{data_set}_embeddings_pool.csv', header=None)
    embs = data.iloc[:, :-1]
    plot_embeddings(embs, data_set)

