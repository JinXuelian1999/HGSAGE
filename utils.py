import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
from gravity import gravity_matrix
import pandas as pd

from dgl.data.utils import download
from pprint import pprint
from scipy import sparse
from scipy import io as sio


default_configure = {
    'lr': 0.001,             # Learning rate
    'num_heads': [8],        # Number of attention heads for node-level attention
    'hidden_units': 16,
    'dropout': 0.6,
    'weight_decay': 0.001,
    'num_epochs': 200,
    'patience': 100,
    'dataset': 'IMDB',
    'aggregator_type': 'pool'
}

sampling_configure = {
    'batch_size': 20
}


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix


def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format(args['dataset'], date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'

    mkdir_p(log_dir)
    return log_dir


def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args)
    return args


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def load_acm(remove_self_loop):
    assert not remove_self_loop
    data_path = './dataset/ACM/'
    with open(data_path + 'edges.pkl', 'rb') as f:
        edges = pickle.load(f)
    with open(data_path + 'labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open(data_path + 'node_features.pkl', 'rb') as f:
        node_features = pickle.load(f)

    p_vs_a = edges[0].todense().nonzero()     # paper-field
    a_vs_p = edges[1].todense().nonzero()     # field-paper
    p_vs_l = edges[2].todense().nonzero()     # paper-author
    l_vs_p = edges[3].todense().nonzero()     # auther-paper
    # 构建异构图
    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): p_vs_a,     # paper-author构成边，关系='pa'
        ('author', 'ap', 'paper'): a_vs_p,
        ('paper', 'pf', 'field'): p_vs_l,
        ('field', 'fp', 'paper'): l_vs_p
    })
    # print(hg.edges(etype='pa'))
    # 节点特征
    features = node_features[hg.nodes('paper'), ]
    pap_g = dgl.metapath_reachable_graph(hg, ['pa', 'ap'])
    pap_adj = pap_g.adjacency_matrix()   # 邻接矩阵
    pap_edge_list = list(zip(pap_adj._indices().cpu().numpy()[0, :], pap_adj._indices().cpu().numpy()[1, :]))  # 边列表
    pap_gravity = gravity_matrix(pap_edge_list)
    features = np.insert(features, features.shape[1], values=pap_gravity, axis=1)

    plp_g = dgl.metapath_reachable_graph(hg, ['pf', 'fp'])
    plp_adj = plp_g.adjacency_matrix()
    plp_edge_list = list(zip(plp_adj._indices().cpu().numpy()[0, :], plp_adj._indices().cpu().numpy()[1, :]))
    plp_gravity = gravity_matrix(plp_edge_list)
    features = np.insert(features, features.shape[1], values=plp_gravity, axis=1)

    features = torch.FloatTensor(features)
    # 标签
    node_labels = np.append(labels[0], labels[1], axis=0)          # 拼接矩阵
    node_labels = np.append(node_labels, labels[2], axis=0)
    node_labels = node_labels[np.lexsort([node_labels.T[0]])]       # 矩阵按照节点序号排序
    node_labels = torch.LongTensor(node_labels[:, 1])

    num_classes = 3

    # train, val, test
    train_idx = labels[0][:, 0]
    val_idx = labels[1][:, 0]
    test_idx = labels[2][:, 0]

    num_nodes = hg.number_of_nodes('paper')     # 节点数量
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    return hg, features, node_labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask


def load_dblp(remove_self_loop):
    assert not remove_self_loop
    data_path = './dataset/DBLP/'
    with open(data_path + 'edges.pkl', 'rb') as f:
        edges = pickle.load(f)
    with open(data_path + 'labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open(data_path + 'node_features.pkl', 'rb') as f:
        node_features = pickle.load(f)

    p_vs_a = edges[0].todense().nonzero()     # paper-author
    a_vs_p = edges[1].todense().nonzero()     # author-paper
    p_vs_c = edges[2].todense().nonzero()     # paper-conference
    c_vs_p = edges[3].todense().nonzero()     # conference-paper

    # 构建异构图
    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): p_vs_a,     # paper-author构成边，关系='pa'
        ('author', 'ap', 'paper'): a_vs_p,
        ('paper', 'pc', 'conference'): p_vs_c,
        ('conference', 'cp', 'paper'): c_vs_p
    })

    # 节点特征
    features = node_features[hg.nodes('author'), ]

    apa_g = dgl.metapath_reachable_graph(hg, ['ap', 'pa'])
    apa_adj = apa_g.adjacency_matrix()  # 邻接矩阵
    apa_edge_list = list(zip(apa_adj._indices().cpu().numpy()[0, :], apa_adj._indices().cpu().numpy()[1, :]))  # 边列表
    apa_gravity = gravity_matrix(apa_edge_list)
    features = np.insert(features, features.shape[1], values=apa_gravity, axis=1)

    apcpa_g = dgl.metapath_reachable_graph(hg, ['ap', 'pc', 'cp', 'pa'])
    apcpa_adj = apcpa_g.adjacency_matrix()
    apcpa_edge_list = list(zip(apcpa_adj._indices().cpu().numpy()[0, :], apcpa_adj._indices().cpu().numpy()[1, :]))
    apcpa_gravity = gravity_matrix(apcpa_edge_list)
    features = np.insert(features, features.shape[1], values=apcpa_gravity, axis=1)

    features = torch.FloatTensor(features)
    # 标签
    node_labels = np.append(np.array(labels[0]), np.array(labels[1]), axis=0)          # 拼接矩阵
    node_labels = np.append(node_labels, np.array(labels[2]), axis=0)
    node_labels = node_labels[np.lexsort([node_labels.T[0]])]       # 矩阵按照节点序号排序
    node_labels = torch.LongTensor(node_labels[:, 1])

    num_classes = 4

    # train, val, test
    train_idx = np.array(labels[0])[:, 0]
    val_idx = np.array(labels[1])[:, 0]
    test_idx = np.array(labels[2])[:, 0]

    num_nodes = hg.number_of_nodes('author')     # 节点数量
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    return hg, features, node_labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask


def load_imdb(remove_self_loop):
    assert not remove_self_loop
    data_path = './dataset/IMDB/'
    with open(data_path + 'edges.pkl', 'rb') as f:
        edges = pickle.load(f)
    with open(data_path + 'labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open(data_path + 'node_features.pkl', 'rb') as f:
        node_features = pickle.load(f)

    m_vs_d = edges[0].todense().nonzero()     # movie-actor
    d_vs_m = edges[1].todense().nonzero()     # actor-movie
    m_vs_a = edges[2].todense().nonzero()     # movie-director
    a_vs_m = edges[3].todense().nonzero()     # director-movie

    # 构建异构图
    hg = dgl.heterograph({
        ('movie', 'ma', 'actor'): m_vs_a,     # movie-actor构成边，关系='ma'
        ('actor', 'am', 'movie'): a_vs_m,
        ('movie', 'md', 'director'): m_vs_d,
        ('director', 'dm', 'movie'): d_vs_m
    })

    # 标签
    node_labels = np.append(np.array(labels[0]), np.array(labels[1]), axis=0)          # 拼接矩阵
    node_labels = np.append(node_labels, np.array(labels[2]), axis=0)
    node_labels = node_labels[np.lexsort([node_labels.T[0]])]       # 矩阵按照节点序号排序
    movie_nodes = node_labels[:, 0]
    movie_index = dict(zip(movie_nodes, range(len(movie_nodes))))
    node_labels = torch.LongTensor(node_labels[:, 1])
    num_classes = 3

    # 节点特征
    features = node_features[hg.nodes('movie'), ]

    mam_g = dgl.metapath_reachable_graph(hg, ['ma', 'am'])
    mam_adj = mam_g.adjacency_matrix()  # 邻接矩阵
    mam_edge_list = list(zip(mam_adj._indices().cpu().numpy()[0, :], mam_adj._indices().cpu().numpy()[1, :]))  # 边列表
    mam_gravity = gravity_matrix(mam_edge_list)
    features = np.insert(features, features.shape[1], values=mam_gravity, axis=1)

    mdm_g = dgl.metapath_reachable_graph(hg, ['md', 'dm'])
    mdm_adj = mdm_g.adjacency_matrix()
    mdm_edge_list = list(zip(mdm_adj._indices().cpu().numpy()[0, :], mdm_adj._indices().cpu().numpy()[1, :]))
    mdm_gravity = gravity_matrix(mdm_edge_list)
    features = np.insert(features, features.shape[1], values=mdm_gravity, axis=1)

    features = torch.FloatTensor(features)

    # train, val, test
    train_idx = np.array(labels[0])[:, 0]
    val_idx = np.array(labels[1])[:, 0]
    test_idx = np.array(labels[2])[:, 0]

    train_idx = np.array(pd.DataFrame(train_idx, columns=['movie'])['movie'].map(movie_index))
    val_idx = np.array(pd.DataFrame(val_idx, columns=['movie'])['movie'].map(movie_index))
    test_idx = np.array(pd.DataFrame(test_idx, columns=['movie'])['movie'].map(movie_index))

    num_nodes = len(movie_nodes)     # 节点数量
    movie_nodes = torch.tensor(movie_nodes)
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    return hg, features, node_labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask, movie_nodes


def load_data(dataset, remove_self_loop=False):
    if dataset == 'ACM':
        return load_acm(remove_self_loop)
    elif dataset == 'DBLP':
        return load_dblp(remove_self_loop)
    elif dataset == 'IMDB':
        return load_imdb(remove_self_loop)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset))


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
