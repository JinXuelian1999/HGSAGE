import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import SAGEConv

import pandas as pd


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)              # (M, 1)
        beta = torch.softmax(w, dim=0)           # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)   # (N, M, 1)

        return (beta * z).sum(1)                 # (N, D * K)


class HGSAGELayer(nn.Module):
    """
        HGSAGE layer.

        Arguments
        ---------
        meta_paths : list of metapaths, each as a list of edge types
        in_size : input feature dimension
        out_size : output feature dimension
        dropout : Dropout probability

        Inputs
        ------
        g : DGLHeteroGraph
            The heterogeneous graph
        h : tensor
            Input features

        Outputs
        -------
        tensor
            The output feature
        """
    def __init__(self, meta_paths, in_size, out_size, dropout, aggregator_type):
        super(HGSAGELayer, self).__init__()

        # Two GraphSAGE layers for meta path based adjacency matrices
        self.sage_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            temp = nn.ModuleList()
            temp.append(SAGEConv(in_size, out_size, aggregator_type, dropout, activation=F.elu))
            temp.append(SAGEConv(out_size, out_size, aggregator_type, dropout, activation=F.elu))
            self.sage_layers.append(temp)
        self.semantic_attention = SemanticAttention(in_size=out_size)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)     # 将meta-path转换成元组形式

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:      # 第一次，建立一张metapath下的异构图
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                    g, meta_path)       # 构建异构图的邻居
        # self._cached_coalesced_graph 多个metapath下的异构图
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]       # meta-path下的节点邻居图
            semantic_embeddings.append(self.sage_layers[i][1](new_g, self.sage_layers[i][0](new_g, h)).flatten(1))     # 采样和聚合
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)            # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)           # (N, D * K)


class HGSAGE(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, dropout, aggregator_type):
        super(HGSAGE, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HGSAGELayer(meta_paths, in_size, hidden_size, dropout, aggregator_type))
        self.predict = nn.Linear(hidden_size, out_size)

    def forward(self, g, h, args):
        for gnn in self.layers:
            h = gnn(g, h)
        pd.DataFrame(h.cpu().detach().numpy()).to_csv(f"{args['dataset']}_embeddings_{args['aggregator_type']}+g.csv",
                                                      index=False, header=False)
        return self.predict(h)

