import networkx as nx
import numpy as np
from scipy import sparse
import torch


def gravity_matrix(edge_list):
    """根据邻接矩阵计算引力矩阵"""
    graph = nx.Graph()
    graph.add_edges_from(edge_list)   # 生成图
    distance_m = nx.floyd_warshall_numpy(graph)    # 生成距离矩阵
    # np.save("distance_m.npy", distance_m)
    # distance_m = np.load("distance_m.npy")
    distance_m = distance_m + np.diag([np.inf] * distance_m.shape[0])     # 自己到自己的距离为无穷

    degrees = [d for n, d in nx.degree(graph)]   # 得到每个节点的度
    temp = np.reshape(np.array(degrees), (len(degrees), 1))
    degrees_mul = np.dot(temp, temp.T)  # 度相乘矩阵

    gravity_m = degrees_mul / distance_m   # 生成引力矩阵
    # return torch.FloatTensor(gravity_m)
    gravity_of_nodes = np.sum(gravity_m, axis=1)        # 行求和
    mean = np.mean(gravity_of_nodes)
    gravities = (gravity_of_nodes >= mean).astype(int)
    result = [0] * (max(graph.nodes()) + 1)
    nodes = list(graph.nodes())
    for i in range(len(nodes)):
        result[nodes[i]] = gravities[i]
    return np.array(result)



