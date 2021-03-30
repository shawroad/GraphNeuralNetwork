"""
@file   : data_process.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-03-30
"""
import os
import pickle
import torch
import numpy as np
from collections import namedtuple
from torch_sparse import coalesce
from torch_geometric.utils import loop
from itertools import repeat


def read_data(path):
    name = os.path.basename(path)
    if name == "ind.cora.test.index":
        out = np.genfromtxt(path, dtype="int64")
        return out
    else:
        out = pickle.load(open(path, "rb"), encoding="latin1")
        out = out.toarray() if hasattr(out, "toarray") else out
        return out


def process_data(path):
    # 首先加载八种数据
    filenames = ["ind.cora.{}".format(name) for name in ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]
    x, tx, allx, y, ty, ally, graph, test_index = [read_data(os.path.join(path, file)) for file in filenames]
    # 这里的x是allx的子集，具体数据说明请参考../data/readme.txt

    train_index = np.arange(y.shape[0])   # 前140个节点为训练节点
    valid_index = np.arange(y.shape[0], y.shape[0]+500)  # 140-540为验证节点
    sorted_test_index = sorted(test_index)
    # 将所有数据拼接 也就是加入了测试集
    x = np.concatenate((allx, tx), axis=0)
    y = np.concatenate((ally, ty), axis=0).argmax(axis=1)   # 将one-hot标签转为标量显示

    x[test_index] = x[sorted_test_index]
    y[test_index] = y[sorted_test_index]

    num_nodes = x.shape[0]   # 总的节点数
    train_mask = np.zeros(num_nodes, dtype=np.bool)   # 标记哪些是训练节点
    valid_mask = np.zeros(num_nodes, dtype=np.bool)   # 标记哪些是验证节点
    test_mask = np.zeros(num_nodes, dtype=np.bool)   # 标记哪些是测试节点

    train_mask[train_index] = True
    valid_mask[valid_index] = True
    test_mask[test_index] = True

    # 构图
    edge_index = construct_edge(graph, num_nodes)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return Data(x=x, y=y, edge_index=edge_index,
                train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)


def construct_edge(graph_dict, num_nodes):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    edge_index, _ = loop.remove_self_loops(edge_index)   # 去除重复的边
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index


if __name__ == '__main__':
    Data = namedtuple('Data', ['x', 'y', 'edge_index', 'train_mask', 'valid_mask', 'test_mask'])
    path = '../data/cora/raw/'
    data = process_data(path)

    save_file = os.path.join(path, 'cora_data_processed_cached.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(data, f)