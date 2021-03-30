"""
@file   : data_process.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-03-19
"""
import os
import pickle
import itertools
from scipy import sparse
import numpy as np
from collections import namedtuple


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
    sorted_test_index = sorted(test_index)    # 测试集索引

    # 将所有数据拼接 也就是加入了测试集
    x = np.concatenate((allx, tx), axis=0)
    y = np.concatenate((ally, ty), axis=0).argmax(axis=1)   # 将one-hot标签转为标量显示

    # x[test_index] = x[sorted_test_index]
    # y[test_index] = y[sorted_test_index]

    num_nodes = x.shape[0]   # 总的节点数
    train_mask = np.zeros(num_nodes, dtype=np.bool)   # 标记哪些是训练节点
    valid_mask = np.zeros(num_nodes, dtype=np.bool)   # 标记哪些是验证节点
    test_mask = np.zeros(num_nodes, dtype=np.bool)   # 标记哪些是测试节点

    train_mask[train_index] = True
    valid_mask[valid_index] = True
    test_mask[test_index] = True

    # 构图
    adjacency = build_adjacency(graph)
    print('Node feature shape:', x.shape)
    print('Node label shape:', y.shape)
    print('Adjacency shape:', adjacency.shape)
    print('train node num:', train_mask.sum())
    print('valid node num:', valid_mask.sum())
    print('test node num:', test_mask.sum())

    return Data(x=x, y=y, adjacency=adjacency,
                train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)


def build_adjacency(graph):
    # 构图
    edge_index = []
    # print(adj_dict)
    num_nodes = len(graph)

    for src, dst in graph.items():
        # 字典的key是起点，字典中的value是终点
        edge_index.extend([src, v] for v in dst)
        edge_index.extend([v, src] for v in dst)   # 相当于无向

    # 去掉重复边
    edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
    edge_index = np.asarray(edge_index)
    # 压缩存储图
    adjacency = sparse.coo_matrix((np.ones(len(edge_index)),
                                   (edge_index[:, 0], edge_index[:, 1])),
                                  shape=(num_nodes, num_nodes), dtype='float32')
    return adjacency


if __name__ == '__main__':
    Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'valid_mask', 'test_mask'])
    path = '../data/cora'
    data = process_data(path)

    save_file = os.path.join(path, 'cors_data_processed_cached.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(data, f)

