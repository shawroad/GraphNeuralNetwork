"""
@file   : train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-03-19
"""
import os
import itertools
import torch
import pickle
from torch import nn
from torch import optim
import numpy as np
from model import GCNNet
from scipy import sparse
from config import set_args
from collections import namedtuple
import matplotlib.pyplot as plt


def test(mask):
    model.eval()
    with torch.no_grad():
        logits = model(tensor_adjacency, tensor_x)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, tensor_y[mask]).float().mean()
    return accuarcy, test_mask_logits.cpu().numpy(), tensor_y[mask].cpu().numpy()


def train(model):
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = tensor_y[train_mask]
    train_acc_list = []
    val_acc_list = []
    for epoch in range(args.Epochs):
        logits = model(tensor_adjacency, tensor_x)  # 前向传播
        train_mask_logits = logits[train_mask]   # 只选择训练节点进行监督
        loss = criterion(train_mask_logits, train_y)    # 计算损失值
        optimizer.zero_grad()
        loss.backward()     # 反向传播计算参数的梯度
        optimizer.step()    # 使用优化方法进行梯度更新
        train_acc, _, _ = test(train_mask)     # 计算当前模型训练集上的准确率
        val_acc, _, _ = test(valid_mask)     # 计算当前模型在验证集上的准确率

        # 记录训练过程中损失值和准确率的变化，用于画图
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch, loss.item(), train_acc.item(), val_acc.item()))
    return loss_history, val_acc_history


def normalization(adjacency):
    """计算 L=D^-0.5 * (A+I) * D^-0.5"""
    adjacency += sparse.eye(adjacency.shape[0])    # 增加自连接
    degree = np.array(adjacency.sum(1))
    d_hat = sparse.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()


if __name__ == "__main__":
    args = set_args()

    Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'valid_mask', 'test_mask'])

    # 加载数据
    save_file = os.path.join(args.data_path, 'cors_data_processed_cached.pkl')
    dataset = pickle.load(open(save_file, 'rb'))

    # 对节点上的特征向量进行归一化
    node_feature = dataset.x / dataset.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
    num_nodes, input_dim = node_feature.shape

    tensor_x, tensor_y = torch.from_numpy(node_feature), torch.from_numpy(dataset.y)
    train_mask, valid_mask, test_mask = torch.from_numpy(dataset.train_mask), torch.from_numpy(dataset.valid_mask), torch.from_numpy(dataset.test_mask)

    # 对邻接矩阵规范化
    normalize_adjacency = normalization(dataset.adjacency)
    # print(normalize_adjacency.shape)   # (2708, 2708)
    indices = torch.from_numpy(np.asarray([normalize_adjacency.row, normalize_adjacency.col]).astype('int64')).long()
    values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, (num_nodes, num_nodes))

    # 模型定义：Model, Loss, Optimizer
    model = GCNNet(input_dim)

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_history, val_acc_history = train(model)
    plt.plot(loss_history)
    plt.plot(val_acc_history)
    plt.show()





