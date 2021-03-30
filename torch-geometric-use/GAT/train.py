"""
@file   : train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-03-30
"""
import pickle
import os
from collections import namedtuple
import torch
from model import GATModel
import torch.nn.functional as F
from config import set_args


def evaluate(data):
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    for mask in [data.train_mask, data.valid_mask, data.test_mask]:
        pred = out[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


if __name__ == '__main__':
    args = set_args()
    # 加载数据
    Data = namedtuple('Data', ['x', 'y', 'edge_index', 'train_mask', 'valid_mask', 'test_mask'])
    save_file = os.path.join(args.save_data_path, 'cora_data_processed_cached.pkl')
    dataset = pickle.load(open(save_file, 'rb'))

    x = dataset.x
    # x = dataset.x / dataset.x.sum(1, keepdim=True).clamp(min=1)   # 是否进行归一化
    edge_index = dataset.edge_index

    # 实例化模型
    model = GATModel()
    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_acc = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        optimizer.zero_grad()
        if torch.cuda.is_available():
            x, edge_index = x.cuda(), edge_index.cuda()
        out = model(x, edge_index)
        loss = F.nll_loss(out[dataset.train_mask], dataset.y[dataset.train_mask])
        loss.backward()
        optimizer.step()
        train_acc, val_acc, test_acc = evaluate(dataset)
        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
              f'Test: {test_acc:.4f}')

        if test_acc > best_acc:
            best_acc = test_acc
            # 保存模型
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            os.makedirs(args.save_model, exist_ok=True)
            output_model_file = os.path.join(args.save_model, "best_pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)


