"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-03-19
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/cora', help='data path')
    parser.add_argument('--Epochs', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=0.1, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="weight_decay")
    args = parser.parse_args()
    return args
