"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-03-30
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    # num_train_epochs
    parser.add_argument('--num_train_epochs', default=100, type=str, help='code will operate in this gpu')
    parser.add_argument("--save_data_path", default='../data/cora/raw/', type=str,)
    parser.add_argument("--learning_rate", default=0.005, type=float,)
    parser.add_argument("--weight_decay", default=5e-4, type=float,)
    parser.add_argument("--save_model", default='./save_model', type=str,)
    args = parser.parse_args()
    return args
