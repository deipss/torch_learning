import os.path
import torch
import torch.nn as nn
import platform
import numpy as np
import random

from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, MixHopConv, DirGNNConv, AntiSymmetricConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import TUDataset
import time

_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data_path = '/data/ai_data' if platform.system() == 'Linux' else '../data'
seed = 1024
torch.manual_seed(seed)
torch.manual_seed(seed)  # 为当前CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
np.random.seed(seed)  # 为NumPy设置随机种子
random.seed(seed)  # 为Python的random模块设置随机种子
import argparse
from basics import *

#########################################################################
# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="Process some files.")
# 添加位置参数,位置参数（positional arguments）是指那些没有前缀（如 - 或 --）的命令行参数。它们通常用于指定必填的参数，顺序固定，且必须按顺序提供。
# parser.add_argument('filename', type=str, help='The name of the file to process')
# 添加可选参数
parser.add_argument('--name', type=str, default='mlp')
parser.add_argument('--gname', type=str, default='mlp')
# https://chrsmrrs.github.io/datasets/docs/datasets/
parser.add_argument('--ds', type=str, default='MUTAG', help='IMDB-BINARY,REDDIT-BINARY,PROTEINS')
parser.add_argument('--max_acc', type=float, default=0.01)
parser.add_argument('--ep', type=int, default=1000)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--lr', default=1e-2, type=float, help='Learning rate')
parser.add_argument('--drop', type=float, default=0.7)
parser.add_argument('--loss', type=float, default=0.001)
parser.add_argument('--dim', type=int, default=64)
parser.add_argument('--h_layer', type=int, default=2)
parser.add_argument('--min_acc', type=int, default=0.52)
parser.add_argument('--debug', type=bool, default=False)
# 解析命令行参数
args = parser.parse_args()


#########################################################################

def load_data(start_index=0):
    dataset = TUDataset(root=os.path.join(data_path, 'TUDataset'), name=args.ds)
    dataset.to(device=_device)
    # Gather some statistics about the first graph.

    dataset = dataset.shuffle()
    all_len = len(dataset)
    split_len = all_len // 5

    gap_start = start_index * split_len
    gap_end = (start_index + 1) * split_len

    train_dataset = []
    train_dataset.append(dataset[:gap_start])
    train_dataset.append(dataset[gap_end:])
    train_dataset = [item for sublist in train_dataset for item in sublist]
    test_dataset = dataset[gap_start: gap_end]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader, dataset


def get_block_model(model_name, feature, hidden_channels):
    # GCNConv, GATConv, TransformerConv, MixHopConv, DirGNNConv, AntiSymmetricConv
    if model_name == 'GCNConv':
        return GCNConv(feature, hidden_channels)
    elif model_name == 'GATConv':
        return GATConv(feature, hidden_channels)
    elif model_name == 'TransformerConv':
        return TransformerConv(feature, hidden_channels)
    else:
        print(f'f model not found,model_name:{model_name}')
        return None


class BlockGNN(torch.nn.Module):
    def __init__(self, hidden_channels, dataset, hidden_layer: int, model_name: str, res_graph=None):
        super(BlockGNN, self).__init__()
        self.to_hidden = get_block_model(model_name, dataset.num_node_features, hidden_channels)
        self.sequence = nn.Sequential()

        for i in range(hidden_layer):
            self.sequence.add_module(f'{model_name}{i}',
                                     get_block_model(model_name, hidden_channels,
                                                     hidden_channels))
        self.lin = nn.Linear(hidden_channels, dataset.num_classes)
        self.res_graph = res_graph

    def forward(self, x, edge_index, batch, graph_hidden=None):
        # 1. Obtain node embeddings
        x = self.to_hidden(x, edge_index)
        x = x.relu()
        # 2. deep train layer
        for model in self.sequence:
            x = F.relu(model(x, edge_index))

        # 3. Readout layer
        global_mean = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        if self.res_graph and graph_hidden is not None:  # use preview train result of graph embedding
            global_mean = global_mean + graph_hidden
        # 4. Apply a final classifier
        y = F.dropout(global_mean, p=args.drop, training=self.training)
        y = self.lin(y)

        return y, global_mean


class ResBlockGnn(torch.nn.Module):
    def __init__(self, hidden_channels, dataset, hidden_layer, model_name, res_graph=None):
        super(ResBlockGnn, self).__init__()
        self.to_hidden = get_block_model(model_name, dataset.num_node_features, hidden_channels)
        self.sequence = nn.Sequential()
        for i in range(hidden_layer):
            self.sequence.add_module(f'{model_name}{i}',
                                     get_block_model(model_name, hidden_channels,
                                                     hidden_channels))
        self.lin = nn.Linear(hidden_channels, dataset.num_classes)
        self.res_graph = res_graph

    def forward(self, x, edge_index, batch, graph_hidden=None):
        # 1. Obtain node embeddings
        x_cur = F.relu(self.to_hidden(x, edge_index))

        x_pre = torch.zeros(x_cur.shape, device=_device)
        for i, m in enumerate(self.sequence):
            x_temp = x_cur
            x_cur = F.relu(m(x_cur + x_pre, edge_index))
            x_pre = x_temp

        global_mean = global_mean_pool(x_cur, batch)  # [batch_size, hidden_channels]
        if self.res_graph and graph_hidden is not None:  # use preview train result of graph embedding
            global_mean = global_mean + graph_hidden
        # 3. Apply a final classifier
        y = F.dropout(global_mean, p=args.drop, training=self.training)
        y = self.lin(y)

        return y, global_mean


class ResGraphBlockGnn(torch.nn.Module):
    def __init__(self, hidden_channels, dataset, hidden_layer, model_name):
        super(ResGraphBlockGnn, self).__init__()
        self.inner_model1 = ResBlockGnn(hidden_channels, dataset, hidden_layer, model_name, res_graph=True)
        self.inner_model2 = ResBlockGnn(hidden_channels, dataset, hidden_layer, model_name, res_graph=True)
        self.lin = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        _, g = self.inner_model1(x, edge_index, batch)
        y, g = self.inner_model2(x, edge_index, batch, g)
        y = self.lin(g)
        return y, g


class GraphBlockGnn(torch.nn.Module):
    def __init__(self, hidden_channels, dataset, hidden_layer, model_name):
        super(GraphBlockGnn, self).__init__()
        self.inner_model1 = BlockGNN(hidden_channels, dataset, hidden_layer, model_name, res_graph=True)
        self.inner_model2 = BlockGNN(hidden_channels, dataset, hidden_layer, model_name, res_graph=True)
        self.lin = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        _, g = self.inner_model1(x, edge_index, batch)
        y, g = self.inner_model2(x, edge_index, batch, g)
        y = self.lin(g)
        return y, g


def train_model(start_index):
    train_loader, test_loader, dataset = load_data(start_index)
    if args.gname == 'BlockGNN':
        model = BlockGNN(hidden_channels=args.dim, dataset=dataset, hidden_layer=args.h_layer, model_name=args.name)
    elif args.gname == 'ResBlockGnn':
        model = ResBlockGnn(hidden_channels=args.dim, dataset=dataset, hidden_layer=args.h_layer, model_name=args.name)
    elif args.gname == 'ResGraphBlockGnn':
        model = ResGraphBlockGnn(hidden_channels=args.dim, dataset=dataset, hidden_layer=args.h_layer,
                                 model_name=args.name)
    elif args.gname == 'GraphBlockGnn':
        model = GraphBlockGnn(hidden_channels=args.dim, dataset=dataset, hidden_layer=args.h_layer,
                              model_name=args.name)
    else:
        print(f'no model name {args.gname}')
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device=_device)
    criterion.to(device=_device)

    def train():
        model.train()
        min_loss = 1e6
        for data in train_loader:  # Iterate in batches over the training dataset.
            out, _ = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            min_loss = min(loss, min_loss)
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
        return min_loss

    def test(loader):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out, _ = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    epoch = 0
    max_acc = -1
    records = []
    loss = 1e6
    while loss > 0.0001 and epoch < args.ep:
        loss = train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        if (max_acc < test_acc):
            max_acc = test_acc
            if (max_acc > args.min_acc):
                save_model(model=model, is_debug=args.debug, **vars(args))
        if epoch % 5 == 0:
            print_loss(epoch=epoch, is_debug=args.debug, loss=loss.item(), test_acc=test_acc, train_acc=train_acc,
                       max_acc=max_acc)
            records.append({'epoch': epoch, 'loss': loss.item(), 'test_acc': test_acc, 'train_acc': train_acc})
        epoch += 1
    args.max_acc = max_acc
    save_json(records=records, is_debug=args.debug, **vars(args))
    return max_acc


def download_dataset():
    # ['MUTAG', 'DD', 'COIL-RAG', 'MSRC_9', 'AIDS', 'Mutagenicity']
    # ['MUTAG', 'DD', 'COIL-RAG', 'MSRC_9', 'AIDS', 'naphthalene', 'QM9', 'salicylic_acid', 'Mutagenicity']:
    for i in ['MUTAG', 'DD', 'COIL-RAG', 'MSRC_9', 'AIDS', 'Mutagenicity']:
        args.ds = i
        load_data()


def debug():
    """
    MixHopConv有问题，存在一个power，需要对隐藏层的输出层的形状，作调整
    """
    import traceback
    args.debug = True
    args.ep = 1
    results = []
    models = ['GCNConv', 'GATConv', 'TransformerConv']
    g_models = ['BlockGNN', 'ResBlockGnn', 'ResGraphBlockGnn', 'GraphBlockGnn']
    start_index_list = [0, 1, 2, 3, 4]
    for m in models:
        for gm in g_models:
            args.ds = 'MUTAG'
            args.name = m
            args.dim = 8
            args.h_layer = 1
            args.gname = gm
            start_time = time.time()
            acc_list = []
            for start_index in start_index_list:
                try:
                    acc = train_model(start_index)
                    acc_list.append(acc)
                except Exception as e:
                    print(e)
            execution_time = time.time() - start_time
            avg_acc = sum(acc_list) / len(acc_list)
            line = f'gm={gm},model={m},h={1},ds=mutag,dim={16},acc={avg_acc:.5f},acc0={acc_list[0]:.5f},acc1={acc_list[1]:.5f},acc2={acc_list[2]:.5f},acc3={acc_list[3]:.5f},acc4={acc_list[4]:.5f},execution_time={execution_time:.5f}'
            print(line)
            results.append(line)
            fp = 'graph_classify_v2_5_fold_1118_debug.txt'
            with open(fp, 'a') as file:
                file.writelines(line + '\n')
    save_records(records=results, is_debug=args.debug, file_name='graph_class')


if __name__ == '__main__':
    # debug()
    # args.debug = True
    # args.ep = 1
    models = ['GCNConv', 'GATConv', 'TransformerConv']
    g_models = ['ResBlockGnn', 'BlockGNN', 'ResGraphBlockGnn', 'GraphBlockGnn']
    ds_list = ['MUTAG', 'DD', 'MSRC_9', 'AIDS']
    h_list = [1, 2, 3, 4, 5, 6, 7]
    start_index_list = [0, 1, 2, 3, 4]
    acc = 0
    results = []
    line = ''
    for ds in ds_list:
        for dim in [32, 64]:
            for h in h_list:
                for m in models:
                    for gm in g_models:
                        args.ds = ds
                        args.name = m
                        args.dim = dim
                        args.gname = gm
                        args.h_layer = h
                        start_time = time.time()
                        acc_list = []
                        for start_index in start_index_list:
                            try:
                                acc = train_model(start_index)
                                acc_list.append(acc)
                            except Exception as e:
                                print(f'gm={gm},model={m},h={h},ds={ds},dim={dim},e={e}')
                        execution_time = time.time() - start_time
                        avg_acc = sum(acc_list) / len(acc_list)
                        line = f'gm={gm},model={m},h={h},ds={ds},dim={dim},acc={avg_acc:.5f},acc0={acc_list[0]:.5f},acc1={acc_list[1]:.5f},acc2={acc_list[2]:.5f},acc3={acc_list[3]:.5f},acc4={acc_list[4]:.5f},execution_time={execution_time:.5f}'
                        print(line)
                        results.append(line)
                        fp = 'graph_classify_v2_5_fold_1118.txt'
                        with open(fp, 'a') as file:
                            file.writelines(line + '\n')
    save_records(records=results, is_debug=args.debug, file_name='graph_class')
