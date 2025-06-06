import os.path
import torch
import torch.nn as nn
import platform
import numpy as np
import random

from torch_geometric.loader import DataLoader
import torch.nn.functional as F
# MixHopConv, DirGNNConv, AntiSymmetricConv
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
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
torch.autograd.set_detect_anomaly(True)
import argparse
import traceback
from datetime import datetime
import json
import torch
import os
from torch.utils.tensorboard import SummaryWriter

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx


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
parser.add_argument('--ep', type=int, default=1000 * 1.5)
parser.add_argument('--lr', default=1e-2, type=float, help='Learning rate')
parser.add_argument('--drop', type=float, default=0.6)
parser.add_argument('--loss', type=float, default=0.001)
parser.add_argument('--dim', type=int, default=64)
parser.add_argument('--h_layer', type=int, default=2)
parser.add_argument('--min_acc', type=int, default=0.1)
parser.add_argument('--debug', type=bool, default=False)
# 解析命令行参数
args = parser.parse_args()
#########################################################################


separator = '__'


def print_loss(epoch=0, is_debug=False, **param):
    """
    输出loss到电脑屏障
    param = {'loss':'111',acc:'3333'}
    """
    if is_debug:
        return
    # 将时间格式化为字符串
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs = ""
    for k, v in param.items():
        if isinstance(v, float):
            logs += "{0}={1:.4f}\t".format(k, v)
        elif isinstance(v, int):
            logs += "{0}={1:03d}\t".format(k, v)
        else:
            logs += "{0}={1}\t".format(k, v)
    print(f'{formatted_time}\t epoch={epoch}\t{logs}')  # 输出格式：2023-10-21 22:30:45


def save_json(records=None, is_debug=False, **param):
    """
    保存训练记录
    """
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs = ""
    for k, v in param.items():
        if isinstance(v, float):
            logs += "{0}={1:.4f}_".format(k, v)
        elif isinstance(v, int):
            logs += "{0}={1:03d}_".format(k, v)
        else:
            logs += "{0}={1}_".format(k, v)
    f_name = separator.join([logs, formatted_time])
    f_name += '.json'
    fpath = os.path.join(data_path, 'logs', f_name)
    if is_debug:
        print("save_json f_name" + f_name)
        return f_name
    _ = {'records': records, 'param': param}
    with open(fpath, 'w') as file:
        json.dump(_, file)
    return f_name


def save_records(records=None, is_debug=False, file_name=''):
    """
    保存训练记录
    """
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    f_name = separator.join([file_name, formatted_time])
    f_name += '.json'
    fpath = os.path.join(data_path, 'records', f_name)
    _ = {'records': records}
    if is_debug:
        print("save_records fpath=" + fpath)
        return
    with open(fpath, 'w') as file:
        json.dump(_, file)
    print(f_name + " are saved")


def save_model(model=None, is_debug=False, **model_param):
    """
    保存模型
    """
    if is_debug:
        return
    formatted_time = datetime.now().strftime("%Y%m%d_%H")
    dict_format = separator.join(["{0}={1}_".format(k, v) for k, v in model_param.items()])
    f_name = separator.join([dict_format, formatted_time])
    f_name += '.pkl'
    fpath = os.path.join(data_path, 'pkls', f_name)
    torch.save(model.state_dict, fpath)
    print(f_name + ' model saved on ' + fpath)


def statistic_dataset(ds):
    dataset = TUDataset(root=os.path.join(data_path, 'TUDataset'), name=ds)
    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')


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

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

        # x = x.relu()
        # 2. deep train layer
        for model in self.sequence:
            if self.res_graph and graph_hidden is not None:
                with torch.no_grad():
                    for i, b in enumerate(batch):
                        x[i] = x[i] + graph_hidden[b]
            x = F.relu(model(x, edge_index))
            x = F.dropout(x, p=args.drop, training=self.training)

        # 3. Readout layer
        global_mean = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        if graph_hidden is not None:  # use preview train result of graph embedding
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
            if self.res_graph and graph_hidden is not None:
                with torch.no_grad():
                    for i, b in enumerate(batch):
                        x[i] = x[i] + graph_hidden[b]
            x_cur = F.relu(m(x_cur + x_pre, edge_index))
            x_cur = F.dropout(x_cur, p=args.drop, training=self.training)
            x_pre = x_temp

        global_mean = global_mean_pool(x_cur, batch)  # [batch_size, hidden_channels]
        if graph_hidden is not None:  # use preview train result of graph embedding
            global_mean = global_mean + graph_hidden
        # 3. Apply a final classifier
        y = F.dropout(global_mean, p=args.drop, training=self.training)
        y = self.lin(y)

        return y, global_mean


class CrossBlockGnn(torch.nn.Module):
    def __init__(self, hidden_channels, dataset, hidden_layer, model_name, res_graph=None):
        super(CrossBlockGnn, self).__init__()
        self.to_hidden_1 = get_block_model(model_name, dataset.num_node_features, hidden_channels)
        self.to_hidden_2 = get_block_model(model_name, dataset.num_node_features, hidden_channels)
        self.sequence = nn.Sequential()
        for i in range(hidden_layer * 2):
            self.sequence.add_module(f'{model_name}{i}',
                                     get_block_model(model_name, hidden_channels,
                                                     hidden_channels))
        self.lin = nn.Linear(hidden_channels, dataset.num_classes)
        self.res_graph = res_graph

    def forward(self, x, edge_index, batch, graph_hidden=None):
        # 1. Obtain node embeddings
        x_cur_1 = F.relu(self.to_hidden_1(x, edge_index))
        x_cur_2 = F.relu(self.to_hidden_2(x, edge_index))

        x_pre_1 = torch.zeros(x_cur_1.shape, device=_device)
        x_pre_2 = torch.zeros(x_cur_2.shape, device=_device)
        i = 0
        while i < len(self.sequence):
            x_temp_1 = x_cur_1
            x_temp_2 = x_cur_2

            x_cur_1 = F.relu(self.sequence[i](x_cur_1 + x_pre_2, edge_index))
            x_cur_1 = F.dropout(x_cur_1, p=args.drop, training=self.training)

            x_cur_2 = F.relu(self.sequence[i + 1](x_cur_2 + x_pre_1, edge_index))
            x_cur_2 = F.dropout(x_cur_2, p=args.drop, training=self.training)

            x_pre_1 = x_temp_1
            x_pre_2 = x_temp_2
            i += 2
        x_cur = x_cur_1 + x_cur_2
        global_mean = global_mean_pool(x_cur, batch)  # [batch_size, hidden_channels]
        if graph_hidden is not None:  # use preview train result of graph embedding
            global_mean = global_mean + graph_hidden
        # 3. Apply a final classifier
        y = F.dropout(global_mean, p=args.drop, training=self.training)
        y = self.lin(y)

        return y, global_mean


class GraphBlockGnn(torch.nn.Module):
    def __init__(self, hidden_channels, dataset, hidden_layer, model_name):
        super(GraphBlockGnn, self).__init__()
        self.inner_model1 = BlockGNN(hidden_channels, dataset, hidden_layer, model_name, res_graph=False)
        self.inner_model2 = BlockGNN(hidden_channels, dataset, hidden_layer, model_name, res_graph=False)
        self.lin = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        _, g = self.inner_model1(x, edge_index, batch)
        y, g = self.inner_model2(x, edge_index, batch, g)
        y = self.lin(g)
        return y, g


class ResGraphBlockGnn(torch.nn.Module):
    def __init__(self, hidden_channels, dataset, hidden_layer, model_name):
        super(ResGraphBlockGnn, self).__init__()
        self.inner_model1 = BlockGNN(hidden_channels, dataset, hidden_layer, model_name, res_graph=True)
        self.inner_model2 = BlockGNN(hidden_channels, dataset, hidden_layer, model_name, res_graph=True)
        self.inner_model3 = BlockGNN(hidden_channels, dataset, hidden_layer, model_name, res_graph=True)
        self.lin = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        _, g = self.inner_model1(x, edge_index, batch)
        y, g = self.inner_model2(x, edge_index, batch, g)
        y, g = self.inner_model3(x, edge_index, batch, g)
        y = self.lin(g)
        return y, g


class CrossGraphBlockGnn(torch.nn.Module):
    def __init__(self, hidden_channels, dataset, hidden_layer, model_name):
        super(CrossGraphBlockGnn, self).__init__()
        self.sequence = nn.Sequential()
        # todo 使用可以配套的参数
        for i in range(4):
            self.sequence.add_module(f'{model_name}{i}',
                                     BlockGNN(hidden_channels, dataset, hidden_layer, model_name, res_graph=False))
        self.lin = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        g_1, g_2 = None, None
        i = 0
        while i < len(self.sequence):
            _, global_mean_1 = self.sequence[i](x, edge_index, batch, g_1)
            _, global_mean_2 = self.sequence[i + 1](x, edge_index, batch, g_2)

            g_1 = global_mean_2
            g_2 = global_mean_1
            i += 2
        global_mean = g_1 + g_2
        y = F.dropout(global_mean, p=args.drop, training=self.training)
        y = self.lin(y)

        return y, global_mean


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
    elif args.gname == 'CrossBlockGnn':
        model = CrossBlockGnn(hidden_channels=args.dim, dataset=dataset, hidden_layer=args.h_layer,
                              model_name=args.name)
    elif args.gname == 'CrossGraphBlockGnn':
        model = CrossGraphBlockGnn(hidden_channels=args.dim, dataset=dataset, hidden_layer=args.h_layer,
                                   model_name=args.name)
    else:
        print(f'no model name {args.gname}')
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device=_device)
    criterion.to(device=_device)

    # 初始化 SummaryWriter
    log_dir = os.path.join(data_path, 'runs', f"{args.gname}_{args.name}_{args.ds}_{args.dim}_{args.h_layer}")
    writer = SummaryWriter(log_dir)

    # 保存模型结构
    data = next(iter(train_loader))
    writer.add_graph(model, (data.x.to(_device), data.edge_index.to(_device), data.batch.to(_device)))

    def train(loader):
        model.train()
        correct = 0
        cnt = 1
        data_size = len(loader.dataset)
        min_loss = 1e6
        for data in loader:  # Iterate in batches over the training dataset.
            out, _ = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.

            predict = out.argmax(dim=1)  # Use the class with the highest probability.
            correct += int((predict == data.y).sum())  # Check against ground-truth labels.
            # todo 可视化
            # plt_train = visualize_embedding(_, color=data.y)
            # writer.add_figure('EmbeddingVisualization', plt_train.gcf(), global_step=cnt)
            loss_train = criterion(out, data.y)  # Compute the loss.
            writer.add_scalar('Loss/train', loss_train.item(), epoch * data_size + cnt)
            loss_train.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            min_loss = min(loss_train, min_loss)
            cnt += 1
        return min_loss, correct / data_size

    def test(loader):
        model.eval()
        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out, _ = model(data.x, data.edge_index, data.batch)
            predict = out.argmax(dim=1)  # Use the class with the highest probability.
            correct += int((predict == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    epoch = 0
    max_acc = -1
    records = []
    loss = 1e6
    while loss > 0.0001 and epoch < args.ep:
        loss,train_acc = train(train_loader)
        test_acc = test(test_loader)
        if (max_acc < test_acc):
            max_acc = test_acc

        # 记录训练 loss 和测试 acc
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        # 记录模型参数和梯度
        for name, param in model.named_parameters():
            writer.add_histogram(f"Weights/{name}", param.data.cpu().numpy(), epoch)
            if param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad.cpu().numpy(), epoch)

        records.append({'epoch': epoch, 'loss': loss.item(), 'test_acc': test_acc, 'train_acc': train_acc})
        epoch += 1
    args.max_acc = max_acc
    f_name = save_json(records=records, is_debug=args.debug, **vars(args))

    # 关闭 SummaryWriter
    writer.close()

    return max_acc, f_name


def download_dataset():
    # ['MUTAG', 'DD', 'COIL-RAG', 'MSRC_9', 'AIDS', 'Mutagenicity']
    # ['MUTAG', 'DD', 'COIL-RAG', 'MSRC_9', 'AIDS', 'naphthalene', 'QM9', 'salicylic_acid', 'Mutagenicity']:
    for i in ['MUTAG', 'DD', 'MSRC_9', 'AIDS']:
        args.ds = i
        load_data()


def print_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated(device=_device) / 1024 / 1024 / 1024  # 转换为GB
    reserved = torch.cuda.memory_reserved(device=_device) / 1024 / 1024 / 1024  # 转换为GB
    return allocated, reserved


def count_lines(file_path):
    # 检查文件是否存在
    if os.path.exists(file_path):
        print(f"文件 {file_path} 存在。")

        # 计算文件中的行数
        with open(file_path, 'r', encoding='utf-8') as file:
            line_count = sum(1 for line in file)
        print(f"文件共有 {line_count} 行。")
        return line_count
    else:
        print(f"文件 {file_path} 不存在。")
        return 0


def debug():
    """ python graph_classify_v2.py --dim 64 --name GCNConv --ds AIDS --ep 1 --debug Ture
    MixHopConv有问题，存在一个power，需要对隐藏层的输出层的形状，作调整
    """
    args.debug = False
    args.ep = 1
    results = []
    models = ['GCNConv', 'GATConv', 'TransformerConv']
    g_models = ['CrossBlockGnn', 'CrossGraphBlockGnn', 'ResBlockGnn', 'ResGraphBlockGnn', 'BlockGNN', 'GraphBlockGnn']
    start_index_list = [0]
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
                    acc, f_name = train_model(start_index)
                    print(f'acc={acc},gm={gm}')
                    acc_list.append(acc)
                except Exception as e:
                    print(e)
                    print(f'error gm ={gm}')
                    traceback.print_exc()
            execution_time = time.time() - start_time
            avg_acc = sum(acc_list) / len(acc_list)
            line = f'gm={gm},model={m},h={1},ds=mutag,dim={16},acc={avg_acc:.5f},execution_time={execution_time:.5f}'
            print(line)


def true_train():
    global args
    # models = ['GCNConv', 'GATConv', 'TransformerConv']
    g_models = ['CrossBlockGnn', 'CrossGraphBlockGnn', 'ResBlockGnn', 'ResGraphBlockGnn', 'BlockGNN', 'GraphBlockGnn']
    # ds_list = ['MUTAG', 'AIDS', 'DD', 'MSRC_9']
    start_index_list = [0, 1, 2, 3, 4]
    results = []
    count = 1
    filename = "graph_classify_v4_5_fold_0421_" + "_".join([args.ds, args.name, str(args.dim)])
    fp = '../records/' + filename
    print(f'file name ={fp}')
    exist_row = count_lines(fp)
    for h in [1, 2, 3, 4, 5]:
        for gm in g_models:
            args.gname = gm
            args.h_layer = h
            if count < exist_row:
                print(f'{count} < {exist_row} ,continue')
                count += 1
                continue
            start_time = time.time()
            accuracies = []
            files = []
            for start_index in start_index_list:
                try:
                    acc, f_name = train_model(start_index)
                    accuracies.append(acc)
                    files.append(f_name)
                except Exception as e:
                    print(f'gm={gm},model={args.name},h={h},ds={args.ds},dim={args.dim},e={e}')
            execution_time = time.time() - start_time
            avg_acc = sum(accuracies) / len(accuracies)
            line = (
                f'gm={gm},model={args.name},h={h},ds={args.ds},dim={args.dim},'
                f'acc={avg_acc:.5f}'
                f',acc0={accuracies[0]:.5f}'
                f',acc1={accuracies[1]:.5f}'
                f',acc2={accuracies[2]:.5f}'
                f',acc3={accuracies[3]:.5f}'
                f',acc4={accuracies[4]:.5f}'
                f',f0={files[0]}'
                f',f1={files[1]}'
                f',f2={files[2]}'
                f',f3={files[3]}'
                f',f4={files[4]}'
                f',execution_time={execution_time:.3f}'
            )
            print(line)
            results.append(line)
            with open(fp, 'a') as file:
                file.writelines(line + '\n')
            save_records(records=results, is_debug=args.debug, file_name=filename)
    send_email(args.ds + " " + args.dim, 'DONE')


def pre_check_train():
    global args
    # models = ['GCNConv', 'GATConv', 'TransformerConv']
    g_models = [ 'BlockGNN', 'GraphBlockGnn']
    # ds_list = ['MUTAG', 'AIDS', 'DD', 'MSRC_9']
    start_index_list = [0, 1, 2, 3, 4]
    results = []
    args.debug = True
    args.ds = 'MUTAG'
    args.ep = 20
    args.name = 'GATConv'
    args.dim = 8
    args.h_layer = 3
    count = 1
    filename = "pre_graph_classify_v3_5_fold_0421_" + "_".join([args.ds, args.name, str(args.dim)])
    fp = '../records/' + filename
    print(f'file name ={fp}')
    exist_row = count_lines(fp)
    for h in [1, 2, 3, 4, 5]:
        for gm in g_models:
            args.gname = gm
            args.h_layer = h
            if count < exist_row:
                print(f'{count} < {exist_row} ,continue')
                count += 1
                continue
            start_time = time.time()
            accuracies = []
            files = []
            for start_index in start_index_list:
                try:
                    acc, f_name = train_model(start_index)
                    accuracies.append(acc)
                    files.append(f_name)
                except Exception as e:
                    print(f'Exception when training:gm={gm},model={args.name},h={h},ds={args.ds},dim={args.dim},e={e}')
            execution_time = time.time() - start_time
            avg_acc = sum(accuracies) / len(accuracies)
            line = (
                f'gm={gm},model={args.name},h={h},ds={args.ds},dim={args.dim},'
                f'acc={avg_acc:.5f}'
                f',acc0={accuracies[0]:.5f}'
                f',acc1={accuracies[1]:.5f}'
                f',acc2={accuracies[2]:.5f}'
                f',acc3={accuracies[3]:.5f}'
                f',acc4={accuracies[4]:.5f}'
                f',f0={files[0]}'
                f',f1={files[1]}'
                f',f2={files[2]}'
                f',f3={files[3]}'
                f',f4={files[4]}'
                f',execution_time={execution_time:.3f}'
            )
            print(line)
            results.append(line)
            with open(fp, 'a') as file:
                file.writelines(line + '\n')
            save_records(records=results, is_debug=args.debug, file_name=filename)


def debug_one():
    args.debug = True
    args.ep = 1
    results = []
    models = ['GCNConv', 'GATConv', 'TransformerConv']
    g_models = ['ResGraphBlockGnn', 'CrossGraphBlockGnn']
    start_index_list = [0]
    args.ds = 'MUTAG'
    args.name = 'GCNConv'
    args.dim = 8
    args.h_layer = 3
    args.gname = 'CrossGraphBlockGnn'
    start_time = time.time()
    acc_list = []
    for start_index in start_index_list:
        try:
            acc, f_name = train_model(start_index)
            acc_list.append(acc)
            print(f'acc={acc},gm={args.gname}')
        except Exception as e:
            print(e)
            print(f'error gm ={args.gname}')
            traceback.print_exc()
    print(acc_list)


def tools():
    shell = 'nohup python graph_classify_v2.py --dim {dim} --name {m} --ds {ds} > graph_classify_v3_{dim}_{m}_{ds}.log 2>&1 &'
    models = ['GCNConv', 'GATConv', 'TransformerConv']
    ds_list = ['MUTAG', 'AIDS', 'DD', 'MSRC_9']
    for ds in ds_list:
        for dim in [32, 64]:
            for m in models:
                print(shell.format(dim=dim, m=m, ds=ds))


# 使用示例    send_email('MUTUD', "DONE")
def send_email(subject, body):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    try:
        # 创建一个多部分的邮件对象
        msg = MIMEMultipart()
        from_mail = 'huxl@ltzy.edu.cn'
        to_addrs = 'deipss@qq.com'
        msg['From'] = from_mail
        msg['To'] = to_addrs
        msg['Subject'] = subject + ' GNN AI Train DONE'

        # 添加邮件正文
        msg.attach(MIMEText(body, 'plain'))

        # 连接到SMTP服务器
        server = smtplib.SMTP('smtp.qiye.163.com', 587)
        server.starttls()  # 启用TLS加密
        server.login(from_mail, 'Y9ugNr9csf6hvEHN')  # 登录到邮箱

        # 发送邮件
        server.sendmail(from_mail, to_addrs, msg.as_string())
        server.close()
    except Exception as e:
        print(e)


def summary_writer_demo():
    global Net, step
    # 创建SummaryWriter实例，指定日志保存路径
    writer = SummaryWriter('runs/demo_experiment')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    # 模拟训练过程，记录100个迭代的数据
    for step in range(110):
        # 生成一些随机的loss和accuracy数据
        loss = random.random() * 2  # 随机生成一个0到2之间的数作为loss
        accuracy = random.random()  # 随机生成一个0到1之间的数作为accuracy

        # 使用add_scalar方法将数据写入日志
        writer.add_scalar('Loss/train', loss, step)
        writer.add_scalar('Loss/test', loss, step)
        writer.add_scalar('Accuracy/train', accuracy, step)
        writer.add_scalar('Accuracy/test', accuracy, step)
        writer.add_scalar('Accuracy/train/a', accuracy, step)
        writer.add_pr_curve('Accuracy/pr', accuracy, step)
        # 记录每个层的权重分布
        for name, param in net.named_parameters():
            writer.add_histogram(
                tag=f"Weights/{name}",
                values=param.clone().detach().cpu().numpy(),
                global_step=step
            )

        # 记录梯度分布（需确保参数可训练）
        for name, param in net.named_parameters():
            # if param.grad is not None:
            writer.add_histogram(
                tag=f"Gradients/{name}",
                values=param.clone().detach().cpu().numpy(),
                global_step=step
            )

        # 生成一张随机的示例图片（假设为3通道，28x28的RGB图像）
        # 注意：TensorBoard要求图像格式为 [C, H, W]，且像素值在 [0, 1] 或 [-1, 1] 范围内
        image = torch.rand(3, 28, 28)  # 生成一个随机的3通道28x28图像（假设使用PyTorch张量）
        writer.add_image('Training_Images', image, global_step=step)
    images = torch.randn(1, 28, 28)
    # 关闭writer，确保所有内容都被正确写入到磁盘上
    writer.add_graph(net, images)
    writer.close()





def visualize_graph(data, color):
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()


def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    return plt

if __name__ == '__main__':
    pre_check_train()
