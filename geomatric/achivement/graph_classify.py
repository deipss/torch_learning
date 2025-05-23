import os.path
import torch
import torch.nn as nn
import platform
import numpy as np
import random

from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv,TransformerConv,MixHopConv,DirGNNConv,AntiSymmetricConv
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
# https://chrsmrrs.github.io/datasets/docs/datasets/
parser.add_argument('--ds', type=str, default='MUTAG', help='IMDB-BINARY,REDDIT-BINARY,PROTEINS')
parser.add_argument('--max_acc', type=float, default=0.01)
parser.add_argument('--ep', type=int, default=2048)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--lr', default=1e-2, type=float, help='Learning rate')
parser.add_argument('--drop', type=float, default=0.7)
parser.add_argument('--loss', type=float, default=0.001)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--min_acc', type=int, default=0.52)
parser.add_argument('--debug', type=bool, default=False)
# 解析命令行参数
args = parser.parse_args()


#########################################################################

def load_data():
    dataset = TUDataset(root=os.path.join(data_path, 'TUDataset'), name=args.ds)
    dataset.to(device=_device)
    print()
    print(
        f'ds: {dataset},Number of graphs: {len(dataset)},Number of features: {dataset.num_features},Number of classes: {dataset.num_classes}')
    data = dataset[0]  # Get the first graph object.
    print(data)
    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes},')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    dataset = dataset.shuffle()
    all_len = len(dataset)
    len10 = all_len // 10
    train_dataset = dataset[:all_len - 2 * len10]
    val_dataset = dataset[all_len - 2 * len10:all_len - len10]
    test_dataset = dataset[all_len - len10:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()
    return train_loader, test_loader, dataset


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # self.conv4 = GCNConv(hidden_channels, hidden_channels)
        # self.conv5 = GCNConv(hidden_channels, hidden_channels)
        # self.conv6 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        # x = x.relu()
        # x = self.conv4(x, edge_index)
        # x = x.relu()
        # x = self.conv5(x, edge_index)
        # x = x.relu()
        # x = self.conv6(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=args.drop, training=self.training)
        x = self.lin(x)

        return x


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels=dataset.num_node_features, out_channels=hidden_channels, heads=args.heads)
        self.conv2 = GATConv(in_channels=hidden_channels * args.heads, out_channels=hidden_channels, heads=args.heads)
        self.conv3 = GATConv(in_channels=hidden_channels * args.heads, out_channels=hidden_channels, heads=1)
        # self.conv4 = GCNConv(hidden_channels, hidden_channels)
        # self.conv5 = GCNConv(hidden_channels, hidden_channels)
        # self.conv6 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        # x = x.relu()
        # x = self.conv4(x, edge_index)
        # x = x.relu()
        # x = self.conv5(x, edge_index)
        # x = x.relu()
        # x = self.conv6(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=args.drop, training=self.training)
        x = self.lin(x)

        return x


class ResGCN(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super(ResGCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.conv6 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = F.relu(self.conv3(x2 + x1, edge_index))
        x4 = F.relu(self.conv4(x3 + x2, edge_index))
        x5 = F.relu(self.conv5(x3 + x4, edge_index))
        x6 = self.conv6(x4 + x5, edge_index)

        # 2. Readout layer
        x6 = global_mean_pool(x6, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x6 = F.dropout(x6, p=args.drop, training=self.training)
        y = self.lin(x6)

        return y


class ResGAT(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super(ResGAT, self).__init__()
        self.conv1 = GATConv(in_channels=dataset.num_features, out_channels=hidden_channels, heads=args.heads)
        self.conv2 = GATConv(in_channels=hidden_channels * args.heads, out_channels=hidden_channels, heads=args.heads)
        self.conv3 = GATConv(in_channels=hidden_channels * args.heads, out_channels=hidden_channels, heads=args.heads)
        self.conv4 = GATConv(in_channels=hidden_channels * args.heads, out_channels=hidden_channels, heads=args.heads)
        self.conv5 = GATConv(in_channels=hidden_channels * args.heads, out_channels=hidden_channels, heads=args.heads)
        self.conv6 = GATConv(in_channels=hidden_channels * args.heads, out_channels=hidden_channels, heads=1)
        self.lin = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = F.relu(self.conv3(x2 + x1, edge_index))
        x4 = F.relu(self.conv4(x3 + x2, edge_index))
        x5 = F.relu(self.conv5(x3 + x4, edge_index))
        x6 = self.conv6(x4 + x5, edge_index)

        # 2. Readout layer
        x6 = global_mean_pool(x6, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x6 = F.dropout(x6, p=args.drop, training=self.training)
        y = self.lin(x6)

        return y


def train_model():
    train_loader, test_loader, dataset = load_data()
    if args.name == 'GCN':
        model = GCN(hidden_channels=args.hidden, dataset=dataset)
    elif args.name == 'ResGCN':
        model = ResGCN(hidden_channels=args.hidden, dataset=dataset)
    elif args.name == 'GAT':
        model = GAT(hidden_channels=args.hidden, dataset=dataset)
    elif args.name == 'ResGAT':
        model = ResGAT(hidden_channels=args.hidden, dataset=dataset)
    else:
        print(f'no model name {args.name}')
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device=_device)
    criterion.to(device=_device)

    def train():
        model.train()
        min_loss = 1e6
        for data in train_loader:  # Iterate in batches over the training dataset.
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
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
            out = model(data.x, data.edge_index, data.batch)
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
        if epoch % 9 == 0:
            print_loss(epoch=epoch, is_debug=args.debug, loss=loss.item(), test_acc=test_acc, train_acc=train_acc,
                       max_acc=max_acc)
            records.append({'epoch': epoch, 'loss': loss.item(), 'test_acc': test_acc, 'train_acc': train_acc})
        epoch += 1
    args.max_acc = max_acc
    save_json(records=records, is_debug=args.debug, **vars(args))
    print(max_acc)
    return max_acc


def download_dataset():
    # ['MUTAG', 'DD', 'COIL-RAG', 'MSRC_9', 'AIDS', 'Mutagenicity']
    # ['MUTAG', 'DD', 'COIL-RAG', 'MSRC_9', 'AIDS', 'naphthalene', 'QM9', 'salicylic_acid', 'Mutagenicity']:
    for i in ['MUTAG', 'DD', 'COIL-RAG', 'MSRC_9', 'AIDS', 'Mutagenicity']:
        args.ds = i
        load_data()


if __name__ == '__main__':
    """
    name=ResGCN_ds=MUTAG_ds_split=public_max_acc=0.89474_
    name=GCN_ds=MUTAG_ds_split=public_max_acc=0.86842_
    ds=naphthalene发生了一个异常: expected scalar type Long but found Float,
    ds=QM9发生了一个异常: The size of tensor a (64) must match the size of tensor b (19) at non-singleton dimension 1,
    ds=salicylic_acid发生了一个异常: expected scalar type Long but found Float,
    toluene dataset need more memory
    Mutagenicity so long time
    COIL-RAG low precision
    """
    # args.debug = True
    # args.ep = 1
    models = ['GCN', 'ResGCN', 'GAT', 'ResGAT']
    ds_list = ['MUTAG', 'DD', 'MSRC_9', 'AIDS']
    results = []

    for ds in ds_list:
        for hi in [16, 32, 64, 128, 256]:
            for m in models:
                args.ds = ds
                args.name = m
                args.hidden = hi
                start_time = time.time()
                try:
                    acc = train_model()
                except Exception as e:
                    print(f"ds={ds},models={m}发生了一个异常: {str(e)},")
                execution_time = time.time() - start_time
                print(f'model={m},ds={ds},hidden={hi},acc={acc:.5f},execution_time={execution_time:.5f}')
                results.append(f'model={m},ds={ds},hidden={hi},acc={acc:.5f},execution_time={execution_time:.5f}')
    save_records(records=results, is_debug=args.debug, file_name='graph_class')
