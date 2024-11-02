import os.path
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse

from basics import *

_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data_root_path = '/data/ai_data' if platform.system() == 'Linux' else '../data'
seed = 1024
torch.manual_seed(seed)
torch.manual_seed(seed)  # 为当前CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
np.random.seed(seed)  # 为NumPy设置随机种子
random.seed(seed)  # 为Python的random模块设置随机种子

"""
# python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
# python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
# pip uninstall pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv 
libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
- https://stackoverflow.com/questions/77664847/although-installed-pyg-lib-successfully-getting-error-while-importing
- Reference: What's the difference between "pip install" and "python -m pip install"?



TransformerConv
AGNNConv
FastRGCNConv
GCN

10-29 
在Core数据集上，使用Conv1d做残差网络的效果并不好 
在Core数据集上，使用隐藏层直接相加，作为残差，的效果比不上，直接输入上一层的，不做残差
在随机生成的数据集上， 使用隐藏层直接相加，作为残差，的效果优胜于，直接输入上一层的，不做残差

10-30 
使用AGN+GCN双塔结构
使用GCN+MLP双塔结构
使用AGN+MLP双塔结构 the experiment is not good 

['ind.cora.x', 'ind.cora.tx', 'ind.cora.allx', 'ind.cora.y', 'ind.cora.ty', 'ind.cora.ally', 'ind.cora.graph', 'ind.cora.test.index']

10-31 图结点的分类问题，有最佳的模型
- https://paperswithcode.com/paper/optimization-of-graph-neural-networks-with
- https://github.com/russellizadi/ssp
- https://arxiv.org/pdf/2008.09624

11-1
- 通过实验，发现rest-net是有效的
- 通过实验，发现gat+gcn是有效的
不足之处：
1、模型不能过于简单
2、工作量

11-2 为此，准备2部分工作
- 工作量：加数据集、加任务、
- 模型：要设计时，复杂一些
"""
#########################################################################
# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="Process some files.")
# 添加位置参数,位置参数（positional arguments）是指那些没有前缀（如 - 或 --）的命令行参数。它们通常用于指定必填的参数，顺序固定，且必须按顺序提供。
# parser.add_argument('filename', type=str, help='The name of the file to process')
# 添加可选参数
parser.add_argument('--name', type=str, default='mlp')
parser.add_argument('--ds', type=str, default='PubMed', help='CiteSeer,Cora,PubMed')
parser.add_argument('--ds_split', type=str, default='public', help=' to see Planetoid')
parser.add_argument('--max_acc', type=float, default=0.01)
parser.add_argument('--ep', type=int, default=4096)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--drop', type=float, default=0.5)
parser.add_argument('--loss', type=float, default=0.01)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--min_acc', type=int, default=0.52)
# 解析命令行参数
args = parser.parse_args()


#########################################################################

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


def load_data():
    if platform.system() != 'Linux':
        print('user faker graph data by make_traph()')
        return make_graph()
    dataset = Planetoid(root=os.path.join(data_root_path, 'Planetoid'), name=args.ds,
                        split=args.ds_split,
                        transform=NormalizeFeatures())
    """
    https://github.com/kimiyoung/planetoid/raw/master/data/ind.pubmed.x
    https://github.com/kimiyoung/planetoid/raw/master/data/ind.pubmed.tx
    https://github.com/kimiyoung/planetoid/raw/master/data/ind.pubmed.allx
    https://github.com/kimiyoung/planetoid/raw/master/data/ind.pubmed.y
    https://github.com/kimiyoung/planetoid/raw/master/data/ind.pubmed.ty
    https://github.com/kimiyoung/planetoid/raw/master/data/ind.pubmed.ally
    https://github.com/kimiyoung/planetoid/raw/master/data/ind.pubmed.graph
    https://github.com/kimiyoung/planetoid/raw/master/data/ind.pubmed.test.index
 
    """
    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('===========================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    dataset.to(device=_device)
    return dataset


def make_graph():
    from torch_geometric.utils import barabasi_albert_graph
    node = 1024
    edge = node // 70
    num_classes = 8
    x = torch.randint(0, 8, (node, 32), dtype=torch.float)
    edge_index = barabasi_albert_graph(node, edge)
    y = torch.randint(0, num_classes, (node,), dtype=torch.long)

    train_mask = torch.randint(0, 2, (node,), dtype=torch.bool)
    val_mask = torch.randint(0, 2, (node,), dtype=torch.bool)
    test_mask = torch.randint(0, 2, (node,), dtype=torch.bool)

    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.num_classes = num_classes
    print(f'num_features={data.num_features},num_classes={data.num_classes}')
    print(f'data.x= {data.x}')
    print(f'data.y= {data.y}')
    return data


def show_graph():
    import matplotlib.pyplot as plt
    import networkx as nx

    # 生成 Barabási-Albert 图
    n = 50  # 节点数
    m = 3  # 每个新节点连接的边数
    G = nx.barabasi_albert_graph(n, m)
    print(G)

    # 绘制图形
    plt.figure(figsize=(10, 8))
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray', font_size=10)
    plt.title('Barabási-Albert Graph')
    plt.show()


class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        self.lin1 = nn.Linear(dataset.num_features, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x):
        x1 = self.lin1(x)
        x1 = x1.relu()
        x1 = F.dropout(x1, p=args.drop, training=self.training)
        y = self.lin2(x1)  # acc = 0.55
        # y = self.lin2(x1+x)
        return y


class RestGCNEqualHidden(torch.nn.Module):
    # max_eva=0.26262626262626265 max_eva=0.825
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = x1.relu()
        x1 = F.dropout(x1, p=args.drop, training=self.training)

        x2 = self.conv2(x1, edge_index)
        x2 = x2.relu()
        x2 = F.dropout(x2, p=args.drop, training=self.training)

        x3 = self.conv3(x2 + x1, edge_index)
        x3 = x3.relu()
        x3 = F.dropout(x3, p=args.drop, training=self.training)

        x4 = self.conv4(x2 + x3, edge_index)
        x4 = x4.relu()
        x4 = F.dropout(x4, p=args.drop, training=self.training)

        x5 = self.conv4(x4 + x3, edge_index)
        y = x5.relu()
        return y


class GCN(torch.nn.Module):
    # max_eva=0.2404040404040404 Accuracy: 0.8280
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = x1.relu()
        x1 = F.dropout(x1, p=args.drop, training=self.training)
        y = self.conv2(x1, edge_index)
        return y


class GAT_GCN(torch.nn.Module):
    # Max Accuracy: 0.8210
    def __init__(self, hidden_channels, dataset, training=True):
        super().__init__()
        self.training = training
        self.conv1_t = GATConv(in_channels=dataset.num_features, out_channels=hidden_channels, heads=args.heads,
                               dropout=args.drop)
        self.conv2_t = GATConv(in_channels=args.heads * hidden_channels, out_channels=dataset.num_classes, heads=1, dropout=args.drop)

        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

        self.lin1 = nn.Linear(dataset.num_classes * 2, hidden_channels * 2)
        self.lin2 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, dataset.num_classes)

        self.weight_t = torch.tensor(0.611, requires_grad=True)
        self.weight_c = torch.tensor(0.212, requires_grad=True)

    def forward(self, x, edge_index):
        # gat
        x_gat = F.dropout(x, p=args.drop, training=self.training)
        x_gat = self.conv1_t(x_gat, edge_index)
        x_gat = F.elu(x_gat)
        x_gat = F.dropout(x_gat, p=args.drop, training=self.training)
        x_gat = self.conv2_t(x_gat, edge_index)

        x_gcn = self.conv1(x, edge_index)
        x_gcn = x_gcn.relu()
        x_gcn = F.dropout(x_gcn, p=args.drop, training=self.training)
        x_gcn = self.conv2(x_gcn, edge_index)

        x_cat = torch.cat([x_gcn * self.weight_c, x_gat * self.weight_t], dim=1)
        y = self.lin1(x_cat)
        y = self.lin2(y)
        y = self.lin3(y)

        return y


class GAT(torch.nn.Module):
    # 0.8220
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        self.conv1 = GATConv(in_channels=dataset.num_features, out_channels=hidden_channels, heads=args.heads)
        self.conv2 = GATConv(in_channels=hidden_channels * args.heads, out_channels=dataset.num_classes, heads=1)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=args.drop, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=args.drop, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_mlp():
    data = load_data()
    model = MLP(hidden_channels=args.hidden, dataset=data)
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)  # Define optimizer.

    def train():
        model.train()
        model.to(device=_device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x)  # Perform a single forward pass.
        loss = criterion(out[data.train_mask],
                         data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def eva():
        model.eval()
        out = model(data.x)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
        return test_acc

    max_acc = -1
    records = []
    for cur_epoch in range(1, 100):
        loss = train()
        test_acc = eva()
        if (max_acc < test_acc):
            max_acc = test_acc
            if (max_acc > args.min_acc):
                save_model(model=model, **vars(args))
        if (cur_epoch % 30 == 0):
            print_loss(epoch=cur_epoch, loss=loss.item(), test_acc=test_acc)
            records.append({'epoch': cur_epoch, 'loss': loss.item(), 'test_acc': test_acc})
    args.max_acc = max_acc
    save_json(records=records, **vars(args))
    print(max_acc)


def train_gcn():
    model = None
    data = load_data()
    # Core
    # CiteSeer  GAT_GCN=0.6970 GCN=0.704
    # PubMed GAT_GCN=0.7940 GCN=0.7970 GAT=0.7870
    if (args.name == 'GCN'):
        model = GCN(hidden_channels=args.hidden, dataset=data)
    if (args.name == 'GAT'):
        model = GAT(hidden_channels=args.hidden, dataset=data)
    if (args.name == 'RestGCNEqualHidden'):
        model = RestGCNEqualHidden(hidden_channels=args.hidden, dataset=data)
    if (args.name == 'GAT_GCN'):
        model = GAT_GCN(hidden_channels=args.hidden, dataset=data)
    model.to(device=_device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device=_device)

    def train():
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index)  # Perform a single forward pass.
        inner_loss = criterion(out[data.train_mask],
                               data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        inner_loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return inner_loss

    def eva():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
        return test_acc

    loss = train()
    # 47223  0.2160
    # max_eva=0.2448559670781893
    epoch = 0
    max_acc = -1
    records = []
    while loss > 0.0001 and epoch < args.ep:
        loss = train()
        test_acc = eva()
        if (max_acc < test_acc):
            max_acc = test_acc
            if (max_acc > args.min_acc):
                save_model(model=model, **vars(args))
        if epoch % 9 == 0:
            print_loss(epoch=epoch, loss=loss.item(), test_acc=test_acc, max_acc=max_acc)
            records.append({'epoch': epoch, 'loss': loss.item(), 'test_acc': test_acc})
        epoch += 1
    args.max_acc = max_acc
    save_json(records=records, **vars(args))
    print(max_acc)
    return max_acc


if __name__ == '__main__':
    args.name = 'GAT_GCN'
    ds_list = ['CiteSeer', 'Cora', 'PubMed']
    ds_split = ['full', 'random', 'public']
    models = ['GAT', 'GAT_GCN','RestGCNEqualHidden','GCN'  ]
    for m in models:
        for s in ds_split:
            for ds in ds_list:
                args.ds = ds
                args.ds_split = s
                args.name = m

                acc = train_gcn()
                print(f'model={m},ds={ds},ds_split={s},acc={acc:.4f}')
