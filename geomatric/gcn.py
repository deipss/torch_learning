import os.path
import torch
import torch.nn as nn
import platform
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
使用AGN+MLP双塔结构

['ind.cora.x', 'ind.cora.tx', 'ind.cora.allx', 'ind.cora.y', 'ind.cora.ty', 'ind.cora.ally', 'ind.cora.graph', 'ind.cora.test.index']


"""
def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


def load_data():
    dataset = Planetoid(root=os.path.join(data_root_path, 'Planetoid'), name='Cora', transform=NormalizeFeatures())

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
    dataset.to(device=device)
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
        x1 = F.dropout(x1, p=0.5, training=self.training)
        y = self.lin2(x + x1)
        return y

    # def forward(self, x):
    #     x = self.lin1(x)
    #     x = x.relu()
    #     x = F.dropout(x, p=0.5, training=self.training)
    #     y = self.lin2(x)
    #     return y


class RestGCNEqualHidden(torch.nn.Module):
    # max_eva=0.26262626262626265
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = x1.relu()
        x1 = F.dropout(x1, p=0.5, training=self.training)
        y = self.conv2(x1 + x, edge_index)
        return y

class GCN(torch.nn.Module):
    # max_eva=0.2404040404040404
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = x1.relu()
        x1 = F.dropout(x1, p=0.5, training=self.training)
        y = self.conv2(x1, edge_index)
        return y


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        self.conv1_t = GATConv(in_channels=dataset.num_features,out_channels=hidden_channels,heads=4,dropout=0.2)
        self.conv2_t = GATConv(in_channels=hidden_channels*4,out_channels=dataset.num_classes,heads=1,dropout=0.2)

        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

        self.lin1 = nn.Linear(dataset.dataset.num_classes*2, dataset.num_classes)

        self.weight_t = torch.tensor(0.5, requires_grad=True)
        self.weight_c = torch.tensor(0.5, requires_grad=True)

    def forward(self, x, edge_index):
        # gat
        x_gat = F.dropout(x, p=0.6, training=self.training)
        x_gat = self.conv1_t(x_gat, edge_index)
        x_gat = F.elu(x_gat)
        x_gat = F.dropout(x_gat, p=0.6, training=self.training)
        x_gat = self.conv2_t(x_gat, edge_index)

        x_gcn = self.conv1(x, edge_index)
        x_gcn = x_gcn.relu()
        x_gcn = F.dropout(x_gcn, p=0.5, training=self.training)
        x_gcn = self.conv2(x_gcn, edge_index)

        x = torch.cat([x_gcn*self.weight_c, x_gat*self.weight_t])
        y = self.lin1(x)

        return y



class GAT_GCN(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        self.conv1 = GATConv(in_channels=dataset.num_features,out_channels=hidden_channels,heads=4,dropout=0.2)
        self.conv2 = GATConv(in_channels=hidden_channels*4,out_channels=dataset.num_classes,heads=1,dropout=0.2)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x



def train_mlp():
    data = load_data()
    model = MLP(hidden_channels=16, dataset=data)
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.

    def train():
        model.train()
        model.to(device=device)
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

    for epoch in range(1, 1001):
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    test_acc = eva()
    print(f'Test Accuracy: {test_acc:.4f}')


def train_gcn():
    data = load_data()
    model = RestGCNEqualHidden(hidden_channels=data.num_features, dataset=data)
    model.to(device=device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device=device)

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
    max_eva = -1
    while loss > 0.1 and epoch < 47223:
        loss = train()
        if epoch % 99 == 0:
            test_acc = eva()
            print(f'Epoch: {epoch:010d}, Loss: {loss:.4f},Test Accuracy: {test_acc:.4f}')
            max_eva=max(test_acc, max_eva)
        epoch += 1
    print(f'max_eva={max_eva}')


if __name__ == '__main__':
    train_gcn()
