import os.path

from torch_geometric.datasets import Planetoid
import torch
import torch.nn as nn
import platform
import os.path as path
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nwx
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data_root_path = '/data/ai_data' if platform.system() == 'Linux' else '../data'
torch.manual_seed(1234567)



"""
# python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
# pip uninstall pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv 
libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
- https://stackoverflow.com/questions/77664847/although-installed-pyg-lib-successfully-getting-error-while-importing
- Reference: What's the difference between "pip install" and "python -m pip install"?



TransformerConv
AGNNConv
FastRGCNConv
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
    return dataset


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


class RestGCNEqualHidden(torch.nn.Module):
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



def train_mlp():
    data = load_data()
    model = MLP(hidden_channels=16)
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.

    def train():
        model.train()
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

    for epoch in range(1, 201):
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    test_acc = eva()
    print(f'Test Accuracy: {test_acc:.4f}')


def train_gcn():
    data = make_graph()
    model = GCN(hidden_channels=data.num_features, dataset=data)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index)  # Perform a single forward pass.
        loss = criterion(out[data.train_mask],
                         data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

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
    load_data()
