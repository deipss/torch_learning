import os.path
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
import time
from basics import *

_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data_path = '/data/ai_data' if platform.system() == 'Linux' else '../data'
############################### seed ##########################################
seed = 1024
torch.manual_seed(seed)  # 为当前CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
np.random.seed(seed)  # 为NumPy设置随机种子
random.seed(seed)  # 为Python的random模块设置随机种子
################################# args ########################################
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
parser.add_argument('--debug', type=bool, default=False)
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
    dataset = Planetoid(root=os.path.join(data_path, 'Planetoid'), name=args.ds,
                        split=args.ds_split,
                        transform=NormalizeFeatures())
    """
    ['ind.cora.x', 'ind.cora.tx', 'ind.cora.allx', 'ind.cora.y', 'ind.cora.ty', 'ind.cora.ally', 'ind.cora.graph', 'ind.cora.test.index']
    https://github.com/kimiyoung/planetoid/raw/master/data
 
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


class ResGCN(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
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
        return x5


class ResGAT(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        self.conv1 = GATConv(in_channels=dataset.num_features, out_channels=hidden_channels, heads=args.heads)
        self.conv2 = GATConv(in_channels=hidden_channels*args.heads, out_channels=hidden_channels, heads=args.heads)
        self.conv3 = GATConv(in_channels=hidden_channels*args.heads, out_channels=hidden_channels, heads=args.heads)
        self.conv4 = GATConv(in_channels=hidden_channels*args.heads, out_channels=hidden_channels, heads=args.heads)
        self.conv5 = GATConv(in_channels=hidden_channels * args.heads, out_channels=dataset.num_classes, heads=1)

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
        return x5


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=args.drop, training=self.training)
        # x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, p=args.drop, training=self.training)
        # x = F.relu(self.conv3(x, edge_index))
        # x = F.dropout(x, p=args.drop, training=self.training)
        # x = F.relu(self.conv4(x, edge_index))
        x = F.dropout(x, p=args.drop, training=self.training)
        y = self.conv5(x, edge_index)
        return y



class GAT(torch.nn.Module):
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


def train_gcn():
    model = None
    data = load_data()
    if (args.name == 'GCN'):
        model = GCN(hidden_channels=args.hidden, dataset=data)
    if (args.name == 'GAT'):
        model = GAT(hidden_channels=args.hidden, dataset=data)
    if (args.name == 'ResGCN'):
        model = ResGCN(hidden_channels=args.hidden, dataset=data)
    if (args.name == 'ResGAT'):
        model = ResGAT(hidden_channels=args.hidden, dataset=data)
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

    epoch = 0
    max_acc = -1
    records = []
    while loss > 0.0001 and epoch < args.ep:
        loss = train()
        test_acc = eva()
        if (max_acc < test_acc):
            max_acc = test_acc
            if (max_acc > args.min_acc):
                save_model(model=model, is_debug=args.debug, **vars(args))
        if epoch % 9 == 0:
            print_loss(epoch=epoch, is_debug=args.debug, loss=loss.item(), test_acc=test_acc, max_acc=max_acc)
            records.append({'epoch': epoch, 'loss': loss.item(), 'test_acc': test_acc})
        epoch += 1
    args.max_acc = max_acc
    save_json(records=records, is_debug=args.debug, **vars(args))
    print(max_acc)
    return max_acc


if __name__ == '__main__':
    args.ep=1
    args.debug=True
    ds_list = ['CiteSeer', 'Cora', 'PubMed']
    ds_split = ['full', 'random', 'public']
    models = ['GAT', 'ResGAT', 'ResGCN', 'GCN']
    results=[]
    for s in ds_split:
        for ds in ds_list:
            for m in models:
                args.ds = ds
                args.ds_split = s
                args.name = m

                start_time = time.time()
                acc = train_gcn()
                execution_time = time.time() - start_time
                results.append( f'model={m},ds={ds},ds_split={s},acc={acc:.5f},execution_time={execution_time:.5f}')
                print(f'model={m},ds={ds},ds_split={s},acc={acc:.5f},execution_time={execution_time:.5f}')
    save_records(records=results, is_debug=args.debug, file_name='node_class')