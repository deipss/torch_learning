

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets.graph_generator import *
from torch_geometric.datasets.motif_generator import *
from torch_geometric.data import Data
import torch

def motif_graph_nx():
    # 创建一个示例图
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5)])

    # 提取三节点模体
    def extract_motifs(graph, size=3):
        motifs = []
        nodes = list(graph.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                for k in range(j + 1, len(nodes)):
                    subgraph = graph.subgraph([nodes[i], nodes[j], nodes[k]])
                    if subgraph.number_of_edges() > 0:
                        motifs.append(subgraph)
        return motifs

    # 提取模体
    motifs = extract_motifs(G)
    # 统计模体频率
    motif_counts = {}
    for motif in motifs:
        motif_tuple = tuple(sorted(motif.edges()))
        if motif_tuple in motif_counts:
            motif_counts[motif_tuple] += 1
        else:
            motif_counts[motif_tuple] = 1
    # 打印模体频率
    for motif, count in motif_counts.items():
        print(f"Motif {motif}: {count} times")
    # 构建模体图
    motif_graph = nx.Graph()
    for motif, count in motif_counts.items():
        motif_graph.add_node(motif, count=count)
    # 添加模体之间的边（这里以共享节点为例）
    for motif1 in motif_counts.keys():
        for motif2 in motif_counts.keys():
            if set(motif1).intersection(set(motif2)):
                motif_graph.add_edge(motif1, motif2)
    # 绘制模体图
    pos = nx.spring_layout(motif_graph)
    labels = {motif: f"{motif}: {motif_counts[motif]}" for motif in motif_graph.nodes()}
    nx.draw(motif_graph, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=3000, font_size=10)
    plt.title('Motif Graph')
    plt.show()


def barabasi_albert_graph_nx():
    # 生成BA图
    G = nx.barabasi_albert_graph(100, 3)
    # 绘制图
    plt.figure(figsize=(8, 8))
    nx.draw(G, node_color='lightblue', with_labels=False, node_size=50)
    plt.title('BA Graph')
    plt.show()

def er_graph_nx():
    # 生成ER图
    G = nx.erdos_renyi_graph(100, 0.1)

    # 绘制图
    plt.figure(figsize=(8, 8))
    nx.draw(G, node_color='lightblue', with_labels=False, node_size=50)
    plt.title('ER Graph')
    plt.show()

def grid_graph_nx():
    # 生成Grid图
    G = nx.grid_2d_graph(10, 10)

    # 绘制图
    plt.figure(figsize=(8, 8))
    pos = {(x, y): (x, y) for x, y in G.nodes()}  # 设置节点位置
    nx.draw(G, pos, node_color='lightblue', with_labels=False, node_size=50)
    plt.title('Grid Graph')
    plt.show()
def tree_graph_nx():
    # 生成Tree图
    G = nx.balanced_tree(2, 3)  # 生成一个2叉树，深度为3

    # 绘制图
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)  # 设置节点位置
    nx.draw(G, pos, node_color='lightblue', with_labels=True, node_size=500)
    plt.title('Tree Graph')
    plt.show()
    print(1)

def torch_geometric_graph():
    bg_graph = BAGraph(18,10)
    er_graph = ERGraph(18,10)
    grid_graph = GridGraph(6,6)
    tree_graph = TreeGraph(3,2)
    cycle_motif_graph = CycleMotif(18)
    grid_motif_graph = GridMotif()
    house_motif_graph = HouseMotif()
    torch_geometric_show(house_motif_graph)
    print(1)
def torch_geometric_show(graph_name:GraphGenerator):
    data = graph_name()
    # 将 PyTorch Geometric 的 Data 对象转换为 NetworkX 图
    G = nx.Graph()
    for i in range(data.num_nodes):
        G.add_node(i)

    for edge in data.edge_index.t().tolist():
        G.add_edge(edge[0], edge[1])

    # 绘制图结构
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)  # 计算节点的位置
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=15, font_weight='bold')
    plt.title('Graph Structure')
    plt.show()
def torch_geometric_generate():
    # 创建图数据
    # 节点特征（假设每个节点有一个特征）
    x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)

    # 边索引（每列表示一条边，第一列是源节点，第二列是目标节点）
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                               [1, 0, 2, 1, 3, 2]], dtype=torch.long)

    # 边属性（假设每条边有一个属性）
    edge_attr = torch.tensor([[0.5], [0.3], [0.7], [0.4], [0.6], [0.2]], dtype=torch.float)

    # 创建 Data 对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # 将 PyTorch Geometric 的 Data 对象转换为 NetworkX 图
    G = nx.Graph()
    for i in range(data.num_nodes):
        G.add_node(i)

    for (src, dst), attr in zip(data.edge_index.t().tolist(), data.edge_attr.tolist()):
        G.add_edge(src, dst, weight=attr[0])

    # 绘制图结构
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)  # 计算节点的位置

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)

    # 绘制边
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color='r', alpha=0.5)

    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=15, font_weight='bold')

    # 绘制边标签
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')

    plt.title('Graph with Edge Attributes')
    plt.show()
if __name__ == '__main__':

    torch_geometric_generate()