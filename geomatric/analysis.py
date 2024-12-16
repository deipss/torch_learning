import numpy as np
import json
from os import path
import os
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
import platform

data_path = '/data/ai_data' if platform.system() == 'Linux' else '../data'


def statistic_dataset():
    """
    statistic_dataset()
    """
    ds_list = ['MUTAG', 'DD', 'MSRC_9', 'AIDS']
    for ds in ds_list:
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


def analysis_data(data_path, data_name, data_type='train'):
    """
    analysis_data('../records', 'graph_class__20241113_074727_391.json')
    """
    with open(path.join(data_path, data_name), 'r') as file:
        data = json.load(file)
        records = data['records']
        data_list = []
        for record in records:
            m = {}
            arr = record.split(',')
            for e in arr:
                pair = e.split('=')
                m[pair[0]] = pair[1]
            data_list.append(m)
        ds_list = ['MUTAG', 'DD', 'MSRC_9', 'AIDS']
        for i in ds_list:
            mutag = filter(lambda x: x['ds'] == i, data_list)
            mutag = sorted(mutag, key=lambda x: (float(x['acc']), x['model'], -int(x['h']), -int(x['dim'])),
                           reverse=True)
            print(f'ds={i}')
            print(''.join(f'{key:<20}' for key in mutag[0].keys()))
            for m in mutag[:70]:
                print(''.join(f'{key:<20}' for key in m.values()))


def analysis_fold_data(data_path, data_name, data_type='train'):
    """
    analysis_fold_data('../records', 'graph_class__20241201_112212_301.json')
    """
    with open(path.join(data_path, data_name), 'r') as file:
        data = json.load(file)
        records = data['records']
        data_list = []
        for record in records:
            m = {}
            arr = record.split(',')
            for e in arr:
                pair = e.split('=')
                if pair[0] in ['execution_time', 'f_name']:
                    continue
                m[pair[0]] = pair[1]
            accs = [float(m['acc0']), float(m['acc1']), float(m['acc2']), float(m['acc3']), float(m['acc4'])]
            m['std'] = round(np.std(accs), 4)
            data_list.append(m)
        ds_list = ['MUTAG', 'DD', 'MSRC_9', 'AIDS']
        for i in ds_list:
            mutag = filter(lambda x: x['ds'] == i, data_list)

            mutag = sorted(mutag, key=lambda x: (float(x['acc']), x['model'], -int(x['h']), -int(x['dim'])),
                           reverse=True)
            if len(mutag) < 1:
                continue
            print(f'ds={i}')
            print(''.join(f'{key:<15}' for key in mutag[0].keys()))
            for m in mutag[:70]:
                print(''.join(f'{key:<15}' for key in m.values()))


def show_acc(data_path, data_name, ):
    """

    """
    with open(path.join(data_path, data_name), 'r') as file:
        data = json.load(file)
        records = data['records']
        params = data['param']
        sorted(records, key=lambda x: x['epoch'])
        # 提取epoch和loss数据
        epochs = [i['epoch'] for i in records]
        losses = [i['test_acc'] for i in records]

        # 使用Matplotlib绘制曲线图
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, marker='o', linestyle='-', color='b')
        plt.title(f'{params["name"]}-{params["ds"]}-{params["dim"]}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()


def search_min_epoch(path):
    """
    search_min_epoch('../data/logs')
    max_epoch = 1535
    """
    files = os.listdir(path)
    max_epoch = -1
    for f in files:
        if 'gname' not in f: continue
        full_path = os.path.join(path, f)
        with open(full_path, 'r') as file:
            data = json.load(file)
            records = data['records']
            params = data['param']
            records = sorted(records, key=lambda x: (-x['test_acc'], x['epoch']))
            # 提取epoch和loss数据
            max_epoch = max(max_epoch, records[0]['epoch'])
            if (records[0]['epoch'] > 1000):
                print(params, records[0]['epoch'], records[0]['test_acc'], f)
    print(max_epoch)


def show_loss(file=None):
    with open(file, 'r') as file:
        data = json.load(file)
        records = data['records']
        params = data['param']
        sorted(records, key=lambda x: x['epoch'])
        # 提取epoch和loss数据
        epochs = [i['epoch'] for i in records]
        losses = [i['loss'] for i in records]

        # 使用Matplotlib绘制曲线图
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, marker='o', linestyle='-', color='b')
        plt.title(f'{params["name"]}-{params["ds"]}-{params["hidden"]}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    analysis_fold_data('../records', 'graph_class__20241213_192005_415.json')
