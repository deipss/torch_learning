import numpy as np
import json
from os import path
import os
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
import platform
from tabulate import tabulate


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




def parse_line(line):
    """解析一行数据为字典，并计算 acc0-acc4 的标准差"""
    fields = line.strip().split(',')
    result = {}
    acc_values = []  # 用于存储 acc0-acc4 的数值部分

    for field in fields:
        if '=' in field:
            key, value = field.split('=', 1)  # 仅分割第一个等号
            result[key] = value

            # 提取 acc0-acc4 的数值部分
            if key.startswith('acc') and key[3:].isdigit():
                acc_values.append(float(value))

    # 计算 acc0-acc4 的标准差
    if acc_values:
        std_dev = np.std(acc_values)
        result['acc_std_dev'] = std_dev
    else:
        result['acc_std_dev'] = 0.0  # 如果没有 acc0-acc4 字段，标准差设为 0

    return result

def process_files_in_folder(folder_path):
    """处理文件夹中的所有文件"""
    all_data = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.startswith('graph'):  # 确保是文本文件
            with open(file_path, 'r') as f:
                for line in f:
                    # 解析每一行的数据
                    data = parse_line(line)
                    if data:  # 确保解析成功
                        all_data.append(data)

    return all_data


def group_and_sort_data(data_list):
    """根据 ds, model, dim, h 分组，并按 gm 和 acc 降序排序"""
    grouped_data = {}

    for data in data_list:
        ds = data.get('ds')
        model = data.get('model')
        dim = data.get('dim')
        h = data.get('h')
        gm = data.get('gm', '')  # 默认为空字符串
        acc = float(data.get('acc', 0.0))  # 将 acc 转换为浮点数

        # 创建分组键
        group_key = (ds, model, dim, h)

        if group_key not in grouped_data:
            grouped_data[group_key] = []

        # 将数据添加到对应的分组
        grouped_data[group_key].append(data)

    # 对每个分组按 gm 和 acc 降序排序
    for group_key in grouped_data:
        grouped_data[group_key].sort(
            key=lambda x: (-float(x.get('acc', 0.0)), x.get('gm', '')),
            reverse=True
        )

    return grouped_data


def print_as_table(grouped_data):
    """以表格形式打印分组数据"""
    for group_key, data_list in grouped_data.items():
        print(f"Group: ds={group_key[0]}, model={group_key[1]}, dim={group_key[2]}, h={group_key[3]}")

        # 准备表格数据
        table_data = []
        headers = ['gm', 'acc', 'execution_time']

        for data in data_list:
            row = [
                data.get('gm', ''),
                data.get('acc', ''),
                data.get('execution_time', '')
            ]
            table_data.append(row)

        # 打印表格
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print('-' * 50)


def save_to_excel():
    import pandas as pd
    """将数据保存到 Excel 文件中"""
    # 将数据转换为 DataFrame
    all_data = process_files_in_folder('../records' )

    df = pd.DataFrame(all_data)

    # 确保 Excel 文件存在
    df.to_excel('../records/result.xlsx', index=False)


if __name__ == '__main__':
    import json
    import matplotlib.pyplot as plt

    # 从文件名中提取参数
    filename = "name=GATConv_gname=CrossBlockGnn_ds=AIDS_max_acc=0.7700_ep=1500.0000_lr=0.0100_drop=0.6000_loss=0.0010_dim=032_h_layer=005_min_acc=0.1000_debug=000___20250312_225850.json"
    params = {}
    for part in filename.split("___")[0].split("_"):  # 提取参数部分（忽略时间戳）
        if "=" in part:
            key, value = part.split("=", 1)
            params[key] = value

    file_path='/Users/deipss/workspace/ai/torch_learning/log/AIDS_GATConv32/'+filename
    # 2. 读取JSON文件内容
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # 正确使用 json.load() 读取文件对象
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在！")
        exit()
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 不是有效的JSON格式！")
        exit()

    records = data["records"]

    # 提取数据
    epochs = [record["epoch"] for record in records]
    loss_values = [record["loss"] for record in records]
    test_acc_values = [record["test_acc"] for record in records]
    train_acc_values = [record["train_acc"] for record in records]

    # 生成图表标题
    title = (
        f"Model: {params.get('gname', 'N/A')} on {params.get('ds', 'N/A')} Dataset\n"
        f"LR={params.get('lr', 'N/A')}, Dropout={params.get('drop', 'N/A')}, "
        f"Dim={params.get('dim', 'N/A')}, Hidden Layers={params.get('h_layer', 'N/A')}"
    )

    # 绘制图表
    plt.figure(figsize=(12, 8))

    # Loss 曲线
    plt.subplot(3, 1, 1)
    plt.plot(epochs, loss_values, color="red")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Test/Train Accuracy 曲线
    plt.subplot(3, 1, 2)
    plt.plot(epochs, test_acc_values, label="Test Accuracy", color="blue")
    plt.plot(epochs, train_acc_values, label="Train Accuracy", color="green")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # 全局标题
    plt.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()

