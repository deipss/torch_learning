import pandas as pd
import numpy as np
import json
from os import path
import os
import matplotlib.pyplot as plt


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
            mutag = sorted(mutag, key=lambda x: (x['acc'], x['h'], x['dim']), reverse=True)
            print(mutag)


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


if __name__ == '__main__':
    search_min_epoch('../data/logs')
