from datetime import datetime
import json
import torch
import os
from torchsummary import summary
import platform
import matplotlib.pyplot as plt

data_root_path = '/data' if platform.system() == 'Linux' else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
separator = '__'


def print_model(model, input_size):
    """
    输出模型结构
    """
    summary(model, input_size)


def print_loss(epoch=0, is_debug=False, **param):
    """
    输出loss
    param = {'loss':'111',acc:'3333'}
    """
    if is_debug:
        return
    # 将时间格式化为字符串
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    logs = ""
    for k, v in param.items():
        if isinstance(v, float):
            logs += "{0}={1:.5f}\t".format(k, v)
        elif isinstance(v, int):
            logs += "{0}={1:04d}\t".format(k, v)
        else:
            logs += "{0}={1}\t".format(k, v)
    print(f'{formatted_time}\t epoch={epoch}\t{logs}')  # 输出格式：2023-10-21 22:30:45


def save_json(records=None, is_debug=False, **param):
    """
    保存训练记录
    """
    if is_debug:
        return
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    logs = ""
    for k, v in param.items():
        if isinstance(v, float):
            logs += "{0}={1:.5f}_".format(k, v)
        elif isinstance(v, int):
            logs += "{0}={1:08d}_".format(k, v)
        else:
            logs += "{0}={1}_".format(k, v)
    f_name = separator.join([logs, formatted_time])
    f_name += '.json'
    fpath = os.path.join(data_root_path, 'logs', f_name)
    _ = {'records': records, 'param': param}
    with open(fpath, 'w') as file:
        json.dump(_, file)
    print(f_name + " area saved on " + fpath)


def save_records(records=None, is_debug=False, file_name=''):
    """
    保存训练记录
    """
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    f_name = separator.join([file_name, formatted_time])
    f_name += '.json'
    fpath = os.path.join(data_root_path, 'records', f_name)
    _ = {'records': records}
    with open(fpath, 'w') as file:
        json.dump(_, file)
    print(f_name + " are saved on " + fpath)


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
    fpath = os.path.join(data_root_path, 'pkls', f_name)
    torch.save(model.state_dict, fpath)
    print(f_name + ' model saved on ' + fpath)


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

def show_acc(file=None):
    with open(file, 'r') as file:
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
        plt.title(f'{params["name"]}-{params["ds"]}-{params["hidden"]}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    show_acc(
        '/Users/deipss/workspace/ai/torch_learning/data/log/name=GAT_ds=PubMed_ds_split=public_max_acc=0.21600_ep=00002048_heads=00000004_lr=0.00100_drop=0.50000_loss=0.01000_hidden=00000064_min_acc=0.12000___20241101_183743_916.json')
