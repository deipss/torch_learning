from datetime import datetime
import json
import torch
import os
from torchsummary import summary
import platform

data_root_path = '/data' if platform.system() == 'Linux' else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
separator = '__'


def print_model(model, input_size):
    """
    输出模型结构
    """
    summary(model, input_size)


def print_loss(epoch=0, **param):
    """
    输出loss
    param = {'loss':'111',acc:'3333'}
    """
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


def save_json(records=None, **param):
    """
    保存训练记录
    """
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


def save_model(model=None, **model_param):
    """
    保存模型
    """
    formatted_time = datetime.now().strftime("%Y%m%d_%H")
    dict_format = separator.join(["{0}={1}_".format(k, v) for k, v in model_param.items()])
    f_name = separator.join([dict_format, formatted_time])
    f_name += '.pkl'
    fpath = os.path.join(data_root_path, 'pkls', f_name)
    torch.save(model.state_dict, fpath)
    #print(f_name + ' model saved on ' + fpath)

#
# if __name__ == '__main__':
#     print_loss(1, 1, **{'a': 'b', 'c': 'd', 'loss': '0.4343'})
#     save_json(filename='aaa', model_name='cnn_model', json_data={1: 2}, **{'lr': '0.002', 'hidden': '16'})
#     save_model(model_name='cnn', model=torch.nn.Linear(3, 4), **{'a': 'b', 'c': 'd'})
