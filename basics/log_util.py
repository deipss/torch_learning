from datetime import datetime
import json
import torch
import os
from torchsummary import summary
root_path = '../data'

def print_model(model,input_size):
    summary(model, input_size)


def print_loss(epoch=0, idx=0, **param):
    # 将时间格式化为字符串
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    logs = ""
    for k, v in param.items():
        logs += "{0}={1}\t".format(k, v)
    print(formatted_time + '\t' + logs)  # 输出格式：2023-10-21 22:30:45
    pass


def save_json(filename='', modelname='', json_data=None, **param):
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    dict_format = '_'.join(["{0}={1}_".format(k, v) for k, v in param.items()])
    fname = '_'.join([modelname, filename, dict_format, formatted_time])
    fname += '.json'
    fpath = os.path.join(root_path, 'log', fname)
    with open(fpath, 'w') as file:
        json.dump(json_data, file)
    print(fname + " area saved")


def save_model(filename='', model=None, replace=True, **model_param):
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    dict_format = '_'.join(["{0}={1}_".format(k, v) for k, v in model_param.items()])
    fname = '_'.join([filename, dict_format, formatted_time])
    fname += '.pkl'
    fpath = os.path.join(root_path, 'pkl', fname)
    torch.save(model.state_dict(), fpath)



if __name__ == '__main__':
    print_loss(1, 1, **{'a': 'b', 'c': 'd'})
    save_json(filename='aaa', modelname='cnn_model', json_data={1: 2}, **{'a': 'b', 'c': 'd'})
    save_model(filename='aaa', model=torch.nn.Linear(3,4), **{'a': 'b', 'c': 'd'})
