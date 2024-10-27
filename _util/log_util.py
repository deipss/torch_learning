from datetime import datetime
import json
import torch
from torchsummary import summary
import matplotlib.pyplot as plt
import os
import random
from torchvision.io import read_image
import platform
data_root_path = '/data/ai_data' if platform.system()=='Linux' else '../data'

def show_img_by_folder(path: str, col=5):
    paths = os.listdir(path)
    random_elements = random.sample(paths, min(20, len(paths)))
    fig, axes = plt.subplots(nrows=max(1, len(random_elements) // col), ncols=col, figsize=(15, 15))
    ax = axes.flatten()
    for i, e in enumerate(random_elements):
        # 将图像张量转换为 NumPy 数组
        image_np = read_image(os.path.join(path, e)).numpy()
        # 调整图像的维度顺序，从 (C, H, W) 转换为 (H, W, C)
        image_np = image_np.transpose(1, 2, 0)
        ax[i].imshow(image_np)
        ax[i].set_title(e)
    plt.show()


def show_img_by_map(img_map: dict, col=5):
    """
    map ={}
    for i in range(10) :
        random_image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        map[i]=random_image

    :param img_map:
    :param col:
    :return:
    """
    fig, axes = plt.subplots(nrows=max(1, len(img_map) // col), ncols=col, figsize=(5, 5))
    ax = axes.flatten()
    for i, (k, v) in enumerate(img_map.items()):
        ax[i].imshow(v)
        ax[i].set_title(k)
    plt.show()


def show_img_by_list(imgs: list, col=5):
    """
    for i in range(10):
        imgs.append( np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8))
    """
    fig, axes = plt.subplots(nrows=max(1, len(imgs) // col), ncols=col, figsize=(5, 5))
    ax = axes.flatten()
    for i, img in enumerate(imgs):
        ax[i].imshow(img)
        ax[i].set_title(i)
    plt.show()


def show_img_by_path(path: str, row=5):
    image_np = read_image(path).numpy()
    # 调整图像的维度顺序，从 (C, H, W) 转换为 (H, W, C)
    image_np = image_np.transpose(1, 2, 0)
    plt.figure(figsize=(12, 12))
    plt.imshow(image_np)
    plt.show()


def print_model(model, input_size):
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
    fpath = os.path.join(data_root_path, 'log', fname)
    with open(fpath, 'w') as file:
        json.dump(json_data, file)
    print(fname + " area saved")


def save_model(filename='', model=None, replace=True, **model_param):
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    dict_format = '_'.join(["{0}={1}_".format(k, v) for k, v in model_param.items()])
    fname = '_'.join([filename, dict_format, formatted_time])
    fname += '.pkl'
    fpath = os.path.join(data_root_path, 'pkl', fname)
    torch.save(model.state_dict(), fpath)


if __name__ == '__main__':
    # print_loss(1, 1, **{'a': 'b', 'c': 'd'})
    # save_json(filename='aaa', modelname='cnn_model', json_data={1: 2}, **{'a': 'b', 'c': 'd'})
    # save_model(filename='aaa', model=torch.nn.Linear(3,4), **{'a': 'b', 'c': 'd'})
    # show_img_by_path('/Users/deipss/workspace/ai/torch_learning/data/PennFudanPed/PNGImages/FudanPed00002.png')
    pass