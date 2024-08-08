
import matplotlib.pyplot as plt
import numpy as np


def show_1():
    # 创建一些示例数据
    data = np.random.rand(10, 10)
    # 使用imshow展示数据分布
    plt.imshow(data, cmap='viridis')  # 选择颜色映射
    plt.colorbar()  # 显示颜色条
    plt.show()


def show2():
    # 创建一些示例数据
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    # 根据y值设置点的颜色
    colors = y - y.min()  # 归一化颜色值
    # 使用scatter绘制散点图，并应用颜色映射
    plt.scatter(x, y, c=colors, cmap='coolwarm')
    plt.colorbar()  # 显示颜色条
    plt.show()


def show3():
    from matplotlib.colors import LinearSegmentedColormap

    # 定义自定义颜色映射
    cmap = LinearSegmentedColormap.from_list('mycmap', ['blue', 'cyan', 'green', 'yellow', 'red'])

    # 使用自定义颜色映射展示数据
    data = np.random.rand(10, 10)
    plt.imshow(data, cmap=cmap)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':

    show3()