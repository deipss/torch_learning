
import matplotlib.pyplot as plt
import numpy as np


def show_1():
    # 创建一些示例数据
    # 生成一个10x10的随机数矩阵，用于后续的数据可视化
    data = np.random.rand(10, 10)

    # 使用imshow展示数据分布
    # 这里省略了实际的绘图代码，通常会包括导入matplotlib库并使用plt.imshow(data)等语句
    plt.imshow(data, cmap='viridis')  # 选择颜色映射
    plt.colorbar()  # 显示颜色条
    plt.show()


def show2():
    # 创建一些示例数据
    # 生成一个从-1到1的等差数列，包含100个元素
    x = np.linspace(-1, 1, 100)
    # 打印x数组的所有元素
    print(x)
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

def show4():
    import numpy as np
    import matplotlib.pyplot as plt

    # 定义x的范围
    # 生成一个等差数列，用于后续的绘图或计算
    x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    # 计算y值
    y_cos = np.cos(x)
    y_sin = np.sin(x)
    y_tan = np.tan(x)

    # 创建一个新的图形
    plt.figure(figsize=(14, 6))

    # 绘制cos函数
    plt.subplot(1, 3, 1)
    plt.plot(x, y_cos, label='cos(x)', color='blue')
    plt.title('Cosine Function')
    plt.xlabel('x')
    plt.ylabel('cos(x)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.legend()

    # 绘制sin函数
    plt.subplot(1, 3, 2)
    plt.plot(x, y_sin, label='sin(x)', color='red')
    plt.title('Sine Function')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.legend()

    # 绘制tan函数, 注意避开不连续点
    plt.subplot(1, 3, 3)
    # 找到tan函数中绝对值很大的点，这些点附近是不连续的
    mask = np.abs(y_tan) < 10  # 设置一个阈值来过滤掉接近不连续点的数据
    plt.plot(x[mask], y_tan[mask], label='tan(x)', color='green')
    plt.title('Tangent Function')
    plt.xlabel('x')
    plt.ylabel('tan(x)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.legend()

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()

if __name__ == '__main__':

    show4()