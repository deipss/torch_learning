from torchvision import datasets
from torchvision.transforms import ToTensor,transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else DEVICE

'''
在Matplotlib中，颜色映射（Colormap）用于将数据值映射到颜色空间，以便在可视化中表示不同的数值或类别。除了`'gray'`之外，Matplotlib提供了许多预定义的颜色映射选项。以下是一些常用的颜色映射：

1. **'viridis'**：一种对颜色盲友好的颜色映射，从蓝色到黄色渐变。
2. **'plasma'**：从蓝色到红色的鲜艳颜色映射。
3. **'magma'**：类似于plasma，但颜色更暖。
4. **'inferno'**：从黄色到橙色再到红色的渐变。
5. **'cividis'**：对颜色盲友好的颜色映射，从浅蓝色到深蓝色渐变。
6. **'Blues'**：蓝色调的颜色映射。
7. **'Greens'**：绿色调的颜色映射。
8. **'Reds'**：红色调的颜色映射。
9. **'Purples'**：紫色调的颜色映射。
10. **'Oranges'**：橙色调的颜色映射。
11. **'YlOrBr'**：黄-橙-棕色的颜色映射。
12. **'YlGn'**：黄-绿颜色映射。
13. **'OrRd'**：橙-红色映射。
14. **'PuBu'**：紫色-蓝色映射。
15. **'BuPu'**：蓝色-紫色映射。
16. **'GnBu'**：绿色-蓝色映射。
17. **'PuRd'**：紫色-红色映射。
18. **'YlGnBu'**：黄-绿-蓝色映射。
19. **'PuBuGn'**：紫色-蓝色-绿色映射。
20. **'cool'**：从深蓝色到浅蓝色渐变。
21. **'coolwarm'**：从深蓝色到深红色的双色调映射，适合表示接近中心对称的数据分布。
22. **'bwr'**：蓝-白-红色映射，常用于表示正负差异。
23. **'seismic'**：从绿色到红色的地震色彩映射。

此外，Matplotlib还允许用户创建自定义颜色映射，以满足特定的可视化需求。可以使用`matplotlib.cm`模块中的`LinearSegmentedColormap`类来定义自己的颜色映射。

例如，要使用Jet颜色映射，你可以这样做：

```python
import matplotlib.pyplot as plt

plt.imshow(data, cmap='jet')
plt.show()
```

在这段代码中，`data`是你的图像数据，`cmap='jet'`指定了使用Jet颜色映射。使用不同的`cmap`参数值就可以应用不同的颜色映射。

'''




def showImg():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        print(sample_idx)
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        print(img.size())  # (1,28,28)
        print(img.squeeze().size())  # (28,28)
        plt.imshow(img.squeeze(), cmap="coolwarm")
    plt.show()


if __name__ == '__main__':
    showImg()

