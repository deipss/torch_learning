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


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        # 第一部分
        self.features = nn.Sequential(
            nn.Conv2d(1, 58, kernel_size=3, stride=1, padding=2),  # 输入通道数为3（RGB），输出通道数为96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 池化层
            nn.Conv2d(58, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # 第二部分，全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # 自适应平均池化
        self.classifier = nn.Sequential(
            nn.Dropout(),  # 丢弃层
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)  # 输出层，类别数为num_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


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


def train():
    num_epochs = 1000
    # 定义数据转换操作
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将[0, 255]的像素值转换为[0.0, 1.0]
        transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]
    ])

    # 加载FashionMNIST训练集和测试集
    train_dataset = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


    # 实例化AlexNet模型
    model = AlexNet(num_classes=10).to(DEVICE)  # FashionMNIST有10个类别

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 训练模型
    model.train()  # 设置模型为训练模式
    # for epoch in range(num_epochs):
    #     for images, labels in train_loader:
    #         images=images.to(DEVICE)
    #         labels=labels.to(DEVICE)
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #
    #         # 反向传播和优化
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images= images.to(DEVICE)
            labels= labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the model on the test images: {100 * correct / total}%")

    # 保存模型
    torch.save(model.state_dict(), "data/alexnet_fashionmnist.pth")


if __name__ == '__main__':
    train()

