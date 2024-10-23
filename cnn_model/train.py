
from torchvision import datasets
from torchvision.transforms import ToTensor,transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from AlexNet import AlexNet

def train():
    num_epochs = 10
    # 定义数据转换操作
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将[0, 255]的像素值转换为[0.0, 1.0]
        transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]
    ])

    # 加载FashionMNIST训练集和测试集
    train_dataset = datasets.FashionMNIST(
        root="../data", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform)

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
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images=images.to(DEVICE)
            labels=labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

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
    torch.save(model.state_dict(), "../data/alexnet_fashionmnist.pth")


if __name__ == '__main__':
    train()
