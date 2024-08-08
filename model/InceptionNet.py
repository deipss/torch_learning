


import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()
        # 1x1 convolution branch
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        # 1x1 to 3x3 convolution branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        # 1x1 to 5x5 convolution branch
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        # Pooling and projection branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch3 = self.branch3(x)
        branch5 = self.branch5(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch1, branch3, branch5, branch_pool]
        return torch.cat(outputs, 1)

class InceptionNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.aux1 = self._make_aux_classifier()
        self.aux2 = self._make_aux_classifier()
        self.inception4a = self._make_inception(160, 64, 112, 224, 256, 64)
        self.inception4b = self._make_inception(192, 128, 192, 192, 224, 64)
        self.inception4c = self._make_inception(208, 160, 112, 224, 224, 64)
        self.inception4d = self._make_inception(256, 128, 256, 256, 256, 64)
        self.inception4e = self._make_inception(384, 192, 192, 384, 384, 128)
        self.inception5a = self._make_inception(384, 256, 384, 384, 384, 128)
        self.inception_res = self._make_inception_resnet(256, 256, 256, 256, 384, 128)
        self.avgpool = nn.AvgPool2d(5, stride=3)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(1024, num_classes)

    def _make_inception(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        return InceptionModule(in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj)

    def _make_inception_resnet(self, in_channels, ch1x1, ch3x3, ch5x5, pool_proj):
        # Inception module with residual connections
        inception = InceptionModule(in_channels, ch1x1, ch3x3, ch5x5, pool_proj)
        # Residual connection
        resnet = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(ch3x3, ch5x5, kernel_size=3, padding=1)
        )
        return nn.Sequential(inception, nn.ReLU(True), resnet)

    def _make_aux_classifier(self):
        return nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(5, stride=3),
            nn.Conv2d(128, 768, kernel_size=1),
            nn.ReLU(True),
            nn.AvgPool2d(5, stride=6),
            nn.Flatten(),
            nn.Linear(768, 768),
            nn.ReLU(True),
            nn.Linear(768, 1000)
        )

    def forward(self, x):
        out = self.pre_layers(x)
        aux1 = self.aux1(out)
        out = self.inception4a(out)
        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        out = self.inception4e(out)
        out = self.inception5a(out)
        out = self.inception_res(out)
        out = self.avgpool(out)
        out = self.dropout(out)
        out = torch.flatten(out, 1)
        return out

# 实例化模型
model = InceptionNet(num_classes=1000)

# 创建一个随机初始化的输入张量，模拟一个批量大小为64的图像批次
# 假设输入图像大小为224x224，通道数为3（RGB）
input_tensor = torch.randn(64, 3, 224, 224)

# 前向传播以获取模型输出
output = model(input_tensor)
print(output.size())  # 应该输出: torch.Size([64, 1000])