import torch
import numpy as np
from torch import nn

def cnn_kernel_learning():
    img = torch.ones(6,8)
    img[:,2:4] = 0
    Y = torch.rand(6,7)
    print(img)
    print(Y)
    K = torch.tensor([[1.0, -1.0]])
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
    img = img.reshape((1,1,6,8))
    Y = Y.reshape((1,1,6,7))
    lr = 3e-2
    for i in range(10):
        Y_hat = conv2d(img)
        l =  (Y_hat-Y)**2
        conv2d.zero_grad()
        l.sum().backward()
        conv2d.weight.data[:] -=lr*conv2d.weight.grad
        if(i%2==0):
            print(f'epoch {i} ,loss = {l.sum():.3f}')

class Conv2D(nn.Module):
    def __init__(self, kernel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = nn.Parameter(torch.rand(), kernel)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.weight + self.bias


def zip_show():
    t1 = torch.arange(1, 13).reshape(3, 4)
    t2 = torch.arange(12, 24).reshape(3, 4)
    print(t1)
    print(t2)
    t3 = zip(t1, t2)
    for a, b in t3:
        print(a, b)
        print(*a)

    d1 = torch.cumsum(t1, dim=0)
    print(d1)
    d2 = torch.cumsum(d1, dim=1)
    print(d2)


def corr2d():
    X = torch.arange(1, 13).reshape(3, 4)
    K = torch.tensor([[1, 2], [3, 4]])
    h, w = K.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = torch.sum(X[i:i + h, j:j + w] * K)

    print(Y)


def param():
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    X = torch.rand(size=(2, 4))
    net(X)
    print(net)
    print(net.state_dict())
    print(net.named_parameters())
    print(net[2].bias)
    print(net[2].bias.data)
    print(net[2].weight)


def param_share():
    share = nn.Linear(8, 8)
    net = nn.Sequential(share, nn.ReLU(), share, nn.ReLU(), nn.Linear(8, 1))
    x = torch.randn((2, 8))
    a = net(x)
    print(net[0].weight.data[0] == net[2].weight.data[0])


def torch_tensor():
    global data
    x = torch.rand(4, 3)
    print(x)
    print("相比reshape，官方建议使用clone后，再view")
    y = torch.zeros(4, 3, 4, dtype=torch.int32)
    print(y)
    y_c = y.clone().view(-1, 2)
    print("y_c")
    print(y_c)
    z = y.new_zeros(4, 3, dtype=torch.float32)
    print(z)
    k = torch.randn(4, 3, dtype=torch.float32)
    print(k)
    print("in-place，原值修改")
    v = k.copy_(z)
    print(k)
    print(v)
    # 广播
    x = torch.arange(5, 30).view(5, 5)
    print(x)
    y = torch.arange(1, 6).view(5, 1)
    print(y)
    print(x + y)
    print("自动求导")
    g = torch.arange(5, 30).view(5, 5).float().requires_grad_(True)
    print(g)
    g = g + 1
    a = g.sum()
    a.backward()
    print(a)
    print(g, g.grad_fn)
    g = g * 2 / (g - 1)
    print(g, g.grad_fn)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 指明调用的GPU为1号
    print(device)
    print("grad============")
    x = torch.randn((3), dtype=torch.float32, requires_grad=True)
    y = torch.randn((3), dtype=torch.float32, requires_grad=True)
    z = torch.randn((3), dtype=torch.float32, requires_grad=True)
    t = x + y
    loss = t.dot(z)  # 求向量的内积
    loss.backward(retain_graph=True)
    print(z, x.grad, y.grad)
    print(t, z.grad)
    #    print(t.grad)
    print("torch.mv tensor multiply a vector")
    A = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])
    v = torch.tensor([7, 8, 9])
    # Compute matrix-vector multiplication
    result = torch.mv(A, v)
    print(result)  # Output: tensor([ 50, 122])
    print("np array 与 tensor 互相转换")
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    print(id(data[1][1]), (data[1][1]))
    print(id(x_data[1][1]), x_data[1][1])
    print(id(np_array[1][1]), np_array[1][1])
    print(id(x_np[1][1]), x_np[1][1])


if __name__ == '__main__':
    cnn_kernel_learning()
