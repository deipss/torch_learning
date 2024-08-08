import torch
import numpy as np
if __name__ == '__main__':
    x = torch.rand(4, 3)
    print(x)

    print("相比reshape，官方建议使用clone后，再view")
    y = torch.zeros(4, 3, 4, dtype=torch.int32)
    print(y)
    y_c= y.clone().view(-1,2)
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
    print(g,g.grad_fn)

    g = g * 2 /(g-1)
    print(g,g.grad_fn)

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

    print(id(data[1][1]),(data[1][1]))
    print(id(x_data[1][1]),x_data[1][1])
    print(id(np_array[1][1]),np_array[1][1])
    print(id(x_np[1][1]),x_np[1][1])