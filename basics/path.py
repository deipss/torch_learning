import os

import torch


def path_print():
    print(os.path.join('..', 'data', 'c'))
    print(os.path.join('..', 'data'))
    print(os.path.join('../', 'data'))
    print(os.path.join('..', '/data'))
    print(os.path.join('..', '/data'))


def math():
    a = torch.arange(1, 7).reshape(2, 3)
    b = torch.arange(7, 13).reshape(2, 3)

    print(a)
    print(b)
    print(a * b)
    print(torch.sum(a * b))
    print(torch.mm(a, b.T))
    print(torch.norm(a.to(dtype=float), dim=0))


if __name__ == '__main__':
    pairs = [(1, 'a'), (2, 'b'), (3, 'c')]

    # *解压 zip进行元组压缩
    numbers, letters = zip(*pairs)
    print(numbers)  # 输出: (1, 2, 3)
    print(letters)  # 输出: ('a', 'b', 'c')
