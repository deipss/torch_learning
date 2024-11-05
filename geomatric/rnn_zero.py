import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.nn import init as init
from d2l import torch as d2l


def init_data():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    X = torch.arange(10).reshape((2, 5))
    Y = F.one_hot(X.T, 28)
    print(Y.shape)


def get_params(vocab_size, num_hiddens, device):
    num_input = num_output = vocab_size

    W_xh = nn.Parameter(torch.randn((num_input, num_hiddens), device=device))
    init.normal_(W_xh, mean=0, std=0.01)
    W_hh = nn.Parameter(torch.randn((num_hiddens, num_hiddens), device=device))
    init.normal_(W_hh, mean=0, std=0.01)
    b_h = torch.zero(num_hiddens, device=device)

    W_hq = nn.Parameter(torch.randn((num_hiddens, num_output), device=device))
    init.normal_(W_hq, mean=0, std=0.01)
    b_q = torch.zero(num_output, device=device)

    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])


def get_gru_params(vocab_size, num_hiddens, device):
    num_input = num_output = vocab_size

    def three():
        W_xh = nn.Parameter(torch.randn((num_input, num_hiddens), device=device))
        init.normal_(W_xh, mean=0, std=0.01)
        W_hh = nn.Parameter(torch.randn((num_hiddens, num_hiddens), device=device))
        init.normal_(W_hh, mean=0, std=0.01)
        b_h = torch.zero(num_hiddens, device=device)
        return W_xh, W_hh, b_h

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数

    W_hq = nn.Parameter(torch.randn((num_hiddens, num_output), device=device))
    init.normal_(W_hq, mean=0, std=0.01)
    b_q = torch.zero(num_output, device=device)

    return nn.ParameterList([W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q])


def get_lstm_params(vocab_size, num_hiddens, device):
    num_input = num_output = vocab_size

    def three():
        W_xh = nn.Parameter(torch.randn((num_input, num_hiddens), device=device))
        init.normal_(W_xh, mean=0, std=0.01)
        W_hh = nn.Parameter(torch.randn((num_hiddens, num_hiddens), device=device))
        init.normal_(W_hh, mean=0, std=0.01)
        b_h = torch.zero(num_hiddens, device=device)
        return W_xh, W_hh, b_h

    W_xf, W_hf, b_f = three()  # 遗忘门
    W_xi, W_hi, b_i = three()  # 输入门
    W_xo, W_ho, b_o = three()  # 输出门
    W_xc, W_hc, b_c = three()  # 记忆

    W_hq = nn.Parameter(torch.randn((num_hiddens, num_output), device=device))
    init.normal_(W_hq, mean=0, std=0.01)
    b_q = torch.zero(num_output, device=device)

    return nn.ParameterList([W_xf, W_hf, b_f, W_xi, W_hi, b_i, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q])

    return


def lstm(inputs, state, params, condition):
    W_xf, W_hf, b_f, W_xi, W_hi, b_i, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        F = torch.sigmoid(torch.mm(X, W_xf) + torch.mm(H, W_hf) + b_f)
        I = torch.sigmoid(torch.mm(X, W_xi) + torch.mm(H, W_hi) + b_i)
        O = torch.sigmoid(torch.mm(X, W_xo) + torch.mm(H, W_ho) + b_o)
        C_tilda = torch.tanh(torch.mm(X, W_xc) + torch.mm(H, W_hc) + b_c)
        C = F * condition + C_tilda * I
        H = O * torch.tanh(C)
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        R = torch.sigmoid(torch.mm(X, W_xr) + torch.mm(H, W_hr) + b_r)
        Z = torch.sigmoid(torch.mm(X, W_xz) + torch.mm(H, W_hz) + b_z)
        H_tilda = torch.tanh(torch.mm(X, W_xh) + torch.mm(R * H, W_hh) + b_h)
        H = Z @ H + (1 - Z) @ H_tilda
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


class RNNModelScratch:  # @save
    """从零开始实现的循环神经网络模型"""

    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


def grad_clipping(net, theta):  # @save
    params = []
    if isinstance(net, nn.Module):
        for param in net.parameters():
            if param.requires_grad:
                params.append(param)
            else:
                params = net.parameters()
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


if __name__ == '__main__':
    num_hiddens = 512
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    X = torch.arange(10).reshape((2, 5))
    net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                          init_rnn_state, rnn)
    state = net.begin_state(X.shape[0], d2l.try_gpu())
    Y, new_state = net(X.to(d2l.try_gpu()), state)
    Y.shape, len(new_state), new_state[0].shape
