# -*- coding: utf-8 -*-
import numpy as np
import math


def arr():
    a = np.array([1, 2, 3])
    print(a)
    b = np.array([[1, 2, 3], [2, 3, 4]])
    print(b)
    c = np.zeros((480, 120, 3), np.uint8)
    print(c)
    d = np.ones((480, 120, 3), np.uint8)
    print(d)
    e = np.full((480, 120, 3), 4, np.uint8)
    print(e)
    f = np.identity(7)
    print(f)
    g = np.eye(3, k=1)
    print(g)


def grad_loss():
    # Create random input and output data
    x = np.linspace(-math.pi, math.pi, 2000)
    y = np.sin(x)
    # Randomly initialize weights
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    d = np.random.randn()
    learning_rate = 1e-6
    for t in range(2000):
        # Forward pass: compute predicted y
        # y = a + b x + c x^2 + d x^3
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # Compute and print loss 平方和
        loss = np.square(y_pred - y).sum()
        if t % 100 == 99:
            print(t, loss)

        # Backprop to compute gradients of a, b, c, d with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x ** 2).sum()
        grad_d = (grad_y_pred * x ** 3).sum()

        # Update weights
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d
    print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')


if __name__ == '__main__':
    grad_loss()
