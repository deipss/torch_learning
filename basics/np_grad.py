# -*- coding: utf-8 -*-
import numpy as np
import math

def arr():
    """
    Demonstrates the creation and printing of arrays using NumPy.

    This function showcases various ways to create arrays, including 1D and 2D arrays with specific values,
    arrays filled with zeros, ones, a specific value, identity matrices, and arrays with specific patterns.
    """
    # Create and print a 1D array
    a = np.array([1, 2, 3])
    print(a)

    # Create and print a 2D array
    b = np.array([[1, 2, 3], [2, 3, 4]])
    print(b)

    # Create and print a 3D array filled with zeros
    c = np.zeros((480, 120, 3), np.uint8)
    print(c)

    # Create and print a 3D array filled with ones
    d = np.ones((480, 120, 3), np.uint8)
    print(d)

    # Create and print a 3D array filled with a specific value
    e = np.full((480, 120, 3), 4, np.uint8)
    print(e)
    # print matrix E

    # Create and print an identity matrix
    f = np.identity(7)
    print(f)

    # Create and print a matrix with a specific diagonal
    #建5x4的矩阵，主对角线以下的第 2 个对角线上的元素为1
    g = np.eye(3, k=-2)
    print(g)


def grad_loss():
    """
    Gradually decrease the loss function through gradient descent to fit the sine curve.
    This function does not accept parameters, but initializes weights randomly and uses gradient descent to minimize the loss function.
    """
    # Create random input and output data
    x = np.linspace(-math.pi, math.pi, 2000)
    y = np.sin(x)
    # Randomly initialize weights
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    d = np.random.randn()
    # Define learning rate
    learning_rate = 1e-6
    # Gradient descent iteration
    for t in range(2000):
        # Forward pass: compute predicted y
        # y = a + b x + c x^2 + d x^3
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # Compute and print loss 平方和
        loss = np.square(y_pred - y).sum()
        # Print loss every 100 iterations
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
    # Print final fitting result
    print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')


if __name__ == '__main__':
    arr()
