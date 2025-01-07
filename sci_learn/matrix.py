import numpy as np
import matplotlib.pyplot as plt


def gram_schmidt(vectors):
    """
    使用施密特正交化方法将一组线性无关的向量转化为标准正交基。

    参数:
    vectors (list of lists): 线性无关的向量列表

    返回:
    list of np.ndarray: 标准正交基
    """
    vectors = [np.array(v).astype(float) for v in vectors]
    orthogonal_vectors = []

    for i, v in enumerate(vectors):
        w = v.copy()
        for u in orthogonal_vectors:
            projection = np.dot(w, u) / np.dot(u, u) * u
            w -= projection
        if not np.allclose(w, 0):
            orthogonal_vectors.append(w / np.linalg.norm(w))

    return orthogonal_vectors


def show_2dim():
    # 定义矩阵 A 和 B
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    # 左乘 AB
    left_multiplication = np.dot(A, B)
    print("左乘 AB:\n", left_multiplication)
    # 右乘 BA
    right_multiplication = np.dot(B, A)
    print("右乘 BA:\n", right_multiplication)
    # 可视化原始矩阵 B
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title('Original Matrix B')
    plt.quiver([0, 0], [0, 0], B[:, 0], B[:, 1], angles='xy', scale_units='xy', scale=1, color=['r', 'b'])
    plt.xlim(-10, 20)
    plt.ylim(-10, 20)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    # 可视化左乘结果 AB
    plt.subplot(1, 3, 2)
    plt.title('Left Multiplication AB')
    plt.quiver([0, 0], [0, 0], left_multiplication[:, 0], left_multiplication[:, 1], angles='xy', scale_units='xy',
               scale=1, color=['r', 'b'])
    plt.xlim(-10, 60)
    plt.ylim(-10, 60)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    # 可视化右乘结果 BA
    plt.subplot(1, 3, 3)
    plt.title('Right Multiplication BA')
    plt.quiver([0, 0], [0, 0], right_multiplication[:, 0], right_multiplication[:, 1], angles='xy', scale_units='xy',
               scale=1, color=['r', 'b'])
    plt.xlim(-10, 60)
    plt.ylim(-10, 60)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()


def eigen_vector():
    import numpy as np
    import matplotlib.pyplot as plt

    # 定义矩阵 A
    A = np.array([[2, 1],
                  [1, 2]])

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(A)

    print("eigen value:", eigenvalues)
    print("eigen vector:\n", eigenvectors)

    # 可视化原始向量和变换后的向量
    plt.figure(figsize=(8, 8))

    # 绘制坐标轴
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    # 绘制原始特征向量
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, label=f'raw $v_{i + 1}$',
                   color=['r', 'g'][i])

    # 绘制变换后的特征向量
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        Av = A @ v
        plt.quiver(0, 0, Av[0], Av[1], angles='xy', scale_units='xy', scale=1, label=f'transform $Av_{i + 1}$',
                   color=['b', 'm'][i])

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('transform')
    plt.legend()
    plt.show()


def orthonormal_show():
    vectors = [
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ]
    orthonormal_basis = gram_schmidt(vectors)
    print("原始向量:")
    for v in vectors:
        print(v)
    print("\n标准正交基:")
    for u in orthonormal_basis:
        print(u)


if __name__ == '__main__':
    pass
