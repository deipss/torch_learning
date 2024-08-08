import ctypes

if __name__ == '__main__':


    lib = ctypes.CDLL('data/libhello.so')

    # 调用 hello 函数

    # 调用 add 函数
    result = lib.add(3, 4)
    print(result)