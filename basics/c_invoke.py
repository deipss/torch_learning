import ctypes

if __name__ == '__main__':

    '''调用C语言的代码，需要将libhello.so文件放在对应的目录中'''
    lib = ctypes.CDLL('../data/libhello.so')

    # 调用 hello 函数

    # 调用 add 函数
    result = lib.add(3, 4)
    print(result)