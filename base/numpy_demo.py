import numpy as np

def arr():
    a=np.array([1,2,3])
    print(a)
    b=np.array([[1,2,3],[2,3,4]])
    print(b)
    c= np.zeros((480,120,3),np.uint8)
    print(c)
    d=np.ones((480,120,3),np.uint8)
    print(d)
    e=np.full((480,120,3),4,np.uint8)
    print(e)
    f=np.identity(7)
    print(f)
    g=np.eye(3,k=1)
    print(g)

if __name__ == '__main__':
    arr();