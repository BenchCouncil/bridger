import pandas as pd
import numpy as np 
import torch
from tensor_ra.utils import Timer

def test_addition(length):
    a = np.random.rand(length)
    b = np.random.rand(length)
    pandas_a =  pd.DataFrame(data=a)
    pandas_b =  pd.DataFrame(data=b)
    torch_a = torch.from_numpy(a)
    torch_b = torch.from_numpy(b)
    with Timer() as t1:
        pandas_out = pandas_a.add(pandas_b)
    with Timer() as t2:
        torch_out = torch.add(torch_a, torch_b)
    return t1.interval * 1000, t2.interval * 1000   

def test_matmul(size):
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    pandas_a =  pd.DataFrame(data=a)
    pandas_b =  pd.DataFrame(data=b)
    torch_a = torch.from_numpy(a)
    torch_b = torch.from_numpy(b)
    with Timer() as t1:
        pandas_out = pandas_a.dot(pandas_b)
    with Timer() as t2:
        torch_out = torch.matmul(torch_a, torch_b)
    return t1.interval * 1000, t2.interval * 1000   


def bench_linear():
    length = 10000000
    t1, t2 = test_addition(length)
    print("pandas,addition,"+str(length)+","+str(t1))
    print("tensorra,addition,"+str(length)+","+str(t2))
    size = 4096
    t1, t2 = test_matmul(size)
    print("pandas,matmul,"+str(size)+","+str(t1))
    print("tensorra,matmul,"+str(size)+","+str(t2))
    return 0

bench_linear()

