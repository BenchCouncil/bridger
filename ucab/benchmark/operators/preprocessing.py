"""test preprocessing"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
from sklearn.preprocessing import Binarizer,LabelBinarizer,Normalizer,LabelEncoder
import numpy as np
from ucab.topi.preprocessing import binarizer,label_binarizer,normalizer,label_encoder
from benchmark.utils import bench_sklearn,bench_hb,bench_hb_all,bench_ucab
import time
import os 
from ucab.topi.x86.common import schedule_fuse_parallel,schedule_parallel_vectorize
from ucab.topi.x86.preprocessing import schedule_normalizer
# Add nvcc path
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"

def bench_binarizer(*shape, threshold, number, dtype="float32", target="llvm"):
    # training model
    a_np = np.random.randn(*shape).astype(dtype=dtype)
    transformer = Binarizer(threshold=threshold, copy=False).fit(a_np)
    
    # tvm implements
    A = te.placeholder((shape), name="A", dtype=dtype)
    B = binarizer(A, threshold, dtype)
    #s = ut.x86.preprocessing.schedule_binarizer(B)
    s = schedule_fuse_parallel(B)
    ctx = tvm.context(target, 0)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    
    func = tvm.build(s, [A, B], target, name = "binarizer")
    #print(tvm.lower(s, (A, B), simple_mode=True))
    sk_time = bench_sklearn(transformer, a_np, number=number)
    hb_time = bench_hb_all(transformer, a_np, number=number)
    ucab_time = bench_ucab(func, number, (a,b))
    print(ucab_time, sk_time, hb_time)

def bench_label_binarizer(shape, number, target="llvm", dtype="int64"):
    # training model
    a_np = np.random.randn(shape).astype(dtype=dtype)
    transformer = LabelBinarizer().fit(a_np)
    classes = transformer.classes_
    b_np = np.random.randn(shape).astype(dtype=dtype)
    
    # tvm implements
    A = te.placeholder((shape,), dtype=dtype)
    C = te.placeholder((len(classes),), dtype=dtype) 
    Y = label_binarizer(A, C)
    s = schedule_fuse_parallel(Y)
    ctx = tvm.context(target, 0)
    a = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(classes, ctx)
    y = tvm.nd.array(np.zeros(get_const_tuple(Y.shape), dtype=Y.dtype), ctx)
    func = tvm.build(s, [A, C, Y], target, name = "label_binarizer")
    
    sk_time = bench_sklearn(transformer, b_np, number=number)
    #hummingbird unsupported
    #hb_time = bench_hb_all(transformer, b_np, number=number)
    ucab_time = bench_ucab(func, number, (a,c,y))
    print(ucab_time, sk_time)

def bench_norm(*shape, norm, number, dtype="float32", target="llvm"):
    # training model
    a_np = np.random.randn(*shape).astype(dtype=dtype)
    transformer = Normalizer(norm=norm).fit(a_np)
    
    # tvm implements
    A = te.placeholder((shape), name="A", dtype=dtype)
    B = normalizer(A, norm=norm)
    s = schedule_normalizer(B)
    ctx = tvm.context(target, 0)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func = tvm.build(s, [A, B], target, name = "normalizer")
    print(tvm.lower(s, (A, B), simple_mode=True))
    print(*B.op.axis)
    print(len(B.op.axis))
    sk_time = bench_sklearn(transformer, a_np, number=number)
    hb_time = bench_hb_all(transformer, a_np, number=number)
    print("start")
    ucab_time = bench_ucab(func, number, (a,b))
    print(ucab_time, sk_time, hb_time)

def test_label_encoder(shape, target="llvm", dtype="int64"):
    # sklearn implements
    a_np = np.random.randn(shape).astype(dtype=dtype)
    transformer = LabelEncoder().fit(a_np)
    classes = transformer.classes_
    #b_np = np.random.randn(shape).astype(dtype=dtype)
    b_np = a_np
    y_np = transformer.transform(b_np)
    # tvm implements
    A = te.placeholder((shape,), dtype=dtype)
    C = te.placeholder((len(classes),), dtype=dtype) 
    Y = label_encoder(A, C)
    s = te.create_schedule(Y.op)
    ctx = tvm.context(target, 0)
    a = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(classes, ctx)
    y = tvm.nd.array(np.zeros(get_const_tuple(Y.shape), dtype=Y.dtype), ctx)
    func = tvm.build(s, [A, C, Y], target, name = "label_binarizer")
    func(a, c, y)
    # check
    try:
        tvm.testing.assert_allclose(y.asnumpy(), y_np, rtol=1e-5)
        print("pass")
    except Exception as e:
        print("error")
        print(e)

#bench_binarizer(100000, 1000, number=1, threshold=0, target="llvm")
#bench_label_binarizer(1000, number=1, target="llvm", dtype="int64")
norm_list = ["l1", "l2", "max"]
for norm in norm_list:
    bench_norm(100000, 1000, number=1, norm=norm, target="llvm -mcpu=core-avx2")

def bench_log_softmax(*shape, dtype="float32", target="llvm"):
    # training model
    a_np = np.random.randn(*shape).astype(dtype=dtype)
    
    # tvm implements
    A = te.placeholder((shape), name="A", dtype=dtype)
    B = topi.nn.log_softmax(A)
    s = topi.x86.schedule_softmax(B)
    ctx = tvm.context(target, 0)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    func = tvm.build(s, [A, B], target, name = "normalizer")
    print(tvm.lower(s, (A, B), simple_mode=True))

#bench_log_softmax(10000, 1000)
