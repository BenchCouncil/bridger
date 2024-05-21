import numpy as np

import tvm
from tvm import te
from tvm import rpc
from tvm.contrib import utils

n = tvm.runtime.convert(1024)
A = te.placeholder((n,), name="A")
B = te.compute((n,), lambda i: A[i] + 1.0, name="B")
s = te.create_schedule(B.op)
print("233")
#target = tvm.target.Target("llvm -device=arm_cpu -model=bcm2711 -mtriple=armv8l-linux-gnueabihf -mattr=+neon -mcpu=cortex-a72")
target = "llvm"
#target = "llvm -device=arm_cpu -model=bcm2711 -mtriple==armv7l-linux-gnueabihfn"
func = tvm.build(s, [A, B], target=target, name="add_one")
print("sa")
# save the lib at a local temp folder
temp = utils.tempdir()
path = temp.relpath("lib.tar")
func.export_library(path)
host = "10.130.10.42"
port = 9091
remote = rpc.connect(host, port)
remote.upload(path)
func = remote.load_module("lib.tar")
print("2333")
# create arrays on the remote device
dev = remote.cpu()
a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), dev)
b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), dev)
# the function will run on the remote device
func(a, b)
print(b)