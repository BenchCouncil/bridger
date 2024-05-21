import numpy as np

import tvm
from tvm import te
from tvm import rpc
from tvm.contrib import utils

n = tvm.runtime.convert(1024)
A = te.placeholder((n,), name="A")
B = te.compute((n,), lambda i: A[i] + 1.0, name="B")
s = te.create_schedule(B.op)

#target = "llvm -mtriple=riscv64-unknown-linux-gnu"
target = "llvm -keys=arm_cpu,cpu -device=arm_cpu -mabi=lp64d -mcpu=sifive-u74 -model=sifive-74 -mtriple=riscv64-unknown-linux-gnu"
#target = tvm.target.riscv_cpu()
print(target)
func = tvm.build(s, [A, B], target=target, name="add_one")
print("1212")
# save the lib at a local temp folder
temp = utils.tempdir()
path = temp.relpath("lib.tar")
func.export_library(path)
print("233")
host = "10.30.5.181"
port = 9090
remote = rpc.connect(host, port)

remote.upload(path)
func = remote.load_module("lib.tar")

# create arrays on the remote device
dev = remote.cpu()
a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), dev)
b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), dev)
# the function will run on the remote device
func(a, b)
np.testing.assert_equal(b.numpy(), a.numpy() + 1)
print(b)

time_f = func.time_evaluator(func.entry_name, dev, number=10)
cost = time_f(a, b).mean
print("%g secs/op" % cost)