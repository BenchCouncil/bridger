from tvm import testing
import os
import json
import tarfile
import pathlib
import tempfile
import numpy as np
import tvm
from tvm import relay
from tvm import rpc
import tvm.contrib.utils
from tvm.contrib import graph_executor
from tvm.micro import export_model_library_format
import time

def create_module(batch_size, n_feature, n_class, weight_sample):
    data_shape = (batch_size, n_feature)
    weight_shape = (n_class, n_feature)
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    weight = relay.var("weight", shape=weight_shape, dtype=dtype)
    y = relay.nn.dense(data, weight, units=n_class)
    f = tvm.relay.Function([data, weight], y)
    relay_mod = tvm.IRModule.from_expr(f)
    relay_mod = tvm.relay.transform.InferType()(relay_mod)
    params = {"weight": weight_sample}
    return relay_mod, params

batch_size = 1
n_feature = 128
n_class = 1
data = np.random.uniform(size=(batch_size, n_feature))
weight = np.random.uniform(size=(n_class, n_feature))
mod, params = create_module(batch_size, n_feature, n_class, weight)

def run_host(data, mod, params):
    target = "llvm"
    #target = tvm.target.riscv_cpu()
    RUNTIME = tvm.relay.backend.Runtime("crt", {"system-lib": True})
    dev = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target,runtime=RUNTIME, params=params)
    c_source_module = lib.get_lib()
    c_source_code = c_source_module.get_source()
    print(c_source_code)
    print(mod)
    model = graph_executor.GraphModule(lib["default"](dev))
    model.set_input("data", data)
    a = time.perf_counter()
    model.run()
    b = time.perf_counter()
    out = model.get_output(0, tvm.nd.empty((batch_size, n_class), "float32"))
    return out, b-a


def run_rpc(data, mod, params, host, port):
    """
    RPC for RISC-V
    """
    target = "llvm -keys=arm_cpu,cpu -device=arm_cpu -mabi=lp64d -mcpu=sifive-u74 -model=sifive-74 -mtriple=riscv64-unknown-linux-gnu"
    #target = tvm.target.riscv_cpu()
    print(mod)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params)
    # save the lib at a local temp folder
    temp_dir = tvm.contrib.utils.tempdir()
    path = temp_dir.relpath("lib.tar")
    lib.export_library(path)
    remote = rpc.connect(host, port)
    remote.upload(path)
    rlib = remote.load_module("lib.tar")
    # create arrays on the remote device
    dev = remote.cpu()
    model = graph_executor.GraphModule(rlib["default"](dev))
    model.set_input("data", data)
    a = time.perf_counter()
    model.run()
    b = time.perf_counter()
    out = model.get_output(0)
    return out, b-a

def run_simulator(data, mod, params):
    """
    Run Gem5 for RISC-V
    """
    boards_file = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr")) / "boards.json"
    with open(boards_file) as f:
        boards = json.load(f)
    #BOARD = os.getenv("TVM_MICRO_BOARD", default="qemu_riscv64")
    BOARD = os.getenv("TVM_MICRO_BOARD", default="mps2_an521")
    print(BOARD)
    TARGET = tvm.target.target.micro(boards[BOARD]["model"])
    #TARGET = "llvm -keys=arm_cpu,cpu -mcpu=cortex-m33 -model=mps2_an521"
    #TARGET = "llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=generic-rv64 -mabi=lp64d -mattr=+64bit,+m,+a,+f,+d,+c"
    #TARGET = "llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=generic-rv64"
    print(TARGET)
    RUNTIME = tvm.relay.backend.Runtime("crt", {"system-lib": True})
    
    with tvm.transform.PassContext(
        #opt_level=3, config={"tir.disable_vectorize": True}, disabled_pass=["AlterOpLayout"]
        opt_level=3
    ):
        module = relay.build(mod, target=TARGET, runtime=RUNTIME, params=params)
    #Print source code
    c_source_module = module.get_lib().imported_modules[0]
    c_source_code = c_source_module.get_source()
    print(c_source_code)
    temp_dir = tvm.contrib.utils.tempdir()
    model_tar_path = temp_dir / "model.tar"
    export_model_library_format(module, model_tar_path)
    with tarfile.open(model_tar_path, "r:*") as tar_f:
        print("\n".join(f" - {m.name}" for m in tar_f.getmembers())) 
    template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr"))
    project_options = {
        "project_type": "host_driven",
        "board": BOARD,
        "config_main_stack_size": 4096,
        "zephyr_base": os.getenv("ZEPHYR_BASE", default="/content/zephyrproject/zephyr"),
    }
    # Create a temporary directory
    temp_dir = tvm.contrib.utils.tempdir()
    generated_project_dir = temp_dir / "generated-project"
    #generated_project_dir = "project/generated-project"
    generated_project = tvm.micro.generate_project(
        template_project_path, module, generated_project_dir, project_options
    )
    # Build and flash the project
    generated_project.build()
    generated_project.flash()
    with tvm.micro.Session(transport_context_manager=generated_project.transport()) as session:
        model = tvm.micro.create_local_graph_executor(
            module.get_graph_json(), session.get_system_lib(), session.device
        )

        # Set the model parameters using the lowered parameters produced by `relay.build`.
        model.set_input(**module.get_params())
        model.set_input("data", data)
        a = time.perf_counter()
        model.run()
        b = time.perf_counter()
        out = model.get_output(0, tvm.nd.empty((batch_size, n_class), "float32"))
    return out, b-a

#host_out, host_time = run_host(data, mod, params)
#print(host_out)
#print(host_time)
#simulator_out, simulator_time = run_simulator(data, mod, params)
#host = "10.30.5.181"
#port = 9090
#rpc_out, rpc_time = run_rpc(data, mod, params, host, port)
#print(rpc_out)
#print(rpc_time)
#print(simulator_time)
#print(host_time, simulator_time)
#print(host_out)
#print(np.matmul(data, weight.T))