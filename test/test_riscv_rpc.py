"""
Test ucab on riscv using rpc
"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
from tvm.contrib import graph_executor
from sklearn.preprocessing import Binarizer,LabelBinarizer,Normalizer,LabelEncoder
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,StandardScaler,RobustScaler
from sklearn.feature_selection import VarianceThreshold,SelectKBest,SelectPercentile
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,Perceptron,RidgeClassifier,RidgeClassifierCV,SGDClassifier,LinearRegression,Ridge,RidgeCV,SGDRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,ExtraTreeClassifier,ExtraTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.svm import LinearSVC,LinearSVR,NuSVC,NuSVR,SVC,SVR
import numpy as np
import pandas as pd
from ucab.utils.supported_ops import tree_clf,ensemble_clf,preprocessing_op,feature_selectors
from benchmark.utils import bench_sklearn,bench_hb,bench_hb_all,bench_ucab,generate_dataset,check_data
import time
import os 
import pickle
from ucab.model import build_model,tune_model,load_tune,tune_log_name
import sys
import tarfile
import pathlib
import tempfile
from tvm import rpc

os.environ["TVM_BACKTRACE"] = "1"

def load_model(model_dir, func_name, dataset):
    # Load model
    filename = func_name + "_" + dataset + ".sav"
    filename = os.path.join(model_dir, filename)
    clf = pickle.load(open(filename, 'rb'))
    return clf

def load_data(data_dir, dataset):
    # Load data
    data_name = os.path.join(data_dir, dataset + ".dat")
    data = pickle.load(open(data_name, 'rb'))
    return data

def run_rpc(data, lib, host, port, target):
    """
    RPC for RISC-V
    """
    #target = tvm.target.riscv_cpu()
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
    data = tvm.nd.array(data, device=dev)
    start_time = time.perf_counter()
    model.set_input("data", data)
    model.run()
    out = model.get_output(0)
    end_time = time.perf_counter()
    return end_time - start_time

def test_model(sklearn_func, dataset, model_dir, data_dir, target, host, port):
    func_name = str(sklearn_func).split("\'")[1].split(".")[-1]
    data = load_data(data_dir, dataset)
    if((sklearn_func in preprocessing_op) or (sklearn_func in feature_selectors)):
        if(sklearn_func in [LabelEncoder]):
            data = data[:,0]
        clf = sklearn_func().fit(data)
    else:
        clf = load_model(model_dir, func_name, dataset)
    input_data = np.array(data)
    ucab_model = build_model(clf, data.shape, batch_size=None, target=target, sparse_replacing=True, dtype_converting=True, elimination=True)
    exec_time = run_rpc(data, ucab_model.lib, host, port, target)
    return exec_time

def test_framework(models, n_repeat, dataset, model_dir, data_dir, target, host, port):
    columns = ["model", "time"]
    df = pd.DataFrame(columns=columns)
    for model in models:
        for i in range(n_repeat):
            func_name = str(model).split("\'")[1].split(".")[-1]
            print("executing " + func_name)
            exec_time = test_model(model, dataset, model_dir, data_dir, target, host, port)
            result = [func_name, exec_time]
            df.loc[len(df)] = result
    return df

models = [
        Binarizer,
        Normalizer,
        MinMaxScaler,
        RobustScaler,
        LinearRegression, 
        LogisticRegression, 
        SGDClassifier,
        DecisionTreeClassifier, 
        DecisionTreeRegressor, 
        #RandomForestClassifier,
        ExtraTreeClassifier,
        LinearSVR,
        LinearSVC
    ]
n_repeat = int(sys.argv[1])
savefile = sys.argv[2]
dataset = "year"
model_dir = "../ucab/benchmark/operators/depth4"
data_dir = "../ucab/benchmark/operators/test_datasets"
#target = "llvm -keys=arm_cpu,cpu -device=arm_cpu -mabi=lp64d -mcpu=sifive-u74 -model=sifive-74 -mtriple=riscv64-unknown-linux-gnu"
#target = tvm.target.Target("llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=rocket-rv64 -mabi=lp64d -mattr=+64bit,+m,+a,+f,+d,+c")
target = "llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=rocket-rv64 -mabi=lp64d -mattr=+64bit,+m,+a,+f,+d,+c"

host = "10.30.5.181"
port = 9090

df = test_framework(models, n_repeat, dataset, model_dir, data_dir, target, host, port)
print(df)
df.to_csv(savefile, mode = "a", index=False, header=True)
