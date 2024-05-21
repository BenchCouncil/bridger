"""
Test ucab on x86
"""
import tvm
from tvm import te, topi, relay
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
from tvm import rpc, autotvm
from tvm.autotvm.tuner import XGBTuner
from tvm.contrib.debugger.debug_executor import GraphModuleDebug

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

def run_model(data, lib, target):
    dev = tvm.cpu(0)
    model = graph_executor.GraphModule(lib["default"](dev))
    #model = GraphModuleDebug(lib["debug_create"]("default", dev), [dev], lib.graph_json, dump_root="/tmp/tvmdbg",)
    start_time = time.perf_counter()
    model.set_input("data", data)
    model.run()
    out = model.get_output(0)
    end_time = time.perf_counter()
    return end_time - start_time

def test_model(sklearn_func, dataset, model_dir, data_dir, target):
    func_name = str(sklearn_func).split("\'")[1].split(".")[-1]
    log_file = "x86_schedule.log"
    data = load_data(data_dir, dataset)
    if((sklearn_func in preprocessing_op) or (sklearn_func in feature_selectors)):
        if(sklearn_func in [LabelEncoder]):
            data = data[:,0]
        clf = sklearn_func().fit(data)
    else:
        clf = load_model(model_dir, func_name, dataset)
    input_data = np.array(data)
    ucab_model = build_model(clf, data.shape, batch_size=None, target=target, sparse_replacing=True, dtype_converting=True, elimination=True)
    mod = ucab_model.mod
    params = ucab_model.params
    #print(ucab_model.params)
    with autotvm.apply_history_best(log_file):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)  
    data = tvm.nd.array(data)
    exec_time = run_model(data, lib, target)
    return exec_time

def test_framework(models, n_repeat, dataset, model_dir, data_dir, target):
    columns = ["model", "time"]
    df = pd.DataFrame(columns=columns)
    for model in models:
        for i in range(n_repeat):
            func_name = str(model).split("\'")[1].split(".")[-1]
            exec_time = test_model(model, dataset, model_dir, data_dir, target)
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
target = "llvm"

df = test_framework(models, n_repeat, dataset, model_dir, data_dir, target)
print(df)
df.to_csv(savefile, mode = "a", index=False, header=True)
