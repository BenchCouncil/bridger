"""testing models"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
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

# Add nvcc path
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"
os.environ["TVM_BACKTRACE"] = "1"
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
np.set_printoptions(threshold=1000)

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

def _rewriting(df, sklearn_func, dataset, model_dir, data_dir, batch_size=None):
    func_name = str(sklearn_func).split("\'")[1].split(".")[-1]
    data = load_data(data_dir, dataset)
    if(batch_size == None):
        batch_size = data.shape[0]
    clf = load_model(model_dir, func_name, dataset)
    results = [func_name, dataset]
    for target in target_list:
        results.append(target)
        model = build_model(clf, data.shape, sparse_replacing=False, dtype_converting=False, batch_size=batch_size, target=target)
        load_time, exec_time, store_time, take_time, ucab_time = bench_ucab(model, 1, data, check_flag=False)
        results.append(ucab_time)
        for sparse_type in  ["csr", "bsr", "csc", "coo", "dia", "lil"]:
            model = build_model(clf, data.shape, sparse_replacing=sparse_type, dtype_converting=True, batch_size=batch_size, target=target)
            if(func_name in ["RandomForestClassifier", "ExtraTreesClassifier", "RandomForestRegressor", "ExtraTreesRegressor"] and sparse_type == "dia"):
                ucab_time = -1
            else:
                load_time, exec_time, store_time, take_time, ucab_time = bench_ucab(model, 1, data, check_flag=False)
            results.append(ucab_time)
        df.loc[len(df)] = results
        results = [func_name, dataset]
    return df

def test_rewriting(models, datasets, target_list, n_repeat, model_dir, data_dir, batch_size):
    """
    test the influence of graph rewriting
    "sparse" means only using sparse replacing
    "dtype" means only using dtype converting
    "both" means using both
    """
    columns = ["model", "dataset", "target", "base", "csr", "bsr", "csc", "coo", "dia", "lil"]
    df = pd.DataFrame(columns=columns)
    for model in models:
        for dataset in datasets:
            for j in range(n_repeat):
                df = _rewriting(df, model, dataset, model_dir, data_dir, batch_size)
    return df

models = [
        #Binarizer,
        #Normalizer,
        #MinMaxScaler,
        #MaxAbsScaler,
        #StandardScaler,
        #RobustScaler,
        #LinearRegression, 
        #LogisticRegression, 
        DecisionTreeClassifier, 
        #DecisionTreeRegressor, 
	    #ExtraTreeClassifier,
	    #ExtraTreeRegressor,
        #RandomForestClassifier,
        #RandomForestRegressor,
	    #ExtraTreesClassifier,
	    #ExtraTreesRegressor,
        #LinearSVR,
        #LinearSVC,
        ]

datasets = [
    #"fraud", 
    "year",
    #"higgs", 
    #"epsilon", 
    #"airline" 
    ]
#target_list = ["llvm -mcpu=core-avx2", "llvm"]
target_list = ["llvm"]
#model_dir = "models"
model_dir = "depth4"
#batchsize_list = [1721, 3442, 5163, 8605, 10326, 17210, 25815, 51630]

num_repeat = int(sys.argv[1])
savefile = sys.argv[2]
elim_savefile = sys.argv[3]
batchsize_list = [1721, 3442, 5163, 8605, 10326, 17210, 25815, 51630]
#for batch in batchsize_list:
#    df = test_rewriting(models, datasets, target_list, 1, model_dir, "test_datasets", batch)
#    print(df)
df = test_rewriting(models, datasets, target_list, num_repeat, model_dir, "test_datasets", 10326)
#df = test_rewriting(models, datasets, target_list, num_repeat, model_dir, "test_datasets", 1721)
df.to_csv(savefile, mode = "a", index=False)
#print(df)
#models = [SGDClassifier, LogisticRegression]
#df = run_elimination(models, datasets, target_list, num_repeat, "depth4", "test_datasets", None)
#df.to_csv(elim_savefile, mode = "a", index=False)
#print(df)
