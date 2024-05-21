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
from ucab.utils.tree_common import convert_decision_tree,convert_random_forest
from scipy.sparse import dia_matrix

def cal_sparsity(x):
    non_zero = np.count_nonzero(x)
    total_val = np.prod(x.shape)
    return (total_val - non_zero) / total_val

def cal_diag_rate(x):
    x = dia_matrix(x)
    return x.offsets.shape[0] * 1.0 / (np.sum(x.shape) - 1)

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

def get_sparsitry(model_dir, models, datasets):
    columns = ["name", "sparsity", "diag_rate", "row", "col"]
    df = pd.DataFrame(columns=columns)
    for dataset in datasets:
        data = load_data("test_datasets", dataset)
        for model in models:
            results = []
            func_name = str(model).split("\'")[1].split(".")[-1]
            name = func_name + "_" + dataset
            results.append(name)
            clf = load_model(model_dir, func_name, dataset)
            if(func_name in ["DecisionTreeClassifier", "ExtraTreeClassifier"]):
                weights = convert_decision_tree(data.shape[1], clf, "tree_clf", "float32", "llvm", False, False)
            elif(func_name in ["DecisionTreeRegressor", "ExtraTreeRegressor"]):
                weights = convert_decision_tree(data.shape[1], clf, "tree_reg", "float32", "llvm", False, False)
            elif(func_name in ["RandomForestClassifier", "ExtraTreesClassifier"]):
                weights = convert_random_forest(data.shape[1], clf, "forest_clf", "float32", "llvm", False, False)
            elif(func_name in ["RandomForestRegressor", "ExtraTreesRegressor"]):
                weights = convert_random_forest(data.shape[1], clf, "forest_reg", "float32", "llvm", False, False)
            S = weights[0]
            S = S.asnumpy()
            results.append(cal_sparsity(S))
            results.append(None)
            #results.append(cal_diag_rate(S))
            results.append(S.shape[0])
            results.append(S.shape[1])
            df.loc[len(df)] = results
    return df


sparsity_tree = get_sparsitry("test_models", [DecisionTreeClassifier, ExtraTreeClassifier, ExtraTreeRegressor], ["year", "higgs", "epsilon", "airline"])
sparsity_dtr = get_sparsitry("test_models", [DecisionTreeRegressor], ["year", "higgs", "epsilon"])
sparsity_forest = get_sparsitry("depth4", [RandomForestClassifier, RandomForestRegressor], ["year", "higgs", "epsilon", "airline", "airline"])
sparsity = pd.concat([sparsity_tree, sparsity_dtr, sparsity_forest])
print(sparsity)
sparsity.to_csv("CMLSparse.csv", index=False)