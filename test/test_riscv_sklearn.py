"""
Test spark on riscv
"""
from sklearn.preprocessing import Binarizer,LabelBinarizer,Normalizer,LabelEncoder
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,StandardScaler,RobustScaler
from sklearn.feature_selection import VarianceThreshold,SelectKBest,SelectPercentile
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,Perceptron,RidgeClassifier,RidgeClassifierCV,SGDClassifier,LinearRegression,Ridge,RidgeCV,SGDRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,ExtraTreeClassifier,ExtraTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.svm import LinearSVC,LinearSVR,NuSVC,NuSVR,SVC,SVR
import numpy as np
import pandas as pd
import time
import os 
import pickle
import sys
"""
Supported operators
"""

# Preprocessing operatros
preprocessing_op = [
        Binarizer,
        LabelBinarizer,
        Normalizer,
        LabelEncoder,
        MaxAbsScaler,
        MinMaxScaler,
        StandardScaler,
        RobustScaler
        ]

# Feature Selectors
feature_selectors = [
        SelectKBest,
        VarianceThreshold
        ]

# Linear-based classifier
linear_clf = [
        LogisticRegression,
        LogisticRegressionCV,
        Perceptron,
        RidgeClassifier,
        RidgeClassifierCV,
        SGDClassifier,
        LinearSVC
        ]

# Linear-based regressor
linear_reg = [
        LinearRegression,
        Ridge,
        RidgeCV,
        SGDRegressor,
        LinearSVR
        ]

# Linear-based models
linear_ops = linear_clf + linear_reg

# Tree-based classifier
tree_clf = [
        DecisionTreeClassifier,
        ExtraTreeClassifier
        ]

# Tree-based regressor
tree_reg = [
        DecisionTreeRegressor,
        ExtraTreeRegressor
        ]

# Tree-based models
tree_ops = tree_clf + tree_reg

# Ensemble Classifier
ensemble_clf = [
        RandomForestClassifier,
        ExtraTreesClassifier
        ]

# Ensemble Regressor
ensemble_reg = [
        RandomForestRegressor,
        ExtraTreesRegressor
        ]

# Ensemble models
ensemble_ops = ensemble_clf + ensemble_reg

#SVM Classifer
svm_clf = [
        SVC,
        NuSVC
        ]

# SVM Regressor
svm_reg = [
        SVR,
        NuSVR
        ]

# SVM
svm_ops = svm_clf + svm_reg

# All Classifiers
clf_ops = tree_clf + linear_clf + ensemble_clf + svm_clf

# All Regressors
reg_ops = tree_reg + linear_reg + ensemble_reg + svm_reg

# All operators
ops = clf_ops + reg_ops

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

def test_model(sklearn_func, dataset, model_dir, data_dir):
    func_name = str(sklearn_func).split("\'")[1].split(".")[-1]
    data = load_data(data_dir, dataset)
    if((sklearn_func in preprocessing_op) or (sklearn_func in feature_selectors)):
        if(sklearn_func in [LabelEncoder]):
            data = data[:,0]
        clf = sklearn_func().fit(data)
    else:
        clf = load_model(model_dir, func_name, dataset)
    input_data = np.array(data)
    try:
        clf.transform = clf.predict
    except Exception as e:
        pass
    start_time = time.perf_counter()
    out_data = clf.transform(input_data)
    end_time = time.perf_counter()
    return end_time - start_time

def test_framework(models, n_repeat, dataset, model_dir, data_dir):
    columns = ["model", "time"]
    df = pd.DataFrame(columns=columns)
    for model in models:
        for i in range(n_repeat):
            func_name = str(model).split("\'")[1].split(".")[-1]
            exec_time = test_model(model, dataset, model_dir, data_dir)
            result = [func_name, exec_time]
            df.loc[len(df)] = result
    return df

models = [
        #Binarizer,
        #Normalizer,
        #MinMaxScaler,
        #RobustScaler,
        LinearRegression, 
        #LogisticRegression, 
        #SGDClassifier,
        #DecisionTreeClassifier, 
        #DecisionTreeRegressor, 
        #RandomForestClassifier,
        #ExtraTreeClassifier,
        #ExtraTreesClassifier,
        #LinearSVR,
        #LinearSVC
    ]
n_repeat = int(sys.argv[1])
savefile = sys.argv[2]
dataset = "year"
model_dir = "depth4"
data_dir = "test_datasets"

df = test_framework(models, n_repeat, dataset, model_dir, data_dir)
print(df)
df.to_csv(savefile, mode = "a", index=False, header=True)



