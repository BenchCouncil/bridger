"""benchmarking linear models"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,Perceptron,RidgeClassifier,RidgeClassifierCV,SGDClassifier,LinearRegression,Ridge,RidgeCV,SGDRegressor
from sklearn.datasets import make_classification,make_multilabel_classification,make_regression
import numpy as np
from sklearn.model_selection import train_test_split
from ucab.utils.common_parser import parse_linear
from benchmark.utils import bench_sklearn,bench_hb,bench_hb_all,bench_ucab
import time
import os 
from ucab.topi.linear import logistic_regression,logistic_regression_cv,ridge_classifier,ridge_classifier_cv,sgd_classifier,perceptron,linear_regression,ridge,ridge_cv,sgd_regressor
from ucab.topi.x86.preprocessing import schedule_normalizer
# Add nvcc path
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"

def bench_linear_model(sklearn_func, tvm_func, n_samples, n_features, n_classes=1, n_labels=1, number=1, dtype="float32", target="llvm"):
    if sklearn_func in [LinearRegression, Ridge, RidgeCV, SGDRegressor]:
        X, y = make_regression(n_samples=n_samples, n_features=n_features)
    else:
        if(n_labels == 1):
            n_informative = int(n_features / 2)
            X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_informative=n_informative)
        else:
            X, y = make_multilabel_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_labels=n_labels)
            
    # load dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # training model
    clf = sklearn_func()
    clf.fit(X_train, y_train)
    X_shape = X_train.shape[1]
    c, b = parse_linear(clf)
    
    # tvm implements
    X_tvm = te.placeholder((X_test.shape), name="X", dtype=dtype)
    C = te.placeholder((c.shape), name="C", dtype=dtype)
    B = te.placeholder((b.shape), name="B", dtype=dtype)
    Y_tvm = tvm_func(X_tvm, C, B)
    s = te.create_schedule(Y_tvm.op)
    ctx = tvm.context(target, 0)
    X_test = X_test.astype(np.float32)
    x_tvm = tvm.nd.array(X_test, ctx)
    y_tvm = tvm.nd.array(np.zeros(get_const_tuple(Y_tvm.shape), dtype=Y_tvm.dtype), ctx)
    func = tvm.build(s, [X_tvm, C, B, Y_tvm], target)
    sk_time = bench_sklearn(clf, X_test, number=number)
    hb_time = bench_hb_all(clf, X_test, number=number)
    ucab_time = bench_ucab(func, number, (x_tvm, c, b, y_tvm))
    print(ucab_time, sk_time, hb_time)


bench_linear_model(LogisticRegression, logistic_regression, n_samples=1000, n_features=100, n_classes=10, n_labels=1, number=1)
"""
bench_linear_model(RidgeClassifier, ridge_classifier, n_samples=1000, n_features=100, n_classes=10, n_labels=1)
bench_linear_model(SGDClassifier, sgd_classifier, n_samples=1000, n_features=100, n_classes=10, n_labels=1)
bench_linear_model(Perceptron, perceptron, n_samples=1000, n_features=100, n_classes=10, n_labels=1)
bench_linear_model(LogisticRegressionCV, logistic_regression_cv, n_samples=1000, n_features=100, n_classes=10, n_labels=1)
bench_linear_model(RidgeClassifierCV, ridge_classifier_cv, n_samples=1000, n_features=100, n_classes=10, n_labels=1)

bench_linear_model(LinearRegression, linear_regression, n_samples=1000, n_features=100)
bench_linear_model(Ridge, ridge, n_samples=1000, n_features=100)
bench_linear_model(SGDRegressor, sgd_regressor, n_samples=1000, n_features=100)
bench_linear_model(RidgeCV, ridge_cv, n_samples=1000, n_features=100)
"""
