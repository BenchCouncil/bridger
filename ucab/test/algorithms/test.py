"""test tree models based on gemm, including Decision Tree and Extra Tree"""
import tvm
from tvm import te, topi
from tvm.topi.utils import get_const_tuple
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,ExtraTreeClassifier,ExtraTreeRegressor
from sklearn.datasets import make_classification,make_regression
import numpy as np
from sklearn.model_selection import train_test_split
#from hummingbird.ml import convert
from ucab.model import build_model
import os
import tvm.testing
from parse_tree import parse_tree
from sklearn import tree
import pickle

# Add nvcc path
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"
os.environ["TVM_BACKTRACE"] = "1"

def new_tree(x, S, T, B, L):
    """
        
    """
    y = np.matmul(x, S.T)
    y = np.greater(y, T)
    y = np.matmul(y, B.T)
    y = np.argmax(y, axis=-1)
    y = np.take(L, y)
    return y  

def test_tree_gemm(sklearn_func, max_depth, n_samples, n_features, n_classes, dtype="float32", target="llvm"):
    """
    Testing tree based on gemm
    Input: sklearn function and corresponding tvm function
    Output: The result equals or not 
    """
    if(sklearn_func in [DecisionTreeRegressor, ExtraTreeRegressor]):
        classification = False
        X, y = make_regression(n_samples=n_samples, n_features=n_features)
        out_dtype = "float32"
    else:
        classification = True
        n_informative = int(n_features / 2)
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_informative=n_informative)
        out_dtype = "int8"
    
    # load dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # sklearn implements
    clf = sklearn_func(max_depth=max_depth, random_state=0)
    clf.fit(X_train, y_train)
    #print(type(clf.classes_))
    #clf = pickle.load(open("clf.sav", "rb"))
    S, T, B, L = parse_tree(n_features, clf, "tree_reg", "float32")
    y = new_tree(X_test, S, T, B, L)
    #print(y)
    Y_test = clf.predict(X_test)
    #print(Y_test)
    for i in range(len(y)):
        if(y[i] != Y_test[i]):
            print("233")
    """
    print(clf.tree_.children_left)
    print(clf.tree_.children_right)
    print(clf.tree_.feature)
    print(clf.tree_.threshold)
    print(clf.tree_.value)
    fig = plt.figure(figsize=(25,20))
    tree.plot_tree(clf)
    fig.tight_layout()
    fig.savefig("decistion_tree.png")
    """
    #pickle.dump(clf, open("clf.sav", "wb"))
    """
    model = build_model(clf, X_test.shape, out_dtype=out_dtype, target="cuda", sparse_replacing=False, dtype_converting=False)
    y_tvm = model.run(X_test)
    # check
    y_np = np.array(Y_test)

    try:
        tvm.testing.assert_allclose(y_tvm, y_np, rtol=1e-5)
        print("pass")
    except Exception as e:
        print("error")
        print(e)
    """
#test_tree_gemm(DecisionTreeRegressor, max_depth=8, n_samples=1000, n_features=100, n_classes=10)
#test_tree_gemm(ExtraTreeRegressor, max_leaf_nodes=10, n_samples=1010, n_features=10, n_classes=2)
test_tree_gemm(DecisionTreeClassifier, max_depth=4, n_samples=100, n_features=10, n_classes=4)
#test_tree_gemm(ExtraTreeClassifier, max_leaf_nodes=10, n_samples=1000, n_features=10, n_classes=2)

