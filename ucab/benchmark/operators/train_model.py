"""training sklearn models"""
from ucab.utils.common_parser import convert_from_sklearn
from benchmark.utils import generate_dataset
import os 
import pickle
from ucab.utils.supported_ops import ops, clf_ops, reg_ops


def train_linear_model(sklearn_func, model_dir, *data):
    X, y = data
    # training model
    
    clf = sklearn_func()
    clf.fit(X, y)
    func_name = str(sklearn_func).split("\'")[1].split(".")[-1]
    filename = func_name + ".sav"
    filename = os.path.join(model_dir, filename)
    pickle.dump(clf, open(filename, 'wb'))

def train_models(n_samples, model_dir):
    # Using generated data
    print("generating data\n")
    X_clf, y_clf = generate_dataset(n_samples, n_features=100, n_classes=10, n_labels=1, regression=False)
    X_reg, y_reg = generate_dataset(n_samples, n_features=100, n_classes=10, n_labels=1, regression=True)
    #for model in [LogisticRegression,LogisticRegressionCV,Perceptron,RidgeClassifier,RidgeClassifierCV,SGDClassifier]:
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    for model in ops:
        func_name = str(model).split("\'")[1].split(".")[-1]
        print("training " + func_name + "\n")
        if model in reg_ops:
            train_linear_model(model, model_dir, X_reg, y_reg)
        elif model in clf_ops:
            train_linear_model(model, model_dir, X_clf, y_clf)

train_models(3000000, "models")
