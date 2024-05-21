import pickle
import os
from train import *

def load_test(data_dir, dataset):
    data_name = os.path.join(data_dir, dataset)
    X_test = pickle.load(open(data_name, 'rb'))
    print(X_test[0:1])
    print(X_test.shape)
    print(X_test.dtype)

def save_test(dataset):
    """
    Convert datasets trained by hummingbird to sklearn test datasets
    """
    print(dataset)
    data = pickle.load(open(os.path.join("datasets",dataset+"-pickle.dat"), "rb"))
    X_test = data.X_test
    print(X_test[0:1])
    print(X_test.shape)
    print(X_test.dtype)
    pickle.dump(X_test, open(os.path.join("test_datasets",dataset+".dat"), "wb"))

def save_model(model_name, model_dir, out_dir):
    """
    Convert models trained by hummingbird to sklearn models
    """
    print(model_name)
    model = pickle.load(open(os.path.join(model_dir,model_name + ".sav"), "rb")).model
    pickle.dump(model, open(os.path.join(out_dir,model_name + ".sav"), "wb"))

def extract_data():
    for dataset in ["fraud","year","airline","epsilon","higgs"]:
        save_test(dataset)
        load_test("test_datasets", dataset+".dat")

def extract_model(model_dir, out_dir):
    for i in os.listdir(model_dir):
        name = i.split(".")[0]
        save_model(name, model_dir, out_dir)

extract_model("new_models", "new_models_converted")
#extract_data()


