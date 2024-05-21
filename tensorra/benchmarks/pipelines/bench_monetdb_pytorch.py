import pandas as pd
import time
import torch
from torch.utils import data

def bench_bixi_linear_regression(n_epoch, size):
    filename = "./tmp/monetdb_pytorch_lr_" + size + ".csv"
    a = time.perf_counter()
    trip_station = torch.tensor(pd.read_csv(filename).values)
    b = time.perf_counter()
    #Calculate Euclidean Distance
    distance = torch.sqrt(torch.sum(torch.pow((trip_station[:,4:5] - trip_station[:,7:8]), 2), axis=1))
    max_distance = torch.max(distance)
    distance = torch.div(distance, max_distance)
    duration = trip_station[:,2]
    max_duration = torch.max(duration)
    duration = torch.div(duration, max_duration)
    #Split dataset, 80% train data, 20% test data
    dataset = data.TensorDataset(distance, duration)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])
    train_x, train_y = train_dataset[:]
    #Train model: linear regression y = a * x + b 
    param_a = 1 
    param_b = 0
    #Train model: linear regression y = a * x + b 
    def square_err(y, pred_y):
        #Square error
        return torch.sum(torch.pow(y - pred_y, 2)) / (2 * (y.shape[0]))
    def grad_update(alpha, param_a, param_b, y, pred_y, x):
        #Gradint descending to update parameters
        param_a = param_a - alpha * torch.dot(pred_y - y, x) / x.shape[0]
        param_b = param_b - alpha * torch.sum(pred_y - y) / x.shape[0]
        return param_a, param_b
    x_train, y_train = train_dataset[:]
    x_test, y_test = test_dataset[:]
    alpha = 0.1
    for epoch in range(n_epoch):
        y_pred = torch.mul(x_train, param_a)
        loss = square_err(y_train, y_pred)
        param_a, param_b = grad_update(alpha, param_a, param_b, y_train, y_pred, x_train)
    #Test model
    y_pred = torch.mul(x_test, param_a) + param_b
    loss = square_err(y_test, y_pred)
    c = time.perf_counter()
    return b-a, c-b

def bench_monetdb_pytorch(nrepeat):
    #Bench bixi linear regression
    number_str = {1000:"1k", 10000:"10k", 100000:"100k", 1000000:"1m", 10000000:"10m"}
    #for i in [1000, 10000, 100000, 1000000, 10000000]:
    for i in [1000]:
        for j in range(nrepeat):
            conversion_time, linear_time = bench_bixi_linear_regression(100, number_str[i])
            print("monetdb_pytorch,linear_regression," + str(i) + "," + str(conversion_time) + "," + str(linear_time))

    """
    #Bench conference covariance
    ranking = pd.read_csv("../dataset/ranking.csv")
    ranking = ranking[["Acronym", "GGS Rating"]]
    number_str = {100:"100", 1000:"1k", 10000:"10k", 100000:"100k", 1000000:"1m"}
    for i in [100, 1000, 10000, 100000, 1000000]:
        publish = pd.read_csv("../dataset/publish_" + number_str[i] + ".csv")
        for j in range(nrepeat):
            conversion_time, relational_time, linear_time = bench_conference_covariance(publish, ranking)
            print("tensorra,covariance," + str(i) + "," + str(conversion_time) + "," + str(relational_time) + "," + str(linear_time))
    """

bench_monetdb_pytorch(5)
