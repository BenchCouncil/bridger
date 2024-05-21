import pandas as pd
import time
import torch
from torch.utils import data

def bench_bixi_linear_regression(n_epoch, station, trip):
    """
    Bench linear regression, using BIXI dataset
    """
    a = time.perf_counter()
    #Relational operators, including one selection and two inner joins
    trip = trip[trip["start_station_code"]!=trip["end_station_code"]] 
    station_trip = pd.merge(trip, station, left_on="start_station_code", right_on="code", sort=True).rename({"latitude":"start_latitude", "longitude": "start_longitude"}, axis="columns")
    station_trip = pd.merge(station_trip, station, left_on="end_station_code", right_on="code", sort=True).rename({"latitude":"end_latitude", "longitude": "end_longitude"}, axis="columns")
    #station_trip = station_trip[["duration_sec", "start_latitude", "start_longitude", "end_latitude", "end_longitude"]]
    b = time.perf_counter()
    trip_station = torch.tensor(station_trip.values)
    c = time.perf_counter()
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
    d = time.perf_counter()
    return b-a, c-b, d-c

def bench_conference_covariance(publish, ranking):
    """
    Bench conference covariance computation,using DBLP dataset 
    """
    ranking = ranking.rename(columns={"Acronym":"conference", "GGS Rating":"rating"})
    a = time.perf_counter()
    publish_tensor = torch.tensor(publish.values)

    b = time.perf_counter()
    #Calculate covariance
    publish_tensor = torch.cov(publish_tensor.T)
    c = time.perf_counter()
    publish_new = pd.DataFrame(publish_tensor.numpy(), columns=publish.columns)
    publish_new["conference"] = publish.columns
    publish = publish_new
    d = time.perf_counter()
    #Join
    res_conf = pd.merge(publish, ranking, left_on="conference", right_on="conference", sort=True)
    res_conf = res_conf[res_conf["rating"] == "A++"]
    e = time.perf_counter()
    return b - a + d - c, e - d, c - b

def bench_pandas_pytorch(nrepeat):
    #Bench bixi linear regression
    station = pd.read_csv("../dataset/station.csv")
    trip = pd.read_csv("../dataset/trip.csv")
    number_str = {1000:"1k", 10000:"10k", 100000:"100k", 1000000:"1m", 10000000:"10m"}
    for i in [1000, 10000, 100000, 1000000, 10000000]:
        trip = pd.read_csv("../dataset/trip_" + number_str[i] + ".csv")
        for j in range(nrepeat):
            relational_time, conversion_time, linear_time = bench_bixi_linear_regression(100, station, trip)
            print("pandas_pytorch,linear_regression," + number_str[i] + "," + str(conversion_time) + "," + str(linear_time) + "," + str(relational_time))

    #Bench conference covariance
    ranking = pd.read_csv("../dataset/ranking.csv")
    ranking = ranking[["Acronym", "GGS Rating"]]
    number_str = {100:"100", 1000:"1k", 10000:"10k", 100000:"100k", 1000000:"1m"}
    for i in [100, 1000, 10000, 100000, 1000000]:
        publish = pd.read_csv("../dataset/publish_" + number_str[i] + ".csv")
        for j in range(nrepeat):
            conversion_time, relational_time, linear_time = bench_conference_covariance(publish, ranking)
            print("pandas_pytorch,covariance," + number_str[i] + "," + str(conversion_time) + "," + str(linear_time) + "," + str(relational_time))

bench_pandas_pytorch(5)
