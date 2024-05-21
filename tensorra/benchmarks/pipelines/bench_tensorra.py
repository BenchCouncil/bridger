import pandas as pd
import time
from tensor_ra.tensor_table import TensorTable
from tensor_ra.join import join, cross_join
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils import data

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

def bench_bixi_linear_regression(n_epoch, station, trip):
    """
    Bench linear regression, using BIXI dataset
    """
    a = time.perf_counter()
    tensor_station = TensorTable()
    tensor_trip = TensorTable()
    tensor_station.from_pandas(station) 
    tensor_trip.from_pandas(trip)
    tensor_trip.tensor = tensor_trip.tensor.float()
    tensor_station.tensor = tensor_station.tensor.float()
    b = time.perf_counter()
    #Relational operators, including one selection and two inner joins
    tensor_trip.selection("not_equal", "start_station_code", "end_station_code")
    trip_station = join(tensor_trip, "start_station_code", tensor_station, "code", "inner_join", False)
    trip_station = join(trip_station, "end_station_code", tensor_station, "code", "inner_join", False)
    trip_station.rename(["start_station_code", "end_station_code", "duration_sec", "start_code", "start_latitude", "start_longitude", "end_code", "end_latitude", "end_longitude"])
    c = time.perf_counter()
    #Calculate Euclidean Distance
    distance_duration = TensorTable()
    distance = torch.sqrt(torch.sum(torch.pow((trip_station.tensor[:,4:5] - trip_station.tensor[:,7:8]), 2), axis=1))
    max_distance = torch.max(distance)
    distance = torch.div(distance, max_distance)
    duration = trip_station.tensor[:,2]
    max_duration = torch.max(duration)
    duration = torch.div(duration, max_duration)
    #distance_duration.tensor  = torch.cat((distance.reshape(-1,1), duration.reshape(-1,1)), -1)
    #distance_duration.column_name = ["distance", "duration"]
    #Split dataset, 80% train data, 20% test data
    dataset = data.TensorDataset(distance, duration)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])
    train_x, train_y = train_dataset[:]
    """
    model = linearRegression(1, 1)
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    #x_train, y_train = train_dataset[:]
    #train_size = 32
    train_loader = data.DataLoader(train_dataset, train_size)
    for epoch in range(n_epoch):
        for x_train, y_train in train_loader:
            #optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            #print(loss)
            #loss.backward()
            #optimizer.step()
    #Test model
    """
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
    end = time.perf_counter()
    return b - a, d - c, c - b

def bench_conference_covariance(publish, ranking):
    """
    Bench conference covariance computation,using DBLP dataset 
    """
    ranking = ranking.rename(columns={"Acronym":"conference", "GGS Rating":"rating"})
    a = time.perf_counter()
    tensor_publish = TensorTable()
    tensor_publish.from_pandas(publish)
    conference = pd.unique(ranking["conference"])
    conference_dict = {conference[i]:i for i in range(len(conference))}
    ranking["conference"] = ranking["conference"].map(conference_dict)
    rating = pd.unique(ranking["rating"])
    rating_dict = {rating[i]:i for i in range(len(rating))}
    ranking["rating"] = ranking["rating"].map(rating_dict)
    #tensor_publish.tensor = tensor_publish.tensor.float()
    tensor_ranking = TensorTable()
    tensor_ranking.from_pandas(ranking)
    b = time.perf_counter()
    #Calculate covariance
    tensor_publish.tensor = torch.cov(tensor_publish.tensor.T)
    tensor_conference = torch.Tensor([conference_dict[i] for i in tensor_publish.column_name])
    tensor_publish.tensor = torch.cat((tensor_conference.reshape(-1, 1), tensor_publish.tensor), 1)
    tensor_publish.column_name.insert(0, "conference")
    c = time.perf_counter()
    #Join
    tensor_publish = join(tensor_publish, "conference", tensor_ranking, "conference", "inner_join", False)
    tensor_publish.selection("equal", "rating", threshold=rating_dict["A++"])
    d = time.perf_counter()
    return b - a, c - b, d - c

#t_linear_regression = bench_bixi_linear_regression(100) 
#print("tensorra,linear_regression," + str(t_linear_regression))
#t_covariance = bench_conference_covariance()
#print("pandas,covariance," + str(t_covariance))

def bench_tensorra(nrepeat):
    #Bench bixi linear regression
    station = pd.read_csv("../dataset/station.csv")
    #trip = pd.read_csv("../dataset/trip.csv")
    number_str = {1000:"1k", 10000:"10k", 100000:"100k", 1000000:"1m", 10000000:"10m"}
    for i in [1000, 10000, 100000, 1000000, 10000000]:
        trip = pd.read_csv("../dataset/trip_" + number_str[i] + ".csv")
        for j in range(nrepeat):
            conversion_time, linear_time, relational_time = bench_bixi_linear_regression(100, station, trip)
            print("tensorra,linear_regression," + number_str[i] + "," + str(conversion_time) + "," + str(linear_time) + "," + str(relational_time))
    #Bench conference covariance
    ranking = pd.read_csv("../dataset/ranking.csv")
    ranking = ranking[["Acronym", "GGS Rating"]]
    number_str = {100:"100", 1000:"1k", 10000:"10k", 100000:"100k", 1000000:"1m"}
    for i in [100, 1000, 10000, 100000, 1000000]:
        publish = pd.read_csv("../dataset/publish_" + number_str[i] + ".csv")
        for j in range(nrepeat):
            conversion_time, linear_time, relational_time = bench_conference_covariance(publish, ranking)
            print("tensorra,covariance," + number_str[i] + "," + str(conversion_time) + "," + str(linear_time) + "," + str(relational_time))

bench_tensorra(5)
