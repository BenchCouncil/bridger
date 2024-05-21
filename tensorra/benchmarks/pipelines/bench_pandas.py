import pandas as pd
import numpy as np
import time 

def bench_bixi_linear_regression(n_epoch, station, trip):
    """
    Bench linear regression, using BIXI dataset
    """
    a = time.perf_counter()
    #Relational operators, including one selection and two inner joins
    trip = trip[trip["start_station_code"]!=trip["end_station_code"]] 
    station_trip = pd.merge(trip, station, left_on="start_station_code", right_on="code", sort=True).rename({"latitude":"start_latitude", "longitude": "start_longitude"}, axis="columns")
    station_trip = pd.merge(station_trip, station, left_on="end_station_code", right_on="code", sort=True).rename({"latitude":"end_latitude", "longitude": "end_longitude"}, axis="columns")
    station_trip = station_trip[["duration_sec", "start_latitude", "start_longitude", "end_latitude", "end_longitude"]]
    b = time.perf_counter()
    #Calculate Euclidean Distance
    station_trip["distance"] = np.linalg.norm(station_trip[["start_latitude", "start_longitude"]].values - station_trip[["end_latitude", "end_longitude"]].values, axis=1)
    distance_duration = station_trip[["distance", "duration_sec"]]
    max_distance = distance_duration["distance"].max()
    distance_duration["distance"] = distance_duration["distance"] / max_distance
    max_duration = distance_duration["duration_sec"].max()
    distance_duration["duration_sec"] = distance_duration["duration_sec"] / max_duration
    #print(distance_duration.sort_values(by=["distance", "duration_sec"]))
    #Split dataset, 80% train data, 20% test data
    train_data = distance_duration.sample(frac=0.8)
    test_data = distance_duration.drop(train_data.index)
    train_x = train_data[["distance"]]
    train_y = train_data[["duration_sec"]]
    test_x = test_data[["distance"]]
    test_y = test_data[["duration_sec"]]
    #Train model: linear regression y = a * x + b 
    param_a = pd.DataFrame({"distance":[1]}, index={"duration_sec"})
    param_b = 0
    def square_err(y, pred_y):
        #Square error
        return ((y - pred_y) ** 2).sum() / (2 * (y.shape[0]))
    alpha = 0.1
    def grad_update(alpha, param_a, param_b, y, pred_y, x):
        #Gradint descending to update parameters
        param_a = param_a - alpha * (pred_y - y).T.dot(x) / x.shape[0]
        param_b = param_b - alpha * (pred_y - y).sum() / x.shape[0]
        return param_a, param_b
    for i in range(0, n_epoch):
        pred_y = train_x.dot(param_a.T).add(param_b)
        loss = square_err(train_y, pred_y)
        param_a, param_b = grad_update(alpha, param_a, param_b, train_y, pred_y, train_x)
    #Test model
    pred_y = test_x.dot(param_a.T).add(param_b)
    loss = square_err(test_y, pred_y)
    c = time.perf_counter()
    return c - b, b - a

def bench_conference_covariance(publish, ranking):
    """
    Bench conference covariance computation,using DBLP dataset 
    """
    ranking = ranking.rename(columns={"Acronym":"conference", "GGS Rating":"rating"})
    a = time.perf_counter()
    #Calculate covariance
    publish = publish.cov()
    b = time.perf_counter()
    #Join
    res_conf = pd.merge(publish, ranking, left_index=True, right_on="conference", sort=True)
    res_conf = res_conf[res_conf["rating"] == "A++"]
    c = time.perf_counter()
    return b - a, c - b

#t_linear_regression = bench_bixi_linear_regression(100) 
#print("pandas,linear_regression," + str(t_linear_regression))

def bench_pandas(nrepeat):
    #Bench bixi linear regression
    """
    station = pd.read_csv("../dataset/station.csv")
    trip = pd.read_csv("../dataset/trip.csv")
    number_str = {1000:"1k", 10000:"10k", 100000:"100k", 1000000:"1m", 10000000:"10m"}
    for i in [1000, 10000, 100000, 1000000, 10000000]:
        trip = pd.read_csv("../dataset/trip_" + number_str[i] + ".csv")
        for j in range(nrepeat):
            linear_time, relational_time = bench_bixi_linear_regression(100, station, trip)
            print("pandas,linear_regression," + str(i) + ",0," + str(linear_time) + "," + str(relational_time))
    """
    #Bench conference covariance
    ranking = pd.read_csv("../dataset/ranking.csv")
    ranking = ranking[["Acronym", "GGS Rating"]]
    number_str = {100:"100", 1000:"1k", 10000:"10k", 100000:"100k", 1000000:"1m"}
    #for i in [100, 1000, 10000, 100000, 1000000]:
    for i in [100]:
        publish = pd.read_csv("../dataset/publish_" + number_str[i] + ".csv")
        for j in range(nrepeat):
            linear_time, relational_time = bench_conference_covariance(publish, ranking)
            print("pandas,covariance," + str(i) + ",0," + str(linear_time) + "," + str(relational_time))

bench_pandas(5)
