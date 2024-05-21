import pymonetdb
import time
import numpy as np
import pandas as pd

def bench_bixi_linear_regression(n_epoch, cursor, trip_name):
    relational_start = time.perf_counter()
    cursor.execute("\
                   SELECT t.duration_sec, \
                   s1.latitude AS start_latitude, \
                   s1.longitude AS start_longitude, \
                   s2.latitude AS end_latitude, \
                   s2.longitude AS end_longitude \
                   FROM (SELECT * FROM "
                   + trip_name +
                   " WHERE start_station_code <> end_station_code) t \
                   JOIN station s1 ON t.start_station_code = s1.code \
                   JOIN station s2 ON t.end_station_code = s2.code;")
    relational_end = time.perf_counter()
    rows = cursor.fetchall()
    data_array = np.array(rows)
    linear_start = time.perf_counter()
    #Calculate Euclidean Distance
    distance = np.linalg.norm(data_array[:, 1:2] - data_array[:, 3:4], axis=1)
    duration = data_array[:, 0]
    max_distance = distance.max()
    distance = distance / max_distance
    max_duration = duration.max()
    duration = duration / max_duration
    #Split dataset, 80% train data, 20% test data
    split_ratio = 0.8
    train_size = int(len(duration) * split_ratio)
    random_indices = np.random.permutation(len(duration))
    train_indices = random_indices[:train_size]
    test_indices = random_indices[train_size:]
    train_x = distance[train_indices]
    test_x = distance[test_indices]
    train_y = duration[train_indices]
    test_y = duration[test_indices]
    #Train model: linear regression y = a * x + b 
    param_a = 1
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
        pred_y = train_x * param_a + param_b
        #print(pred_y)
        loss = square_err(train_y, pred_y)
        param_a, param_b = grad_update(alpha, param_a, param_b, train_y, pred_y, train_x)
    #Test model
    pred_y = test_x * param_a + param_b
    loss = square_err(test_y, pred_y)   
    linear_end = time.perf_counter()
    return linear_end - linear_start, relational_end - relational_start

def bench_conference_covariance(cursor, publish, ranking):
    """
    Bench conference covariance computation,using DBLP dataset 
    """
    ranking = ranking.rename(columns={"Acronym":"conference", "GGS Rating":"rating"})
    x = np.array(publish)
    linear_start = time.perf_counter()
    #Calculate covariance
    y = np.cov(x, rowvar=False)
    #print(publish)
    linear_end = time.perf_counter()
    #Join
    # AIDA loses contextual information when performing LA on TabularData containing non-numeric columns. This results in some relational algebra operations failing to execute correctly after linear algebra computations. As a result, the second mixed pipeline can only execute the first half, encountering errors in the latter part.
    return linear_end - linear_start

def bench_aida(nrepeat):
    connection = pymonetdb.connect(username="monetdb", password="monetdb", hostname="localhost", database="bench", port="23332")
    cursor = connection.cursor()   
    #Bench bixi linear regression
    """
    for i in ["1k", "10k", "100k", "1m", "10m"]:
        for j in range(nrepeat):
            linear_time, relational_time = bench_bixi_linear_regression(100, cursor, "trip_" + i)
            print("aida,linear_regression," + i + ",0," + str(linear_time) + "," + str(relational_time))
    connection.close()
    """
    #Bench conference covariance
    ranking = pd.read_csv("../dataset/ranking.csv")
    ranking = ranking[["Acronym", "GGS Rating"]]
    number_str = {100:"100", 1000:"1k", 10000:"10k", 100000:"100k", 1000000:"1m"}
    for i in [100, 1000, 10000, 100000, 1000000]:
    #for i in [100]:
        publish = pd.read_csv("../dataset/publish_" + number_str[i] + ".csv")
        for j in range(nrepeat):
            linear_time = bench_conference_covariance(cursor, publish, ranking)
            print("aida,covariance," + number_str[i] + ",0," + str(linear_time) + ",0")
    connection.close()

bench_aida(5)
     