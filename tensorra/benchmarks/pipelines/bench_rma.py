import pymonetdb
import time
import numpy as np
import pandas as pd
import os

def bench_bixi_linear_regression(n_epoch, cursor, table_name):
    cursor.execute("SELECT * FROM " + table_name + " ;")
    rows = cursor.fetchall()
    conversion_start = time.perf_counter()
    data_array = np.array(rows)
    conversion_end = time.perf_counter()
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
    conversion_time = conversion_end - conversion_start
    if(table_name != "station_trip_1k"):
        conversion_time = conversion_time / 5
    return linear_end - linear_start, conversion_time

def bench_conference_covariance(connection, publish, ranking, table_name):
    """
    Bench conference covariance computation,using DBLP dataset 
    """
    #connection.commit()
    conference_name = publish.columns
    x = np.array(publish)
    linear_start = time.perf_counter()
    #Calculate covariance
    y = np.cov(x, rowvar=False)
    linear_end = time.perf_counter()
    publish_new = pd.DataFrame(y, columns=conference_name)
    publish_new["conference"] = conference_name
    #Save tmp data and create table, Comment out for repeated experiments
    if(os.path.exists(table_name + ".csv")):
        pass
    else:
        publish_new.to_csv(table_name + ".csv", index=False)
        cursor = connection.cursor()
        create_sql = "CREATE TABLE " + table_name + "( "
        for i in conference_name:
            create_sql = create_sql + "\"" + i + "\"" + " FLOAT, "
        create_sql = create_sql + "conference STRING);" 
        cursor.execute(create_sql)
        connection.commit()    
    return linear_end - linear_start, 0

def read_rma_relational(filename):
    f = open(filename)
    lines = f.readlines()
    i = 0
    res = []
    for l in lines:
        if("run" in l):
            sql_time = l.split(" ")[0].split(":")[1]
            opt_time = l.split(" ")[1].split(":")[1]
            run_time = l.split(" ")[2].split(":")[1]
            exec_time = float(sql_time) + float(opt_time) + float(run_time)
            res.append(exec_time)
    return res

def bench_aida(nrepeat):
    connection = pymonetdb.connect(username="monetdb", password="monetdb", hostname="localhost", database="bench", port="23332")
    cursor = connection.cursor()    
    #Bench bixi linear regression
    """
    line = ["1k", "10k", "100k", "1m", "10m"]
    relational_time = read_rma_relational("rma_relational_lr.csv")
    for i in line:
        for j in range(nrepeat):
            linear_time, conversion_time = bench_bixi_linear_regression(100, cursor, "station_trip_" + i)
            print("rma,linear_regression," + i + "," + str(conversion_time) + "," + str(linear_time) + "," + str(relational_time[line.index(i) + j * 5] / 1000))
    """
    ranking = pd.read_csv("../dataset/ranking.csv")
    ranking = ranking[["Acronym", "GGS Rating"]]
    ranking = ranking.rename(columns={"Acronym":"conference", "GGS Rating":"rating"})
    ranking.to_csv("../dataset/ranking_cleaned.csv", index=False)
    number_str = {100:"100", 1000:"1k", 10000:"10k", 100000:"100k", 1000000:"1m"}
    for i in [100, 1000, 10000, 100000, 1000000]:
    #for i in [100]:
        publish = pd.read_csv("../dataset/publish_" + number_str[i] + ".csv")
        for j in range(nrepeat):
            linear_time, conversion_time = bench_conference_covariance(connection, publish, ranking, "conference_" + number_str[i])
            print("rma,covariance," + number_str[i] + "," + str(conversion_time) + "," + str(linear_time) + ",0")
    connection.close()

#bench_aida(5)
    
def analysis_aida():
    f = open("rma_cov_sql.log")
    lines = f.readlines()
    i = 0
    conversion = []
    relational = []
    for l in lines:
        if("run" in l):
            sql_time = l.split(" ")[0].split(":")[1]
            opt_time = l.split(" ")[1].split(":")[1]
            run_time = l.split(" ")[2].split(":")[1]
            exec_time = float(sql_time) + float(opt_time) + float(run_time)
            if(i % 15 >= 10):
                relational.append(exec_time)
            elif((i - int((i / 15)) * 15) % 2 == 1):
                conversion.append(exec_time)
            i = i + 1
    conversion = np.array(conversion).reshape((5, 5)).T.reshape((25,)) / 1000·
    conversion[0:10] = conversion[0:10] / 100
    conversion[10:20] = conversion[10:20] / 10
    relational = np.array(relational).reshape((5, 5)).T.reshape((25,)) / 1000
    df = pd.read_csv("rma_cov_linear.csv")
    df["conversion_time"] = conversion
    df["relational_time"] = relational
    print(df)
    df.to_csv("pipeline_result.csv", index=False, mode="a")
    return 0

analysis_aida()
     