"""
Bench pandas and tensorra
"""
import pandas as pd
from tensor_ra.tensor_table import TensorTable
from tensor_ra.utils import Timer
from tensor_ra.join import join, cross_join
import torch

def test_selection(trip, tensor_trip):
    # test selection
    # test pandas
    with Timer() as t1:
        res_pd = trip[trip["start_station_code"] != trip["end_station_code"]]
    # test tensorra
    with Timer() as t2:
        tensor_trip.selection("not_equal", "start_station_code", "end_station_code")
    return t1.interval * 1000, t2.interval * 1000   

def test_projection(trip, tensor_trip):
    # test projection
    # test pandas
    with Timer() as t1:
        res_pd = trip[["start_station_code", "end_station_code"]]
    # test tensorra
    with Timer() as t2:
        tensor_trip.projection(["start_station_code", "end_station_code"])
    return t1.interval * 1000, t2.interval * 1000   

def test_cross_join(left_table, right_table):
    # test cross join
    # test pandas
    with Timer() as t1:
        left_table = left_table.merge(right_table, how="cross")
    # test tensorra
    tensor_table_left = TensorTable()
    tensor_table_left.from_pandas(left_table)
    tensor_table_right = TensorTable()
    tensor_table_right.from_pandas(right_table)
    table = TensorTable() 
    with Timer() as t2:
        table = cross_join(tensor_table_left, tensor_table_right, True, 12)
    return t1.interval * 1000, t2.interval * 1000   

def test_inner_join(left_table, right_table, tensor_table_left, tensor_table_right):
    # test inner join
    # test pandas
    with Timer() as t1:
        left_table = left_table.merge(right_table, left_on="start_station_code", right_on="code", how="inner", sort=True)
    # test tensorra
    tensor_table_out = TensorTable()
    with Timer() as t2:
        tensor_table_out = join(tensor_table_left, "start_station_code", tensor_table_right, "code", "inner_join", False)
    return t1.interval * 1000, t2.interval * 1000   

def test_groupby(trip, tensor_trip):
    # test groupby
    # test pandas
    with Timer() as t1:
        res_pd = trip.groupby(["start_station_code"])["start_station_code"].count()
    # test tensorra
    with Timer() as t2:
        res = tensor_trip.groupby_count("start_station_code")
    return t1.interval * 1000, t2.interval * 1000   

def test_aggregration(trip, tensor_trip):
    # test aggregration
    # test pandas
    with Timer() as t1:
        res_pd = trip["duration_sec"].sum()
    # test tensorra
    with Timer() as t2:
        res = tensor_trip.aggregration("sum", "duration_sec")
        #print("unsupported")
    return t1.interval * 1000, t2.interval * 1000

def bench_pandas_and_tensorra(nrepeat):
    station = pd.read_csv("../dataset/station.csv")
    number_str = {1000:"1k", 10000:"10k", 100000:"100k", 1000000:"1m", 10000000:"10m"}
    total_trip = pd.read_csv("../dataset/trip.csv")
    for i in [1000, 10000, 100000, 1000000, 10000000]:
        trip = total_trip[0:i]
        for j in range(nrepeat):
            tensor_trip = TensorTable()
            tensor_station = TensorTable()
            tensor_trip.from_pandas(trip)
            tensor_station.from_pandas(station)
            pd_time, tensorra_time = test_inner_join(trip, station, tensor_trip, tensor_station)
            print("pandas,inner_join," + str(i) + "," + str(pd_time))
            print("tensorra,inner_join," + str(i) + "," + str(tensorra_time))
            tensor_trip.from_pandas(trip)
            tensor_station.from_pandas(station)
            pd_time, tensorra_time = test_selection(trip, tensor_trip)
            print("pandas,selection," + str(i) + "," + str(pd_time))
            print("tensorra,selection," + str(i) + "," + str(tensorra_time))
            tensor_trip.from_pandas(trip)
            tensor_station.from_pandas(station)
            pd_time, tensorra_time = test_projection(trip, tensor_trip)
            print("pandas,projection," + str(i) + "," + str(pd_time))
            print("tensorra,projection," + str(i) + "," + str(tensorra_time))
            #pd_time, tensorra_time = test_cross_join(trip, station)
            #print("pandas,cross_join," + str(i) + "," + str(pd_time))
            #print("tensorra,cross_join," + str(i) + "," + str(tensorra_time))
            
            tensor_trip.from_pandas(trip)
            tensor_station.from_pandas(station)
            pd_time, tensorra_time = test_aggregration(trip, tensor_trip)
            print("pandas,aggregation," + str(i) + "," + str(pd_time))
            print("tensorra,aggregation," + str(i) + "," + str(tensorra_time))
            tensor_trip.from_pandas(trip)
            tensor_station.from_pandas(station)
            pd_time, tensorra_time = test_groupby(trip, tensor_trip)
            print("pandas,groupby," + str(i) + "," + str(pd_time))
            print("tensorra,groupby," + str(i) + "," + str(tensorra_time))

bench_pandas_and_tensorra(1)
