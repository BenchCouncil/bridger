import pandas as pd
from tensor_ra.tensor_table import TensorTable
from tensor_ra.utils import Timer
import torch

trip = pd.read_csv("../benchmarks/dataset/trip.csv")
station = pd.read_csv("../benchmarks/dataset/station.csv")

trip = trip[0:10]
tableA = TensorTable()
tableB = TensorTable()
tableA.from_pandas(trip)
tableB.from_pandas(station)
# test native pandas
with Timer() as t:
    res_pd = trip.groupby(["start_station_code"])
    print(res_pd)
print("pandas: ", t.interval)
print(trip)
# test tensor_ra
with Timer() as t:
    print(tableA.groupby("start_station_code"))
print("tensor_ra: ", t.interval)


for i in [1000, 10000, 100000, 1000000, 10000000]:
    duration = trip[0:i]
    #print(duration)
    #print(station)
    print("datasize:")
    print(i)    
    tableA = TensorTable()
    tableB = TensorTable()
    tableA.from_pandas(duration)
    tableB.from_pandas(station)
    #test_join("full_join", duration, station, tableA, tableB)
    test_join("inner_join", duration, station, tableA, tableB)
