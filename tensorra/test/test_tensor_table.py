#from ..python.tensor_ra.table_tensor import table_tensor
import pandas as pd
import torch
from tensor_ra.tensor_table import TensorTable
from tensor_ra.join import join, cross_join
trip = pd.read_csv("../dataset/OD_2014.csv")
#print(trip)
station = pd.read_csv("../dataset/Stations_2014.csv")[["code", "latitude", "longitude"]]
  
duration = trip[["start_station_code", "end_station_code", "duration_sec"]]
duration = duration[0:12]
station = station[0:12]
tableA = TensorTable()
tableB = TensorTable()
tableA.from_pandas(duration)
tableB.from_pandas(station)
# Using to test inner join
tableB.tensor[1,0] = 6209
tableB.tensor[3,0] = 6214
tableB.tensor[-1,0] = 6214
tableB.tensor[-2,0] = 6082
#print(tableA.column_name, tableA.tensor)
#print(tableB.column_name, tableB.tensor)
#tableA.projection(["start_station_code", "end_station_code"])
#tableA.selection("greater", "duration_sec", threshold=1000)
#tableA.selection("equal", "start_station_code", "end_station_code")
#table = join(tableA, "start_station_code", tableB, "code", "inner_join")
#table = join(tableA, "start_station_code", tableB, "code", "full_join")
#table = join(tableA, "start_station_code", tableB, "code", "left_join")
#table = join(tableA, "start_station_code", tableB, "code", "right_join")
tableA.tensor = tableA.tensor[:5, :]
tableB.tensor = tableB.tensor[:5, :]
print(tableA.column_name, tableA.tensor)
print(tableB.column_name, tableB.tensor)
table = cross_join(tableA, tableB)
print(table.column_name)
print(table.tensor)
