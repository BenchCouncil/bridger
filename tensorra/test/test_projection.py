import pandas as pd
from tensor_ra.tensor_table import TensorTable
from tensor_ra.join import join, cross_join
from tensor_ra.utils import Timer
import pandas.io.sql as psql
import pymonetdb.sql
import torch

hostname = "localhost"
dbname = "bench"
username = "monetdb"
password = "monetdb"
port = 2333

con = pymonetdb.Connection(dbname, hostname=hostname, port=port, username=username, password=password, autocommit=True)

trip = pd.read_csv("../dataset/OD_2014.csv")
station = pd.read_csv("../dataset/Stations_2014.csv")[["code", "latitude", "longitude"]]
  
duration = trip[["start_station_code", "end_station_code", "duration_sec"]]
tableA = TensorTable()
tableB = TensorTable()
tableA.from_pandas(duration)
tableB.from_pandas(station)

# Test projection
# Test pandas sql
with Timer() as t:
     table_psql = pd.DataFrame(psql.read_sql_query('\
        select start_station_code, end_station_code\
        from trip2014;', con))
print("pandas_sql: ", t.interval)

# Test native pandas
with Timer() as t:
    table_pd = duration[["start_station_code", "end_station_code"]]
    #tableA.selection("equal", "start_station_code", "end_station_code")
print("pandas: ", t.interval)

# Test tensor_ra
with Timer() as t:
    tableA.projection(["start_station_code", "end_station_code"])
    #tableA.selection("equal", "start_station_code", "end_station_code")
print("tensor_ra: ", t.interval)

tensor_psql = TensorTable()
tensor_pd = TensorTable()
tensor_psql.from_pandas(table_psql)
tensor_pd.from_pandas(table_pd)
print(torch.equal(tensor_psql.tensor, tableA.tensor))
print(torch.equal(tensor_pd.tensor, tableA.tensor))

"""
# Test projection
# Test pandas sql
with Timer() as t:
     table_psql = pd.DataFrame(psql.read_sql_query('\
        select distinct start_station_code, end_station_code\
        from trip2014\
        order by start_station_code, end_station_code;', con))
print("pandas_sql: ", t.interval)

# Native pandas do not support 

# Test tensor_ra
with Timer() as t:
    tableA.projection(["start_station_code", "end_station_code"])
    #tableA.selection("equal", "start_station_code", "end_station_code")
print("tensor_ra: ", t.interval)

tensor_psql = TensorTable()
tensor_psql.from_pandas(table_psql)
print(torch.equal(tensor_psql.tensor, tableA.tensor))
"""
