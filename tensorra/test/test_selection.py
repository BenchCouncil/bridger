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

# test projection
# test pandas sql
with Timer() as t:
     table_psql = pd.DataFrame(psql.read_sql_query('\
        select start_station_code, end_station_code, duration_sec\
        from trip2014\
        where start_station_code <> end_station_code;', con))
print("pandas_sql: ", t.interval)

# test native pandas
with Timer() as t:
    table_pd = duration[duration["start_station_code"] != duration["end_station_code"]]
print("pandas: ", t.interval)

# test tensor_ra
with Timer() as t:
    tableA.selection("not_equal", "start_station_code", "end_station_code")
print("tensor_ra: ", t.interval)

tensor_psql = TensorTable()
tensor_pd = TensorTable()
tensor_psql.from_pandas(table_psql)
tensor_pd.from_pandas(table_pd)
print(torch.equal(tensor_psql.tensor, tableA.tensor))
print(torch.equal(tensor_pd.tensor, tableA.tensor))