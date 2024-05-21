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


#trip = pd.read_csv("../dataset/OD_2014.csv")
#station = pd.read_csv("../dataset/Stations_2014.csv")[["code", "latitude", "longitude"]]
#trip = trip[["start_station_code", "end_station_code", "duration_sec"]]
trip = pd.read_csv("../benchmarks/dataset/trip.csv")
station = pd.read_csv("../benchmarks/dataset/station.csv")

"""
con = pymonetdb.Connection(dbname, hostname=hostname, port=port, username=username, password=password, autocommit=True)
with Timer() as t:
     table_psql = pd.DataFrame(psql.read_sql_query('\
        select * from trip_small\
        inner join station_small\
        on trip_small.start_station_code = station_small.code\
        ;', con))
print("pandas_sql: ", t.interval)
print(table_psql)
"""

def test_join(join_type, left_pandas, right_pandas, left_table, right_table):
    if(join_type == "cross_join"):
        with Timer() as t:
            left_pandas = left_pandas.merge(right_pandas, how="cross")
        print(t.interval)
        table_single = TensorTable() 
        table_multi = TensorTable()
        with Timer() as t:
            table_single = cross_join(left_table, right_table, False)
        print(t.interval)
        with Timer() as t:
            table_multi = cross_join(left_table, right_table, True, 12)
        print(t.interval)
    else:
        pandas_join_type = {"inner_join":"inner", "full_join":"outer", "left_join":"left", "right_join":"right"}
        with Timer() as t1:
            left_pandas = left_pandas.merge(right_pandas, left_on="start_station_code", right_on="code", how=pandas_join_type[join_type], sort=True)
        table_single = TensorTable() 
        #table_multi = TensorTable()
        with Timer() as t2:
            table_single = join(left_table, "start_station_code", right_table, "code", join_type, False)
        with Timer() as t3:
            #table_multi = join(left_table, "start_station_code", right_table, "code", join_type, True, 12)
            print("233")
        #print(t1.interval, t2.interval, t3.interval)
        print(t1.interval, t2.interval)
    table_pandas = TensorTable()
    table_pandas.from_pandas(left_pandas)
    #print(table_pandas.tensor)
    #print(table.tensor)
    print(torch.equal(table_pandas.tensor.type(torch.float32), table_single.tensor.type(torch.float32)))
    #print(torch.equal(table_pandas.tensor.type(torch.float32), table_multi.tensor.type(torch.float32)))
    return 0

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
    table_single = TensorTable()
    #test_join("full_join", duration, station, tableA, tableB)
    test_join("inner_join", duration, station, tableA, tableB)
