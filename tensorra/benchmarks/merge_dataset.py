"""
Merge four csv files into single csv file
"""
import pandas as pd

trip2014 = pd.read_csv("dataset/OD_2014.csv", low_memory=False)[["start_station_code", "end_station_code", "duration_sec"]]
trip2015 = pd.read_csv("dataset/OD_2015.csv", low_memory=False)[["start_station_code", "end_station_code", "duration_sec"]]
trip2016 = pd.read_csv("dataset/OD_2016.csv", low_memory=False)[["start_station_code", "end_station_code", "duration_sec"]]
trip2017 = pd.read_csv("dataset/OD_2017.csv", low_memory=False)[["start_station_code", "end_station_code", "duration_sec"]]
station2014 = pd.read_csv("dataset/Stations_2014.csv", low_memory=False)[["code", "latitude", "longitude"]]
station2015 = pd.read_csv("dataset/Stations_2015.csv", low_memory=False)[["code", "latitude", "longitude"]]
station2016 = pd.read_csv("dataset/Stations_2016.csv", low_memory=False)[["code", "latitude", "longitude"]]
station2017 = pd.read_csv("dataset/Stations_2017.csv", low_memory=False)[["code", "latitude", "longitude"]]

trip = pd.concat([trip2014, trip2015, trip2016, trip2017], ignore_index=True)
station = pd.concat([station2014, station2015, station2016, station2017], ignore_index=True)
station = station.drop_duplicates(ignore_index=True)
trip.to_csv("dataset/trip.csv", index=False)
station.to_csv("dataset/station.csv", index=False)


for i in [1000, 10000, 100000, 1000000, 10000000]:
    tmp = trip[0:i]
    number_str = {1000:"1k", 10000:"10k", 100000:"100k", 1000000:"1m", 10000000:"10m"}
    filename = "dataset/" + "trip_" + number_str[i] + ".csv"
    tmp.to_csv(filename, index=False)