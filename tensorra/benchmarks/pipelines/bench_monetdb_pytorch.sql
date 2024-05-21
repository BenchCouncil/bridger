CREATE TABLE res_1k 
AS SELECT 
trip_1k.start_station_code,
trip_1k.end_station_code,
trip_1k.duration_sec,
start_station.code as start_code,
start_station.latitude as start_latitude,
start_station.longitude as start_longitude,
end_station.code as end_code,
end_station.latitude as end_latitude,
end_station.longitude as end_longitude
FROM trip_1k 
inner JOIN station start_station 
ON trip_1k.start_station_code = start_station.code 
inner JOIN station end_station 
ON trip_1k.end_station_code = end_station.code;
COPY SELECT * FROM res_1k INTO '/home/xwen/tensor_ra/benchmarks/pipelines/tmp/monetdb_pytorch_lr_1k.csv' USING DELIMITERS ',' , '\n' , '"';
CREATE TABLE res_10k 
AS SELECT 
trip_10k.start_station_code,
trip_10k.end_station_code,
trip_10k.duration_sec,
start_station.code as start_code,
start_station.latitude as start_latitude,
start_station.longitude as start_longitude,
end_station.code as end_code,
end_station.latitude as end_latitude,
end_station.longitude as end_longitude
FROM trip_10k 
inner JOIN station start_station 
ON trip_10k.start_station_code = start_station.code 
inner JOIN station end_station 
ON trip_10k.end_station_code = end_station.code;
COPY SELECT * FROM res_10k INTO '/home/xwen/tensor_ra/benchmarks/pipelines/tmp/monetdb_pytorch_lr_10k.csv' USING DELIMITERS ',' , '\n' , '"';
CREATE TABLE res_100k 
AS SELECT 
trip_100k.start_station_code,
trip_100k.end_station_code,
trip_100k.duration_sec,
start_station.code as start_code,
start_station.latitude as start_latitude,
start_station.longitude as start_longitude,
end_station.code as end_code,
end_station.latitude as end_latitude,
end_station.longitude as end_longitude
FROM trip_100k 
inner JOIN station start_station 
ON trip_100k.start_station_code = start_station.code 
inner JOIN station end_station 
ON trip_100k.end_station_code = end_station.code;
COPY SELECT * FROM res_100k INTO '/home/xwen/tensor_ra/benchmarks/pipelines/tmp/monetdb_pytorch_lr_100k.csv' USING DELIMITERS ',' , '\n' , '"';
CREATE TABLE res_1m 
AS SELECT 
trip_1m.start_station_code,
trip_1m.end_station_code,
trip_1m.duration_sec,
start_station.code as start_code,
start_station.latitude as start_latitude,
start_station.longitude as start_longitude,
end_station.code as end_code,
end_station.latitude as end_latitude,
end_station.longitude as end_longitude
FROM trip_1m 
inner JOIN station start_station 
ON trip_1m.start_station_code = start_station.code 
inner JOIN station end_station 
ON trip_1m.end_station_code = end_station.code;
COPY SELECT * FROM res_1m INTO '/home/xwen/tensor_ra/benchmarks/pipelines/tmp/monetdb_pytorch_lr_1m.csv' USING DELIMITERS ',' , '\n' , '"';
CREATE TABLE res_10m 
AS SELECT 
trip_10m.start_station_code,
trip_10m.end_station_code,
trip_10m.duration_sec,
start_station.code as start_code,
start_station.latitude as start_latitude,
start_station.longitude as start_longitude,
end_station.code as end_code,
end_station.latitude as end_latitude,
end_station.longitude as end_longitude
FROM trip_10m 
inner JOIN station start_station 
ON trip_10m.start_station_code = start_station.code 
inner JOIN station end_station 
ON trip_10m.end_station_code = end_station.code;
COPY SELECT * FROM res_10m INTO '/home/xwen/tensor_ra/benchmarks/pipelines/tmp/monetdb_pytorch_lr_10m.csv' USING DELIMITERS ',' , '\n' , '"';
