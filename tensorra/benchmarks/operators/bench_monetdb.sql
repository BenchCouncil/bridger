/*
    Selection for monetdb
*/
CREATE TABLE selection_1k AS SELECT * FROM trip_1k WHERE start_station_code <> end_station_code;
CREATE TABLE selection_10k AS SELECT * FROM trip_10k WHERE start_station_code <> end_station_code;
CREATE TABLE selection_100k AS SELECT * FROM trip_100k WHERE start_station_code <> end_station_code;
CREATE TABLE selection_1m AS SELECT * FROM trip_1m WHERE start_station_code <> end_station_code;
CREATE TABLE selection_10m AS SELECT * FROM trip_10m WHERE start_station_code <> end_station_code;
CREATE TABLE projection_1k AS SELECT start_station_code, end_station_code FROM trip_1k;
CREATE TABLE projection_10k AS SELECT start_station_code, end_station_code FROM trip_10k;
CREATE TABLE projection_100k AS SELECT start_station_code, end_station_code FROM trip_100k;
CREATE TABLE projection_1m AS SELECT start_station_code, end_station_code FROM trip_1m;
CREATE TABLE projection_10m AS SELECT start_station_code, end_station_code FROM trip_10m;
CREATE TABLE inner_join_1k AS SELECT * FROM trip_1k innner JOIN station ON start_station_code = code;
CREATE TABLE inner_join_10k AS SELECT * FROM trip_10k innner JOIN station ON start_station_code = code;
CREATE TABLE inner_join_100k AS SELECT * FROM trip_100k innner JOIN station ON start_station_code = code;
CREATE TABLE inner_join_1m AS SELECT * FROM trip_1m innner JOIN station ON start_station_code = code;
CREATE TABLE inner_join_10m AS SELECT * FROM trip_10m innner JOIN station ON start_station_code = code;
/*
CREATE TABLE groupby_1k AS SELECT start_station_code, end_station_code FROM trip_1k GROUP BY start_station_code, end_station_code;
CREATE TABLE groupby_10k AS SELECT start_station_code, end_station_code FROM trip_10k GROUP BY start_station_code, end_station_code;
CREATE TABLE groupby_100k AS SELECT start_station_code, end_station_code FROM trip_100k GROUP BY start_station_code, end_station_code;
CREATE TABLE groupby_1m AS SELECT start_station_code, end_station_code FROM trip_1m GROUP BY start_station_code, end_station_code;
CREATE TABLE groupby_10m AS SELECT start_station_code, end_station_code FROM trip_10m GROUP BY start_station_code, end_station_code;
*/
CREATE TABLE groupby_1k AS SELECT start_station_code, COUNT(*) AS "count" FROM trip_1k GROUP BY start_station_code;
CREATE TABLE groupby_10k AS SELECT start_station_code, COUNT(*) AS "count" FROM trip_10k GROUP BY start_station_code;
CREATE TABLE groupby_100k AS SELECT start_station_code, COUNT(*) AS "count" FROM trip_100k GROUP BY start_station_code;
CREATE TABLE groupby_1m AS SELECT start_station_code, COUNT(*) AS "count" FROM trip_1m GROUP BY start_station_code;
CREATE TABLE groupby_10m AS SELECT start_station_code, COUNT(*) AS "count" FROM trip_10m GROUP BY start_station_code;
SELECT SUM(duration_sec) from trip_1k; 
SELECT SUM(duration_sec) from trip_10k; 
SELECT SUM(duration_sec) from trip_100k; 
SELECT SUM(duration_sec) from trip_1m; 
SELECT SUM(duration_sec) from trip_10m; 