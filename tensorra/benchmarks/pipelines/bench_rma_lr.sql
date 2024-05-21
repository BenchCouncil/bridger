CREATE TABLE station_trip_1k AS 
SELECT t.duration_sec, s1.latitude AS start_latitude, s1.longitude AS start_longitude, s2.latitude AS end_latitude, s2.longitude AS end_longitude FROM 
(SELECT * FROM trip_1k WHERE start_station_code <> end_station_code) t 
JOIN station s1 ON t.start_station_code = s1.code 
JOIN station s2 ON t.end_station_code = s2.code;
CREATE TABLE station_trip_10k AS 
SELECT t.duration_sec, s1.latitude AS start_latitude, s1.longitude AS start_longitude, s2.latitude AS end_latitude, s2.longitude AS end_longitude FROM 
(SELECT * FROM trip_10k WHERE start_station_code <> end_station_code) t 
JOIN station s1 ON t.start_station_code = s1.code 
JOIN station s2 ON t.end_station_code = s2.code;
CREATE TABLE station_trip_100k AS 
SELECT t.duration_sec, s1.latitude AS start_latitude, s1.longitude AS start_longitude, s2.latitude AS end_latitude, s2.longitude AS end_longitude FROM 
(SELECT * FROM trip_100k WHERE start_station_code <> end_station_code) t 
JOIN station s1 ON t.start_station_code = s1.code 
JOIN station s2 ON t.end_station_code = s2.code;
CREATE TABLE station_trip_1m AS 
SELECT t.duration_sec, s1.latitude AS start_latitude, s1.longitude AS start_longitude, s2.latitude AS end_latitude, s2.longitude AS end_longitude FROM 
(SELECT * FROM trip_1m WHERE start_station_code <> end_station_code) t 
JOIN station s1 ON t.start_station_code = s1.code 
JOIN station s2 ON t.end_station_code = s2.code;
CREATE TABLE station_trip_10m AS 
SELECT t.duration_sec, s1.latitude AS start_latitude, s1.longitude AS start_longitude, s2.latitude AS end_latitude, s2.longitude AS end_longitude FROM 
(SELECT * FROM trip_10m WHERE start_station_code <> end_station_code) t 
JOIN station s1 ON t.start_station_code = s1.code 
JOIN station s2 ON t.end_station_code = s2.code;