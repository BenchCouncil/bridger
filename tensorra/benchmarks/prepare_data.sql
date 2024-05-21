/*
    Load BIXI dataset from csv to database
*/
CREATE TABLE trip(
    start_station_code INT,
    end_station_code INT,
    duration_sec INT
);
COPY OFFSET 2 INTO trip FROM 'dataset/trip.csv' ON CLIENT USING DELIMITERS ',', E'\n', '"';
CREATE TABLE station(
    code INT,
    latitude FLOAT,
    longitude FLOAT
);
COPY OFFSET 2 INTO station FROM 'dataset/station.csv' ON CLIENT USING DELIMITERS ',', E'\n', '"';
CREATE TABLE trip_1k AS SELECT * FROM trip LIMIT 1000;
CREATE TABLE trip_10k AS SELECT * FROM trip LIMIT 10000;
CREATE TABLE trip_100k AS SELECT * FROM trip LIMIT 100000;
CREATE TABLE trip_1m AS SELECT * FROM trip LIMIT 1000000;
CREATE TABLE trip_10m AS SELECT * FROM trip LIMIT 10000000;


