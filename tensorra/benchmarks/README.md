# Benchmark 

Compared with pandas, spark and AIDA

## Install 

### Install monetdb

Following https://www.monetdb.org/documentation-Jan2022/admin-guide/installation/

### Install AIDA


## Build database 

### Run shell scripts
`./build.sh`

### Build by yourself
`monetdbd create db`

`monetdbd set port=2333 db`

`monetdbd start db`

`monetdb create bench`

`monetdb release bench`

`mclient -u monetdb -d bench`

## Load data into dataset
`./prepare_data.sh`

## Delete data from dataset
`./delete_data.sh`