./run_spark.sh
python bench_pandas_and_tensorra.py >> operator_result.csv
./run_monetdb.sh monetdb.log
python parse_monetdb.py monetdb.log >> operator_result.csv
python plot_operator.py