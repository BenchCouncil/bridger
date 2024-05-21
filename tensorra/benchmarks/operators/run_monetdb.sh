#Run monetdb
cp ../.monetdb .
nrepeat=5
for ((i=0; i<$nrepeat; i++))
do
    mclient -d bench < delete_result.sql
    mclient -d bench --timer performance < bench_monetdb.sql &>> $1
    mclient -d bench < delete_result.sql
done