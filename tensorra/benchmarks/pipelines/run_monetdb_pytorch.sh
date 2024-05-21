#Run monetdb
cp ../.monetdb .
nrepeat=1
for ((i=0; i<$nrepeat; i++))
do
    rm tmp/*
    mclient -d bench < delete_result.sql
    mclient -d bench --timer performance < bench_monetdb_pytorch.sql &>> $1
    mclient -d bench < delete_result.sql
    rm tmp/*
done
