nrepeat=5
# Bench RMA linear regerssion relational operators
#for ((i=0; i<$nrepeat; i++))
#do
#    mclient -d bench < delete_result.sql
#    mclient -d bench --timer performance < bench_rma_lr.sql &>> $1
#done

# Bench RMA conference covariance relational operators
for ((i=0; i<$nrepeat; i++))
do
    mclient -d bench < delete_result.sql
    mclient -d bench --timer performance < load_rma_data.sql
    mclient -d bench --timer performance < bench_rma_cov.sql
done
