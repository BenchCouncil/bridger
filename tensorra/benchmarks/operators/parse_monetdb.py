from ast import operator
import sys

operators = ["selection", "projection", "inner_join", "groupby", "aggregation"]

row_list = [1000, 10000, 100000, 1000000, 10000000]

f = open(sys.argv[1])
lines = f.readlines()
i = 0
for l in lines:
    if("run" in l):
        sql_time = l.split(" ")[0].split(":")[1]
        opt_time = l.split(" ")[1].split(":")[1]
        run_time = l.split(" ")[2].split(":")[1]
        exec_time = float(sql_time) + float(opt_time) + float(run_time)
        row_num = i % 5        
        nrepeat  = i // 25
        operator_num = (i - 25 * nrepeat) // 5       
        print("monetdb," + str(operators[operator_num]) + "," + str(row_list[row_num]) + "," + str(exec_time))
        i = i + 1

