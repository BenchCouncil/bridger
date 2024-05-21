/*
Bench Spark
*/
import org.apache.spark.sql.DataFrame
import java.io._

// warmup
val path = "../dataset/trip_10m.csv"
val df = spark.read.option("header", "true").csv(path)
val df_res = df.filter(col("start_station_code") =!= col("end_station_code"))
val count = df_res.count()

def bench_selection(df: DataFrame) : Double = {
    val start_time = System.nanoTime()
    val df_res = df.filter(col("start_station_code") =!= col("end_station_code"))
    val count = df_res.count()
    df_res.show()
    val duration = (System.nanoTime() - start_time) / 10e6
    return duration
}

def bench_projection(df: DataFrame) : Double = {
    val start_time = System.nanoTime()
    val df_res = df.select("start_station_code", "end_station_code")
    val count = df_res.count()
    df_res.show(1)
    val duration = (System.nanoTime() - start_time) / 10e6
    return duration
}

def bench_cross_join(left_table: DataFrame, right_table: DataFrame) : Double = {
    val start_time = System.nanoTime()
    val df_res = left_table.crossJoin(right_table)
    val count = df_res.count()
    df_res.show(1)
    val duration = (System.nanoTime() - start_time) / 10e6
    return duration
}

def bench_inner_join(left_table: DataFrame, right_table: DataFrame) : Double = {
    val start_time = System.nanoTime()
    val df_res = left_table.join(right_table, left_table("start_station_code") === right_table("code"), "inner")
    val count = df_res.count()
    df_res.show(1)
    val duration = (System.nanoTime() - start_time) / 10e6
    return duration
}

def bench_groupby(df: DataFrame) : Double = {
    val start_time = System.nanoTime()
    val df_res = df.groupBy("start_station_code").count()
    //val df_res = df.groupBy("start_station_code")
    df_res.show(1)
    val duration = (System.nanoTime() - start_time) / 10e6
    return duration
}

def bench_aggregration(df: DataFrame) : Double = {
    val start_time = System.nanoTime()
    val df_sum = df.agg(sum("duration_sec"))
    df_sum.show(1)
    val duration = (System.nanoTime() - start_time) / 10e6
    return duration
}

val station = spark.read.option("header", "true").csv("../dataset/station.csv")
val len_list = List(1000, 10000, 100000, 1000000, 10000000)
val num_str = Map(1000 -> "1k", 10000 -> "10k", 100000 -> "100k", 1000000 -> "1m", 10000000 -> "10m")
val i = 0
val writer = new PrintWriter("operator_result.csv")
writer.write("framework,operator,row_num,exec_time\n")
for (i <- len_list){
    val path = "../dataset/trip_" + num_str(i) + ".csv"
    val df = spark.read.option("header", "true").csv(path)
    val nrepeat = 5
    for (j <- 1 to nrepeat){      
        val t_selection = bench_selection(df)
        writer.write("spark,selection," + i + "," + t_selection + "\n")
        val t_projection = bench_projection(df)
        writer.write("spark,projection," + i + "," + t_projection + "\n")
        //val t_cross_join = bench_cross_join(df, station)
        //writer.write("spark,cross_join," + i + "," + t_cross_join + "\n")
        val t_inner_join = bench_inner_join(df, station)
        writer.write("spark,inner_join," + i + "," + t_inner_join + "\n")
        val t_groupby = bench_groupby(df)
        writer.write("spark,groupby," + i + "," + t_groupby + "\n")
        val t_aggregration = bench_aggregration(df)
        writer.write("spark,aggregation," + i + "," + t_aggregration + "\n")
    }
}
writer.close()
