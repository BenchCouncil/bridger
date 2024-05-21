/*
Bench Spark
*/
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, BlockMatrix}
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.ml.feature.VectorAssembler
import spark.sqlContext.implicits._
import java.io._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import scala.collection.mutable.ArrayBuffer

//warm up
val path = "../dataset/trip.csv"
val df = spark.read.option("header", "true").csv(path)
val df_res = df.filter(col("start_station_code") =!= col("end_station_code"))
val count = df_res.count()

def bench_bixi_linear_regression(nepoch: Int, station: DataFrame, trip:DataFrame) : (Double, Double, Double) = {
    //Bench linear regression, using BIXI dataset
    val a = System.nanoTime()
    //Selection and Join
    val station_trip = trip.filter(col("start_station_code") =!= col("end_station_code")).
        join(station, col("start_station_code") === col("code"), "inner").
        withColumnRenamed("latitude", "start_latitude").withColumnRenamed("longitude", "start_longitude").
        drop("code").
        join(station, col("end_station_code") === col("code"), "inner").
        withColumnRenamed("latitude", "end_latitude").withColumnRenamed("longitude", "end_longitude").
        drop("start_station_code").drop("end_station_code").drop("code")
    station_trip.show()
    val b = System.nanoTime()
    val relational_time = b - a
    val conversion_time = 0
    val linear_time = 0 
    return (conversion_time/10e9, linear_time/10e9, relational_time/10e9)
    /*
    //Calculate distance
    val distance = (x1: Double, y1: Double, x2: Double, y2: Double) => math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))
    val distanceUDF = udf(distance)
    val duration_distance = station_trip.withColumn("distance", distanceUDF(col("start_latitude"), col("start_longitude"), col("end_latitude"), col("end_longitude"))).select("duration_sec", "distance")
    //Split dataset, 80% train data, 20% test data
    val maxdist = duration_distance.agg(max(col("distance"))).head().getDouble(0)
    val maxduration = duration_distance.agg(max(col("duration_sec"))).head().getInt(0)
    val duration_distance_norm = duration_distance.select(col("distance") / maxdist as "distance", col("duration_sec") / maxduration as "duration_sec")
    val split = duration_distance_norm.randomSplit(Array(4, 1))
    val train_data = split(0)
    val test_data = split(1) 
    //Train model: linear regression y = a * x + b 
    def dataframeToMatrix(df: Dataset[Row]) : BlockMatrix = {
        val assembler = new VectorAssembler().setInputCols(df.columns).setOutputCol("vector")
        val df2 = assembler.transform(df)
        return new IndexedRowMatrix(df2.select("vector").rdd.map{case Row(v: Vector) => Vectors.fromML(v)}.zipWithIndex.map { case (v, i) => IndexedRow(i, v) }).toBlockMatrix()
    }
    def squaredErr(actual: BlockMatrix, predicted: BlockMatrix) : Double = {
        var s: Double = 0
        val it = actual.subtract(predicted).toLocalMatrix().rowIter
        while (it.hasNext) {
            s += scala.math.pow(it.next.apply(0), 2)
        }
        return s / (2 * actual.numRows())
    }
    def gradDesc(actual: BlockMatrix, predicted: BlockMatrix,indata: BlockMatrix) : Seq[Double] = {
        val m = predicted.subtract(actual).transpose.multiply(indata).toLocalMatrix()
        val n = actual.numRows()
        return Seq(m.apply(0, 0) / n, m.apply(0, 1) / n)
    }
    val trainDataSet = train_data.select("distance").withColumn("x0", lit(1)).select("x0","distance")
    val trainDataSetDuration = train_data.select("duration_sec")
    var params = Seq(1.0).toDF("a").withColumn("b", lit(0))
    val coversion_a = System.nanoTime()
    val trainDataSetMat = dataframeToMatrix(trainDataSet)
    val trainDataSetDurationMat = dataframeToMatrix(trainDataSetDuration)
    val coversion_b = System.nanoTime()
    //trainDataSetMat.cache()
    //trainDataSetDurationMat.cache()
    val alpha = 0.1
    for (i <- 0 to nepoch) {
        val paramsMat = dataframeToMatrix(params)
        val pred = trainDataSetMat.multiply(paramsMat.transpose)
        val update = gradDesc(trainDataSetDurationMat, pred, trainDataSetMat)
        params = params.select(col("a") - alpha * update(0) as "a", col("b") - alpha * update(1) as "b")
    }
    // Test model

    val testDataSet = test_data.select("distance").withColumn("x0", lit(1)).select("x0","distance")
    val testDataSetDuration = test_data.select("duration_sec")
    val coversion_c = System.nanoTime()
    val testDataSetMat = dataframeToMatrix(testDataSet)
    val testDataSetDurationMat = dataframeToMatrix(testDataSetDuration)
    val coversion_d = System.nanoTime()
    val paramsMat = dataframeToMatrix(params)
    val testpred = testDataSetMat.multiply(paramsMat.transpose)
    val testsqerr = squaredErr(testDataSetDurationMat, testpred)
    val end = System.nanoTime()
    val conversion_time = coversion_b - coversion_a + coversion_d - coversion_c
    val relational_time = b - a
    val linear_time = end - a - relational_time
    return (conversion_time/10e9, linear_time/10e9, relational_time/10e9)
    */
}

def bench_conference_covariance(publish:DataFrame, ranking_redundant:DataFrame) : (Double, Double, Double) = {
    //Bench conference covariance, using DBLP dataset
    val ranking = ranking_redundant.select("Acronym", "GGS Rating").withColumnRenamed("Acronym", "conference").withColumnRenamed("GGS Rating", "rating")
    val a = System.nanoTime()
    val matrixColumns = publish.columns.map(col(_))
    //Covert dataframe to matrix
    val rdd = publish.select(array(matrixColumns:_*).as("arr")).as[Array[String]].rdd.zipWithIndex().map{ case(arr, index) => IndexedRow(index, Vectors.dense(arr.map(_.toDouble)))}
    val dm = new IndexedRowMatrix(rdd).toRowMatrix()
    val b = System.nanoTime()
    //Compute covariance matrix
    val cov: Matrix = dm.computeCovariance()
    //Covert matrix to dataframe
    val c = System.nanoTime()
    val rdd2 = sc.parallelize(cov.colIter.toSeq).map(x => {
      Row.fromSeq(x.toArray.toSeq)
    })
    var schema = new StructType()
    val ids = ArrayBuffer[String]()
    for (i <- 0 until cov.rowIter.size) {
        schema = schema.add(StructField(i.toString(), DoubleType, true))
        ids.append(i.toString())
    }
    val spark: SparkSession = SparkSession.builder.master("local").getOrCreate
    val df_matrix = spark.createDataFrame(rdd2, schema)
    val df_name = spark.sparkContext.parallelize(publish.columns.toList).toDF("conference")
    val df_covariance = df_matrix.withColumn("row_id", monotonically_increasing_id()).join(df_name.withColumn("row_id", monotonically_increasing_id()), ("row_id")).drop("row_id")
    val d = System.nanoTime()
    //Join and selection
    //df_covariance.show()
    //ranking.show()    
    val df_res = df_covariance.join(ranking, df_covariance("conference") === ranking("conference"), "inner")
    val f = System.nanoTime()
    val df_res2 = df_res.filter("rating = 'A++'")
    //val count = df_res2.count()
    val e = System.nanoTime()
    //df_res2.show()
    //println((f-d)/10e9)
    //println((e-f)/10e9)
    val conversion_time = b - a + d - c 
    return (conversion_time / 10e9, (c - b) / 10e9, (e - d) /10e9)
}

def bench_scala(nrepeat: Int) : Double = {
    spark.conf.set("spark.sql.optimizer.maxIterations", "1000");
    val start_time = System.nanoTime()
    val writer = new PrintWriter("pipeline_result.csv")
    writer.write("framework,pipeline,datasize,conversion_time,linear_time,relational_time\n")
    println("framework,pipeline,datasize,conversion_time,linear_time,relational_time")
    //Bench bixi linear regression
    //val len_list_linear_regression = List(1000)
    val len_list_linear_regression = List(1000, 10000, 100000, 1000000, 10000000)
    //val len_list_linear_regression = List(10000000)
    val num_str_linear_regression = Map(1000 -> "1k", 10000 -> "10k", 100000 -> "100k", 1000000 -> "1m", 10000000 -> "10m")
    val station = spark.read.option("header", "true").csv("../dataset/station.csv")
    for(i <- len_list_linear_regression){
        val trip_path = "../dataset/trip_" + num_str_linear_regression(i) + ".csv"
        val trip = spark.read.option("header", "true").csv(trip_path)
        val trip_df = trip.withColumn("duration_sec", col("duration_sec").cast("int"))
        for(j <- 1 to nrepeat){
            val (conversion_time, linear_time, relational_time) = bench_bixi_linear_regression(100, station, trip_df)
            writer.write("spark,linear_regression," + i + "," + conversion_time + "," + linear_time + "," + relational_time + "\n")
            println("spark,linear_regression," + i + "," + conversion_time + "," + linear_time + "," + relational_time)
        }
    }
   /*
    //Bench conference covariance
    val len_list_conference_covariance = List(10000, 100000, 1000000)
    val num_str_conference_covariance = Map(100 -> "100", 1000 -> "1k", 10000 -> "10k", 100000 -> "100k", 1000000 -> "1m")
    val ranking = spark.read.option("header", "true").csv("../dataset/ranking.csv")
    for(i <- len_list_conference_covariance){
        val publish_path = "../dataset/publish_" + num_str_conference_covariance(i) + ".csv"
        val publish = spark.read.option("header", "true").csv(publish_path)
        for(j <- 1 to nrepeat){
            val (conversion_time, linear_time, relational_time) = bench_conference_covariance(publish, ranking)
            writer.write("spark,covariance," + i + "," + conversion_time + "," + linear_time + "," + relational_time + "\n")
            println("spark,covariance," + i + "," + conversion_time + "," + linear_time + "," + relational_time)
        }
    }
    */
    writer.close()   
    return (System.nanoTime() - start_time) / 10e9
}

val exec_time = bench_scala(1)
print(exec_time)
