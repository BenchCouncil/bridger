import org.apache.spark.mllib.linalg.{Matrix,DenseMatrix,DenseVector}
import org.apache.spark.mllib.random.RandomRDDs
import java.io._
import java.util.Random

def bench_addition(length: Int): Double = {
  val rng = new Random(24);
  val a = RandomRDDs.normalRDD(sc, length)
  val b = RandomRDDs.normalRDD(sc, length)
  val start_time = System.nanoTime()
  val out = a.zip(b).map{ case(i, j) => i + j}
  val duration = (System.nanoTime() - start_time) / 10e6
  return duration
}


def bench_matmul(size: Int): Double = {
  val rng = new Random(24);
  val a = DenseMatrix.randn(size, size, rng)
  val b = DenseMatrix.randn(size, size, rng)
  val start_time = System.nanoTime()
  val out = a.multiply(b)
  val duration = (System.nanoTime() - start_time) / 10e6
  return duration
}

val length = 10000000
val t_add = bench_addition(length)
println("spark,addition," + length+","+t_add)
//val size = 4096
//val t_matmul = bench_matmul(size)
//println("spark,matmul," + size+","+t_matmul)
