from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import udf
import math
from pyspark.sql.types import DoubleType


def calc_vec_len(item_scores):
    """calculates the length of the vector"""
    length = 0.0
    for score in item_scores.values():
        length += (score * score)
    return math.sqrt(length)


def cos_sim(vector1, vector2, vec_len1, vec_len2):
    """calculate cosine simulate of two vectors"""
    dot = 0.0
    for user_id1, score1 in vector1.items():
        score2 = vector2.get(user_id1, 0.0)
        dot += (score1 * score2)

    return dot / (vec_len1 * vec_len2)


if __name__ == '__main__':
    spark = SparkSession.builder.appName("item_cf").master("local[*]").getOrCreate()
    sc = spark.sparkContext

    df = sc.parallelize([
        Row(userId='Bob', itemId="A", score=5.0), \
        Row(userId='Bob', itemId="D", score=1.0), \
        Row(userId='Bob', itemId="E", score=1.0), \
        Row(userId='Ali', itemId="C", score=1.0), \
        Row(userId='Ali', itemId="E", score=4.0), \
        Row(userId='Tom', itemId="A", score=4.0), \
        Row(userId='Tom', itemId="B", score=2.0), \
        Row(userId='Tom', itemId="E", score=2.0), \
        Row(userId='Zoe', itemId="A", score=2.0), \
        Row(userId='Zoe', itemId="B", score=5.0), \
        Row(userId='Amy', itemId="B", score=0.0001), \
        Row(userId='Amy', itemId="C", score=0.001), \
        Row(userId='Amy', itemId="D", score=4.0)]).toDF()

    # (itemId, (userId, score))
    item_score_df = df.rdd.map(lambda row: (row[0], (row[2], row[1]))) \
        .groupByKey() \
        .map(lambda kv: Row(item_id = kv[0], scores = dict(kv[1]))) \
        .toDF()

    calc_vec_len_udf = udf(calc_vec_len, returnType=DoubleType())
    item_score_len_df = item_score_df \
        .withColumn("vec_len", calc_vec_len_udf(item_score_df["scores"])) \
        .cache()

    item_score_len_df.show(truncate = False)

    item2item_df = item_score_len_df.alias("t1").crossJoin(item_score_len_df.alias("t2")) \
        .where("t1.item_id < t2.item_id") \
        .cache()
    item2item_df.show(truncate = False)

    cos_sim_udf = udf(cos_sim, returnType=DoubleType())
    item_sim_df = item2item_df \
        .withColumn("cos_sim", cos_sim_udf(item2item_df["t1.scores"], item2item_df["t2.scores"], item2item_df["t1.vec_len"], item2item_df["t2.vec_len"])) \
        .select("t1.item_id", "t2.item_id", "cos_sim")
    item_sim_df.show(truncate = False)

    spark.stop()
    sc.stop()