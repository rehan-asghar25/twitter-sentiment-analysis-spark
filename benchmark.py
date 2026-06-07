import os
import sys
import time
import json
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import pyspark.sql.functions as F
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, NGram, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

def run_benchmark(cores):
    print(f"\n{'='*50}")
    print(f"🚀 STARTING BENCHMARK: {cores} CORE(S)")
    print(f"{'='*50}\n")
    
    # Configure Spark to use the specific number of cores passed by the user
    master_config = f"local[{cores}]" if cores != "*" else "local[*]"
    
    spark = SparkSession.builder \
        .appName(f"PDC-Benchmark-{cores}-Cores") \
        .master(master_config) \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    # Set to ERROR to hide standard warnings so we can focus on the timer
    spark.sparkContext.setLogLevel("ERROR")

    # 1. Ingest Data
    schema = StructType([
        StructField("target", IntegerType(), True),
        StructField("id", StringType(), True),
        StructField("date", StringType(), True),
        StructField("flag", StringType(), True),
        StructField("user", StringType(), True),
        StructField("text", StringType(), True)
    ])
    
    data_path = "data/training.1600000.processed.noemoticon.csv"
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        spark.stop()
        return

    raw_df = spark.read.csv(data_path, schema=schema, header=False)
    
    # 2. Clean Data
    cleaned_df = raw_df.select(
        F.when(F.col("target") == 4, 1).otherwise(0).alias("label"),
        F.col("text")
    ).withColumn(
        "clean_text", F.regexp_replace(F.col("text"), r"@[A-Za-z0-9_]+", "")
    ).withColumn(
        "clean_text", F.regexp_replace(F.col("clean_text"), r"https?://[A-Za-z0-9./]+", "")
    ).withColumn(
        "clean_text", F.regexp_replace(F.col("clean_text"), r"[^a-zA-Z0-9\s]", "")
    ).withColumn(
        "clean_text", F.lower(F.trim(F.col("clean_text")))
    ).filter(
        F.col("clean_text") != ""
    )

    # 3. Build Pipeline
    negation_terms = {"not", "no", "nor", "never", "cannot", "cant", "wont", "dont", "didnt", "isnt", "arent"}
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    default_stopwords = set(StopWordsRemover.loadDefaultStopWords("english"))
    custom_stopwords = sorted(default_stopwords - negation_terms)
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words", stopWords=custom_stopwords)
    ngram = NGram(n=2, inputCol="filtered_words", outputCol="bigrams")
    unigram_tf = HashingTF(inputCol="filtered_words", outputCol="unigram_tf", numFeatures=1 << 18)
    bigram_tf = HashingTF(inputCol="bigrams", outputCol="bigram_tf", numFeatures=1 << 17)
    unigram_idf = IDF(inputCol="unigram_tf", outputCol="unigram_features")
    bigram_idf = IDF(inputCol="bigram_tf", outputCol="bigram_features")
    assembler = VectorAssembler(inputCols=["unigram_features", "bigram_features"], outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
    
    pipeline = Pipeline(
        stages=[tokenizer, remover, ngram, unigram_tf, bigram_tf, unigram_idf, bigram_idf, assembler, lr]
    )

    # 4. Split Data
    train_data, test_data = cleaned_df.randomSplit([0.8, 0.2], seed=42)

    # 5. START TIMING THE TRAINING
    print(f"-> Data prepared. Starting distributed ML training on {cores} core(s)...")
    
    start_time = time.time()
    model = pipeline.fit(train_data)
    end_time = time.time()
    
    # 6. CALCULATE AND PRINT RESULTS
    duration = end_time - start_time
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"⏱️ Execution Time ({cores} cores): {duration:.2f} seconds")
    print(f"{'='*50}\n")

    os.makedirs("results", exist_ok=True)
    benchmark_path = "results/benchmark_results.json"
    if os.path.exists(benchmark_path):
        with open(benchmark_path, "r", encoding="utf-8") as benchmark_file:
            benchmark_data = json.load(benchmark_file)
    else:
        benchmark_data = {"results": []}

    normalized_cores = str(cores)
    benchmark_data["results"] = [
        row for row in benchmark_data.get("results", []) if str(row.get("cores")) != normalized_cores
    ]
    benchmark_data["results"].append(
        {
            "cores": int(cores) if str(cores).isdigit() else str(cores),
            "time_seconds": float(duration),
        }
    )
    benchmark_data["results"].sort(key=lambda row: str(row["cores"]))

    with open(benchmark_path, "w", encoding="utf-8") as benchmark_file:
        json.dump(benchmark_data, benchmark_file, indent=2)
    print(f"-> Benchmark results saved to {benchmark_path}")
    
    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_benchmark(sys.argv[1])
    else:
        print("Error: Please provide the number of cores.")
        print("Usage: python benchmark.py <number_of_cores>")
        print("Example: python benchmark.py 1")