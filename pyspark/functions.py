import csv, os, sys
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from utility import *


# Initialize a spark session.
@metrics
def init_spark():
    global spark
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

@metrics
def load_train_test_rdd(filename_train, filename_test):
    train_rdd = spark.read.csv(filename_train, header=True, multiLine=True, inferSchema=True, escape='"', quote='"')
    test_rdd = spark.read.csv(filename_test, header=True, multiLine=True, inferSchema=True, escape='"', quote='"')
    return train_rdd, test_rdd

@metrics
def transform_rdd_to_df(train_rdd,test_rdd):
    training = train_rdd.rdd \
    .map(lambda x: (x["Title"]+" "+x["Body"], x["Y"])) \
    .toDF(["Question", "Output"])

    testing = test_rdd.rdd \
    .map(lambda x: (x["Title"]+" "+x["Body"], x["Y"])) \
    .toDF(["Question", "Output"])
    
    return training, testing

def get_stop_word_remover(input_col_name, stopwords):
    return StopWordsRemover(inputCol=input_col_name, outputCol="filtered").setStopWords(stopwords)


@metrics
def get_heuristics(stop_word_file):
    # HEURISTIC 1 - Tokenize the words
    regexTokenizer = RegexTokenizer(inputCol="Question", outputCol="words", pattern="\\W")
    
    # HEURISTIC 2 - Remove the stopwords
    stop_words = []
    with open(stop_word_file, "r") as text_file:
        stop_words = text_file.read().split('\n')
    
    stopwordsRemover = get_stop_word_remover("words", stop_words)
    
    return regexTokenizer, stopwordsRemover


@metrics
def get_bag_of_word_model(features_col_name, label_col_name):
    countVectors = CountVectorizer(inputCol=features_col_name, outputCol="features")
    indexed_features = StringIndexer(inputCol = label_col_name, outputCol = "label")
    return countVectors, indexed_features

@metrics
def get_pipeline(*args):
    return Pipeline(stages=[*args])


@metrics
def get_pipeline_model(pipeline, data):
    """ We should use the same pipeline model on training and testing """
    return pipeline.fit(data)

@metrics
def transform_data_through_pipeline(model, data):
    data_transformed = model.transform(data)
    return data_transformed

@metrics
def split_dataset(data, distribution):
    return data.randomSplit([distribution, 1-distribution], seed = 1234)


@metrics
def get_best_smoothing_values(target_col, prediction_col):
    # Create grid to find best smoothing
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    paramGrid = ParamGridBuilder().addGrid(nb.smoothing, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).build()
#     cvEvaluator = BinaryClassificationEvaluator(rawPredictionCol=prediction_col)
    cvEvaluator = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol=prediction_col)

    # Cross-validate all smoothing values
    cv = CrossValidator(estimator=nb, estimatorParamMaps=paramGrid, evaluator=cvEvaluator)
    return cv


@metrics
def train_naive_bayes_model(cv, data):
    model = cv.fit(data)
    return model


@metrics
def predict(model, data):
    predictions = model.transform(data)
    return predictions


@metrics
def evaluate_model(target_col, prediction_col, predictionAndTarget):
#     evaluator = BinaryClassificationEvaluator(rawPredictionCol=prediction_col)
    evaluatorMulti = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol=prediction_col)
#     accuracy = evaluatorMulti.evaluate(predictions)
    # Get metrics
    acc = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "accuracy"})
    f1 = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "f1"})
    weightedPrecision = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedPrecision"})
    weightedRecall = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedRecall"})
#     auc = evaluator.evaluate(predictionAndTarget)
    print("\n******** Metrics **********")
    print ("Model Accuracy: {:.3f}%".format(acc*100))
    print ("Model f1-score: {:.3f}%".format(f1*100))
    print ("Model weightedPrecision: {:.3f}%".format(weightedPrecision*100))
    print ("Model weightedRecall: {:.3f}%".format(weightedRecall*100))
    
    
    
def main_spark():
    print("##########################################")
    print("############# Start Spark ################")
    print("##########################################")
    global spark
    spark = init_spark()
    filename_train = "../dataset/train.csv"
    filename_test = "../dataset/valid.csv"
    stop_word_file = "../dataset/stop_words.txt"
    print("\n#########################################")
    print("####### Load dataset in spark rdd #######")
    print("###########################################")
    train_rdd, test_rdd = load_train_test_rdd(filename_train, filename_test)
    print("\n#########################################")
    print("########## Transform rdd to df ############")
    print("##########################################")
    train_df, test_df = transform_rdd_to_df(train_rdd,test_rdd)
    print("\n##########################################")
    print("########## Create Heuristics #############")
    print("##########################################")
    regexTokenizer, stopwordsRemover = get_heuristics(stop_word_file)
    countVectors_h1, indexed_features_h1 = get_bag_of_word_model("words", "Output")
    countVectors_h2, indexed_features_h2 = get_bag_of_word_model("filtered", "Output")
    print("\n##########################################")
    print("############ Construct pipeline ############")
    print("############################################")
    pipeline = get_pipeline(regexTokenizer, stopwordsRemover, countVectors_h2, indexed_features_h2)
    print("\n##########################################")
    print("########### Train pipeline model ###########")
    print("############################################")
    model_pipeline = get_pipeline_model(pipeline, train_df)
    process_info()
    print("\n#############################################")
    print("####### Transform train data through pipeline #####")
    print("################################################")
    train = transform_data_through_pipeline(model_pipeline, train_df)
    print("\n##################################################")
    print("####### Transform test data through pipeline #######")
    print("##################################################")
    test = transform_data_through_pipeline(model_pipeline, test_df)
    print("\n###################################################")
    print("####### Train naive base classifier model #######")
    print("####################################################")
    cv = get_best_smoothing_values("label", "prediction")
    nb_model = train_naive_bayes_model(cv,train)
    print("\n##############################################################")
    print("### Predict test data using naive base classifier model ######")
    print("################################################################")
    predictions = predict(nb_model, test)
    print("####################################")
    print("####### Evaluate predictions #######")
    print("#####################################")
    evaluate_model("label", "prediction",predictions.select("label","prediction"))











