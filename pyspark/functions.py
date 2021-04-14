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
def load_data(filename_train, filename_test):
    """ Load data in rdd """
    train_rdd = spark.read.csv(filename_train, header=True, multiLine=True, inferSchema=True, escape='"', quote='"')
    test_rdd = spark.read.csv(filename_test, header=True, multiLine=True, inferSchema=True, escape='"', quote='"')
    return train_rdd, test_rdd

@metrics
def clean_data(train_rdd,test_rdd):
    """ Filter rdd and convert to dataframe """
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
def preprocess_data(stop_word_file):
    """ get heuristics """
    # HEURISTIC 1 - Tokenize the words
    regexTokenizer = RegexTokenizer(inputCol="Question", outputCol="words", pattern="\\W")
    
    # HEURISTIC 2 - Remove the stopwords
    stop_words = []
    with open(stop_word_file, "r") as text_file:
        stop_words = text_file.read().split('\n')
    
    stopwordsRemover = get_stop_word_remover("words", stop_words)
    
    return regexTokenizer, stopwordsRemover


@metrics
def init_bow(features_col_name, label_col_name):
    """ initialize bag of word model """
    countVectors = CountVectorizer(inputCol=features_col_name, outputCol="features")
    indexed_features = StringIndexer(inputCol = label_col_name, outputCol = "label")
    return countVectors, indexed_features


def get_pipeline(*args):
    return Pipeline(stages=[*args])


@metrics
def build_bow(pipeline, data):
    """ We should use the same pipeline model on training and testing 
        Buildng pipeline model where we have the bow 
    """ 
    return pipeline.fit(data)

@metrics
def transform_bow(model, data):
    """transform_data_through_pipeline"""
    data_transformed = model.transform(data)
    return data_transformed

@metrics
def split_dataset(data, distribution):
    return data.randomSplit([distribution, 1-distribution], seed = 1234)


def hypertune(target_col, prediction_col):
    """get_best_smoothing_values"""
    # Create grid to find best smoothing
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    paramGrid = ParamGridBuilder().addGrid(nb.smoothing, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).build()

    cvEvaluator = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol=prediction_col)

    # Cross-validate all smoothing values
    cv = CrossValidator(estimator=nb, estimatorParamMaps=paramGrid, evaluator=cvEvaluator)
    return cv


@metrics
def train(cv, data):
    """ train_naive_bayes_model """
    model = cv.fit(data)
    return model


@metrics
def predict(model, data):
    predictions = model.transform(data)
    return predictions


@metrics
def evaluate(target_col, prediction_col, predictionAndTarget):
    evaluatorMulti = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol=prediction_col)

    # Get metrics
    acc = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "accuracy"})
    f1 = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "f1"})
    weightedPrecision = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedPrecision"})
    weightedRecall = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedRecall"})

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
    train_rdd, test_rdd = load_data(filename_train, filename_test)
    print("\n#########################################")
    print("########## Transform rdd to df ############")
    print("##########################################")
    train_df, test_df = clean_data(train_rdd,test_rdd)
    print("\n##########################################")
    print("########## Create Heuristics #############")
    print("##########################################")
    regexTokenizer, stopwordsRemover = preprocess_data(stop_word_file)
    countVectors, indexed_features = init_bow("filtered", "Output")
    print("\n##########################################")
    print("############ Construct pipeline ############")
    print("############################################")
    pipeline = get_pipeline(regexTokenizer, stopwordsRemover, countVectors, indexed_features)
    print("\n##########################################")
    print("########### Train pipeline model ###########")
    print("############################################")
    model_pipeline = build_bow(pipeline, train_df)
    print("\n#############################################")
    print("####### Transform train data through pipeline #####")
    print("################################################")
    train_data = transform_bow(model_pipeline, train_df)
    print("\n##################################################")
    print("####### Transform test data through pipeline #######")
    print("##################################################")
    test = transform_bow(model_pipeline, test_df)
    print("\n###################################################")
    print("####### Train naive base classifier model #######")
    print("####################################################")
    cv = hypertune("label", "prediction")
    nb_model = train(cv,train_data)
    print("\n##############################################################")
    print("### Predict test data using naive base classifier model ######")
    print("################################################################")
    predictions = predict(nb_model, test)
    print("####################################")
    print("####### Evaluate predictions #######")
    print("#####################################")
    evaluate("label", "prediction",predictions.select("label","prediction"))
    print('\n\n(((((((((((((PROCESSES)))))))))))))))')
    process_info()
    print('((((((((((((((((())))))))))))))))))))')











