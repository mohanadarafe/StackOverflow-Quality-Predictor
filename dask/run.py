import dask.dataframe as dd
import dask.bag as db
import dask.array as da
from dask_ml.feature_extraction.text import CountVectorizer
from dask_ml import preprocessing
from dask_ml.wrappers import ParallelPostFit
from dask_ml.naive_bayes import GaussianNB
import pandas as pd
import csv, sys, multiprocessing
from os import path
from nltk import word_tokenize
sys.path.append('../pyspark')
from utility import *

filename_train = "../dataset/train.csv"
filename_test = "../dataset/valid.csv"
NUMBER_OF_CPU = multiprocessing.cpu_count()
STOPWORDS = [] 
with open("../dataset/stop_words.txt", "r") as r:
    STOPWORDS = r.read().split('\n')

@metrics
def load_data(trainFile, testFile):
    panda_train = pd.read_csv(trainFile)
    panda_test = pd.read_csv(testFile)
    train_df = dd.from_pandas(panda_train, npartitions=NUMBER_OF_CPU)
    test_df = dd.from_pandas(panda_test, npartitions=NUMBER_OF_CPU)
    return train_df, test_df

def get_question(partition):
    title = partition.Title
    body = partition.Body
    return title + " " + body

def get_quality(partition):
    return partition.Y

@metrics
def clean_data(train, test):
    train["X_trn"] = train.map_partitions(get_question, meta=str)
    train["y_trn"] = train.map_partitions(get_quality, meta=str)
    test["X_tst"] = test.map_partitions(get_question, meta=str)
    test["y_tst"] = test.map_partitions(get_quality, meta=str)
    new_train = train.drop(['Id', 'Title', 'Body', 'CreationDate', 'Y', 'Tags'], axis=1)
    new_test = test.drop(['Id', 'Title', 'Body', 'CreationDate', 'Y', 'Tags'], axis=1)
    return new_train, new_test

@metrics
def preprocess_data(training, testing):
    if isinstance(training.head().loc[0, 'X_trn'], str):
        training["X_trn"] = training["X_trn"].str.lower()
        training["X_trn"] = training["X_trn"].replace(to_replace="(\\W)+", value=' ', regex=True)
        training['X_trn'] = training['X_trn'].apply(lambda x: [token for token in x.split(" ")], meta=str)
        training['X_trn'] = training['X_trn'].apply(lambda x: [token for token in x if token not in STOPWORDS], meta=str)
        training['X_trn'] = training['X_trn'].apply(lambda x: [token for token in x if token], meta=str)
        training['X_trn'] = training['X_trn'].apply(lambda x: " ".join(x), meta=str)
        
    if isinstance(testing.head().loc[0, 'X_tst'], str):
        testing["X_tst"] = testing["X_tst"].str.lower()
        testing["X_tst"] = testing["X_tst"].replace(to_replace="(\\W)+", value=' ', regex=True)
        testing['X_tst'] = testing['X_tst'].apply(lambda x: [token for token in x.split(" ")], meta=str)
        testing['X_tst'] = testing['X_tst'].apply(lambda x: [token for token in x if token not in STOPWORDS], meta=str)
        testing['X_tst'] = testing['X_tst'].apply(lambda x: [token for token in x if token], meta=str)
        testing['X_tst'] = testing['X_tst'].apply(lambda x: " ".join(x), meta=str)

def compute_chunks(X_train, y_train, X_test, y_test):
    X_train.compute_chunk_sizes()
    y_train.compute_chunk_sizes()
    X_test.compute_chunk_sizes()
    y_test.compute_chunk_sizes()
    
def convert_X_data(train, test):
    X_train = train.map_blocks(lambda x: x.toarray(), dtype=int)
    X_test = test.map_blocks(lambda x: x.toarray(), dtype=int)
    return X_train, X_test

@metrics
def build_bow_model(training, testing):
    vectorizer = CountVectorizer()
    encoder = preprocessing.LabelEncoder()
    
    print("Converting to Dask Databags...")
    X_train_db = db.from_sequence(training['X_trn'], npartitions=NUMBER_OF_CPU)
    X_test_db = db.from_sequence(testing['X_tst'], npartitions=NUMBER_OF_CPU)

    print("Building BoW...")
    X_model = vectorizer.fit(X_train_db)
    X_train = X_model.transform(X_train_db)
    X_test = X_model.transform(X_test_db)

    print("Indexing strings...")
    y_model = encoder.fit(training['y_trn'])
    y_train = y_model.transform(training['y_trn'])
    y_test = y_model.transform(testing['y_tst'])
    
    print("Computing chunks...")
    compute_chunks(X_train, y_train, X_test, y_test)
    
    print("Re-convert to Dask Array")
    Xtrain, Xtest = convert_X_data(X_train, X_test)
        
    return Xtrain, y_train, Xtest, y_test

@metrics
def train_model(x_train, y_train):
    clf = ParallelPostFit(estimator = GaussianNB(), scoring='accuracy')
    clf.fit(x_train, y_train)
    return clf

print("###############LOADING DATA###############")
train_df, test_df = load_data(filename_train, filename_test)

print("###############CLEANING DATA###############")
training, testing = clean_data(train_df, test_df)
process_info()

print("###############PREPROCESSING DATA###############")
preprocess_data(training, testing)

print("###############BUILDING BOW###############")
X_train, y_train, Xtest, y_test = build_bow_model(training, testing)

print("###############TRAINING DATA###############")
clf = train_model(X_train, y_train)
process_info()
