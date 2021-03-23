import nltk
from pyspark.sql import SparkSession
from pyspark.rdd import RDD
from os import path
from nltk.stem import PorterStemmer 

def init_spark():
    '''
    The following function taken from our lab assignments is used to initialize a Spark session.
    '''
    spark = SparkSession \
        .builder \
        .appName("Spark session for preprocessing data") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

def remove_punctuation(string):
    '''
    The following function inputs a string & returns the string with no punctuation
    '''
    punctuation = '*+,-.@[\\]^_/:;!"#$%&\'()<=>?`{|}~'
    lower_str = string.lower()
    for punc in punctuation:
        lower_str = lower_str.replace(punc, '')
    return lower_str

def tokenize(seq):
    '''
    The following function uses NLTK's library to tokenize a string. It returns an array of strings.
    '''
    return nltk.word_tokenize(seq)

def stem(seq):
    '''
    The following function uses NLTK's library to stem a string using Porter's Stemmer. It returns
    an array of strings.
    '''
    spark = init_spark()
    ps = PorterStemmer()
    rdd = spark.sparkContext.parallelize(tokenize(seq))
    collection = rdd \
        .map(lambda token: ps.stem(token)) \
        .map(lambda token: remove_punctuation(token)) \
        .filter(lambda token: len(token) != 0) \
        .collect()
        
    return collection
    
def stopword_removal(seq):
    '''
    The following function uses a list of 30 most common stopwords & removes them from a string. It
    returns an array of strings.
    '''
    spark = init_spark()
    rdd = spark.sparkContext.parallelize(tokenize(seq))
    stopwords = []
    assert path.exists("utilities/stopwords.txt"), "Make sure the utilities package contains the stopwords!"
    with open("utilities/stopwords.txt", 'r') as fp:
        stopwords = fp.readlines()

    stopwords = [words.replace('\n', "") for words in stopwords]

    collection = rdd \
        .filter(lambda token: token.lower() not in stopwords) \
        .map(lambda token: remove_punctuation(token)) \
        .filter(lambda token: len(token) != 0) \
        .collect()

    return collection

if __name__ == "__main__":
    testStr = "This is a test string that will test all three heuristics for the preprocssing of our data."
    print(f'Tokenizer\n{"-"*60}\n{tokenize(testStr)}\n{"-"*60}\n')
    print(f'Stemmer\n{"-"*60}\n{stem(testStr)}\n{"-"*60}\n')
    print(f'Stopword removal\n{"-"*60}\n{stopword_removal(testStr)}\n{"-"*60}\n')