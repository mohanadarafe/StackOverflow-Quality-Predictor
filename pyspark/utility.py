from time import time
from datetime import timedelta
from sys import argv
import psutil as psu
import csv, os

def system_info():
    info = {
#         "Uptime": timedelta(seconds=time()-psu.boot_time()),
        "CPU in use": "{:.2f}%".format(psu.cpu_percent(interval=.1)),
        "Time on CPU": timedelta(seconds=psu.cpu_times().system+psu.cpu_times().user),
        "Memory in use": "{:.2f}GiB".format(psu.virtual_memory().available/(1024**3)),
        "Disk in use": "{:.2f}%".format(psu.disk_usage('/').percent),
        "Disk free": "{:.2f}GiB".format(psu.disk_usage('/').free/(1024**3)),    
    }
    
    print("\n\n ========== SYSTEM INFO ===========\n\n" + "\n".join(["{}: {}".format(key,value) for key,value in info.items()]))
    
    
def process_info():
    for process in psu.process_iter(attrs=('name','cmdline','pid','create_time','cpu_percent','cpu_times','num_threads','memory_percent')):
#         print(process.info["name"])
        if "python3.5" in process.info["name"]:
            mem = process.info['memory_percent']
            info = {
                "PID": process.info["pid"],
                "Create time": datetime.fromtimestamp(process.create_time()).strftime("%Y-%m-%d %H:%M:%S"),
                "Uptime": timedelta(seconds=time() - process.info["create_time"]),
                "CPU in use": "{:.2f}%".format(process.info['cpu_percent']),
                "Time on CPU": timedelta(seconds=process.info["cpu_times"].system + process.info["cpu_times"].user),
                "Nb of threads": process.info["num_threads"],
                "Memory in use": "{:.2f}%".format(mem),
                "Memory_usage": "{:.2f} GiB".format(psu.virtual_memory().total*(mem/100)/(1024**3)),
            }
            
            print("\n\n ********* PROCESS INFO *********\n\n" + "\n".join(["{}: {}".format(key,value) for key,value in info.items()]))
            

            
def metrics(fun):
    
    def wrapper(*args, **kwargs):
        print('--------- BEFORE CALL TO FUNCTION ---------')
        system_info()
        start = time()
        rv = fun(*args, **kwargs)
        duration = time() - start
        print('\n\n --------- AFTER CALL TO FUNCTION ---------')
        system_info()
#         process_info()
        print("\n----------> Execution Time: {}".format(duration))
        return rv
    
    return wrapper
    
# spark = init_spark()
# filename_train = "../dataset/train.csv"
# filename_test = "../dataset/valid.csv"
# stop_word_file = "../dataset/stop_words.txt"
# train_rdd, test_rdd = load_train_test_rdd(filename_train, filename_test)
# train_df, test_df = transform_rdd_to_df(train_rdd,test_rdd)
# regexTokenizer, stopwordsRemover = get_heuristics(stop_word_file)
# countVectors_h1, indexed_features_h1 = get_bag_of_word_model("words", "Output")
# countVectors_h2, indexed_features_h2 = get_bag_of_word_model("filtered", "Output")
# pipeline = get_pipeline(regexTokenizer, stopwordsRemover, countVectors_h2, indexed_features_h2)

# model_pipeline = get_pipeline_model(pipeline, train_df)
# train = transform_data_through_pipeline(model_pipeline, train_df)
# test = transform_data_through_pipeline(model_pipeline, test_df)

# cv = get_best_smoothing_values("label", "prediction")
# nb_model = train_naive_bayes_model(cv,train)
# predictions = predict(nb_model, test)

# evaluate_model("label", "prediction",predictions.select("label","prediction"))

# system_info()
# process_info()