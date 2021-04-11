from time import time
from datetime import timedelta
from sys import argv
import psutil as psu
import csv, os

def system_info():
    info = {
        "Uptime": timedelta(seconds=time()-psu.boot_time()),
        "CPU in use": "{}%".format(psu.cpu_percent(interval=.1)),
        "Time on CPU": timedelta(seconds=psu.cpu_times().system+psu.cpu_times().user),
        "Memory in use": "{}GiB".format(psu.virtual_memory().available/(1024**3)),
        "Disk in use": "{}%".format(psu.disk_usage('/').percent),
        "Disk free": "{}GiB".format(psu.disk_usage('/').free/(1024**3)),    
    }
    
    print("\n\n SYSTEM INFO\n\n" + "\n".join(["{}: {}".format(key,value) for key,value in info.items()]))
    
    
    
def process_info():
    for process in psu.process_iter(attrs=('name','cmdline','pid','create_time','cpu_percent','cpu_times','num_threads','memory_percent')):
        if "python" in process.info["name"]:
            mem = process.info['memory_percent']
            info = {
                "PID": process.info["pid"],
                "Updtime": timedelta(seconds=time() - process.info["create_time"]),
                "CPU in use": "{}%".format(process.info['cpu_percent']),
                "Time on CPU": timedelta(seconds=process.info["cpu_times"].system + process.info["cpu_times"].user),
                "Nb of threads": process.info["num_threads"],
                "Memory in use": "{}%".format(mem),
                "Memory_usage": "{} GiB".format(psu.virtual_memory().total*(mem/100)/(1024**3)),
            }
            
            print("\n\n PROCESS INFO\n\n" + "\n".join(["{}: {}".format(key,value) for key,value in info.items()]))
            

            
def timer(fun):
    
    def wrapper(*args, **kwargs):
        print('BEFORE CALL TO FUNCTION')
        system_info()
        start = time()
        rv = fun(*args, **kwargs)
        duration = time() - start
        print('\n\n AFTER CALL TO FUNCTION')
        system_info()
        print("Execution Time: {}".format(duration))
        return rv
    
    return wrapper
    
