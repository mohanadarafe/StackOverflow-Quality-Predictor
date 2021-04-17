from time import time
from datetime import timedelta, datetime
from sys import argv
from os import path
import psutil as psu
import csv, os
import json


def system_info():
    info = {
        "CPU in use": "{:.2f}%".format(psu.cpu_percent(interval=.1)),
        # "Time on CPU": timedelta(seconds=psu.cpu_times().system + psu.cpu_times().user),
        "Memory in use": "{:.2f}GiB".format(psu.virtual_memory().available / (1024 ** 3)),
        "Disk in use": "{:.2f}%".format(psu.disk_usage('/').percent),
        "Disk free": "{:.2f}GiB".format(psu.disk_usage('/').free / (1024 ** 3)),
    }

    print("\n\n ========== SYSTEM INFO ===========\n\n" + "\n".join(
        ["{}: {}".format(key, value) for key, value in info.items()]))

    return info


def process_info():
    processes_info = []
    for process in psu.process_iter(attrs=(
            'name', 'cmdline', 'pid', 'create_time', 'cpu_percent', 'cpu_times', 'num_threads', 'memory_percent')):

        if process.info["name"] is not None and ("python" in process.info["name"] or "java" in process.info["name"]):
            mem = process.info['memory_percent']
            info = {
                "PID": process.info["pid"],
                "Process name": process.info["name"],
                "Current time": datetime.fromtimestamp(time()).strftime("%Y-%m-%d %H:%M:%S"),
                "Create time": datetime.fromtimestamp(process.create_time()).strftime("%Y-%m-%d %H:%M:%S"),
                "Uptime": timedelta(seconds=time() - process.info["create_time"]),
                "CPU in use": "{:.2f}%".format(process.info['cpu_percent']),
                "Time on CPU": timedelta(seconds=process.info["cpu_times"].system + process.info["cpu_times"].user),
                "Nb of threads": process.info["num_threads"],
                "Memory in use": "{:.2f}%".format(mem),
                "Memory_usage": "{:.2f} GiB".format(psu.virtual_memory().total * (mem / 100) / (1024 ** 3)),
            }
            processes_info.append(info)
            print("\n\n ********* PROCESS INFO *********\n\n" + "\n".join(
                ["{}: {}".format(key, value) for key, value in info.items()]))

    process_dict = {"processes": processes_info}

    with open('processes_info.json', 'w') as fp:
        json.dump(process_dict, fp, default=str)

    return processes_info


def create_files():
    if os.path.exists("system_info.json"):
        os.remove("system_info.json")

    if os.path.exists("processes_info.json"):
        os.remove("processes_info.json")

    with open("system_info.json", "w") as f:
        f.write("{}")

    with open("processes_info.json", "w") as a:
        a.close()


# decorator
def metrics(fun):
    def wrapper(*args, **kwargs):
        print('--------- BEFORE CALL TO FUNCTION ---------')
        fun_name = fun.__name__
        info_before_call = system_info()
        start = time()
        rv = fun(*args, **kwargs)
        duration = time() - start
        print('\n\n --------- AFTER CALL TO FUNCTION ---------')
        info_after_call = system_info()
        system_dict = {fun_name: [info_before_call, info_after_call, duration]}
        print("\n----------> Execution Time: {:.5f} seconds".format(duration))

        with open("system_info.json", "r+") as file:
            data = json.load(file)
            data.update(system_dict)
            file.seek(0)
            json.dump(data, file, default=str)
        return rv

    return wrapper
