import os
from os import path
assert os.path.basename(os.getcwd()) == "pyspark", "Please run the program from the pyspark directory!"

from utility import *
from functions import *

create_files()
main_spark()
print('\n\n(((((((((((((PROCESSES)))))))))))))))')
process_info()
print('((((((((((((((((())))))))))))))))))))')
