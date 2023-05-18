import os

def set_master_local():
    os.environ["PYSPARK_SUBMIT_ARGS"] = "--master local[*] pyspark-shell" #  "--master local[*]"

def set_master_remote():
    os.environ["PYSPARK_SUBMIT_ARGS"] = "--master spark://remote-master:7077"

# SalviaDivinorum.conf: --master spark://172.28.176.216:7077 --conf spark.driver.host=172.28.176.1
