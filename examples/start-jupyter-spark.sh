#!/bin/bash
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS='notebook --no-browser --allow-root --port=17777 '
pyspark --master spark://spark-test-extra-master:7077
