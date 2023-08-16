#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 08 2023

@author: shong
"""

import sys
import numpy as np
import pandas as pd
import glob
import sys
import h5py
#from netCDF4 import Dataset
from datetime import datetime
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import pyarrow as pa
import pyarrow.parquet as pq

from functools import reduce
import operator
import gc

# PySpark packages
from pyspark import SparkContext   
from pyspark.sql import SparkSession

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark import Row
from pyspark.sql.window import Window as W

from functools import partial 




if __name__ == '__main__':
    # Execution Arguments 
    if len(sys.argv) != 3:
        print("Usage : <command> <infile_hdf5> <outfile_parquet> ")
        sys.exit()
    else:
        print("=== Parsing the input argv")
        for idx, tmparg in enumerate(sys.argv):
            print("argc = %d, argv[%d] = %s" % (idx,idx,tmparg))


    # Define Spark Session
    print("Program Log : Defining SparkSession...")
    
    spark = SparkSession.builder \
    .master("yarn") \
    .appName("convert-h5-parquet") \
    .config("spark.driver.maxResultSize", "32g") \
    .config("spark.driver.memory", "32g") \
    .config("spark.executor.memory", "14g") \
    .config("spark.executor.cores", "2") \
    .config("spark.executor.instances", "50") \
    .getOrCreate()


    sc = spark.sparkContext
    sc.setCheckpointDir("hdfs://spark00:54310/tmp/checkpoints")

    spark.conf.set("spark.sql.debug.maxToStringFields", 500)
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    
        
    # infile and outfile names from sys.argv
    infile  = sys.argv[1]
    outfile = sys.argv[2]
    
    ############## Read h5 ###############
    try:
        h5f = h5py.File(infile, "r")
    except IOError as e:
        print("Error opening HDF5 file:", str(e))
    # Don't forget f.close() when done! 
    
    schema = T.StructType([\
                       T.StructField('filename',T.StringType(), True),\
                       T.StructField('altitude',T.FloatType(), True),\
                       T.StructField('longitude',T.FloatType(), True),\
                       T.StructField('latitude',T.FloatType(), True),\
                       T.StructField('aspect',T.FloatType(), True),\
                       T.StructField('slope',T.FloatType(), True),\
                       T.StructField('sol_az',T.FloatType(), True),\
                       T.StructField('sol_zn',T.FloatType(), True),\
                       T.StructField('xco2',T.FloatType(), True),\
                       T.StructField('time_str',T.StringType(), True)\
                      ])
    numresults = len(h5f['RetrievalResults/xco2'][:])
    
    fname = infile.split('oco2_')[-1]
    fillist = [fname for i in range(numresults)]
    
    altlist = []
    lonlist = []
    latlist = []
    asplist = []
    slolist = []
    sazlist = []
    sznlist = []
    tstrlist = []

    altlist = h5f['RetrievalGeometry/retrieval_altitude'][:].tolist()
    lonlist = h5f['RetrievalGeometry/retrieval_longitude'][:].tolist()
    latlist = h5f['RetrievalGeometry/retrieval_latitude'][:].tolist()
    asplist = h5f['RetrievalGeometry/retrieval_aspect'][:].tolist()
    slolist = h5f['RetrievalGeometry/retrieval_slope'][:].tolist()
    sazlist = h5f['RetrievalGeometry/retrieval_solar_azimuth'][:].tolist()
    sznlist = h5f['RetrievalGeometry/retrieval_solar_zenith'][:].tolist()
    xcolist = h5f['RetrievalResults/xco2'][:].tolist()
    
    
    tstrlist = [ onestr.decode() for onestr in h5f['RetrievalHeader/retrieval_time_string'][:].tolist()]


    
    ################ Save to a parquet 
    sparkdf = spark.createDataFrame(zip(fillist,altlist,lonlist,latlist,asplist,slolist, \
                                        sazlist,sznlist,xcolist,tstrlist),schema)
    
    sparkdf.write.option("compression","snappy").mode("overwrite").save(outfile)
    #sparkdf.write.parquet("s3a://jwpark-spark/bigdata_result/"+outfile,mode="overwrite")
    
    sc.stop()