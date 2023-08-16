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
                       T.StructField('channel_ind',T.IntegerType(), True),\
                       T.StructField('pix_ind',T.IntegerType(), True),\
                       T.StructField('row_ind',T.IntegerType(), True),\
                       T.StructField('altitude',T.FloatType(), True),\
                       T.StructField('longitude',T.FloatType(), True),\
                       T.StructField('latitude',T.FloatType(), True),\
                       T.StructField('aspect',T.FloatType(), True),\
                       T.StructField('slope',T.FloatType(), True),\
                       T.StructField('sol_az',T.FloatType(), True),\
                       T.StructField('sol_zn',T.FloatType(), True),\
                       T.StructField('fo_az',T.FloatType(), True),\
                       T.StructField('fo_zn',T.FloatType(), True),\
                       T.StructField('flag',T.IntegerType(), True),\
                       T.StructField('snr',T.FloatType(), True),\
                       T.StructField('continuum',T.FloatType(), True),\
                       T.StructField('time_str',T.StringType(), True),\
                       T.StructField('spectrum',T.ArrayType(T.DoubleType(),True), True)\
                          ])
        
    numdata, numpix, numchannel = h5f['FootprintGeometry/footprint_altitude'].shape
    
    filename_list = [infile for i in range(numdata*numpix*numchannel)] 
    
    channel_ind_list =[]
    pixel_ind_list =[]
    row_ind_list =[]
    flag_list =[]
    snr_list = []
    cont_list= []
    time_list=[]
    times =[]

    ####################### generate indices
    for k in range(numdata):
        for j in range(numpix):
            for i in range(numchannel):
                channel_ind_list.append(i)
                pixel_ind_list.append(j)
                row_ind_list.append(k)
        
    ## General info 
    alti_list = h5f['FootprintGeometry/footprint_altitude'][:,:,:].flatten().tolist()
    lon_list = h5f['FootprintGeometry/footprint_longitude'][:,:,:].flatten().tolist()
    lat_list = h5f['FootprintGeometry/footprint_latitude'][:,:,:].flatten().tolist()
    
    aspect_list = h5f['FootprintGeometry/footprint_aspect'][:,:,:].flatten().tolist()
    slope_list  = h5f['FootprintGeometry/footprint_slope'][:,:,:].flatten().tolist()
    surf_list   = h5f['FootprintGeometry/footprint_surface_roughness'][:,:,:].flatten().tolist()
    
    sol_az_list = h5f['FootprintGeometry/footprint_solar_azimuth'][:,:,:].flatten().tolist()
    sol_zn_list = h5f['FootprintGeometry/footprint_solar_zenith'][:,:,:].flatten().tolist()
    fo_az_list  = h5f['FootprintGeometry/footprint_azimuth'][:,:,:].flatten().tolist()
    fo_zn_list  = h5f['FootprintGeometry/footprint_zenith'][:,:,:].flatten().tolist()
    times       = h5f['FootprintGeometry/footprint_time_string'][:,:,:].flatten().tolist()
    time_list   = [time.decode() for time in times]
    

    ## merge info
    o2_flg = h5f['FootprintGeometry/footprint_o2_qual_flag'][:,:].flatten().tolist()
    snr_o2 = h5f['/SoundingMeasurements/snr_o2_l1b'][:,:].flatten().tolist()
    o2_cont= h5f['/SoundingMeasurements/rad_continuum_o2'][:,:].flatten().tolist()
    flag_list.append(o2_flg)
    snr_list.append(snr_o2)
    cont_list.append(o2_cont)
    
    wco2_flg = h5f['FootprintGeometry/footprint_weak_co2_qual_flag'][:,:].flatten().tolist()
    snr_wco2 = h5f['/SoundingMeasurements/snr_weak_co2_l1b'][:,:].flatten().tolist()
    wco2_cont= h5f['/SoundingMeasurements/rad_continuum_weak_co2'][:,:].flatten().tolist()
    flag_list.append(wco2_flg)
    snr_list.append(snr_wco2)
    cont_list.append(wco2_cont)
    
    sco2_flg = h5f['FootprintGeometry/footprint_strong_co2_qual_flag'][:,:].flatten().tolist()
    snr_sco2 = h5f['/SoundingMeasurements/snr_strong_co2_l1b'][:,:].flatten().tolist()
    sco2_cont= h5f['/SoundingMeasurements/rad_continuum_strong_co2'][:,:].flatten().tolist()
    flag_list.append(sco2_flg)
    snr_list.append(snr_sco2)
    cont_list.append(sco2_cont)
    
    flag_list =reduce(operator.concat, flag_list)
    snr_list =reduce(operator.concat, snr_list)
    cont_list =reduce(operator.concat, cont_list)


    ## Spectra info
    o2   = h5f['/SoundingMeasurements/radiance_o2'][:,:,:].reshape(-1,1016).tolist()
    wco2 = h5f['/SoundingMeasurements/radiance_weak_co2'][:,:,:].reshape(-1,1016).tolist()
    sco2 = h5f['/SoundingMeasurements/radiance_strong_co2'][:,:,:].reshape(-1,1016).tolist()
    
    temp_spec = list(map(list, zip(o2, wco2, sco2)))
    all_spec_list = [x for sublist in temp_spec for x in sublist]
    
    
    
    ################ Save to a parquet 
    sparkdf = spark.createDataFrame(zip(filename_list,channel_ind_list,pixel_ind_list,row_ind_list,\
                                    alti_list,lon_list,lat_list,aspect_list,slope_list, \
                                    sol_az_list,sol_zn_list,fo_az_list,fo_zn_list, \
                                    flag_list,snr_list,cont_list,time_list,all_spec_list),schema)
    
    sparkdf.write.option("compression","snappy").mode("overwrite").save(outfile)
    #sparkdf.write.parquet("s3a://jwpark-spark/bigdata_result/"+outfile,mode="overwrite")
    
    sc.stop()