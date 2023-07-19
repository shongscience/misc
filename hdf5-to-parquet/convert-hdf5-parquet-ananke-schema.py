#!/usr/bin/python3
"""
Created on July 2023

@author: shong
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy import stats
import gc

from pyspark import SparkContext   
from pyspark.sql import SparkSession

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark import Row

from functools import partial

import pickle

from igraph import *

from ast import literal_eval

import h5py
import pyarrow as pa
import pyarrow.parquet as pq




if __name__ == '__main__':
    # Execution Arguments 
    if len(sys.argv) != 3:
        print("Usage : <command> <infile_name_csv> <outfile_name_parquet_path> ")
        sys.exit()
    else:
        print("=== Parsing the input argv")
        for idx, tmparg in enumerate(sys.argv):
            print("argc = %d, argv[%d] = %s" % (idx,idx,tmparg))


    # Define Spark Session
    print("Program Log : Defining SparkSession...")
    
    spark = SparkSession.builder \
    .master("yarn") \
    .appName("ananke-hdf5-parquet") \
    .config("spark.driver.maxResultSize", "32g") \
    .config("spark.driver.memory", "64g") \
    .config("spark.executor.memory", "7g") \
    .config("spark.executor.cores", "1") \
    .config("spark.executor.instances", "50") \
    .getOrCreate()
    
    sc = spark.sparkContext
    sc.setCheckpointDir("hdfs://spark00:54310/tmp/checkpoints")
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
           
    
    # Read infile
    input_schema = \
    T.StructType([ \
T.StructField('A0', T.DoubleType(), True), \
T.StructField('a_g_val', T.DoubleType(), True), \
T.StructField('age', T.DoubleType(), True), \
T.StructField('alpha', T.DoubleType(), True), \
T.StructField('b', T.DoubleType(), True), \
T.StructField('b_true', T.FloatType(), True), \
T.StructField('bp_g', T.DoubleType(), True), \
T.StructField('bp_g_true', T.DoubleType(), True), \
T.StructField('bp_rp', T.DoubleType(), True), \
T.StructField('bp_rp_true', T.DoubleType(), True), \
T.StructField('calcium', T.DoubleType(), True), \
T.StructField('carbon', T.DoubleType(), True), \
T.StructField('dec', T.DoubleType(), True), \
T.StructField('dec_error', T.DoubleType(), True), \
T.StructField('dec_true', T.FloatType(), True), \
T.StructField('dmod_true', T.DoubleType(), True), \
T.StructField('e_bp_min_rp_val', T.DoubleType(), True), \
T.StructField('ebv', T.DoubleType(), True), \
T.StructField('feh', T.DoubleType(), True), \
T.StructField('flag_wd', T.LongType(), True), \
T.StructField('g_rp', T.DoubleType(), True), \
T.StructField('g_rp_true', T.DoubleType(), True), \
T.StructField('helium', T.DoubleType(), True), \
T.StructField('l', T.DoubleType(), True), \
T.StructField('l_true', T.FloatType(), True), \
T.StructField('logg', T.FloatType(), True), \
T.StructField('lognh', T.DoubleType(), True), \
T.StructField('logteff', T.FloatType(), True), \
T.StructField('lum', T.FloatType(), True), \
T.StructField('mact', T.FloatType(), True), \
T.StructField('magnesium', T.DoubleType(), True), \
T.StructField('mini', T.DoubleType(), True), \
T.StructField('mtip', T.FloatType(), True), \
T.StructField('neon', T.DoubleType(), True), \
T.StructField('nitrogen', T.DoubleType(), True), \
T.StructField('oxygen', T.DoubleType(), True), \
T.StructField('parallax', T.DoubleType(), True), \
T.StructField('parallax_error', T.DoubleType(), True), \
T.StructField('parallax_over_error', T.DoubleType(), True), \
T.StructField('parallax_true', T.DoubleType(), True), \
T.StructField('parentid', T.LongType(), True), \
T.StructField('partid', T.LongType(), True), \
T.StructField('phot_bp_mean_mag', T.DoubleType(), True), \
T.StructField('phot_bp_mean_mag_abs', T.FloatType(), True), \
T.StructField('phot_bp_mean_mag_error', T.DoubleType(), True), \
T.StructField('phot_bp_mean_mag_int', T.DoubleType(), True), \
T.StructField('phot_bp_mean_mag_true', T.DoubleType(), True), \
T.StructField('phot_g_mean_mag', T.DoubleType(), True), \
T.StructField('phot_g_mean_mag_abs', T.FloatType(), True), \
T.StructField('phot_g_mean_mag_error', T.DoubleType(), True), \
T.StructField('phot_g_mean_mag_int', T.DoubleType(), True), \
T.StructField('phot_g_mean_mag_true', T.DoubleType(), True), \
T.StructField('phot_rp_mean_mag', T.DoubleType(), True), \
T.StructField('phot_rp_mean_mag_abs', T.FloatType(), True), \
T.StructField('phot_rp_mean_mag_error', T.DoubleType(), True), \
T.StructField('phot_rp_mean_mag_int', T.DoubleType(), True), \
T.StructField('phot_rp_mean_mag_true', T.DoubleType(), True), \
T.StructField('pmb', T.DoubleType(), True), \
T.StructField('pmb_true', T.DoubleType(), True), \
T.StructField('pmdec', T.DoubleType(), True), \
T.StructField('pmdec_error', T.DoubleType(), True), \
T.StructField('pmdec_true', T.DoubleType(), True), \
T.StructField('pml', T.DoubleType(), True), \
T.StructField('pml_true', T.DoubleType(), True), \
T.StructField('pmra', T.DoubleType(), True), \
T.StructField('pmra_error', T.DoubleType(), True), \
T.StructField('pmra_true', T.DoubleType(), True), \
T.StructField('px_true', T.DoubleType(), True), \
T.StructField('py_true', T.DoubleType(), True), \
T.StructField('pz_true', T.DoubleType(), True), \
T.StructField('ra', T.DoubleType(), True), \
T.StructField('ra_cosdec_error', T.DoubleType(), True), \
T.StructField('ra_error', T.DoubleType(), True), \
T.StructField('ra_true', T.FloatType(), True), \
T.StructField('radial_velocity', T.DoubleType(), True), \
T.StructField('radial_velocity_error', T.DoubleType(), True), \
T.StructField('radial_velocity_error_corr_factor', T.DoubleType(), True), \
T.StructField('radial_velocity_true', T.DoubleType(), True), \
T.StructField('silicon', T.DoubleType(), True), \
T.StructField('sulphur', T.DoubleType(), True), \
T.StructField('vx_true', T.DoubleType(), True), \
T.StructField('vy_true', T.DoubleType(), True), \
T.StructField('vz_true', T.DoubleType(), True) \
                 ])                   
    
    # Basic variables
    h5name = sys.argv[1]
    outpqnameheader = sys.argv[2]
    outpqnametail = '.parquet.snappy'
    
    
    # Read HDF5
    try:
        f = h5py.File(h5name, "r")
    except IOError as e:
        print("Error opening HDF5 file:", str(e))
    # Don't forget f.close() when done! 
    
    # Extracting HDF5 Info
    keylist = list(f.keys())
    keyzero = keylist[0]
    numtotal = f[keyzero].shape[0]
    
    # Converting Variables 
    chunksize = 1000000
    ichunk = 0
    istart=0
    iend=0
    
    ########## Let's Loop Up #########
    # initalize indices 
    ichunk = 0
    istart = ichunk * chunksize
    iend = (ichunk + 1) * chunksize
    
    # Mark Chunk points (or, addresses)
    # ad-hoc iterations; i know this is not a fancy loop
    chunk_address = []
    
    while istart < numtotal:
        if iend > numtotal:
            iend = numtotal
        
        print("ichunk="+str(ichunk)+", istart="+str(istart)+", iend="+str(iend)+" : Total="+str(numtotal))
        chunk_address.append([ichunk,istart,iend])
        
        ichunk=ichunk+1
        istart = ichunk * chunksize
        iend = (ichunk + 1) * chunksize

    # Save parquet parts on hdfs 
    for ichunk, istart, iend in chunk_address:
        print("ichunk="+str(ichunk)+", istart="+str(istart)+", iend="+str(iend)+" : Total="+str(numtotal))
        print(outpqnameheader+"-part"+str(ichunk)+outpqnametail)
        outname = outpqnameheader+"-part"+str(ichunk)+outpqnametail
        
        chunk_data_list = [f[key][()][istart:iend].tolist() for key in f.keys()]
        zipped_chunk_data_list = list(zip(*chunk_data_list))
        
        spark.createDataFrame(zipped_chunk_data_list,schema=input_schema) \
        .write.option("compression", "snappy") \
        .mode("overwrite") \
        .save(outname)
        
    f.close()
    sc.stop()