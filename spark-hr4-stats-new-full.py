#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 13:27:01 2018

Jackknife Resampling


@author: shong
"""

import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import gc


from pyspark import SparkContext   
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark import Row

from functools import partial

import pickle


from graphframes import *


def getNeighbors(px,py,pz,curlinklen, curtree, curgalid):
    lengcut = curlinklen.value
    
    neighborlists = []
    inearest=2 # this is the first nearest neighbor in kdtree 
    
    dummy, inow = curtree.value.query([px,py,pz],k=[inearest])
    
    while (dummy[0] <= lengcut):
            
            neighborlists.append(curgalid.value.loc[inow[0]].item())
            inearest = inearest + 1 # the next nearest one 
            dummy, inow = curtree.value.query([px,py,pz],k=[inearest])
    
    return neighborlists


if __name__ == '__main__':
    
    # Execution Arguments 
    if len(sys.argv) != 1:
        print "Usage : <command> "
        sys.exit()
    else:
        print "=== Parsing the input argv"
        for idx, tmparg in enumerate(sys.argv):
            print "argc = %d, argv[%d] = %s" % (idx,idx,tmparg)

    
    # Define Spark Session
    print "Program Log : Defining SparkSession..."
    
    #spark = SparkSession.builder.appName("largeScaleGstat").getOrCreate()
    spark = SparkSession.builder.appName("largeScaleGstat")\
    .config("spark.driver.maxResultSize","8g")\
    .config("spark.sql.execution.arrow.enabled","true")\
    .config("spark.executor.memoryOverhead","42GB")\
    .getOrCreate()
    
    sc = spark.sparkContext
    sqlsc = SQLContext(sc)
    #spark.sparkContext.setCheckpointDir("hdfs://master:54310/tmp/spark/checkpoints")
    #sc.setCheckpointDir("hdfs://master:54310/tmp/spark/checkpoints")
    sc.setCheckpointDir('gs://shongdata/tmp')
    
    #print "Default Pickle Protocol = ", pickle.DEFAULT_PROTOCOL
    #pickle.DEFAULT_PROTOCOL = 2
    #print "After the change, Default Pickle Protocol = ", pickle.DEFAULT_PROTOCOL

    
    # Read infile
    halo_schema =T\
    .StructType([T.StructField('haloid', T.IntegerType(), False),\
                 T.StructField('px', T.FloatType(), False),\
                 T.StructField('py', T.FloatType(), False),\
                 T.StructField('pz', T.FloatType(), False),\
                 T.StructField('halomass', T.FloatType(), False)])


    ## Linking Length
    linklen = 3.4    
    bcastLinkLen = sc.broadcast(linklen)
    

    ## Let's skip all INFO logs
    sc.setLogLevel("ERROR")
    



    print "Running...  Reading CSV... "

    #csvname =\
    #"gs://shongdata/data/sub0.csv"

    csvname =\
    "gs://shongdata/data/hr4z0.csv"
                        
    halodf = sqlsc.read.csv(csvname,header=True, schema = halo_schema)
    # rename `haloid` to `id` 
    #halodf = halodf.filter(halodf['halomass'] > 5.0E11)
    
    halodf = halodf.filter(halodf['halomass'] > 5.0E11)
    halodf = halodf.withColumnRenamed('haloid','id')
    halodf.cache()
    rentot = np.double(halodf.count()) # result
    print "Total Nodes = ",rentot
    halodf.describe().show()


    print "Running... Creating pandas dataframes for cKDtree"
    ### generating scipy KDtree
    hpdf = halodf.select('px','py','pz').toPandas()
    iddf = halodf.select('id').toPandas()
    
    print "Sizes of positionDF : ",sys.getsizeof(hpdf)
    print "Sizes of idDF : ",sys.getsizeof(iddf)
    
    print hpdf.head()
    #cols = ['px','py','pz']
    #hpdf[cols] = hpdf[cols].applymap(np.float16)
    #print "Reformatting the posotionDF using float16" 
    #print "Sizes of positionDF : ",sys.getsizeof(hpdf)
    #print hpdf.head()
    
    
    
    ## Test google storage
    #print "Saving a test file on GS..."
    #hpdf.to_parquet('/home/shongscience/test.parquet.snappy', compression='snappy')
    
    print "Running... Generating cKDtree"
    hptree = cKDTree(hpdf[['px','py','pz']])
    print "Sizes of cKDtree : ",sys.getsizeof(hptree)
    
    gc.collect()
    
    
    
    
    print "Running... Broadcasting variables..."
    ## Broadcast variables 
    bcastTree = sc.broadcast(hptree)
    bcastID = sc.broadcast(iddf)

                

    print "Running... Applying UDF..."
    getNeighborUDF = \
    F.udf(partial(getNeighbors,\
                  curlinklen=bcastLinkLen,\
                  curtree = bcastTree,\
                  curgalid = bcastID),T.ArrayType(T.IntegerType()))
        
    neighbordf = halodf.withColumn('neighbors',\
                                   getNeighborUDF('px','py','pz'))
                
    edgelist = neighbordf.select('id',F.explode('neighbors').alias('dst'))
    edgedf = edgelist.select(F.col("id").alias("src"),"dst")
    edgedf.cache()
    reedgetot = np.double(edgedf.count()) # result
    print "Total Edges = ",reedgetot
    
        
    g = GraphFrame(halodf,edgedf)
    resultdf = g.triangleCount().join(g.inDegrees, "id")
    tmp = g.connectedComponents().select("id","component")
    finalresult = resultdf.join(tmp,"id")
                
    finalresult.cache()

    print "Running...  Collecting data to toPandas()... "                
    
    resultpd = \
    finalresult\
    .select('id','count','px','py','pz','halomass','inDegree','component')\
    .toPandas()
        
    # rename the triangle counts `count` to `tcount` to remove confusions 
    resultpd.columns = ['id','tcount','px','py','pz','halomass','inDegree','component']
                
    #save pandas/dataframe to parquet 
    resultpd.to_parquet('/home/shongscience/hr4-full-result.parquet.snappy', compression='snappy')
    #resultpd.to_parquet('/home/shongscience/hr4-sub0-result.parquet.snappy', compression='snappy')
    
    
    #print resultpd.describe()
    realpha = np.double(resultpd['inDegree'].sum())
    ren3xtri = np.double(resultpd['tcount'].sum())
    renvee = np.double(resultpd['inDegree'].apply(lambda x: np.double(x*(x-1))).sum()/2.0)
                
    pscomp = resultpd["component"].value_counts()
    regcomp = np.double(pscomp.values[0])
    ren2comp = np.double(len(pscomp[pscomp == 2]))
    ren3comp = np.double(len(pscomp[pscomp == 3]))
    ren4comp = np.double(len(pscomp[pscomp == 4]))
    ren5compplus = np.double(len(pscomp[pscomp >= 5]))

    print "ntot, gcomp, edgetot, alpha, nvee, n3xtri, n2comp, n3comp, n4comp, n5compplus : ",\
    rentot," ",regcomp," ",reedgetot," ",realpha," ",\
    renvee," ",ren3xtri," ",ren2comp," ",\
    ren3comp," ",ren4comp," ",ren5compplus

                
    # Pickle the output 
    outname = '/home/shongscience/hr4-full-stat-new-3150.pickle'
    #outname = '/home/shongscience/hr4-sub0-stat.pickle'
    print "Saving the results as ",outname
    with open(outname,'wb') as f:
        pickle.dump([rentot,regcomp,reedgetot,realpha,renvee,ren3xtri,\
                     ren2comp,ren3comp,ren4comp,ren5compplus],f)
    f.close() 
    ## end of the for loop
    
    sc.stop()
