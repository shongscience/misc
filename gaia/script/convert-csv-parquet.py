#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Aug 2020

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
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark import Row

from functools import partial

import pickle

from igraph import *

from ast import literal_eval


import pyarrow as pa
import pyarrow.parquet as pq




if __name__ == '__main__':
    # Execution Arguments 
    if len(sys.argv) != 3:
        print("Usage : <command> <infile_name_csv> <outfile_name_parquet> ")
        sys.exit()
    else:
        print("=== Parsing the input argv")
        for idx, tmparg in enumerate(sys.argv):
            print("argc = %d, argv[%d] = %s" % (idx,idx,tmparg))


    # Define Spark Session
    print("Program Log : Defining SparkSession...")
    
#    spark = SparkSession.builder.appName("convert-csv-parquet") \
#                        .config("spark.driver.maxResultSize","8g") \
#                        .config("spark.sql.execution.arrow.enabled","true") \
#                        .config("spark.executor.memoryOverhead","8GB") \
#                        .getOrCreate()
    spark = SparkSession.builder.appName("convert-csv-parquet") \
                        .config("spark.driver.maxResultSize","8g") \
                        .config("spark.sql.execution.arrow.enabled","true") \
                        .getOrCreate()
    
        
    sc = spark.sparkContext
    sqlsc = SQLContext(sc)
    #spark.sparkContext.setCheckpointDir("hdfs://master:54310/tmp/spark/checkpoints")
    sc.setCheckpointDir("hdfs://spark00:54310/tmp/checkpoints")
           
    
    # Read infile
    gaia_schema = T.StructType([ \
T.StructField('solution_id', T.LongType(), True), \
T.StructField('designation', T.StringType(), True), \
T.StructField('source_id', T.LongType(), True), \
T.StructField('random_index', T.LongType(), True), \
T.StructField('ref_epoch', T.DoubleType(), True), \
T.StructField('ra', T.DoubleType(), True), \
T.StructField('ra_error', T.FloatType(), True), \
T.StructField('dec', T.DoubleType(), True), \
T.StructField('dec_error', T.FloatType(), True), \
T.StructField('parallax', T.DoubleType(), True), \
T.StructField('parallax_error', T.FloatType(), True), \
T.StructField('parallax_over_error', T.FloatType(), True), \
T.StructField('pm', T.FloatType(), True), \
T.StructField('pmra', T.DoubleType(), True), \
T.StructField('pmra_error', T.FloatType(), True), \
T.StructField('pmdec', T.DoubleType(), True), \
T.StructField('pmdec_error', T.FloatType(), True), \
T.StructField('ra_dec_corr', T.FloatType(), True), \
T.StructField('ra_parallax_corr', T.FloatType(), True), \
T.StructField('ra_pmra_corr', T.FloatType(), True), \
T.StructField('ra_pmdec_corr', T.FloatType(), True), \
T.StructField('dec_parallax_corr', T.FloatType(), True), \
T.StructField('dec_pmra_corr', T.FloatType(), True), \
T.StructField('dec_pmdec_corr', T.FloatType(), True), \
T.StructField('parallax_pmra_corr', T.FloatType(), True), \
T.StructField('parallax_pmdec_corr', T.FloatType(), True), \
T.StructField('pmra_pmdec_corr', T.FloatType(), True), \
T.StructField('astrometric_n_obs_al', T.IntegerType(), True), \
T.StructField('astrometric_n_obs_ac', T.IntegerType(), True), \
T.StructField('astrometric_n_good_obs_al', T.IntegerType(), True), \
T.StructField('astrometric_n_bad_obs_al', T.IntegerType(), True), \
T.StructField('astrometric_gof_al', T.FloatType(), True), \
T.StructField('astrometric_chi2_al', T.FloatType(), True), \
T.StructField('astrometric_excess_noise', T.FloatType(), True), \
T.StructField('astrometric_excess_noise_sig', T.FloatType(), True), \
T.StructField('astrometric_params_solved', T.IntegerType(), True), \
T.StructField('astrometric_primary_flag', T.BooleanType(), True), \
T.StructField('nu_eff_used_in_astrometry', T.FloatType(), True), \
T.StructField('pseudocolour', T.FloatType(), True), \
T.StructField('pseudocolour_error', T.FloatType(), True), \
T.StructField('ra_pseudocolour_corr', T.FloatType(), True), \
T.StructField('dec_pseudocolour_corr', T.FloatType(), True), \
T.StructField('parallax_pseudocolour_corr', T.FloatType(), True), \
T.StructField('pmra_pseudocolour_corr', T.FloatType(), True), \
T.StructField('pmdec_pseudocolour_corr', T.FloatType(), True), \
T.StructField('astrometric_matched_transits', T.IntegerType(), True), \
T.StructField('visibility_periods_used', T.IntegerType(), True), \
T.StructField('astrometric_sigma5d_max', T.FloatType(), True), \
T.StructField('matched_transits', T.IntegerType(), True), \
T.StructField('new_matched_transits', T.IntegerType(), True), \
T.StructField('matched_transits_removed', T.IntegerType(), True), \
T.StructField('ipd_gof_harmonic_amplitude', T.FloatType(), True), \
T.StructField('ipd_gof_harmonic_phase', T.FloatType(), True), \
T.StructField('ipd_frac_multi_peak', T.IntegerType(), True), \
T.StructField('ipd_frac_odd_win', T.IntegerType(), True), \
T.StructField('ruwe', T.FloatType(), True), \
T.StructField('scan_direction_strength_k1', T.FloatType(), True), \
T.StructField('scan_direction_strength_k2', T.FloatType(), True), \
T.StructField('scan_direction_strength_k3', T.FloatType(), True), \
T.StructField('scan_direction_strength_k4', T.FloatType(), True), \
T.StructField('scan_direction_mean_k1', T.FloatType(), True), \
T.StructField('scan_direction_mean_k2', T.FloatType(), True), \
T.StructField('scan_direction_mean_k3', T.FloatType(), True), \
T.StructField('scan_direction_mean_k4', T.FloatType(), True), \
T.StructField('duplicated_source', T.BooleanType(), True), \
T.StructField('phot_g_n_obs', T.IntegerType(), True), \
T.StructField('phot_g_mean_flux', T.DoubleType(), True), \
T.StructField('phot_g_mean_flux_error', T.FloatType(), True), \
T.StructField('phot_g_mean_flux_over_error', T.FloatType(), True), \
T.StructField('phot_g_mean_mag', T.FloatType(), True), \
T.StructField('phot_bp_n_obs', T.IntegerType(), True), \
T.StructField('phot_bp_mean_flux', T.DoubleType(), True), \
T.StructField('phot_bp_mean_flux_error', T.FloatType(), True), \
T.StructField('phot_bp_mean_flux_over_error', T.FloatType(), True), \
T.StructField('phot_bp_mean_mag', T.FloatType(), True), \
T.StructField('phot_rp_n_obs', T.IntegerType(), True), \
T.StructField('phot_rp_mean_flux', T.DoubleType(), True), \
T.StructField('phot_rp_mean_flux_error', T.FloatType(), True), \
T.StructField('phot_rp_mean_flux_over_error', T.FloatType(), True), \
T.StructField('phot_rp_mean_mag', T.FloatType(), True), \
T.StructField('phot_bp_rp_excess_factor', T.FloatType(), True), \
T.StructField('phot_bp_n_contaminated_transits', T.IntegerType(), True), \
T.StructField('phot_bp_n_blended_transits', T.IntegerType(), True), \
T.StructField('phot_rp_n_contaminated_transits', T.IntegerType(), True), \
T.StructField('phot_rp_n_blended_transits', T.IntegerType(), True), \
T.StructField('phot_proc_mode', T.IntegerType(), True), \
T.StructField('bp_rp', T.FloatType(), True), \
T.StructField('bp_g', T.FloatType(), True), \
T.StructField('g_rp', T.FloatType(), True), \
T.StructField('radial_velocity', T.FloatType(), True), \
T.StructField('radial_velocity_error', T.FloatType(), True), \
T.StructField('rv_method_used', T.IntegerType(), True), \
T.StructField('rv_nb_transits', T.IntegerType(), True), \
T.StructField('rv_nb_deblended_transits', T.IntegerType(), True), \
T.StructField('rv_visibility_periods_used', T.IntegerType(), True), \
T.StructField('rv_expected_sig_to_noise', T.FloatType(), True), \
T.StructField('rv_renormalised_gof', T.FloatType(), True), \
T.StructField('rv_chisq_pvalue', T.FloatType(), True), \
T.StructField('rv_time_duration', T.FloatType(), True), \
T.StructField('rv_amplitude_robust', T.FloatType(), True), \
T.StructField('rv_template_teff', T.FloatType(), True), \
T.StructField('rv_template_logg', T.FloatType(), True), \
T.StructField('rv_template_fe_h', T.FloatType(), True), \
T.StructField('rv_atm_param_origin', T.IntegerType(), True), \
T.StructField('vbroad', T.FloatType(), True), \
T.StructField('vbroad_error', T.FloatType(), True), \
T.StructField('vbroad_nb_transits', T.IntegerType(), True), \
T.StructField('grvs_mag', T.FloatType(), True), \
T.StructField('grvs_mag_error', T.FloatType(), True), \
T.StructField('grvs_mag_nb_transits', T.IntegerType(), True), \
T.StructField('rvs_spec_sig_to_noise', T.FloatType(), True), \
T.StructField('phot_variable_flag', T.StringType(), True), \
T.StructField('l', T.DoubleType(), True), \
T.StructField('b', T.DoubleType(), True), \
T.StructField('ecl_lon', T.DoubleType(), True), \
T.StructField('ecl_lat', T.DoubleType(), True), \
T.StructField('in_qso_candidates', T.BooleanType(), True), \
T.StructField('in_galaxy_candidates', T.BooleanType(), True), \
T.StructField('non_single_star', T.IntegerType(), True), \
T.StructField('has_xp_continuous', T.BooleanType(), True), \
T.StructField('has_xp_sampled', T.BooleanType(), True), \
T.StructField('has_rvs', T.BooleanType(), True), \
T.StructField('has_epoch_photometry', T.BooleanType(), True), \
T.StructField('has_epoch_rv', T.BooleanType(), True), \
T.StructField('has_mcmc_gspphot', T.BooleanType(), True), \
T.StructField('has_mcmc_msc', T.BooleanType(), True), \
T.StructField('in_andromeda_survey', T.BooleanType(), True), \
T.StructField('classprob_dsc_combmod_quasar', T.FloatType(), True), \
T.StructField('classprob_dsc_combmod_galaxy', T.FloatType(), True), \
T.StructField('classprob_dsc_combmod_star', T.FloatType(), True), \
T.StructField('teff_gspphot', T.FloatType(), True), \
T.StructField('teff_gspphot_lower', T.FloatType(), True), \
T.StructField('teff_gspphot_upper', T.FloatType(), True), \
T.StructField('logg_gspphot', T.FloatType(), True), \
T.StructField('logg_gspphot_lower', T.FloatType(), True), \
T.StructField('logg_gspphot_upper', T.FloatType(), True), \
T.StructField('mh_gspphot', T.FloatType(), True), \
T.StructField('mh_gspphot_lower', T.FloatType(), True), \
T.StructField('mh_gspphot_upper', T.FloatType(), True), \
T.StructField('distance_gspphot', T.FloatType(), True), \
T.StructField('distance_gspphot_lower', T.FloatType(), True), \
T.StructField('distance_gspphot_upper', T.FloatType(), True), \
T.StructField('azero_gspphot', T.FloatType(), True), \
T.StructField('azero_gspphot_lower', T.FloatType(), True), \
T.StructField('azero_gspphot_upper', T.FloatType(), True), \
T.StructField('ag_gspphot', T.FloatType(), True), \
T.StructField('ag_gspphot_lower', T.FloatType(), True), \
T.StructField('ag_gspphot_upper', T.FloatType(), True), \
T.StructField('ebpminrp_gspphot', T.FloatType(), True), \
T.StructField('ebpminrp_gspphot_lower', T.FloatType(), True), \
T.StructField('ebpminrp_gspphot_upper', T.FloatType(), True), \
T.StructField('libname_gspphot', T.StringType(), True),
                           ])    
    
    # Read CSV
    gaiadf = sqlsc.read.csv(sys.argv[1], comment='#', header=True, schema = gaia_schema)


    # Write Parquet
    gaiadf \
    .write.option("compression", "snappy") \
    .mode("overwrite") \
    .save(sys.argv[2]) 
    
    
    sc.stop()