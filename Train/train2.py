import tensorflow as tf
import numpy as np
import scipy
import h5py
import os
import sys
import glob
import math
import re
import csv
import sklearn.metrics


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SMALL_SIZE = 15
MEDIUM_SIZE = 15
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


files = glob.glob("/vols/cms/mkomm/VTX/E2E/Train/trainData/*.tfrecord")

print ("Input files: ",len(files))

features = {
    "pvz0": tf.io.FixedLenFeature([1], tf.float32),
    "pv2z0": tf.io.FixedLenFeature([1], tf.float32),
    "hitPattern": tf.io.FixedLenFeature([200*11], tf.float32),
}

trackFeatures = [
    'trk_z0',
    'trk_pt',
    'trk_eta', 
    'trk_chi2rphi', 
    'trk_chi2rz', 
    'trk_bendchi2',
    'trk_nstub', 
]

for trackFeature in trackFeatures:
    features[trackFeature] = tf.io.FixedLenFeature([200], tf.float32)

    
def decode_data(raw_data):
    decoded_data = tf.io.parse_example(raw_data,features)
    decoded_data['hitPattern'] = tf.reshape(decoded_data['hitPattern'],[-1,200,11])
    return decoded_data

def setup_pipeline(fileList):
    ds = tf.data.Dataset.from_tensor_slices(fileList)
    ds.shuffle(len(fileList),reshuffle_each_iteration=True)
    ds = ds.interleave(
        lambda x: tf.data.TFRecordDataset(
            x, compression_type='GZIP', buffer_size=100000000
        ),
        cycle_length=6, 
        block_length=250, 
        num_parallel_calls=6
    )
    ds = ds.batch(250) #decode in batches (match block_length?)
    ds = ds.map(decode_data, num_parallel_calls=6)
    ds = ds.unbatch()
    ds = ds.shuffle(50000,reshuffle_each_iteration=True)
    ds = ds.batch(10000)
    ds = ds.prefetch(5)
    
    return ds
    
    
for batch in setup_pipeline(files[:5]):
    pass
    
    
