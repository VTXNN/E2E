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
import vtx


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
    "trk_fromPV":tf.io.FixedLenFeature([250], tf.float32),
    "trk_hitpattern": tf.io.FixedLenFeature([250*11], tf.float32),
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
    features[trackFeature] = tf.io.FixedLenFeature([250], tf.float32)

    
def predictFastHisto(value,weight):
    z0List = []
    halfBinWidth = 0.5*30./256.
    for ibatch in range(value.shape[0]):
        hist,bin_edges = np.histogram(value[ibatch],256,range=(-15,15),weights=weight[ibatch])
        hist = np.convolve(hist,[1,1,1],mode='same')
        z0Index= np.argmax(hist)
        z0 = -15.+30.*z0Index/256.+halfBinWidth
        z0List.append([z0])
    return np.array(z0List,dtype=np.float32)
    
def decode_data(raw_data):
    decoded_data = tf.io.parse_example(raw_data,features)
    decoded_data['trk_hitpattern'] = tf.reshape(decoded_data['trk_hitpattern'],[-1,250,11])
    return decoded_data

def setup_pipeline(fileList):
    ds = tf.data.Dataset.from_tensor_slices(fileList)
    ds.shuffle(len(fileList),reshuffle_each_iteration=True)
    ds = ds.interleave(
        lambda x: tf.data.TFRecordDataset(
            x, compression_type='GZIP', buffer_size=100000000
        ),
        cycle_length=6, 
        block_length=200, 
        num_parallel_calls=6
    )
    ds = ds.batch(200) #decode in batches (match block_length?)
    ds = ds.map(decode_data, num_parallel_calls=6)
    ds = ds.unbatch()
    ds = ds.shuffle(5000,reshuffle_each_iteration=True)
    ds = ds.batch(2000)
    ds = ds.prefetch(5)
    
    return ds
    
network = vtx.nn.E2ERef(
    nbins=256,
    ntracks=250, 
    nfeatures=17, 
    nweights=1, 
    nlatent=0, 
    activation='relu',
    regloss=1e-10
)

model = network.createE2EModel()
optimizer = tf.keras.optimizers.Adam(lr=0.001)
model.compile(
    optimizer,
    loss=[
        tf.keras.losses.MeanAbsoluteError(),
        tf.keras.losses.BinaryCrossentropy(from_logits=True)
    ],
    metrics=[
        tf.keras.metrics.BinaryAccuracy(threshold=0.,name='assoc_acc') #use thres=0 here since logits are used
    ]
)
model.summary()

for epoch in range(50):
    lr = 0.001/(1+0.1*max(0,epoch-10)**1.5)
    tf.keras.backend.set_value(model.optimizer.learning_rate, lr)
    
    print ("Epoch %i"%epoch)
    
    if epoch>0:
        model.load_weights("weights_%i.tf"%(epoch-1))
    
    for step,batch in enumerate(setup_pipeline(files)):
        '''
        #random z0 shift and flip
        z0Shift = np.random.normal(0.0,1.0,size=batch['pvz0'].shape)
        z0Flip = 2.*np.random.randint(2,size=batch['pvz0'].shape)-1.
        batch['trk_z0']=batch['trk_z0']*z0Flip+z0Shift
        batch['pvz0']=batch['pvz0']*z0Flip+z0Shift
        batch['pv2z0']=batch['pv2z0']*z0Flip+z0Shift
        '''
        trackFeatures = np.stack([batch[feature] for feature in [
            'trk_pt','trk_eta', 'trk_chi2rphi', 'trk_chi2rz', 'trk_bendchi2','trk_nstub'
        ]],axis=2)
        trackFeatures = np.concatenate([trackFeatures,batch['trk_hitpattern']],axis=2)
        result = model.train_on_batch(
            [batch['trk_z0'],trackFeatures],
            [batch['pvz0'],batch['trk_fromPV']]
        )
        result = dict(zip(model.metrics_names,result))
        
        if step%10==0:
            predictedZ0_FH = predictFastHisto(batch['trk_z0'],batch['trk_pt'])
        
            predictedZ0_NN, predictedAssoc_NN = model.predict_on_batch(
                [batch['trk_z0'],trackFeatures]
            )
            qz0_NN = np.percentile(predictedZ0_NN-batch['pvz0'],[5,15,50,85,95])
            qz0_FH = np.percentile(predictedZ0_FH-batch['pvz0'],[5,15,50,85,95])
            #print (qz0_NN,qz0_FH)
            print ("Step %02i-%02i: loss=%.3f (z0=%.3f, assoc=%.3f), q68=(%.4f,%.4f), [FH: q68=(%.4f,%.4f)]"%(
                epoch,step,
                result['loss'],result['position_final_loss'],result['association_final_loss'],
                qz0_NN[1],qz0_NN[3],qz0_FH[1],qz0_FH[3]
            ))
    model.save_weights("weights_%i.tf"%(epoch))

                

        
