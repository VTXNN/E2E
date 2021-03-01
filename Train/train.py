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

import gc


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
#files = files[:10]


fTest = 0.2
iTestSplit = int(round(fTest*(len(files)-2)))+1
filesTrain = files[iTestSplit:]
filesTest = files[:iTestSplit]
print ("Input train/test files: %i/%i"%(len(filesTrain),len(filesTest)))

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
    ds = ds.shuffle(len(fileList),reshuffle_each_iteration=True)
    ds = ds.interleave(
        lambda x: tf.data.TFRecordDataset(
            x, compression_type='GZIP', buffer_size=100000000
        ),
        cycle_length=6, 
        block_length=50, 
        num_parallel_calls=6
    )
    ds = ds.batch(50) #decode in batches (match block_length?)
    ds = ds.map(decode_data, num_parallel_calls=6)
    ds = ds.unbatch()
    ds = ds.shuffle(10000,reshuffle_each_iteration=True)
    ds = ds.batch(1000)
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
        tf.keras.losses.BinaryCrossentropy(from_logits=True),
        lambda y,x: 0.
    ],
    metrics=[
        tf.keras.metrics.BinaryAccuracy(threshold=0.,name='assoc_acc') #use thres=0 here since logits are used
    ],
    loss_weights=[1.,1.,0.]
)
model.summary()

trainPipeline = setup_pipeline(filesTrain)
testPipeline = setup_pipeline(filesTest)

for epoch in range(50):
    lr = 0.001/(1+0.1*max(0,epoch-10)**1.5)
    tf.keras.backend.set_value(model.optimizer.learning_rate, lr)
    
    print ("Epoch %i"%epoch)
    
    if epoch>0:
        model.load_weights("weights_%i.hdf5"%(epoch-1))
    
    lossTrainTotal = 0.
    lossTrainPos = 0.
    lossTrainAssoc = 0.
    for stepTrain,batch in enumerate(trainPipeline):

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
        nBatch = batch['pvz0'].shape[0]
        result = model.train_on_batch(
            [batch['trk_z0'],trackFeatures],
            [batch['pvz0'],batch['trk_fromPV'],np.zeros((nBatch,250,1))]
        )
        result = dict(zip(model.metrics_names,result))
        
        lossTrainTotal += result['loss']
        lossTrainPos += result['position_final_loss']
        lossTrainAssoc += result['association_final_loss']
        
        if stepTrain%10==0:
            predictedZ0_FH = predictFastHisto(batch['trk_z0'],batch['trk_pt'])
        
            predictedZ0_NN, predictedAssoc_NN, predictedWeights_NN = model.predict_on_batch(
                [batch['trk_z0'],trackFeatures]
            )
            qz0_NN = np.percentile(predictedZ0_NN-batch['pvz0'],[5,15,50,85,95])
            qz0_FH = np.percentile(predictedZ0_FH-batch['pvz0'],[5,15,50,85,95])
            #print (qz0_NN,qz0_FH)
            print ("Train step %02i-%02i: loss=%.3f (z0=%.3f, assoc=%.3f), q68=(%.4f,%.4f), [FH: q68=(%.4f,%.4f)]"%(
                epoch,stepTrain,
                result['loss'],result['position_final_loss'],result['association_final_loss'],
                qz0_NN[1],qz0_NN[3],qz0_FH[1],qz0_FH[3]
            ))
            gc.collect()
            
    model.save_weights("weights_%i.hdf5"%(epoch))
    '''
    z0DiffFH = []
    z0DiffNN = []
    weightPtHist = np.zeros((50,50))
    weightEtaHist = np.zeros((50,50))
    weightChi2rphiHist = np.zeros((50,50))
    weightChi2rzHist = np.zeros((50,50))
    weightBendChi2Hist = np.zeros((50,50))
    weightNstubHist = np.zeros((50,50))
    '''
    
    
    lossTestTotal = 0.
    lossTestPos = 0.
    lossTestAssoc = 0.
    for stepTest,batch in enumerate(testPipeline):
        trackFeatures = np.stack([batch[feature] for feature in [
            'trk_pt','trk_eta', 'trk_chi2rphi', 'trk_chi2rz', 'trk_bendchi2','trk_nstub'
        ]],axis=2)
        trackFeatures = np.concatenate([trackFeatures,batch['trk_hitpattern']],axis=2)
        nBatch = batch['pvz0'].shape[0]
        result = model.test_on_batch(
            [batch['trk_z0'],trackFeatures],
            [batch['pvz0'],batch['trk_fromPV'],np.zeros((nBatch,250,1))]
        )
        result = dict(zip(model.metrics_names,result))
        
        lossTestTotal += result['loss']
        lossTestPos += result['position_final_loss']
        lossTestAssoc += result['association_final_loss']
        
        predictedZ0_FH = predictFastHisto(batch['trk_z0'],batch['trk_pt'])
        #z0DiffFH.append(predictedZ0_FH-batch['pvz0'])
        
        predictedZ0_NN, predictedAssoc_NN, predictedWeights_NN = model.predict_on_batch(
            [batch['trk_z0'],trackFeatures]
        )
        #z0DiffNN.append(predictedZ0_NN-batch['pvz0'])
        
        if stepTest%10==0:
            
            qz0_NN = np.percentile(predictedZ0_NN-batch['pvz0'],[5,15,50,85,95])
            qz0_FH = np.percentile(predictedZ0_FH-batch['pvz0'],[5,15,50,85,95])
            #print (qz0_NN,qz0_FH)
            print ("Test step %02i-%02i: loss=%.3f (z0=%.3f, assoc=%.3f), q68=(%.4f,%.4f), [FH: q68=(%.4f,%.4f)]"%(
                epoch,stepTest,
                result['loss'],result['position_final_loss'],result['association_final_loss'],
                qz0_NN[1],qz0_NN[3],qz0_FH[1],qz0_FH[3]
            ))
            gc.collect()

    fstat = open("stat.dat","a")
    fstat.write("%i; %.3e; %.3e;%.3e;%.3e; %.3e;%.3e;%.3e\n"%(
        epoch,
        lr,
        lossTrainTotal/stepTrain,
        lossTrainPos/stepTrain,
        lossTrainAssoc/stepTrain,
        
        lossTestTotal/stepTest,
        lossTestPos/stepTest,
        lossTestAssoc/stepTest,
    ))
    fstat.close()
    
