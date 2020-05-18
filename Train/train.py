import tensorflow as tf
import os
import time
import math
import sys
import shutil
import numpy
import keras
import keras.backend as K
import h5py
import logging
import argparse
import random
import inspect
import matplotlib.pyplot as plt
from functools import *

import vtxops
import vtx


parser = argparse.ArgumentParser(description='Training of networks')
parser.add_argument('-i','--input',nargs="+", dest='inputFiles', default=[], action='append',
                    help='Input files',required=True)
parser.add_argument('-o','--output', dest='outputFolder', type=str,
                    help='Output directory', required=True)
parser.add_argument('-f','--force', dest='force', default=False, action='store_true',
                    help='Force override of output directory')
parser.add_argument('-n','--epochs', dest='epochs', default=50, type=int,
                    help='Number of epochs')
parser.add_argument('-b','--batch', dest='batchSize', default=1000, type=int,
                    help='Batch size')
parser.add_argument('--full', dest='trainFull',action='store_true', default=False,
                    help='Train full network (default: only z0)')
parser.add_argument('-t','--testFrac', dest='testFraction', default=0.15, type=float,
                    help='Test fraction.')
parser.add_argument('-v', dest='logLevel', default='Info', type=str,
                    help='Verbosity level: Debug, Info, Warning, Error, Critical')
parser.add_argument('--seed', dest='seed', default=int(time.time()), type=int,
                    help='Random seed')
parser.add_argument('--gpu', dest='gpu', default=False,action="store_true",
                    help='Force GPU usage')
parser.add_argument('--lr', dest='lr', default=0.01,type=float,
                    help='Learning rate')           
parser.add_argument('--kappa', dest='kappa', default=0.9,type=float,
                    help='Learning rate decay')

args = parser.parse_args()

logging.basicConfig(format='%(levelname)s: %(message)s', level=getattr(logging, args.logLevel.upper(), None))



logging.info("Python: %s (%s)"%(sys.version_info,sys.executable))
logging.info("Keras: %s (%s)"%(keras.__version__,os.path.dirname(keras.__file__)))
logging.info("TensorFlow: %s (%s)"%(tf.__version__,os.path.dirname(tf.__file__)))
devices = vtx.Devices(requireGPU=args.gpu)

logging.info("Output folder: %s"%args.outputFolder)
logging.info("Epochs: %i"%args.epochs)
logging.info("Batch size: %i"%args.batchSize)
logging.info("Test fraction: %.2f"%args.testFraction)
logging.info("Train full network: %s"%args.trainFull)
logging.info("Random seed: %i"%args.seed)

random.seed(args.seed)
numpy.random.seed(args.seed)
tf.set_random_seed(args.seed)

inputFiles = [f for fileList in args.inputFiles for f in fileList]

if len(inputFiles)==0:
    logging.critical("No input files specified")
    sys.exit(1)

if os.path.exists(args.outputFolder):
    if not args.force:
        logging.critical("Output folder '%s' already exists. Use --force to override."%(args.outputFolder))
        sys.exit(1)
else:
    logging.info("Creating output folder: "+args.outputFolder)
    os.mkdir(args.outputFolder)
    

inputFiles = vtx.InputFiles(inputFiles)
pipeline = vtx.Pipeline(inputFiles,testFraction=args.testFraction)

#TODO: make each model part a separate model; build the full model by calling them

from vtx.nn import E2ERef as Network
shutil.copyfile(inspect.getsourcefile(Network),os.path.join(args.outputFolder,"Network.py"))

network = Network(
    nbins=256,
    ntracks=250, 
    nfeatures=10, 
    nweights=1, 
    nlatent=0, 
    activation='relu',
    regloss=1e-6
)
'''
inputFeatureLayer = keras.layers.Input(shape=(250,10))
weights = keras.layers.Lambda(lambda x: x[:,1:])(inputFeatureLayer)
for _ in range(2):
    weights = keras.layers.Dense(20,activation='relu')(weights)
    weights = keras.layers.Dropout(0.1)(weights)
weights = keras.layers.Dense(1,activation=None)(weights)

weightModel = keras.models.Model(inputs=[inputFeatureLayer],outputs=[weights])
weightModel.add_loss(tf.reduce_mean(tf.square(weights)))
weightModel.summary()

inputWeightLayer = keras.layers.Input(shape=(250,1))
hists = vtxops.KDELayer()([inputFeatureLayer,inputWeightLayer])

histModel = keras.models.Model(inputs=[inputFeatureLayer,inputWeightLayer],outputs=[hists])
histModel.summary()

positionInput = keras.layers.Input(shape=(256,1))
position = keras.layers.Flatten()(positionInput)
for _ in range(2):
    position = keras.layers.Dense(100,activation='relu')(position)
    position = keras.layers.Dropout(0.1)(position)
position = keras.layers.Dense(1,activation=None)(position)
positionModel = keras.models.Model(inputs=[positionInput],outputs=[position])
positionModel.summary()

weightResult = weightModel([inputFeatureLayer])
histResult = histModel([inputFeatureLayer,weightResult])
positionResult = positionModel([histResult])
model = keras.models.Model(inputs=[inputFeatureLayer],outputs=[positionResult])
model.summary()

optimizer = keras.optimizers.Adam(lr=0.01)
model.compile(optimizer,loss='mae')
#model = network.makeZ0Model(optimizer)
#sys.exit(1)
'''
learning_rate = args.lr

fhAlgo = vtx.FastHisto()

history = {'lr':[],'trainLoss':[],'testLoss':[]}

for epoch in range(args.epochs):
    #distributions = []
    #learning_rate = args.lr/(1+args.kappa*epoch)
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    model = network.createModel(optimizer)
    
    if epoch==0:
        model.summary()
    if epoch>0:
        model.load_weights(os.path.join(args.outputFolder,"weights_%i.hdf5"%(epoch)))

    stepTrain = 0
    totalLossTrain = 0.
    for batch in pipeline.generate(
        batchSize=args.batchSize,
        nFiles=max(3,len(inputFiles)),
        isTraining=True
    ):
        stepTrain+=1
        
        #if stepTrain>10:
        #    break

        

        
        #add random shift for extra regularization
        randomZ0Shift = numpy.random.uniform(-0.5,0.5,size=(batch['X'].shape[0],1,1))
        batch['X']+=numpy.concatenate([
            numpy.repeat(randomZ0Shift,batch['X'].shape[1],axis=1),
            numpy.zeros((batch['X'].shape[0],batch['X'].shape[1],batch['X'].shape[2]-1))
        ],axis=2)
        batch['y']+=randomZ0Shift[:,:,0]
        batch['y_avg']+=randomZ0Shift[:,:,0]
        
        
        for i in range(batch['X'].shape[0]):
            for c in range(batch['X'].shape[1]):
                if batch['X'][i,c,1] > 500:
                    for j in range(batch['X'].shape[2]):
                        batch['X'][i,c,j] = 0.
        '''
        flatX = numpy.reshape(batch['X'],[-1,batch['X'].shape[2]])
        flatX = flatX[flatX[:,1]>0]
        distributions.append(flatX)
        '''
        #print (numpy.amax(numpy.reshape(batch['X'],[-1,batch['X'].shape[2]]),axis=0))
        #print (numpy.mean(numpy.reshape(batch['X'],[-1,batch['X'].shape[2]]),axis=0))
        lossTrain = model.train_on_batch(batch)
        totalLossTrain+=lossTrain
        
        if stepTrain%10==0:
            logging.info("Training %i-%i: loss=%.3e"%(
                stepTrain,
                epoch+1,
                lossTrain
            ))

    totalLossTrain = totalLossTrain/stepTrain if stepTrain>0 else 0
    logging.info("Done training for %i-%i: lr=%.3e total loss=%.3e"%(
        stepTrain,
        epoch+1,
        learning_rate,
        totalLossTrain
    ))
    '''
    flatX = numpy.concatenate(distributions,axis=0)
    minX = numpy.nanmin(flatX,axis=0)
    maxX = numpy.nanmax(flatX,axis=0)
    meanX = numpy.mean(flatX,axis=0)
    names = ['z0','pt','eta','chi2']#,'bendchi2','nstub']
    uselog=[False,True,False,True]#,True,False]
    
    for i in range(flatX.shape[1]):
        if i>=len(names):
            break
        plt.figure(figsize=(9, 3))
        if uselog[i]:
            plt.hist(flatX[:,i], bins=numpy.logspace(math.log10(minX[i]),math.log10(maxX[i]),100))
            plt.xscale('log')
        else:
            plt.hist(flatX[:,i], bins=100,range=(minX[i],maxX[i]))
        plt.yscale('log')
        plt.title("Feature: %s"%names[i])
        plt.savefig("Feature_%s_ref.png"%names[i])
    '''
    model.save_weights(os.path.join(args.outputFolder,"weights_%i.hdf5"%(epoch+1)))
         
    stepTest = 0  
    totalLossTest = 0.
    
    predictedZ0NN = []
    predictedZ0FH = []
    z0FH = fhAlgo.predictZ0(batch['X'][:,:,0],batch['X'][:,:,1])
    trueZ0 = []
    
    for batch in pipeline.generate(
        batchSize=args.batchSize,
        nFiles=1,
        isTraining=False
    ):
        stepTest += 1
        lossTest = model.test_on_batch(batch)
        totalLossTest+=lossTest
        if stepTest%10==0:
            logging.info("Testing %i-%i: loss=%.3e"%(
                stepTest,
                epoch+1,
                lossTest
            ))
        z0NN,assoc = model.predict_on_batch(batch)
        
        predictedZ0NN.append(z0NN)
        predictedZ0FH.append(fhAlgo.predictZ0(
            batch['X'][:,:,0],
            batch['X'][:,:,1]
        ))
        trueZ0.append(batch['y_avg'])
        
    predictedZ0NN = numpy.concatenate(predictedZ0NN,axis=0)
    predictedZ0FH = numpy.concatenate(predictedZ0FH,axis=0)
    trueZ0 = numpy.concatenate(trueZ0,axis=0)
        
    totalLossTest = totalLossTest/stepTest if stepTest>0 else 0
    logging.info("Done testing for %i-%i: total loss=%.3e"%(
        stepTest,
        epoch+1,
        totalLossTest
    ))
    
    history['lr'].append(learning_rate)
    history['trainLoss'].append(totalLossTrain)
    history['testLoss'].append(totalLossTest)
    if len(history["trainLoss"])>5 and numpy.mean(history["trainLoss"][-3:])<totalLossTrain:
        learning_rate = 0.92*learning_rate
            
    print ("Q:  ",list(map(lambda x: "%6.1f%%"%x,[5.,15.87,50.,84.13,95.])))
    print ("NN: ",list(map(lambda x: "%+6.4f"%x,numpy.percentile(predictedZ0NN-trueZ0,[5.,15.87,50.,84.13,95.]))))
    print ("FH: ",list(map(lambda x: "%+6.4f"%x,numpy.percentile(predictedZ0FH-trueZ0,[5.,15.87,50.,84.13,95.]))))
    



