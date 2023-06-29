from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

import qkeras

import sys
import glob

import h5py

import sklearn.metrics as metrics
import vtx
from EvalScripts.eval_funcs import *
import pandas as pd

import yaml

from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({"ZeroSomeWeights": vtx.nn.constraints.ZeroSomeWeights})

def getWeightArray(model):
    allWeights = []
    allWeightsNonRel = []
    allWeightsByLayer = {}
    allWeightsByLayerNonRel = {}
    for layer in model.layers:         
        if layer.name in ['weight_1','weight_2','weight_final','association_0','association_1','association_final']:
            original_w = layer.get_weights()
            weightsByLayer = []
            weightsByLayerNonRel = []
            for my_weights in original_w:                
                if len(my_weights.shape) < 2: # bias term, ignore for now
                    continue
                #l1norm = tf.norm(my_weights,ord=1)
                elif len(my_weights.shape) == 2: # Dense or LSTM
                    tensor_abs = tf.abs(my_weights)
                    tensor_reduce_max_1 = tf.reduce_max(tensor_abs,axis=-1)
                    tensor_reduce_max_2 = tf.reduce_max(tensor_reduce_max_1,axis=-1)
                    #l1norm_val = float(l1norm.eval())
                    tensor_max = float(tensor_reduce_max_2.numpy())
                it = np.nditer(my_weights, flags=['multi_index'], op_flags=['readwrite'])   
                while not it.finished:
                    w = it[0]
                    allWeights.append(abs(w)/tensor_max)
                    allWeightsNonRel.append(abs(w))
                    weightsByLayer.append(abs(w)/tensor_max)
                    weightsByLayerNonRel.append(abs(w))
                    it.iternext()
            if len(weightsByLayer)>0:
                allWeightsByLayer[layer.name] = np.array(weightsByLayer)
                allWeightsByLayerNonRel[layer.name] = np.array(weightsByLayerNonRel)
    return np.array(allWeights), allWeightsByLayer, np.array(allWeightsNonRel), allWeightsByLayerNonRel

if __name__ == "__main__":

    from qkeras.qlayers import QDense, QActivation
    from qkeras.quantizers import quantized_bits, quantized_relu
    from qkeras.utils import _add_supported_quantized_objects
    co = {}
    _add_supported_quantized_objects(co)


    with open(sys.argv[1]+'.yaml', 'r') as f:
            config = yaml.load(f,Loader=yaml.FullLoader)

    max_ntracks = 250   
    nlatent = config["Nlatent"]
    nbins = config['nbins']

    if sys.argv[2] == '1':

        with open(config['QuantisedModelName']+'_prune_iteration_0_WeightQConfig.yaml', 'r') as f:
            weightqconfig = yaml.load(f,Loader=yaml.FullLoader)
        with open(config['QuantisedModelName']+'_prune_iteration_0_PatternQConfig.yaml', 'r') as f:
            patternqconfig = yaml.load(f,Loader=yaml.FullLoader)
        with open(config['QuantisedModelName']+'_prune_iteration_0_AssociationQConfig.yaml', 'r') as f:
            associationqconfig = yaml.load(f,Loader=yaml.FullLoader)

        Qnetwork = vtx.nn.E2EQKerasDiffArgMax(
                    nbins=nbins,
                    ntracks=max_ntracks, 
                    nweightfeatures=len(config["weight_features"]), 
                    nfeatures=len(config["track_features"]), 
                    nlatent = nlatent,
                    l1regloss = (float)(config['l1regloss']),
                    l2regloss = (float)(config['l2regloss']),
                    nweightnodes = config['nweightnodes'],
                    nweightlayers = config['nweightlayers'],
                    nassocnodes = config['nassocnodes'],
                    nassoclayers = config['nassoclayers'],
                    weightqconfig = weightqconfig,
                    patternqconfig = patternqconfig,
                    associationqconfig = associationqconfig,
                )

    else:

        with open(config['QuantisedModelName']+'_prune_iteration_'+str(int(sys.argv[2])-1)+'_WeightQConfig.yaml', 'r') as f:
            weightqconfig = yaml.load(f,Loader=yaml.FullLoader)
        with open(config['QuantisedModelName']+'_prune_iteration_'+str(int(sys.argv[2])-1)+'_PatternQConfig.yaml', 'r') as f:
            patternqconfig = yaml.load(f,Loader=yaml.FullLoader)
        with open(config['QuantisedModelName']+'_prune_iteration_'+str(int(sys.argv[2])-1)+'_AssociationQConfig.yaml', 'r') as f:
            associationqconfig = yaml.load(f,Loader=yaml.FullLoader)

        Qnetwork = vtx.nn.E2EQKerasDiffArgMaxConstraint(
                    nbins=nbins,
                    ntracks=max_ntracks, 
                    nweightfeatures=len(config["weight_features"]), 
                    nfeatures=len(config["track_features"]), 
                    nlatent = nlatent,
                    l1regloss = (float)(config['l1regloss']),
                    l2regloss = (float)(config['l2regloss']),
                    nweightnodes = config['nweightnodes'],
                    nweightlayers = config['nweightlayers'],
                    nassocnodes = config['nassocnodes'],
                    nassoclayers = config['nassoclayers'],
                    weightqconfig = weightqconfig,
                    patternqconfig = patternqconfig,
                    associationqconfig = associationqconfig,
                    h5fName = config['QuantisedModelName']+'_drop_weights_iteration_'+str(int(sys.argv[2])-1)+'.h5'
                )

    Qmodel = Qnetwork.createE2EModel()
    Qmodel.compile(
        tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=[
            tf.keras.losses.MeanAbsoluteError(),
            tf.keras.losses.BinaryCrossentropy(from_logits=True),
            lambda y,x: 0.
        ],
        metrics=[
            tf.keras.metrics.BinaryAccuracy(threshold=0.,name='assoc_acc') #use thres=0 here since logits are used
        ],
        loss_weights=[config['z0_loss_weight'],
                        config['crossentropy_loss_weight'],
                        0]
    )

    QuantisedModelName = config["QuantisedModelName"] 
    Qmodel.load_weights(QuantisedModelName+"_prune_iteration_"+str(int(sys.argv[2])-1)+".tf").expect_partial()

    print("#=========================================#")
    print("|                                         |")
    print("|       Start of Prune "+sys.argv[2]+" Pruning          |")
    print("|                                         |")
    print("#=========================================#")
    
    weightsPerLayer = {}
    droppedPerLayer = {}
    binaryTensorPerLayer = {}
    allWeightsArray,allWeightsByLayer,allWeightsArrayNonRel,allWeightsByLayerNonRel = getWeightArray(Qmodel)

    relative_weight_max = config["relative_weight_max"][int(sys.argv[2])]
        
    for layer in Qmodel.layers:     
        droppedPerLayer[layer.name] = []
        if layer.name in ['weight_1','weight_2','weight_final','association_1','association_2','association_final']:
            original_w = layer.get_weights()
            weightsPerLayer[layer.name] = original_w
            for my_weights in original_w:
                if len(my_weights.shape) < 2: # bias term, skip for now
                    continue
                #l1norm = tf.norm(my_weights,ord=1)
                elif len(my_weights.shape) == 2: # Dense
                    tensor_abs = tf.abs(my_weights)
                    tensor_reduce_max_1 = tf.reduce_max(tensor_abs,axis=-1)
                    tensor_reduce_max_2 = tf.reduce_max(tensor_reduce_max_1,axis=-1)
                    #l1norm_val = float(l1norm.eval())
                    tensor_max = float(tensor_reduce_max_2.numpy())
                it = np.nditer(my_weights, flags=['multi_index'], op_flags=['readwrite'])                
                binaryTensorPerLayer[layer.name] = np.ones(my_weights.shape)
                while not it.finished:
                    w = it[0]
                    if abs(w)/tensor_max < relative_weight_max:
                        #print("small relative weight %e/%e = %e -> 0"%(abs(w), tensor_max, abs(w)/tensor_max))
                        w[...] = 0
                        droppedPerLayer[layer.name].append((it.multi_index, abs(w)))
                        binaryTensorPerLayer[layer.name][it.multi_index] = 0
                    it.iternext()
            #print('%i weights dropped from %s out of %i weights'%(len(droppedPerLayer[layer.name]),layer.name,layer.count_params()))
            #converted_w = convert_kernel(original_w)
            converted_w = original_w
            layer.set_weights(converted_w)


    print('Summary:')
    totalDropped = sum([len(droppedPerLayer[layer.name]) for layer in Qmodel.layers])
    for layer in Qmodel.layers:
        if layer.name in ['weight_1','weight_2','weight_final','association_1','association_2','association_final']:
            print('%i weights dropped from %s out of %i weights'%(len(droppedPerLayer[layer.name]),layer.name, layer.count_params()))
    print('%i total weights dropped out of %i total weights'%(totalDropped,Qmodel.count_params()))
    print('%.1f%% compression'%(100.*totalDropped/Qmodel.count_params()))

    Qmodel.save_weights(QuantisedModelName+"_prune_iteration_"+sys.argv[2]+".tf")

    # save binary tensor in h5 file 
    h5f = h5py.File(QuantisedModelName+'_drop_weights_iteration_'+sys.argv[2]+'.h5','w')
    for layer, binary_tensor in binaryTensorPerLayer.items():
        h5f.create_dataset('%s'%layer, data = binaryTensorPerLayer[layer])
    h5f.close()

    # plot the distribution of weights

    from scipy import stats

    your_percentile = int(stats.percentileofscore(allWeightsArray, relative_weight_max))
    #percentiles = [5,16,50,84,95,your_percentile]
    percentiles = [5,95,your_percentile]
    #colors = ['r','r','r','r','r','g']
    colors = ['r','r','g']
    vlines = np.percentile(allWeightsArray,percentiles,axis=-1)
    xmin = np.amin(allWeightsArray[np.nonzero(allWeightsArray)])
    xmax = np.amax(allWeightsArray)
    xmin = 6e-8
    xmax = 1
    bins = np.linspace(xmin, xmax, 50)
    logbins = np.geomspace(xmin, xmax, 50)

    labels = []
    histos = []
    for key in reversed(sorted(allWeightsByLayer.keys())):
        labels.append(key)
        histos.append(allWeightsByLayer[key])        
    
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    #plt.hist(allWeightsArray,bins=bins)
    #plt.hist(allWeightsByLayer.values(),bins=bins,histtype='bar',stacked=True,label=allWeightsByLayer.keys())
    ax.hist(histos,bins=bins,histtype='step',stacked=False,label=labels,linewidth=2)
    ax.legend(frameon=False)
    #axis = ax.gca()
    ymin, ymax = ax.get_ylim()
    for vline, percentile, color in zip(vlines, percentiles, colors):
        if percentile==0: continue
        if vline < xmin: continue
        ax.axvline(vline, 0, 1, color=color, linestyle='dashed', linewidth=2, label = '%s%%'%percentile)
        ax.text(vline+0.05*(xmax-xmin), ymax-0.05*(ymax-ymin), '%s%%'%percentile, color=color, horizontalalignment='center')
    ax.set_ylabel('Number of Weights',ha="right",y=1)
    ax.set_xlabel('Absolute Relative Weights',ha="right",x=1)
    ax.grid(True)
    plt.savefig(QuantisedModelName+"_prune_iteration_"+sys.argv[2]+'_weight_histogram.png')

        
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    #plt.hist(allWeightsArray,bins=logbins)
    #plt.hist(allWeightsByLayer.values(),bins=logbins,histtype='bar',stacked=True,label=allWeightsByLayer.keys())
    ax.hist(histos,bins=logbins,histtype='step',stacked=False,label=labels,linewidth=2)
    ax.semilogx()
    ax.legend( frameon=False)
    ymin, ymax = ax.get_ylim()
    
    for vline, percentile, color in zip(vlines, percentiles, colors):
        if percentile==0: continue
        if vline < xmin: continue
        xAdd = 0
        yAdd = 0
        #if plotPercentile5 and percentile==84:
        #    xAdd=0.2
        #if plotPercentile16 and percentile==95:
        #    xAdd=1.2
        ax.axvline(vline, 0, 1, color=color, linestyle='dashed', linewidth=2, label = '%s%%'%percentile)
        ax.text(vline+xAdd, ymax-0.05*(ymax-ymin)+yAdd, '%s%%'%percentile, color=color, horizontalalignment='center')
    ax.set_ylabel('Number of Weights',ha="right",y=1)
    ax.set_xlabel('Absolute Relative Weights',ha="right",x=1)
    ax.grid(True)
    #plt.figtext(0.35, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14) 
    plt.savefig(QuantisedModelName+"_prune_iteration_"+sys.argv[2]+'_weight_histogram_logx.png')


    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    labels = []
    histos = []
    for key in reversed(sorted(allWeightsByLayerNonRel.keys())):
        labels.append(key)
        histos.append(allWeightsByLayerNonRel[key])
        
    xmin = np.amin(allWeightsArrayNonRel[np.nonzero(allWeightsArrayNonRel)])
    xmax = np.amax(allWeightsArrayNonRel)
    #bins = np.linspace(xmin, xmax, 100)
    bins = np.geomspace(xmin, xmax, 50)
    #plt.hist(allWeightsArrayNonRel,bins=bins)
    #plt.hist(allWeightsByLayerNonRel.values(),bins=bins,histtype='bar',stacked=True,label=allWeightsByLayer.keys())
    ax.hist(histos,bins=bins,histtype='step',stacked=False,label=labels,linewidth=2)
    ax.semilogx(base=2)
    ax.legend(frameon=False,loc='upper left')
    ax.set_ylabel('Number of Weights',ha="right",y=1)
    ax.set_xlabel('Absolute Value of Weights',ha="right",x=1)
    ax.grid()
    #plt.figtext(0.35, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14) 
    plt.savefig(QuantisedModelName+"_prune_iteration_"+sys.argv[2]+'_weight_nonrel_histogram_logx.png')
