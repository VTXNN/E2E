import glob
import sys
import os
from textwrap import wrap

import comet_ml
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import sklearn.metrics as metrics
import tensorflow as tf
import yaml
import hls4ml


from tensorflow.keras.models import Model

import vtx
from TrainingScripts.train import *
from EvalScripts.eval_funcs import *
from sklearn.metrics import mean_squared_error


nMaxTracks = 250
max_z0 = 15

if __name__=="__main__":

    
    with open(sys.argv[1]+'.yaml', 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    QuantisedModelName = config["QuantisedModelName"] 


    with open(config['QuantisedModelName']+'_prune_iteration_9_WeightQConfig.yaml', 'r') as f:
        weightqconfig = yaml.load(f,Loader=yaml.FullLoader)
    with open(config['QuantisedModelName']+'_prune_iteration_9_PatternQConfig.yaml', 'r') as f:
        patternqconfig = yaml.load(f,Loader=yaml.FullLoader)
    with open(config['QuantisedModelName']+'_prune_iteration_9_AssociationQConfig.yaml', 'r') as f:
        associationqconfig = yaml.load(f,Loader=yaml.FullLoader)

    network = vtx.nn.E2EQKerasDiffArgMaxConstraint(
                nbins=nbins,
                ntracks=max_ntracks, 
                nweightfeatures=len(config["weight_features"]), 
                nfeatures=len(config["track_features"]), 
                nlatent = config['Nlatent'],
                l1regloss = (float)(config['l1regloss']),
                l2regloss = (float)(config['l2regloss']),
                nweightnodes = config['nweightnodes'],
                nweightlayers = config['nweightlayers'],
                nassocnodes = config['nassocnodes'],
                nassoclayers = config['nassoclayers'],
                weightqconfig = weightqconfig,
                patternqconfig = patternqconfig,
                associationqconfig = associationqconfig,
                h5fName = config['QuantisedModelName']+'_drop_weights_iteration_8.h5'
            )

    model = network.createE2EModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
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
        loss_weights=[config['z0_loss_weight'],
                        config['crossentropy_loss_weight'],
                        0]
    )


    model.load_weights(QuantisedModelName+"_prune_iteration_9.tf").expect_partial()

    weightmodel = network.createWeightModel()
    weightmodel.get_layer('weight_1').set_weights    (model.get_layer('weight_1').get_weights())
    weightmodel.get_layer('weight_1_relu').set_weights    (model.get_layer('weight_1_relu').get_weights())
    weightmodel.get_layer('weight_2').set_weights     (model.get_layer('weight_2').get_weights())
    weightmodel.get_layer('weight_2_relu').set_weights     (model.get_layer('weight_2_relu').get_weights())
    weightmodel.get_layer('weight_final').set_weights(model.get_layer('weight_final').get_weights())
    weightmodel.get_layer('weight_final_relu').set_weights(model.get_layer('weight_final_relu').get_weights())

    weightmodel.compile(
                optimizer,
                loss=[
                    lambda y,x: 0.
                ]
        )

    numpt = 127
    numeta = 200

    mva_range = np.linspace(0,7,8)
    pt_range = np.linspace(0,128,numpt)
    eta_range = np.linspace(0,2.4,numeta)

    weight = np.zeros([8,numpt,numeta])

    #contours = (0.0,0.005,0.01,0.012,0.014,0.016,0.018,0.02,0.022,0.024,0.025)
    contours = (0.0,0.005,0.01,0.015,0.02,0.025,0.03)

    for i,mva in enumerate(mva_range):
        for j,pt in enumerate(pt_range):
            for k,eta in enumerate(eta_range):
                print(i,j,k)
                weight[i,j,k] = weightmodel.predict_on_batch(np.expand_dims([pt,mva,eta],axis=0)) 

    for i in range(len(mva_range)):
        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(12,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        image = ax.imshow(weight[i].T,cmap=colormap,extent=[pt_range[0],pt_range[numpt-1],eta_range[0],eta_range[numeta-1]],vmin=0,vmax=np.max(weight),aspect='auto')
        CS = ax.contour(np.flip(weight[i].T,0), contours, colors='k', origin=None, extent=[pt_range[0],pt_range[numpt-1],eta_range[0],eta_range[numeta-1]])
        ax.clabel(CS, inline=1, fontsize=10)
        ax.set_xlabel("$p_T$ [GeV]", horizontalalignment='right', x=1.0)
        ax.set_ylabel("$|\eta|$", horizontalalignment='right', y=1.0)
        cbar = plt.colorbar(image, ax=ax)
        cbar.set_label('Track Weight')
        plt.title("Weights for BDT Output: "+str(i),pad=50)
        plt.tight_layout()
        plt.savefig("WeightsMVA"+str(i)+".png")
        plt.close()
