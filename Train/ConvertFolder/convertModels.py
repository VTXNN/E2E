import tensorflow as tf
import sys
import yaml
import glob
import sys
from textwrap import wrap

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import sklearn.metrics as metrics
import tensorflow as tf
import yaml

import hls4ml

import vtx
#from TrainingScripts.train import *
from EvalScripts.eval import *

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

start = 0
end = 255
bit = True
nMaxTracks = 250

DAnetwork = vtx.nn.E2EDiffArgMax(
            nbins=nbins,
            start=start,
            end=end,
            ntracks=nMaxTracks, 
            nweightfeatures=3, 
            nfeatures=3, 
            nweights=1, 
            nlatent = nlatent,
            l2regloss=1e-10
        )

Qnetwork = vtx.nn.E2EQKerasDiffArgMax(
            nbins=nbins,
            ntracks=max_ntracks, 
            nweightfeatures=len(config["weight_features"]), 
            nfeatures=len(config["track_features"]), 
            nweights=1, 
            nlatent = nlatent,
            l1regloss = (float)(config['l1regloss']),
            l2regloss = (float)(config['l2regloss']),
            nweightnodes = config['nweightnodes'],
            nweightlayers = config['nweightlayers'],
            nassocnodes = config['nassocnodes'],
            nassoclayers = config['nassoclayers'],
            qconfig = config['QConfig']
        )

QPnetwork = vtx.nn.E2EQKerasDiffArgMaxConstraint(
            nbins=nbins,
            ntracks=max_ntracks, 
            nweightfeatures=len(config["weight_features"]),  
            nfeatures=len(config["track_features"]), 
            nweights=1, 
            nlatent = nlatent,
            l1regloss = (float)(config['l1regloss']),
            l2regloss = (float)(config['l2regloss']),
            nweightnodes = config['nweightnodes'],
            nweightlayers = config['nweightlayers'],
            nassocnodes = config['nassocnodes'],
            nassoclayers = config['nassoclayers'],
            qconfig = config['QConfig'],
            h5fName = "NetworkFiles/"+config['QuantisedModelName']+'_drop_weights_iteration_'+str(config['prune_iterations'])+'.h5'
        )

DAmodel = DAnetwork.createE2EModel()
DAmodel.compile(
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

QPmodel = QPnetwork.createE2EModel()
QPmodel.compile(
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

QuantisedPrunedModelName = config["QuantisedPrunedModelName"] 
QuantisedModelName = config["QuantisedModelName"] + "_prune_iteration_0"
UnQuantisedModelName = config["UnquantisedModelName"] 


DAmodel.load_weights("NetworkFiles/"+UnQuantisedModelName+".tf").expect_partial()
Qmodel.load_weights("NetworkFiles/"+QuantisedModelName+".tf").expect_partial()
QPmodel.load_weights("NetworkFiles/"+QuantisedPrunedModelName+".tf").expect_partial()

DAnetwork.load_weights(DAmodel)
Qnetwork.load_weights(Qmodel)
QPnetwork.load_weights(QPmodel)

with open('ModelAccuracies.txt', 'a') as f:
    print('Model Accuracies \n', file=f)

DAnetwork.export_individual_models("ExportedNetworkFiles/"+UnQuantisedModelName)
Qnetwork.export_individual_models("ExportedNetworkFiles/"+QuantisedModelName)
QPnetwork.export_individual_models("ExportedNetworkFiles/"+QuantisedPrunedModelName)

DAnetwork.write_model_graph("ExportedNetworkFiles/"+UnQuantisedModelName)
Qnetwork.write_model_graph("ExportedNetworkFiles/"+QuantisedModelName)
QPnetwork.write_model_graph("ExportedNetworkFiles/"+QuantisedPrunedModelName)

DAnetwork.export_hls_weight_model(UnQuantisedModelName,plot=True)
Qnetwork.export_hls_weight_model(QuantisedModelName,plot=True)
QPnetwork.export_hls_weight_model(QuantisedPrunedModelName,plot=True)

DAnetwork.export_hls_assoc_model(UnQuantisedModelName,plot=True)
Qnetwork.export_hls_assoc_model(QuantisedModelName,plot=True)
QPnetwork.export_hls_assoc_model(QuantisedPrunedModelName,plot=True)

DAnetwork.export_hls_pattern_model(UnQuantisedModelName,plot=True)
Qnetwork.export_hls_pattern_model(QuantisedModelName,plot=True)
QPnetwork.export_hls_pattern_model(QuantisedPrunedModelName,plot=True)





