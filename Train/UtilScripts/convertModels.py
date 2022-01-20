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

import vtx
#from TrainingScripts.train import *
from EvalScripts.evalDA import *

from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.utils import _add_supported_quantized_objects
co = {}
_add_supported_quantized_objects(co)

kf = sys.argv[1]

with open(sys.argv[2]+'.yaml', 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

cnn = sys.argv[3]

max_ntracks = 250   
nlatent = config["Nlatent"]

DAnetwork = vtx.nn.E2EDiffArgMax(
            nbins=256,
            ntracks=max_ntracks, 
            nweightfeatures=len(config['weight_features']), 
            nfeatures=len(config['track_features']), 
            nweights=1, 
            nlatent = nlatent,
            activation='relu',
            l2regloss=1e-10,
            nweightnodes = config['nweightnodes'],
            nweightlayers = config['nweightlayers'],
            nassocnodes = config['nassocnodes'],
            nassoclayers = config['nassoclayers'],
        )

Qnetwork = vtx.nn.E2EQKerasDiffArgMax(
            nbins=256,
            ntracks=max_ntracks, 
            nweightfeatures=len(config["weight_features"]), 
            nfeatures=len(config["track_features"]), 
            nweights=1, 
            nlatent = nlatent,
            activation='relu',
            l1regloss = (float)(config['l1regloss']),
            l2regloss = (float)(config['l2regloss']),
            nweightnodes = config['nweightnodes'],
            nweightlayers = config['nweightlayers'],
            nassocnodes = config['nassocnodes'],
            nassoclayers = config['nassoclayers'],
            qconfig = config['QConfig']
        )

QPnetwork = vtx.nn.E2EQKerasDiffArgMaxConstraint(
            nbins=256,
            ntracks=max_ntracks, 
            nweightfeatures=len(config["weight_features"]),  
            nfeatures=len(config["track_features"]), 
            nweights=1, 
            nlatent = nlatent,
            activation='relu',
            l1regloss = (float)(config['l1regloss']),
            l2regloss = (float)(config['l2regloss']),
            nweightnodes = config['nweightnodes'],
            nweightlayers = config['nweightlayers'],
            nassocnodes = config['nassocnodes'],
            nassoclayers = config['nassoclayers'],
            temperature = 1e-2,
            qconfig = config['QConfig'],
            h5fName = config['QuantisedModelName']+'_drop_weights_iteration_'+str(config['prune_iterations'])+'.h5'
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


DAmodel.load_weights(UnQuantisedModelName+".tf").expect_partial()
Qmodel.load_weights(QuantisedModelName+".tf").expect_partial()
QPmodel.load_weights(QuantisedPrunedModelName+".tf").expect_partial()

DAnetwork.load_weights(DAmodel)
Qnetwork.load_weights(Qmodel)
QPnetwork.load_weights(QPmodel)

if cnn == 'False':
    DAnetwork.export_individual_models(UnQuantisedModelName)
    Qnetwork.export_individual_models(QuantisedModelName)
    QPnetwork.export_individual_models(QuantisedPrunedModelName)

    DAnetwork.write_model_graph(UnQuantisedModelName)
    Qnetwork.write_model_graph(QuantisedModelName)
    QPnetwork.write_model_graph(QuantisedPrunedModelName)

    DAnetwork.export_hls_weight_model(UnQuantisedModelName)
    Qnetwork.export_hls_weight_model(QuantisedModelName)
    QPnetwork.export_hls_weight_model(QuantisedPrunedModelName)

    DAnetwork.export_hls_assoc_model(UnQuantisedModelName)
    Qnetwork.export_hls_assoc_model(QuantisedModelName)
    QPnetwork.export_hls_assoc_model(QuantisedPrunedModelName)

if cnn == 'True':
    DAnetwork.export_hls_pattern_model(UnQuantisedModelName)
    Qnetwork.export_hls_pattern_model(QuantisedModelName)
    QPnetwork.export_hls_pattern_model(QuantisedPrunedModelName)

