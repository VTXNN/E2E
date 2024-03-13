import uproot3 as uproot
import tensorflow as tf
import numpy as np
import math
from math import isnan
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vtx
import yaml
from tensorflow.keras.models import Model
import glob

nMaxTracks = 250
nbins = 256
max_z0 = 20.46912512
import cmsml

def createWeightModel(config,weightqconfig):
    weightInput = tf.keras.layers.Input(shape=(['nweightfeatures']),name="input_weight")
    weightLayers = []
    for ilayer,nodes in enumerate([self.nweightnodes]*self.nweightlayers):
            weightLayers.extend([
                QDense(
                    nodes,
                    trainable=True,
                    kernel_initializer='orthogonal',
                    kernel_regularizer=tf.keras.regularizers.L1L2(self.l1regloss,self.l2regloss),
                    kernel_quantizer=self.weightqconfig['weight_'+str(ilayer+1)]['kernel_quantizer'],
                    bias_quantizer=self.weightqconfig['weight_'+str(ilayer+1)]['bias_quantizer'],
                    kernel_constraint = zero_some_weights(binary_tensor=self.h5f['weight_'+str(ilayer+1)][()].tolist()),
                    activation=None,
                    name='weight_'+str(ilayer+1)
                ),
                QActivation(self.weightqconfig['weight_'+str(ilayer+1)]['activation'],name='weight_'+str(ilayer+1)+'_relu'),
            ])
            
    weightLayers.extend([
            QDense(
                self.nweights,
                kernel_initializer='orthogonal',
                trainable=True,
                kernel_quantizer=self.weightqconfig['weight_final']['kernel_quantizer'],
                bias_quantizer=self.weightqconfig['weight_final']['bias_quantizer'],
                activation=None,
                kernel_regularizer=tf.keras.regularizers.L1L2(self.l1regloss,self.l2regloss),
                kernel_constraint = zero_some_weights(binary_tensor=self.h5f['weight_final'][()].tolist()),
                name='weight_final'
            ),
            QActivation(self.weightqconfig['weight_final']['activation'],name='weight_final_relu')
        ])

    outputs = self.applyLayerList(weightInput,weightLayers)

    return tf.keras.Model(inputs=weightInput,outputs=outputs)

with open('setup.yaml', 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

with open(config['QuantisedModelName']+'_prune_iteration_9_WeightQConfig.yaml', 'r') as f:
        weightqconfig = yaml.load(f,Loader=yaml.FullLoader)
with open(config['QuantisedModelName']+'_prune_iteration_9_PatternQConfig.yaml', 'r') as f:
        patternqconfig = yaml.load(f,Loader=yaml.FullLoader)
with open(config['QuantisedModelName']+'_prune_iteration_9_AssociationQConfig.yaml', 'r') as f:
        associationqconfig = yaml.load(f,Loader=yaml.FullLoader)

QuantisedModelName = config["QuantisedPrunedModelName"] 

weightgraph = cmsml.tensorflow.load_frozen_graph(QuantisedModelName+"_weightModelgraph.pb")
patterngraph = cmsml.tensorflow.load_frozen_graph(QuantisedModelName+"_patternModelgraph.pb")
associationgraph = cmsml.tensorflow.load_frozen_graph(QuantisedModelName+"_associationModelgraph.pb")

print(weightgraph)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(QuantisedModelName+'_weightModelgraph.pb', 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
conf = tf.ConfigProto()
session = tf.Session(graph=detection_graph, config=conf)

layers = [op.name for op in detection_graph.get_operations()]
for layer in layers:
      print(layer)

from tensorflow.python.framework import tensor_util
weight_nodes = [n for n in od_graph_def.node if n.op == 'Const']
for n in weight_nodes:
            print("Name of the node - %s" % n.name)
            print("Value - " )
            print(tensor_util.MakeNdarray(n.attr['value'].tensor))


weightModel = 
weightModel.summary()