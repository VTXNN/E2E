import tensorflow as tf
import sys
import yaml

kf = sys.argv[1]

with open(sys.argv[2]+'.yaml', 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)


QuantisedModelName = config["QuantisedModelName"] 
UnQuantisedModelName = config["UnQuantisedModelName"] 

weightModel = tf.keras.models.load_model(UnQuantisedModelName+'_weightModel')
patternModel = tf.keras.models.load_model(UnQuantisedModelName+'_patternModel')
associationModel = tf.keras.models.load_model(UnQuantisedModelName+'_associationModel')

import cmsml
cmsml.tensorflow.save_graph(UnQuantisedModelName+"_weightModelgraph.pb", weightModel, variables_to_constants=True)
cmsml.tensorflow.save_graph(UnQuantisedModelName+"_patternModelgraph.pb", patternModel, variables_to_constants=True)
cmsml.tensorflow.save_graph(UnQuantisedModelName+"_associationModelgraph.pb", associationModel, variables_to_constants=True)


QweightModel = tf.keras.models.load_model(QuantisedModelName+'_weightModel')
QpatternModel = tf.keras.models.load_model(QuantisedModelName+'_patternModel')
QassociationModel = tf.keras.models.load_model(QuantisedModelName+'_associationModel')

import cmsml
cmsml.tensorflow.save_graph(QuantisedModelName+"_weightModelgraph.pb", QweightModel, variables_to_constants=True)
cmsml.tensorflow.save_graph(QuantisedModelName+"_patternModelgraph.pb", QpatternModel, variables_to_constants=True)
cmsml.tensorflow.save_graph(QuantisedModelName+"_associationModelgraph.pb", QassociationModel, variables_to_constants=True)
