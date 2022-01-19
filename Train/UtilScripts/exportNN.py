import tensorflow as tf
import numpy as np
import scipy
import os
import sys
import vtx
import yaml

kf = sys.argv[1]

with open(sys.argv[2]+'.yaml', 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
retrain = config["retrain"]

trainable = config["trainable"]
trackfeat = config["track_features"] 
weightfeat = config["weight_features"] 
max_ntracks = 250
nlatent = config["Nlatent"]

QuantisedModelName = config["QuantisedModelName"] 
UnQuantisedModelName = config["UnquantisedModelName"] 

if trainable == "QDiffArgMax":
        
        Qnetwork = vtx.nn.E2EQKerasDiffArgMax(
            nbins=256,
            ntracks=max_ntracks, 
            nweightfeatures=len(weightfeat), 
            nfeatures=len(trackfeat), 
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


        network = vtx.nn.E2EDiffArgMax(
            nbins = 256,
            ntracks = max_ntracks, 
            nweightfeatures = len(weightfeat), 
            nfeatures = len(trackfeat), 
            nweights = 1, 
            nlatent = nlatent,
            activation = 'relu',
            l2regloss = 1e-10
        )

elif trainable == "DiffArgMax":
        
        network = vtx.nn.E2EDiffArgMax(
            nbins = 256,
            ntracks = max_ntracks, 
            nweightfeatures = len(weightfeat), 
            nfeatures = len(trackfeat), 
            nweights = 1, 
            nlatent = nlatent,
            activation = 'relu',
            regloss = 1e-10
        )


elif trainable == "FH":

        network = vtx.nn.E2EFH(
            nbins=256,
            ntracks=max_ntracks, 
            nweightfeatures=len(weightfeat), 
            nfeatures=len(trackfeat), 
            nweights=1, 
            activation='relu',
            regloss=1e-10
        )

elif trainable == "FullNetwork":

        network = vtx.nn.E2Ecomparekernel(
            nbins=256,
            ntracks=max_ntracks, 
            nweightfeatures=len(weightfeat), 
            nfeatures=len(trackfeat), 
            nweights=1, 
            nlatent = nlatent,
            activation='relu',
            regloss=1e-10
        )

model = network.createE2EModel()
model.load_weights(UnQuantisedModelName+".tf").expect_partial()

weightModel = network.createWeightModel()
with open(UnQuantisedModelName+"_weightModel.json", 'w') as f:
    f.write(weightModel.to_json())
weightModel.save_weights(UnQuantisedModelName+"_weightModel_weights.hdf5")
weightModel.save(UnQuantisedModelName+"_weightModel")

patternModel = network.createPatternModel()
with open(UnQuantisedModelName+"_patternModel.json", 'w') as f:
    f.write(patternModel.to_json())
patternModel.save_weights(UnQuantisedModelName+"_patternModel_weights.hdf5")
patternModel.save(UnQuantisedModelName+"_patternModel")

associationModel = network.createAssociationModel()
with open(UnQuantisedModelName+"_associationModel.json", 'w') as f:
    f.write(associationModel.to_json())
associationModel.save_weights(UnQuantisedModelName+"_associationModel_weights.hdf5")
associationModel.save(UnQuantisedModelName+"_associationModel")

Qmodel = Qnetwork.createE2EModel()
Qmodel.load_weights(QuantisedModelName+".tf").expect_partial()

weightQModel = Qnetwork.createWeightModel()
with open(QuantisedModelName+"_weightModel.json", 'w') as f:
    f.write(weightQModel.to_json())
weightQModel.save_weights(QuantisedModelName+"_weightModel_weights.hdf5")
weightQModel.save(QuantisedModelName+"_weightModel")

patternQModel = Qnetwork.createPatternModel()
with open(QuantisedModelName+"_patternModel.json", 'w') as f:
    f.write(patternQModel.to_json())
patternQModel.save_weights(QuantisedModelName+"_patternModel_weights.hdf5")
patternQModel.save(QuantisedModelName+"_patternModel")

associationQModel = Qnetwork.createAssociationModel()
with open(QuantisedModelName+"_associationModel.json", 'w') as f:
    f.write(associationQModel.to_json())
associationQModel.save_weights(QuantisedModelName+"_associationModel_weights.hdf5")
associationQModel.save(QuantisedModelName+"_associationModel")

