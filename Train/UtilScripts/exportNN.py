import tensorflow as tf
import numpy as np
import scipy
import os
import sys
import vtx
import yaml

with open(sys.argv[2]+'.yaml', 'r') as f:
        config = yaml.load(f)
retrain = config["retrain"]

trainable = config["trainable"]
trackfeat = config["track_features"] 
weightfeat = config["weight_features"] 
max_ntracks = 250

if trainable == "DA":
        
        nlatent = 2

        network = vtx.nn.E2EDiffArgMax(
            nbins=256,
            ntracks=max_ntracks, 
            nweightfeatures=len(weightfeat), 
            nfeatures=len(trackfeat), 
            nweights=1, 
            nlatent = nlatent,
            activation='relu',
            regloss=1e-10
        )


elif trainable == "FH":
        nlatent = 0

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
        nlatent = 2

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
model.load_weights("../PretrainedModels/NewKFweightsReduced.tf")

weightModel = network.createWeightModel()
with open('weightModelReduced.json', 'w') as f:
    f.write(weightModel.to_json())
weightModel.save_weights("weightModelReduced_weights.hdf5")

patternModel = network.createPatternModel()
with open('patternModelReduced.json', 'w') as f:
    f.write(patternModel.to_json())
patternModel.save_weights("patternModelReduced_weights.hdf5")

if trainable == "FullNetwork":
    positionModel = network.createPositionModel()
    with open('positionModelReduced.json', 'w') as f:
        f.write(positionModel.to_json())
    positionModel.save_weights("positionModelReduced_weights.hdf5")

associationModel = network.createAssociationModel()
with open('asociationModelReduced.json', 'w') as f:
    f.write(associationModel.to_json())
associationModel.save_weights("asociationModelReduced_weights.hdf5")
