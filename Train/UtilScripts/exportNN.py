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
'''
model = network.createE2EModel()
model.load_weights(kf + "best_weights.tf").expect_partial()

weightModel = network.createWeightModel()
with open(kf + 'weightModel.json', 'w') as f:
    f.write(weightModel.to_json())
weightModel.save_weights(kf + "weightModel_weights.hdf5")
weightModel.save('weightModel')

patternModel = network.createPatternModel()
with open(kf + 'patternModel.json', 'w') as f:
    f.write(patternModel.to_json())
patternModel.save_weights(kf + "patternModel_weights.hdf5")
patternModel.save('patternModel')

if trainable == "FullNetwork":
    positionModel = network.createPositionModel()
    with open(kf + 'positionModel.json', 'w') as f:
        f.write(positionModel.to_json())
    positionModel.save_weights(kf + "positionModel_weights.hdf5")

associationModel = network.createAssociationModel()
with open(kf + 'asociationModel.json', 'w') as f:
    f.write(associationModel.to_json())
associationModel.save_weights(kf + "asociationModel_weights.hdf5")
associationModel.save("asociationModel")
'''
Qmodel = Qnetwork.createE2EModel()
Qmodel.load_weights(kf + "best_weights_unpruned.tf")

weightQModel = Qnetwork.createWeightModel()
with open(kf + 'weightQModel.json', 'w') as f:
    f.write(weightQModel.to_json())
weightQModel.save_weights(kf + "weightQModel_weights.hdf5")

patternQModel = Qnetwork.createPatternModel()
with open(kf + 'patternQModel.json', 'w') as f:
    f.write(patternQModel.to_json())
patternQModel.save_weights(kf + "patternQModel_weights.hdf5")

if trainable == "FullNetwork":
    positionQModel = Qnetwork.createPositionModel()
    with open(kf + 'positionQModel.json', 'w') as f:
        f.write(positionQModel.to_json())
    positionQModel.save_weights(kf + "positionQModel_weights.hdf5")

associationQModel = Qnetwork.createAssociationModel()
with open(kf + 'asociationQModel.json', 'w') as f:
    f.write(associationQModel.to_json())
associationQModel.save_weights(kf + "asociationQModel_weights.hdf5")

