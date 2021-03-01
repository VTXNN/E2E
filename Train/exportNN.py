import tensorflow as tf
import numpy as np
import scipy
import os
import sys
import vtx

network = vtx.nn.E2ERef(
    nbins=256,
    ntracks=250, 
    nfeatures=17, 
    nweights=1, 
    nlatent=0, 
    activation='relu',
    regloss=1e-10
)

model = network.createE2EModel()
model.load_weights("weights_10.hdf5")

weightModel = network.createWeightModel()
with open('weightModel.json', 'w') as f:
    f.write(weightModel.to_json())
weightModel.save_weights("weightModel_weights.hdf5")

patternModel = network.createPatternModel()
with open('patternModel.json', 'w') as f:
    f.write(patternModel.to_json())
patternModel.save_weights("patternModel_weights.hdf5")

positionModel = network.createPositionModel()
with open('positionModel.json', 'w') as f:
    f.write(positionModel.to_json())
positionModel.save_weights("positionModel_weights.hdf5")

associationModel = network.createAssociationModel()
with open('asociationModel.json', 'w') as f:
    f.write(associationModel.to_json())
associationModel.save_weights("asociationModel_weights.hdf5")

