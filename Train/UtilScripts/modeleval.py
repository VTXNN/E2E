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
from TrainingScripts.train import *
from EvalScripts.evalDA import *

from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.utils import _add_supported_quantized_objects
co = {}
_add_supported_quantized_objects(co)

#hep.set_style("CMSTex")
#hep.cms.label()
#hep.cms.text("Simulation")
plt.style.use(hep.style.CMS)

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

LEGEND_WIDTH = 31
LINEWIDTH = 3

kf = sys.argv[1]

with open(sys.argv[2]+'.yaml', 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('axes', linewidth=5)              # thickness of axes
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)            # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

matplotlib.rcParams['xtick.major.size'] = 20
matplotlib.rcParams['xtick.major.width'] = 5
matplotlib.rcParams['xtick.minor.size'] = 10
matplotlib.rcParams['xtick.minor.width'] = 4

matplotlib.rcParams['ytick.major.size'] = 20
matplotlib.rcParams['ytick.major.width'] = 5
matplotlib.rcParams['ytick.minor.size'] = 10
matplotlib.rcParams['ytick.minor.width'] = 4

trainable = config["trainable"]
trackfeat = config["track_features"] 
weightfeat = config["weight_features"] 

max_ntracks = 250

if trainable == "QDiffArgMax":
        
    nlatent = config["Nlatent"]

    network = vtx.nn.E2EDiffArgMax(
            nbins=256,
            ntracks=max_ntracks, 
            nweightfeatures=len(weightfeat), 
            nfeatures=len(trackfeat), 
            nweights=1, 
            nlatent = nlatent,
            activation='relu',
            l2regloss=1e-10
        )

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

Qmodel = Qnetwork.createE2EModel()
Qmodel.compile(
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



QuantisedModelName = config["QuantisedModelName"] 
UnQuantisedModelName = config["UnQuantisedModelName"] 

Qmodel.summary()
Qmodel.load_weights(QuantisedModelName+".tf")


weightModel = Qnetwork.createWeightModel()
weightModel.load_weights(QuantisedModelName+".weightModel_weights.hdf5")

patternModel = Qnetwork.createPatternModel()
patternModel.load_weights(QuantisedModelName+".patternModel_weights.hdf5")

associationModel = Qnetwork.createAssociationModel()
associationModel.load_weights(QuantisedModelName+".associationModel_weights.hdf5")


weightModel.summary()
patternModel.summary()
associationModel.summary()

import hls4ml

weightconfig = hls4ml.utils.config_from_keras_model(weightModel, granularity='model')
print(weightconfig)
print("-----------------------------------")
print("Configuration")
#plotting.print_dict(config)
print("-----------------------------------")
random_weight_data = np.random.rand(1000,3)
hls_weight_model = hls4ml.converters.convert_from_keras_model(weightModel,
                                                       hls_config=weightconfig,
                                                       output_dir='/home/cebrown/Documents/Trigger/E2E/Train/'+QuantisedModelName+'_weight/hls4ml_prj',
                                                       fpga_part='xcvu9p-flga2104-2L-e',
                                                       clock_period=2.5)
hls4ml.utils.plot_model(hls_weight_model, show_shapes=True, show_precision=True, to_file=kf+"Weight_model.png")
plt.clf()
ap, wp = hls4ml.model.profiling.numerical(model=weightModel, hls_model=hls_weight_model, X=random_weight_data)
wp.savefig(kf+"Weight_model_activations_profile.png")
ap.savefig(kf+"Weight_model_weights_profile.png")
#####################################################################################################
patternconfig = hls4ml.utils.config_from_keras_model(patternModel, granularity='name')
patternconfig['Model']['Strategy'] = 'Resource'
print("-----------------------------------")
print("Configuration")
#plotting.print_dict(config)
print("-----------------------------------")
random_pattern_data = np.random.rand(1000,256,1)
hls_pattern_model = hls4ml.converters.convert_from_keras_model(patternModel,
                                                       hls_config=patternconfig,
                                                       output_dir='/home/cebrown/Documents/Trigger/E2E/Train/'+QuantisedModelName+'_pattern/hls4ml_prj',
                                                       fpga_part='xcvu9p-flga2104-2L-e',
                                                       clock_period=2.5)
hls4ml.utils.plot_model(hls_pattern_model, show_shapes=True, show_precision=True, to_file=kf+"pattern_model.png")
plt.clf()
ap,wp = hls4ml.model.profiling.numerical(model=patternModel, hls_model=hls_pattern_model, X=random_pattern_data)
wp.savefig(kf+"pattern_model_activations_profile.png")
ap.savefig(kf+"pattern_model_weights_profile.png")
#####################################################################################################
associationconfig = hls4ml.utils.config_from_keras_model(associationModel, granularity='name')
print(associationconfig)
print("-----------------------------------")
print("Configuration")
#plotting.print_dict(config)
print("-----------------------------------")
random_association_data = np.random.rand(1000,4+nlatent)
hls_association_model = hls4ml.converters.convert_from_keras_model(associationModel,
                                                       hls_config=associationconfig,
                                                       output_dir='/home/cebrown/Documents/Trigger/E2E/Train/'+QuantisedModelName+'_association/hls4ml_prj',
                                                       fpga_part='xcvu9p-flga2104-2L-e',
                                                       clock_period=2.5)
hls4ml.utils.plot_model(hls_association_model, show_shapes=True, show_precision=True, to_file=kf+"association_model.png")
plt.clf()
ap,wp = hls4ml.model.profiling.numerical(model=associationModel, hls_model=hls_association_model, X=random_association_data)
wp.savefig(kf+"association_model_activations_profile.png")
ap.savefig(kf+"association_model_weights_profile.png")
#####################################################################################################


#hls_weight_model.compile()
#hls_pattern_model.compile()
#hls_association_model.compile()

#hls_weight_model.build(csim=False,synth=True,vsynth=True)
#hls_pattern_model.build(csim=False,synth=True,vsynth=True)
#hls_association_model.build(csim=False,synth=True,vsynth=True)

#hls4ml.report.read_vivado_report(kf+'model_weight_1_unquantised/hls4ml_prj')
#hls4ml.report.read_vivado_report(kf+'model_association_1_unquantised/hls4ml_prj')
#hls4ml.report.read_vivado_report(kf+'model_pattern_1_unquantised/hls4ml_prj')

