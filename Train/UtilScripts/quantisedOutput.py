import glob
import sys
from textwrap import wrap

import comet_ml
import matplotlib
import pandas as pd
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
from tensorflow.keras.models import Model


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

use_quantised_weights = True
print_extra = True

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

halfBinWidth = 0.5*30./256.

def predictHisto(value,weight,return_hist=False):
    
    hist,bin_edges = np.histogram(value,256,range=(-15,15),weights=weight)
    if return_hist:
        return hist,bin_edges
    else:
        hist = np.convolve(hist,[1,1,1],mode='same')
        z0Index= np.argmax(hist)
        z0 = -15.+30.*z0Index/256.+halfBinWidth
        return z0

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
            regloss=1e-10
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


model.summary()
model.load_weights(kf+"best_weights_unquantised.tf").expect_partial()

Qmodel.summary()
QweightModel = Qnetwork.createWeightModel()
QpatternModel = Qnetwork.createPatternModel()
QassociationModel = Qnetwork.createAssociationModel()

from qkeras.autoqkeras.utils import print_qmodel_summary
print_qmodel_summary(Qmodel)  

if use_quantised_weights:

    Qmodel.load_weights(kf+"best_weights.tf").expect_partial()
    #QweightModel.load_weights(kf+"weightQModel_weights.hdf5")
    #QpatternModel.load_weights(kf+"patternQModel_weights.hdf5")
    #QassociationModel.load_weights(kf+"asociationQModel_weights.hdf5")

    Qmodel.layers[15].set_weights([np.array([[1]], dtype=np.float32), np.array([-halfBinWidth], dtype=np.float32)])

    QweightModel.layers[1].set_weights(Qmodel.layers[1].get_weights())
    QweightModel.layers[2].set_weights(Qmodel.layers[2].get_weights())
    QweightModel.layers[3].set_weights(Qmodel.layers[3].get_weights())
    QweightModel.layers[4].set_weights(Qmodel.layers[4].get_weights())
    QweightModel.layers[5].set_weights(Qmodel.layers[5].get_weights())
    QweightModel.layers[6].set_weights(Qmodel.layers[7].get_weights())

    QpatternModel.layers[1].set_weights(Qmodel.layers[9].get_weights())
    QpatternModel.layers[2].set_weights(Qmodel.layers[10].get_weights())

    QassociationModel.layers[1].set_weights(Qmodel.layers[19].get_weights())
    QassociationModel.layers[2].set_weights(Qmodel.layers[20].get_weights()) 
    QassociationModel.layers[3].set_weights(Qmodel.layers[21].get_weights()) 
    QassociationModel.layers[4].set_weights(Qmodel.layers[22].get_weights()) 
    QassociationModel.layers[5].set_weights(Qmodel.layers[23].get_weights()) 

else:
    model.layers[19].set_weights([np.array([[1]], dtype=np.float32), np.array([0], dtype=np.float32)])

    #QweightModel.layers[1].set_weights(model.layers[1].get_weights())
    #QweightModel.layers[2].set_weights(model.layers[2].get_weights())
    #QweightModel.layers[3].set_weights(model.layers[5].get_weights())
    #QweightModel.layers[4].set_weights(model.layers[6].get_weights())
    #QweightModel.layers[5].set_weights(model.layers[9].get_weights())
    #QweightModel.layers[6].set_weights(model.layers[11].get_weights())

    #QpatternModel.layers[1].set_weights(model.layers[13].get_weights())
    #QpatternModel.layers[2].set_weights(model.layers[14].get_weights())

    #QassociationModel.layers[1].set_weights(model.layers[23].get_weights())
    #QassociationModel.layers[2].set_weights(model.layers[24].get_weights()) 
    #QassociationModel.layers[3].set_weights(model.layers[27].get_weights()) 
    #QassociationModel.layers[4].set_weights(model.layers[28].get_weights()) 
    #QassociationModel.layers[5].set_weights(model.layers[31].get_weights()) 

    Qmodel.layers[0].set_weights(model.layers[0].get_weights())
    Qmodel.layers[1].set_weights(model.layers[1].get_weights())
    Qmodel.layers[2].set_weights(model.layers[2].get_weights())
    Qmodel.layers[3].set_weights(model.layers[5].get_weights())
    Qmodel.layers[4].set_weights(model.layers[6].get_weights())
    Qmodel.layers[5].set_weights(model.layers[9].get_weights())
    Qmodel.layers[6].set_weights(model.layers[10].get_weights())
    Qmodel.layers[7].set_weights(model.layers[11].get_weights())
    Qmodel.layers[8].set_weights(model.layers[12].get_weights())

    Qmodel.layers[9].set_weights(model.layers[13].get_weights())
    Qmodel.layers[10].set_weights(model.layers[14].get_weights())

    Qmodel.layers[11].set_weights(model.layers[15].get_weights())
    Qmodel.layers[12].set_weights(model.layers[16].get_weights())
    Qmodel.layers[13].set_weights(model.layers[17].get_weights())
    Qmodel.layers[14].set_weights(model.layers[18].get_weights())

    Qmodel.layers[15].set_weights(model.layers[19].get_weights())

    Qmodel.layers[16].set_weights(model.layers[20].get_weights())
    Qmodel.layers[17].set_weights(model.layers[21].get_weights())
    Qmodel.layers[18].set_weights(model.layers[22].get_weights())


    Qmodel.layers[19].set_weights(model.layers[23].get_weights())
    Qmodel.layers[20].set_weights(model.layers[24].get_weights()) 
    Qmodel.layers[21].set_weights(model.layers[27].get_weights()) 
    Qmodel.layers[22].set_weights(model.layers[28].get_weights()) 
    Qmodel.layers[23].set_weights(model.layers[31].get_weights()) 

    QweightModel.layers[1].set_weights(Qmodel.layers[1].get_weights())
    QweightModel.layers[2].set_weights(Qmodel.layers[2].get_weights())
    QweightModel.layers[3].set_weights(Qmodel.layers[3].get_weights())
    QweightModel.layers[4].set_weights(Qmodel.layers[4].get_weights())
    QweightModel.layers[5].set_weights(Qmodel.layers[5].get_weights())
    QweightModel.layers[6].set_weights(Qmodel.layers[7].get_weights())

    QpatternModel.layers[1].set_weights(Qmodel.layers[9].get_weights())
    QpatternModel.layers[2].set_weights(Qmodel.layers[10].get_weights())

    QassociationModel.layers[1].set_weights(Qmodel.layers[19].get_weights())
    QassociationModel.layers[2].set_weights(Qmodel.layers[20].get_weights()) 
    QassociationModel.layers[3].set_weights(Qmodel.layers[21].get_weights()) 
    QassociationModel.layers[4].set_weights(Qmodel.layers[22].get_weights()) 
    QassociationModel.layers[5].set_weights(Qmodel.layers[23].get_weights()) 


    Qmodel.layers[15].set_weights([np.array([[1]], dtype=np.float32), np.array([-halfBinWidth], dtype=np.float32)])

layer_names = []
layer_weights = []
layer_biases = []

for i,layer in enumerate(Qmodel.layers):
        get_weights = layer.get_weights()
        if len(get_weights) > 0:
            if "Bin_weight" not in layer.name:  
                weights = get_weights[0].flatten()[get_weights[0].flatten() != 0]
                if len(get_weights) > 1:
                    biases = get_weights[1].flatten()[get_weights[1].flatten() != 0]

                layer_names.append(layer.name)
                layer_weights.append(abs(weights))
                layer_biases.append(abs(biases))

plt.clf()
fig,ax = plt.subplots(1,2,figsize=(20,10))
hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])
ax[0].boxplot(layer_weights,labels=layer_names,notch=True,vert=False)
ax[1].boxplot(layer_biases,labels=layer_names,notch=True,vert=False)
ax[0].grid(True)
ax[0].set_xlabel("Weight",ha="right",x=1)
ax[0].set_ylabel("Layer",ha="right",y=1)
ax[0].set_xscale('log', base=2)

ax[1].grid(True)
ax[1].set_xlabel("Bias",ha="right",x=1)
ax[1].set_ylabel("Layer",ha="right",y=1)
ax[1].set_xscale('log', base=2)

plt.tight_layout()
plt.savefig("Qkeras_weight+bias_profile.png")
plt.close()


import hls4ml

hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

weightconfig = hls4ml.utils.config_from_keras_model(QweightModel, granularity='name')
weightconfig['Model']['Precision'] =  'ap_fixed<'+config['hls4ml_weight']['model']['quantizer']+'>'
weightconfig['Model']['ReuseFactor'] = 1


for layer in weightconfig['LayerName'].keys():
    weightconfig['LayerName'][layer]['Trace'] = True

weightconfig['LayerName']['weight']['Precision']['result'] = 'ap_fixed<'+config['hls4ml_weight']['input']['quantizer']+'>'

weightconfig['LayerName']['weight_1']['Precision']['weight'] = 'ap_fixed<'+config['hls4ml_weight']['weight_1']['kernel_quantizer']+'>'
weightconfig['LayerName']['weight_1']['Precision']['bias'] = 'ap_fixed<'+config['hls4ml_weight']['weight_1']['bias_quantizer']+'>'
weightconfig['LayerName']['q_activation']['Precision']['result'] = 'ap_ufixed<'+config['hls4ml_weight']['weight_1']['activation']+'>'

weightconfig['LayerName']['weight_2']['Precision']['weight'] = 'ap_fixed<'+config['hls4ml_weight']['weight_2']['kernel_quantizer']+'>'
weightconfig['LayerName']['weight_2']['Precision']['bias'] = 'ap_fixed<'+config['hls4ml_weight']['weight_2']['bias_quantizer']+'>'
weightconfig['LayerName']['q_activation_1']['Precision']['result'] = 'ap_ufixed<'+config['hls4ml_weight']['weight_2']['activation']+'>'

weightconfig['LayerName']['weight_final']['Precision']['weight'] = 'ap_fixed<'+config['hls4ml_weight']['weight_final']['kernel_quantizer']+'>'
weightconfig['LayerName']['weight_final']['Precision']['bias'] = 'ap_fixed<'+config['hls4ml_weight']['weight_final']['bias_quantizer']+'>'
weightconfig['LayerName']['q_activation_2']['Precision']['result'] = 'ap_ufixed<'+config['hls4ml_weight']['weight_final']['activation']+'>'

cfg = hls4ml.converters.create_config(backend='Vivado')
cfg['HLSConfig']  = weightconfig
cfg['KerasModel'] = QweightModel
cfg['OutputDir']  = '/home/cebrown/Documents/Trigger/E2E/Train/'+kf+'model_weight_1_unquantised/hls4ml_prj'
cfg['XilinxPart'] = 'xcvu9p-flga2104-2L-e'
hls_weight_model = hls4ml.converters.keras_to_hls(cfg)

#hls_weight_model = hls4ml.converters.convert_from_keras_model(QweightModel,
#                                                       hls_config=weightconfig,
#                                                       output_dir='/home/cebrown/Documents/Trigger/E2E/Train/'+kf+'model_weight_1_unquantised/hls4ml_prj',
#                                                       part='xcvu9p-flga2104-2L-e',
#                                                       clock_period=2.5)
#####################################################################################################
patternconfig = hls4ml.utils.config_from_keras_model(QpatternModel, granularity='name')
patternconfig['Model']['Strategy'] = 'Resource'
patternconfig['Model']['Precision'] = 'ap_fixed<'+config['hls4ml_conv']['model']['quantizer']+'>'

for layer in patternconfig['LayerName'].keys():
    patternconfig['LayerName'][layer]['Trace'] = True

patternconfig['LayerName']['hist']['Precision']['result'] = 'ap_fixed<'+config['hls4ml_conv']['input']['quantizer']+'>'
patternconfig['LayerName']['pattern_1']['Precision']['weight'] = 'ap_fixed<'+config['hls4ml_conv']['conv']['kernel_quantizer']+'>'
patternconfig['LayerName']['q_activation_3']['Precision']['result'] = 'ap_ufixed<'+config['hls4ml_conv']['conv']['activation']+'>'

#hls_pattern_model = hls4ml.converters.convert_from_keras_model(QpatternModel,
#                                                       hls_config=patternconfig,
#                                                       output_dir='/home/cebrown/Documents/Trigger/E2E/Train/'+kf+'model_pattern_1_unquantised/hls4ml_prj',
#                                                       part='xcvu9p-flga2104-2L-e',
#                                                       clock_period=2.5)


cfg = hls4ml.converters.create_config(backend='Vivado')
cfg['HLSConfig']  = patternconfig
cfg['KerasModel'] = QpatternModel
cfg['OutputDir']  = '/home/cebrown/Documents/Trigger/E2E/Train/'+kf+'model_pattern_1_unquantised/hls4ml_prj'
cfg['XilinxPart'] = 'xcvu9p-flga2104-2L-e'
hls_pattern_model = hls4ml.converters.keras_to_hls(cfg)
#####################################################################################################
associationconfig = hls4ml.utils.config_from_keras_model(QassociationModel, granularity='name')
#hls_association_model = hls4ml.converters.convert_from_keras_model(QassociationModel,
#                                                       hls_config=associationconfig,
#                                                       output_dir='/home/cebrown/Documents/Trigger/E2E/Train/'+kf+'model_association_1_unquantised/hls4ml_prj',
#                                                       part='xcvu9p-flga2104-2L-e',
#                                                       clock_period=2.5)


associationconfig['Model']['Precision'] =  'ap_fixed<'+config['hls4ml_assoc']['model']['quantizer']+'>'
associationconfig['Model']['ReuseFactor'] = 1


for layer in associationconfig['LayerName'].keys():
    associationconfig['LayerName'][layer]['Trace'] = True

associationconfig['LayerName']['assoc']['Precision']['result'] = 'ap_fixed<'+config['hls4ml_weight']['input']['quantizer']+'>'

associationconfig['LayerName']['association_0']['Precision']['assoc'] = 'ap_fixed<'+config['hls4ml_assoc']['assoc_1']['kernel_quantizer']+'>'
associationconfig['LayerName']['association_0']['Precision']['bias'] = 'ap_fixed<'+config['hls4ml_assoc']['assoc_1']['bias_quantizer']+'>'
associationconfig['LayerName']['q_activation_4']['Precision']['result'] = 'ap_ufixed<'+config['hls4ml_assoc']['assoc_1']['activation']+'>'

associationconfig['LayerName']['association_1']['Precision']['assoc'] = 'ap_fixed<'+config['hls4ml_assoc']['assoc_2']['kernel_quantizer']+'>'
associationconfig['LayerName']['association_1']['Precision']['bias'] = 'ap_fixed<'+config['hls4ml_assoc']['assoc_2']['bias_quantizer']+'>'
associationconfig['LayerName']['q_activation_5']['Precision']['result'] = 'ap_ufixed<'+config['hls4ml_assoc']['assoc_2']['activation']+'>'

associationconfig['LayerName']['association_final']['Precision']['assoc'] = 'ap_fixed<'+config['hls4ml_assoc']['assoc_final']['kernel_quantizer']+'>'
associationconfig['LayerName']['association_final']['Precision']['bias'] = 'ap_fixed<'+config['hls4ml_assoc']['assoc_final']['bias_quantizer']+'>'

cfg = hls4ml.converters.create_config(backend='Vivado')
cfg['HLSConfig']  = associationconfig
cfg['KerasModel'] = QassociationModel
cfg['OutputDir']  = '/home/cebrown/Documents/Trigger/E2E/Train/'+kf+'model_association_1_unquantised/hls4ml_prj'
cfg['XilinxPart'] = 'xcvu9p-flga2104-2L-e'
hls_association_model = hls4ml.converters.keras_to_hls(cfg)

#####################################################################################################


hls_weight_model.compile()
hls_pattern_model.compile()
hls_association_model.compile()

#hls_weight_model.build(csim=False,synth=True,vsynth=True)
#hls_pattern_model.build(csim=False,synth=True,vsynth=True)
#hls_association_model.build(csim=False,synth=True,vsynth=True)

#hls4ml.report.read_vivado_report(kf+'model_weight_1/hls4ml_prj')
#hls4ml.report.read_vivado_report(kf+'model_association_1/hls4ml_prj')
#hls4ml.report.read_vivado_report(kf+'model_pattern_1/hls4ml_prj')

############### LOAD DATA ######################################
load_data = False

if load_data:

    def decode_data(raw_data):
        decoded_data = tf.io.parse_example(raw_data,features)
        #decoded_data['trk_hitpattern'] = tf.reshape(decoded_data['trk_hitpattern'],[-1,nMaxTracks,11])
        return decoded_data

    def setup_pipeline(fileList):
        ds = tf.data.Dataset.from_tensor_slices(fileList)
        ds.shuffle(len(fileList),reshuffle_each_iteration=True)
        ds = ds.interleave(
                lambda x: tf.data.TFRecordDataset(
                    x, compression_type='GZIP', buffer_size=100000000
                ),
                cycle_length=6, 
                block_length=200, 
                num_parallel_calls=6
            )
        ds = ds.batch(200) #decode in batches (match block_length?)
        ds = ds.map(decode_data, num_parallel_calls=6)
        ds = ds.unbatch()
        ds = ds.shuffle(5000,reshuffle_each_iteration=True)
        ds = ds.batch(2000)
        ds = ds.prefetch(5)

        return ds
    features = {
                "pvz0": tf.io.FixedLenFeature([1], tf.float32),
                "trk_fromPV":tf.io.FixedLenFeature([nMaxTracks], tf.float32)
        }

    trackFeatures = [
                'trk_z0',
                'trk_fake',
                'corrected_trk_z0',
                'bit_trk_pt',
                'bit_trk_eta',
                'rescaled_bit_MVA1',
                'rescaled_bit_trk_z0_res'

            ]

    for trackFeature in trackFeatures:
        features[trackFeature] = tf.io.FixedLenFeature([nMaxTracks], tf.float32)

    test_files = glob.glob(config["data_folder"]+"OldKFData/Test/*.tfrecord")
    for step,batch in enumerate(setup_pipeline(test_files)):
        print(batch['bit_trk_pt'].numpy()[0,:])
        print(batch['bit_trk_eta'].numpy()[0,:])
        print(batch['rescaled_bit_MVA1'].numpy()[0,:])
        print(batch['rescaled_bit_trk_z0_res'].numpy()[0,:])
        print(batch['trk_fake'].numpy()[0,:])
        print(batch['trk_fromPV'].numpy()[0,:])
        print(batch['corrected_trk_z0'].numpy()[0,:])
        print(batch['trk_z0'].numpy()[0,:])
        print(batch['pvz0'].numpy()[0,:])
        #,'bit_trk_eta','rescaled_bit_MVA1','trk_fake','trk_fromPV','pvz0','corrected_trk_z0','trk_z0'
        
        single_Event_DF = pd.DataFrame({'bit_trk_pt':batch['bit_trk_pt'].numpy()[0,:],
                                        'bit_trk_eta':batch['bit_trk_eta'].numpy()[0,:],
                                        'rescaled_bit_MVA1':batch['rescaled_bit_MVA1'].numpy()[0,:],
                                        'rescaled_bit_trk_z0_res':batch['rescaled_bit_trk_z0_res'].numpy()[0,:],
                                        'trk_fake':batch['trk_fake'].numpy()[0,:],  
                                        'trk_fromPV':batch['trk_fromPV'].numpy()[0,:], 
                                        'corrected_trk_z0':batch['corrected_trk_z0'].numpy()[0,:],
                                        'trk_z0':batch['trk_z0'].numpy()[0,:],
                                        'pvz0':batch['pvz0'].numpy()[0,:]*batch['trk_fromPV'].numpy()[0,:]
                                        })

        print(single_Event_DF.head())
        single_Event_DF.to_csv('SingleEvent.csv',index=False)
        break

single_Event_DF = pd.read_csv('SingleEvent.csv')
#print(single_Event_DF)

predictedZ0_NN, predictedAssoc_NN, Weights_NN = model.predict_on_batch([np.expand_dims(single_Event_DF['corrected_trk_z0'],0),
                                                               np.expand_dims(single_Event_DF[['bit_trk_pt','rescaled_bit_MVA1','bit_trk_eta']],0),
                                                               np.expand_dims(single_Event_DF[['bit_trk_pt','rescaled_bit_MVA1','rescaled_bit_trk_z0_res','bit_trk_eta']],0)])
predictedZ0_QNN, predictedAssoc_QNN, Weights_QNN = Qmodel.predict_on_batch([np.expand_dims(single_Event_DF['corrected_trk_z0'],0),
                                                                   np.expand_dims(single_Event_DF[['bit_trk_pt','rescaled_bit_MVA1','bit_trk_eta']],0),
                                                                   np.expand_dims(single_Event_DF[['bit_trk_pt','rescaled_bit_MVA1','rescaled_bit_trk_z0_res','bit_trk_eta']],0)])
print("======================================")
print("z0 Predictions")
print("True: ",single_Event_DF['pvz0'].max())
print("NN: ",predictedZ0_NN[0,0])
print("QNN: ",predictedZ0_QNN[0,0])
print("======================================")

ConvInput_array = np.load("IntermediateArrays/ConvInputArray.npy")
AssocInput_array = np.load("IntermediateArrays/AssocInputArray.npy")
AssocInput_array  = AssocInput_array.reshape(-1, AssocInput_array.shape[-1])


#event_input = single_Event_DF.loc[0] = [0.0 ,0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
event_input = single_Event_DF[['bit_trk_pt','rescaled_bit_MVA1','bit_trk_eta']]

Detached_Weights_QNN = QweightModel.predict_on_batch(np.ascontiguousarray(event_input))
hls4ml_Weights_QNN = hls_weight_model.predict(np.ascontiguousarray(event_input))

hls4ml_pred, hls4ml_weight_trace = hls_weight_model.trace(np.ascontiguousarray(event_input))
hls4ml_pred, hls4ml_conv_trace = hls_pattern_model.trace(np.ascontiguousarray(ConvInput_array))
hls4ml_pred, hls4ml_assoc_trace = hls_association_model.trace(np.ascontiguousarray(AssocInput_array))

keras_trace = hls4ml.model.profiling.get_ymodel_keras(Qmodel,[np.expand_dims(single_Event_DF['corrected_trk_z0'],0),
                                                              np.expand_dims(event_input,0),
                                                              np.expand_dims(event_input,0)])

if print_extra:
    track = 1

    print("hsl4ml input_weight_features: ",np.ascontiguousarray(event_input)[track])
    print("qkeras input_weight_features: ",event_input.iloc[track])

    print("hsl4ml weight 1: ",hls4ml_weight_trace['weight_1'][track])
    print("qkeras weight 1: ",keras_trace['weight_1'][0][track])

    print("hsl4ml alpha 1: ",hls4ml_weight_trace['weight_1_alpha'][track])
    print("hsl4ml activation 0: ",hls4ml_weight_trace['q_activation'][track])
    print("qkeras activation 0: ",keras_trace['q_activation'][0][track])

    print("hsl4ml weight 2: ",hls4ml_weight_trace['weight_2'][track])
    print("qkeras weight 2: ",keras_trace['weight_2'][0][track])

    print("hsl4ml alpha 2: ",hls4ml_weight_trace['weight_2_alpha'][track])
    print("hsl4ml activation 1: ",hls4ml_weight_trace['q_activation_1'][track])
    print("qkeras activation 1: ",keras_trace['q_activation_1'][0][track])

    print("hsl4ml weight final: ",hls4ml_weight_trace['weight_final'][track])
    print("qkeras weight final: ",keras_trace['weight_final'][0][track])

    print("hsl4ml final alpha: ",hls4ml_weight_trace['weight_final_alpha'][track])
    print("hsl4ml activation 2: ",hls4ml_weight_trace['q_activation_2'][track])
    print("qkeras activation 2: ",keras_trace['q_activation_2'][0][track])

    #print("hsl4ml conv: ",hls4ml_conv_trace['pattern_1'][track])
    #print("qkeras conv: ",keras_trace['pattern_1'])
    #print("qkeras conv: ",keras_trace['pattern_1_function'])

    #print("hsl4ml conv alpha: ",hls4ml_conv_trace['pattern_1_alpha'][track])
    #print("hsl4ml conv activation: ",hls4ml_conv_trace['q_activation_3'][track])
    #print("qkeras activation 3: ",keras_trace['q_activation_3'])


network_weights = [hls4ml_weight_trace['q_activation'].flatten(),
                   hls4ml_weight_trace['q_activation_1'].flatten(),
                   hls4ml_weight_trace['q_activation_2'].flatten(),
                   hls4ml_conv_trace['q_activation_3'].flatten(),
                   hls4ml_assoc_trace['q_activation_4'].flatten(),
                   hls4ml_assoc_trace['q_activation_5'].flatten(),
                  ]

network_names = ['q_activation',
                 'q_activation_1',
                 'q_activation_2',
                 'q_activation_3',
                 'q_activation_4',
                 'q_activation_5',
                 ]

plt.clf()
fig,ax = plt.subplots(1,1,figsize=(20,10))
hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
ax.boxplot(network_weights,labels=network_names,notch=True,vert=False)
ax.grid(True)
ax.set_xlabel("x",ha="right",x=1)
ax.set_ylabel("Layer",ha="right",y=1)
ax.set_xscale('log', base=2)
ax.legend(loc=2) 

plt.tight_layout()
plt.savefig("Qkeras_activations_profile.png")
plt.close()

single_Event_DF['Weights_NN'] = Weights_NN[0,:,0]
single_Event_DF['Weights_QNN'] = Weights_QNN[0,:,0]
single_Event_DF['Detached_Weights_QNN'] = Detached_Weights_QNN[:,0]
single_Event_DF['hls4ml_Weights_QNN'] = hls4ml_Weights_QNN[:,0]

print("======================================")
print("Weight Predictions")
print(single_Event_DF[["Weights_NN","Weights_QNN","Detached_Weights_QNN","hls4ml_Weights_QNN"]][0:250])
print("======================================")

NNHisto = predictHisto(single_Event_DF['corrected_trk_z0'],single_Event_DF['Weights_NN'])
QNNHisto = predictHisto(single_Event_DF['corrected_trk_z0'],single_Event_DF['Weights_QNN'])
Detached_QNNHisto = predictHisto(single_Event_DF['corrected_trk_z0'],single_Event_DF['Detached_Weights_QNN'])
hls4ml_QNNHisto = predictHisto(single_Event_DF['corrected_trk_z0'],single_Event_DF['hls4ml_Weights_QNN'])

print("======================================")
print("Histogram Predictions with [1,1,1] convolution")
print("NN Histo: ",NNHisto)
print("QNN Histo: ",QNNHisto)
print("Detached QNN Histo: ",Detached_QNNHisto)
print("hls4ml QNN Histo: ",hls4ml_QNNHisto)
print("======================================")

Detached_QNNHisto_custom,Detached_QNNHisto_custom_be = predictHisto(single_Event_DF['corrected_trk_z0'],single_Event_DF['Detached_Weights_QNN'],True)
hls4ml_QNNHisto_custom,hls4ml_QNNHisto_custom_be = predictHisto(single_Event_DF['corrected_trk_z0'],single_Event_DF['hls4ml_Weights_QNN'],True)
NNHisto_custom,NNHisto_custom_be = predictHisto(single_Event_DF['corrected_trk_z0'],single_Event_DF['Weights_NN'],True)
QNNHisto_custom,QNNHisto_custom_be = predictHisto(single_Event_DF['corrected_trk_z0'],single_Event_DF['Weights_QNN'],True)

plt.clf()
fig,ax = plt.subplots(1,1,figsize=(20,10))
hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
ax.bar(NNHisto_custom_be[:-1],NNHisto_custom,width=30/256,color='b',alpha=0.5, label="Floating Point NN Weight")
ax.bar(QNNHisto_custom_be[:-1],QNNHisto_custom,width=30/256,color='g',alpha=0.5, label="Quantised NN Weight")
ax.bar(Detached_QNNHisto_custom_be[:-1],Detached_QNNHisto_custom,width=30/256,color='yellow',alpha=0.5, label="Detached QNN Weight")
ax.bar(hls4ml_QNNHisto_custom_be[:-1],hls4ml_QNNHisto_custom,width=30/256,color='pink',alpha=0.5, label="hls4ml QNN Weight")
ax.grid(True)
ax.set_xlabel("$z_0$ [cm]",ha="right",x=1)
ax.set_ylabel("Learned Weight",ha="right",y=1)
ax.set_yscale("log")
ax.legend(loc=2) 

plt.tight_layout()
plt.savefig("weights_1_event.png")
plt.close()

Detached_QNNHisto_custom_z0 = QpatternModel.predict_on_batch(np.expand_dims(Detached_QNNHisto_custom,0))
hls4ml_QNNHisto_custom_z0 = hls_pattern_model.predict(np.ascontiguousarray(hls4ml_QNNHisto_custom))

Detached_z0Index= np.argmax(Detached_QNNHisto_custom_z0)
hls4ml_z0Index= np.argmax(hls4ml_QNNHisto_custom_z0)

print("======================================")
print("Histogram Predictions with custom convolution")
print("NN Histo: ",int(((predictedZ0_NN[0,0]-halfBinWidth+15)/30.)*256)," ",predictedZ0_NN[0,0])
print("QNN Histo: ",int(((predictedZ0_QNN[0,0]-halfBinWidth+15)/30.)*256)," ",predictedZ0_QNN[0,0])
print("Detached QNN Histo: ",Detached_z0Index," ",-15.+30.*Detached_z0Index/256.+halfBinWidth)
print("hls4ml QNN Histo: ",hls4ml_z0Index," ",-15.+30.*hls4ml_z0Index/256.+halfBinWidth)
print("======================================")

Detached_Assoc_QNN = QassociationModel.predict_on_batch(np.ascontiguousarray(AssocInput_array))
hls4ml_Assoc_QNN =hls_association_model.predict(np.ascontiguousarray(AssocInput_array))

print("======================================")
print("Association Predictions")
print("NN Assoc: ",predictedAssoc_NN)
print("QNN Assoc: ",predictedAssoc_QNN)
print("Detached QNN Assoc: ",Detached_Assoc_QNN)
print("hls4ml QNN Assoc: ",hls4ml_Assoc_QNN)
print("======================================")