import glob
import sys
from textwrap import wrap

import comet_ml
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import sklearn.metrics as metrics
import tensorflow as tf
import yaml

from tensorflow.keras.models import Model

import vtx
from TrainingScripts.train import *
from EvalScripts.evalDA import *

from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.utils import _add_supported_quantized_objects
co = {}
_add_supported_quantized_objects(co)

#hep.set_style("CMSTex")
#plt.style.use([hep.style.ROOT, hep.style.firamath])

plt.style.use(hep.style.CMS)

colormap = "jet"

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

LEGEND_WIDTH = 31
LINEWIDTH = 3

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

colours=["red","green","blue","orange","purple","yellow"]

if __name__=="__main__":
    with open(sys.argv[2]+'.yaml', 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    kf = sys.argv[1]

    with open(sys.argv[2]+'.yaml', 'r') as f:
            config = yaml.load(f,Loader=yaml.FullLoader)

    if kf == "NewKF":
        test_files = glob.glob(config["data_folder"]+"NewKFData/MET/*.tfrecord")
        z0 = 'trk_z0'
        bit_z0 = 'bit_trk_z0'
    elif kf == "OldKF":
        test_files = glob.glob(config["data_folder"]+"OldKFData/MET/*.tfrecord")
        z0 = 'corrected_trk_z0'
        bit_z0 = 'bit_corrected_trk_z0'

    nMaxTracks = 250
    halfBinWidth = 0.5*30./256.

    save = True
    savingfolder = kf+"SavedArrays/"
    PVROCs = True 

    nlatent = config["Nlatent"]

    outputFolder = kf+config['eval_folder']
    trainable = config["trainable"]
    trackfeat = config["track_features"] 
    weightfeat = config["weight_features"] 


    features = {
            "pvz0": tf.io.FixedLenFeature([1], tf.float32),
            "trk_fromPV":tf.io.FixedLenFeature([nMaxTracks], tf.float32),
    }

    def decode_data(raw_data):
        decoded_data = tf.io.parse_example(raw_data,features)
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

    def predictHisto(value,weight,return_hist=False):
        
        hist,bin_edges = np.histogram(value,256,range=(-15,15),weights=weight)
        if return_hist:
            return hist,bin_edges
        else:
            hist = np.convolve(hist,[1,1,1],mode='same')
            z0Index= np.argmax(hist)
            z0 = -15.+30.*z0Index/256.+halfBinWidth
            return z0
    
    trackFeatures = [
            'trk_z0',
            'trk_pt',
            'trk_eta',
            'trk_fake',
            'corrected_trk_z0',
            'bit_trk_pt',
            'bit_trk_eta',
            'rescaled_bit_MVA1',

        ]

    for trackFeature in trackFeatures:
        features[trackFeature] = tf.io.FixedLenFeature([nMaxTracks], tf.float32)

    qnetwork = vtx.nn.E2EQKerasDiffArgMax(
            nbins=256,
            start=-15,
            end=15,
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

    qmodel = qnetwork.createE2EModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    qmodel.compile(
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
    qmodel.summary()
    qmodel.load_weights(kf+"best_weights.tf").expect_partial()

    QweightModel = qnetwork.createWeightModel()
    QpatternModel = qnetwork.createPatternModel()
    QassociationModel = qnetwork.createAssociationModel()

    QweightModel.layers[1].set_weights(qmodel.layers[1].get_weights())
    QweightModel.layers[2].set_weights(qmodel.layers[2].get_weights())
    QweightModel.layers[3].set_weights(qmodel.layers[3].get_weights())
    QweightModel.layers[4].set_weights(qmodel.layers[4].get_weights())
    QweightModel.layers[5].set_weights(qmodel.layers[5].get_weights())
    QweightModel.layers[6].set_weights(qmodel.layers[7].get_weights())

    QpatternModel.layers[1].set_weights(qmodel.layers[9].get_weights())
    QpatternModel.layers[2].set_weights(qmodel.layers[10].get_weights())

    QassociationModel.layers[1].set_weights(qmodel.layers[19].get_weights())
    QassociationModel.layers[2].set_weights(qmodel.layers[20].get_weights()) 
    QassociationModel.layers[3].set_weights(qmodel.layers[21].get_weights()) 
    QassociationModel.layers[4].set_weights(qmodel.layers[22].get_weights()) 
    QassociationModel.layers[5].set_weights(qmodel.layers[23].get_weights()) 

    #qmodel.layers[15].set_weights([np.array([[1]], dtype=np.float32), np.array([-halfBinWidth], dtype=np.float32)])

    DAnetwork = vtx.nn.E2EDiffArgMax(
            nbins=256,
            start=-15,
            end=15,
            ntracks=max_ntracks, 
            nweightfeatures=len(weightfeat), 
            nfeatures=len(trackfeat), 
            nweights=1, 
            nlatent = nlatent,
            activation='relu',
            regloss=1e-10
        )
        
    DAmodel = DAnetwork.createE2EModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    DAmodel.compile(
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
    DAmodel.summary()
    DAmodel.load_weights(kf + "best_weights_unquantised.tf").expect_partial()

    ####################################################################################################

    predictedZ0_FH = []
    predictedZ0_QNN = []
    predictedZ0_DANN = []
    predictedZ0_splitQNN = []
    predictedZ0_hls4mlNN = []

    predictedAssoc_QNN = []
    predictedAssoc_DANN = []
    predictedAssoc_FH = []
    predictedAssoc_splitQNN = []
    predictedAssoc_hls4mlNN = []

    actual_Assoc = []
    actual_PV = []

    if save:
        for step,batch in enumerate(setup_pipeline(test_files)):

            if step > 0:
                break

            trackFeatures = np.stack([batch[feature] for feature in trackfeat],axis=2)
            WeightFeatures = np.stack([batch[feature] for feature in weightfeat],axis=2)
            nBatch = batch['pvz0'].shape[0]
            FH = predictFastHisto(batch[z0],batch['trk_pt'])
            predictedZ0_FH.append(FH)


            actual_Assoc.append(batch["trk_fromPV"])
            actual_PV.append(batch['pvz0'])
            FHassoc = FastHistoAssoc(predictFastHisto(batch[z0],batch['trk_pt']),batch[z0],batch['trk_eta'],kf)
            predictedAssoc_FH.append(FHassoc)

            #### Q NETWORK #########################################################################################################################

            predictedZ0_QNN_temp, predictedAssoc_QNN_temp, QWeights_QNN = qmodel.predict_on_batch(
                            [batch[z0],WeightFeatures,trackFeatures]
                        )

            predictedAssoc_QNN_temp = tf.math.divide( tf.math.subtract( predictedAssoc_QNN_temp,tf.reduce_min(predictedAssoc_QNN_temp)), 
                                                    tf.math.subtract( tf.reduce_max(predictedAssoc_QNN_temp), tf.reduce_min(predictedAssoc_QNN_temp) ))

            predictedZ0_QNN.append(predictedZ0_QNN_temp)
            predictedAssoc_QNN.append(predictedAssoc_QNN_temp)

            #### DA NETWORK #########################################################################################################################

            predictedZ0_DANN_temp, predictedAssoc_DANN_temp, DAWeights_DANN = DAmodel.predict_on_batch(
                                [batch[z0],WeightFeatures,trackFeatures]
                            )

            predictedAssoc_DANN_temp = tf.math.divide( tf.math.subtract( predictedAssoc_DANN_temp,tf.reduce_min(predictedAssoc_DANN_temp)), 
                                                    tf.math.subtract( tf.reduce_max(predictedAssoc_DANN_temp), tf.reduce_min(predictedAssoc_DANN_temp) ))

            predictedZ0_DANN.append(predictedZ0_DANN_temp)
            predictedAssoc_DANN.append(predictedAssoc_DANN_temp)

            ### Split NETWORK ###############################################################################################################################
            for i,event in enumerate(batch[z0]):
                if (i % 200 == 0):
                    print("Batch: ",step," event: ", i, " out of 2000")
                Detached_Weights_QNN = QweightModel.predict_on_batch(WeightFeatures[i])
                Detached_QNNHisto_custom,Detached_QNNHisto_custom_be = predictHisto(batch[z0][i],Detached_Weights_QNN[:,0],True)
                Detached_QNNHisto_custom_z0 = QpatternModel.predict_on_batch(np.expand_dims(Detached_QNNHisto_custom,0))
                Detached_z0Index = np.argmax(Detached_QNNHisto_custom_z0)
                Detached_Z0 = -15.+30.*Detached_z0Index/256.+halfBinWidth
                predictedZ0_splitQNN.append(Detached_Z0)
                zdiff = abs(batch[z0][i] - Detached_Z0)
                assoc_inputs = np.column_stack((zdiff.numpy(),trackFeatures[i]))
                Detached_Assoc_QNN = QassociationModel.predict_on_batch(assoc_inputs)

                predictedAssoc_splitQNN_temp = tf.math.divide( tf.math.subtract( Detached_Assoc_QNN,tf.reduce_min(Detached_Assoc_QNN)), 
                                                        tf.math.subtract( tf.reduce_max(Detached_Assoc_QNN), tf.reduce_min(Detached_Assoc_QNN) ))

                predictedAssoc_splitQNN.append(predictedAssoc_splitQNN_temp)

                predictedZ0_hls4mlNN.append(Detached_Z0)
                predictedAssoc_hls4mlNN.append(predictedAssoc_splitQNN_temp)

        z0_QNN_array = np.concatenate(predictedZ0_QNN).ravel()
        z0_DANN_array = np.concatenate(predictedZ0_DANN).ravel()
        z0_splitQNN_array = predictedZ0_splitQNN
        z0_hls4mlNN_array = predictedZ0_hls4mlNN
        z0_FH_array = np.concatenate(predictedZ0_FH).ravel()
        z0_PV_array = np.concatenate(actual_PV).ravel()

        assoc_QNN_array = np.concatenate(predictedAssoc_QNN).ravel()
        assoc_DANN_array = np.concatenate(predictedAssoc_DANN).ravel()
        assoc_splitQNN_array = np.concatenate(predictedAssoc_splitQNN).ravel()
        assoc_hls4mlNN_array = np.concatenate(predictedAssoc_hls4mlNN).ravel()
        assoc_FH_array = np.concatenate(predictedAssoc_FH).ravel()
        assoc_PV_array = np.concatenate(actual_Assoc).ravel()

        np.save(savingfolder+"z0_QNN_array",z0_QNN_array)
        np.save(savingfolder+"z0_DANN_array",z0_DANN_array)
        np.save(savingfolder+"z0_splitQNN_array",z0_splitQNN_array)
        np.save(savingfolder+"z0_hls4mlNN_array",z0_hls4mlNN_array)
        np.save(savingfolder+"z0_FH_array",z0_FH_array)
        np.save(savingfolder+"z0_PV_array",assoc_PV_array)

        np.save(savingfolder+"assoc_QNN_array",assoc_QNN_array)
        np.save(savingfolder+"assoc_DANN_array",assoc_DANN_array)
        np.save(savingfolder+"assoc_splitQNN_array",assoc_splitQNN_array)
        np.save(savingfolder+"assoc_hls4mlNN_array",assoc_hls4mlNN_array)
        np.save(savingfolder+"assoc_FH_array",assoc_FH_array)
        np.save(savingfolder+"assoc_PV_array",assoc_PV_array)
        
    else:
        z0_QNN_array = np.load(savingfolder+"z0_QNN_array.npy")
        z0_DANN_array = np.load(savingfolder+"z0_DANN_array.npy")
        z0_splitQNN_array = np.load(savingfolder+"z0_splitQNN_array.npy")
        z0_hls4mlNN_array = np.load(savingfolder+"z0_hls4mlNN_array.npy")

        z0_FH_array = np.load(savingfolder+"z0_FH_array.npy")
        z0_PV_array = np.load(savingfolder+"z0_PV_array.npy")
       
        assoc_QNN_array = np.load(savingfolder+"assoc_QNN_array.npy")
        assoc_DANN_array = np.load(savingfolder+"assoc_DANN_array.npy")
        assoc_splitQNN_array = np.load(savingfolder+"assoc_splitQNN_array.npy")
        assoc_hls4mlNN_array = np.load(savingfolder+"assoc_hls4mlNN_array.npy")

        assoc_FH_array = np.load(savingfolder+"assoc_FH_array.npy")
        assoc_PV_array = np.load(savingfolder+"assoc_PV_array.npy")

    #########################################################################################
    #                                                                                       #
    #                                    Z0 Residual Plots                                  #  
    #                                                                                       #
    #########################################################################################

    #plt.clf()
    figure=plotz0_residual([(z0_PV_array-z0_QNN_array),(z0_PV_array-z0_DANN_array),(z0_PV_array-z0_splitQNN_array),(z0_PV_array-z0_hls4mlNN_array)],
                          [(z0_PV_array-z0_FH_array)],
                          ["NN","QNN","SplitQNN","HLS4ML"],
                          ["Baseline"])
    plt.savefig("Z0Residual.png")
    plt.close()

    plt.clf()
    figure=plotPV_roc(assoc_PV_array,[assoc_DANN_array,assoc_QNN_array,assoc_splitQNN_array,assoc_hls4mlNN_array],
                        [assoc_FH_array],
                        ["NN","QNN","SplitQNN","HLS4ML"],
                        ["Baseline"],Nthresholds=10)
    plt.savefig("PVROC.png")
    plt.close()
