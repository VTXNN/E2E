import glob
import sys
import os
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
from EvalScripts.eval_funcs import *

from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.utils import _add_supported_quantized_objects
co = {}
_add_supported_quantized_objects(co)

nMaxTracks = 250
max_z0 = 20.46912512

def decode_data(raw_data):
    decoded_data = tf.io.parse_example(raw_data,features)
    #decoded_data['trk_hitpattern'] = tf.reshape(decoded_data['trk_hitpattern'],[-1,max_ntracks,11])
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

if __name__=="__main__":
    with open(sys.argv[1]+'.yaml', 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    test_files = glob.glob(config["data_folder"]+"/Test/*.tfrecord")
    z0 = 'int_z0'
    FH_z0 = 'trk_z0'
    start = 0
    end = 255
    bit = True

    save = True
    savingfolder = "SavedArrays/"
    PVROCs = True
    met = False

    nlatent = config["Nlatent"]
    nbins = config['nbins']

    with open('experimentkey.txt') as f:
        first_line = f.readline()

    EXPERIMENT_KEY = first_line

    if (EXPERIMENT_KEY is not None):
        # There is one, but the experiment might not exist yet:
        api = comet_ml.API() # Assumes API key is set in config/env
        try:
            api_experiment = api.get_experiment_by_key(EXPERIMENT_KEY)
        except Exception:
            api_experiment = None

    experiment = comet_ml.ExistingExperiment(
            previous_experiment=EXPERIMENT_KEY,
            log_env_details=True, # to continue env logging
            log_env_gpu=True,     # to continue GPU logging
            log_env_cpu=True,     # to continue CPU logging
        )

    outputFolder = 'plots'
    trackfeat = config["track_features"] 
    weightfeat = config["weight_features"] 

    QuantisedPrunedModelName = config["QuantisedPrunedModelName"] 
    QuantisedModelName = config["QuantisedModelName"] + "_prune_iteration_0"
    UnQuantisedModelName = config["UnquantisedModelName"] 

    features = {
            "pvz0": tf.io.FixedLenFeature([1], tf.float32),
            "trk_fromPV":tf.io.FixedLenFeature([nMaxTracks], tf.float32)
    }

    trackFeatures = [
            'trk_word_chi2rphi', 
            'trk_word_chi2rz', 
            'trk_word_bendchi2',
            'trk_z0_res',
            'trk_gtt_pt',
            'trk_gtt_eta',
            'trk_gtt_phi',
            'trk_fake',
            'trk_z0',
            'int_z0',
            'trk_class_weight',
            'abs_trk_word_pT',
            'abs_trk_word_eta',
            'trk_word_MVAquality',
            'rescaled_trk_word_pT',
            'rescaled_trk_word_eta',
            'rescaled_trk_z0_res'
        ]

    for trackFeature in trackFeatures:
        features[trackFeature] = tf.io.FixedLenFeature([nMaxTracks], tf.float32)

    qnetwork = vtx.nn.E2EQKerasDiffArgMax(
            nbins=nbins,
            start=start,
            end=end,
            max_z0 = max_z0,
            ntracks=nMaxTracks, 
            nweightfeatures=len(weightfeat), 
            nfeatures=len(trackfeat), 
            nweights=1, 
            nlatent = nlatent,
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

    DAnetwork = vtx.nn.E2EDiffArgMax(
            nbins=nbins,
            start=start,
            end=end,
            max_z0 = max_z0,
            ntracks=nMaxTracks, 
            nweightfeatures=len(weightfeat), 
            nfeatures=len(trackfeat), 
            nweights=1, 
            nlatent = nlatent,
            l2regloss=1e-10
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

    QPnetwork = vtx.nn.E2EQKerasDiffArgMaxConstraint(
            nbins=nbins,
            ntracks=nMaxTracks, 
            start=start,
            end=end,
            max_z0 = max_z0,
            nweightfeatures=len(config["weight_features"]),  
            nfeatures=len(config["track_features"]), 
            nweights=1, 
            nlatent = nlatent,
            l1regloss = (float)(config['l1regloss']),
            l2regloss = (float)(config['l2regloss']),
            nweightnodes = config['nweightnodes'],
            nweightlayers = config['nweightlayers'],
            nassocnodes = config['nassocnodes'],
            nassoclayers = config['nassoclayers'],
            temperature = 1e-4,
            qconfig = config['QConfig'],
            h5fName = config['QuantisedModelName']+'_drop_weights_iteration_'+str(config['prune_iterations'])+'.h5'
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

    DAmodel.load_weights(UnQuantisedModelName+".tf").expect_partial()
    qmodel.load_weights(QuantisedModelName+".tf").expect_partial()
    QPmodel.load_weights(QuantisedPrunedModelName+".tf").expect_partial()

    predictedZ0_FH = []
    predictedZ0_FHz0res = []
    predictedZ0_FHz0MVA = []
    predictedZ0_FHnoFake = []
    predictedZ0_QNN = []
    predictedZ0_QPNN = []
    predictedZ0_DANN = []

    predictedQWeights = []
    predictedQPWeights = []
    predictedDAWeights = []

    predictedAssoc_QNN = []
    predictedAssoc_QPNN = []
    predictedAssoc_DANN = []
    predictedAssoc_FH = []
    predictedAssoc_FHres = []
    predictedAssoc_FHMVA = []
    predictedAssoc_FHnoFake = []

    num_threshold = 10
    thresholds = [str(i/num_threshold) for i in range(0,num_threshold)]

    predictedMET_QNN = {key: value for key, value in zip(thresholds, [[] for i in range(0,num_threshold)])}
    predictedMET_QPNN = {key: value for key, value in zip(thresholds, [[] for i in range(0,num_threshold)])}
    predictedMETphi_QNN = {key: value for key, value in zip(thresholds, [[] for i in range(0,num_threshold)])}
    predictedMETphi_QPNN = {key: value for key, value in zip(thresholds, [[] for i in range(0,num_threshold)])}
    predictedMET_DANN = {key: value for key, value in zip(thresholds, [[] for i in range(0,num_threshold)])}
    predictedMETphi_DANN = {key: value for key, value in zip(thresholds, [[] for i in range(0,num_threshold)])}

    predictedMET_FH = []
    predictedMET_FHres = []
    predictedMET_FHMVA = []
    predictedMET_FHnoFake = []

    predictedMETphi_FH = []
    predictedMETphi_FHres = []
    predictedMETphi_FHMVA = []
    predictedMETphi_FHnoFake = []

    actual_Assoc = []
    actual_PV = []
    actual_MET = []
    actual_METphi = []
    actual_trkMET = []
    actual_trkMETphi = []

    trk_z0 = []
    trk_MVA = []
    trk_gtt_phi = []
    trk_gtt_pt = []
    trk_gtt_eta = []
    trk_z0_res = []

    trk_chi2rphi = []
    trk_chi2rz = []
    trk_bendchi2 = []

    threshold = -1

    if save:
        for step,batch in enumerate(setup_pipeline(test_files)):

            trackFeatures = np.stack([batch[feature] for feature in trackfeat],axis=2)
            WeightFeatures = np.stack([batch[feature] for feature in weightfeat],axis=2)
            nBatch = batch['pvz0'].shape[0]

            FH = predictFastHisto(batch[FH_z0],batch['abs_trk_word_pT'],linear_res_function(batch['abs_trk_word_pT']))
            predictedZ0_FH.append(FH)
            FHeta = predictFastHisto(batch[FH_z0],batch['abs_trk_word_pT'],eta_res_function(batch['trk_gtt_eta']))
            predictedZ0_FHz0res.append(FHeta)
            FHz0MVA = predictFastHisto(batch[FH_z0],batch['abs_trk_word_pT'],MVA_res_function(batch['trk_word_MVAquality']))
            predictedZ0_FHz0MVA.append(FHz0MVA)
            FHnoFake = predictFastHisto(batch[FH_z0],batch['abs_trk_word_pT'],fake_res_function(batch['trk_fake']))
            predictedZ0_FHnoFake.append(FHnoFake)

            trk_z0.append(batch[FH_z0])
            trk_MVA.append(batch["trk_word_MVAquality"])
            trk_gtt_pt.append(batch['abs_trk_word_pT'])
            trk_gtt_eta.append(batch['trk_gtt_eta'])
            trk_gtt_phi.append(batch['trk_gtt_phi'])
            trk_z0_res.append(batch['rescaled_trk_z0_res'])

            trk_chi2rphi.append(batch['trk_word_chi2rphi'])
            trk_chi2rz.append(batch['trk_word_chi2rz'])
            trk_bendchi2.append(batch['trk_word_bendchi2'])

            actual_Assoc.append(batch["trk_fromPV"])
            actual_PV.append(batch['pvz0'])

            FHassoc = FastHistoAssoc(FH,batch[FH_z0],batch['trk_gtt_eta'],linear_res_function(batch['trk_gtt_eta'],return_bool=True))
            predictedAssoc_FH.append(FHassoc)

            FHassocres = FastHistoAssoc(FHeta,batch[FH_z0],batch['trk_gtt_eta'],linear_res_function(batch['trk_gtt_eta'],return_bool=True))
            predictedAssoc_FHres.append(FHassocres)

            FHassocMVA = FastHistoAssoc(FHz0MVA,batch[FH_z0],batch['trk_gtt_eta'],MVA_res_function(batch['trk_word_MVAquality'],return_bool=True))
            predictedAssoc_FHMVA.append(FHassocMVA)

            FHassocnoFake = FastHistoAssoc(FHnoFake,batch[FH_z0],batch['trk_gtt_eta'],fake_res_function(batch['trk_fake'],return_bool=True))
            predictedAssoc_FHnoFake.append(FHassocnoFake)

            #for i,event in enumerate(batch[z0]):
            #    if abs(FH[i] - batch['pvz0'][i]) > 100:
            #        figure = plotKDEandTracks(batch['trk_z0'][i],batch['trk_fake'][i],batch['pvz0'][i],FH[i],batch['trk_gtt_pt'][i],weight_label="Baseline",threshold=0.5)
            #        plt.savefig("%s/event.png" % outputFolder)
            #        break

            #### Q NETWORK #########################################################################################################################
            XX = qmodel.input 
            YY = qmodel.layers[5].output
            new_model = Model(XX, YY)

            predictedQWeights_QNN = new_model.predict_on_batch(
                            [batch[z0],WeightFeatures,trackFeatures])

            predictedZ0_QNN_temp, predictedAssoc_QNN_temp, QWeights_QNN = qmodel.predict_on_batch(
                            [batch[z0],WeightFeatures,trackFeatures]
                        )

            predictedAssoc_QNN_temp = tf.math.divide( tf.math.subtract( predictedAssoc_QNN_temp,tf.reduce_min(predictedAssoc_QNN_temp)), 
                                                    tf.math.subtract( tf.reduce_max(predictedAssoc_QNN_temp), tf.reduce_min(predictedAssoc_QNN_temp) ))

            #predictedAssoc_QNN_temp = tf.math.divide( tf.math.subtract( predictedAssoc_QNN_temp,-5),15)

            predictedZ0_QNN.append(predictedZ0_QNN_temp)
            predictedAssoc_QNN.append(predictedAssoc_QNN_temp)
            predictedQWeights.append(predictedQWeights_QNN)

            #### QP NETWORK #########################################################################################################################
            XX = QPmodel.input 
            YY = QPmodel.layers[5].output
            new_model = Model(XX, YY)

            predictedQWeights_QPNN = new_model.predict_on_batch(
                            [batch[z0],WeightFeatures,trackFeatures])

            predictedZ0_QPNN_temp, predictedAssoc_QPNN_temp, QWeights_QPNN = QPmodel.predict_on_batch(
                            [batch[z0],WeightFeatures,trackFeatures]
                        )

            predictedAssoc_QPNN_temp = tf.math.divide( tf.math.subtract( predictedAssoc_QPNN_temp,tf.reduce_min(predictedAssoc_QPNN_temp)), 
                                                    tf.math.subtract( tf.reduce_max(predictedAssoc_QPNN_temp), tf.reduce_min(predictedAssoc_QPNN_temp) ))

            #predictedAssoc_QPNN_temp = tf.math.divide( tf.math.subtract( predictedAssoc_QPNN_temp,-5),15)

            predictedZ0_QPNN.append(predictedZ0_QPNN_temp)
            predictedAssoc_QPNN.append(predictedAssoc_QPNN_temp)
            predictedQPWeights.append(predictedQWeights_QPNN)

            #### DA NETWORK #########################################################################################################################
            XX = DAmodel.input 
            YY = DAmodel.layers[9].output
            new_model = Model(XX, YY)

            predictedDAWeights_DANN = new_model.predict_on_batch(
                                [batch[z0],WeightFeatures,trackFeatures])


            predictedZ0_DANN_temp, predictedAssoc_DANN_temp, DAWeights_DANN = DAmodel.predict_on_batch(
                                [batch[z0],WeightFeatures,trackFeatures]
                            )

            predictedAssoc_DANN_temp = tf.math.divide( tf.math.subtract( predictedAssoc_DANN_temp,tf.reduce_min(predictedAssoc_DANN_temp)), 
                                                    tf.math.subtract( tf.reduce_max(predictedAssoc_DANN_temp), tf.reduce_min(predictedAssoc_DANN_temp) ))

            #predictedAssoc_DANN_temp = tf.math.divide( tf.math.subtract( predictedAssoc_DANN_temp,-5),15)

            predictedZ0_DANN.append(predictedZ0_DANN_temp)
            predictedAssoc_DANN.append(predictedAssoc_DANN_temp)
            predictedDAWeights.append(predictedDAWeights_DANN)

            #actual_MET.append(batch['tp_met_pt'])
            #actual_METphi.append(batch['tp_met_phi'])

            temp_met,temp_metphi = predictMET(batch['abs_trk_word_pT'],batch['trk_gtt_phi'],batch['trk_fromPV'],threshold=0.5,quality_func=linear_res_function(batch['trk_gtt_eta'],return_bool=True))
            actual_trkMET.append(temp_met)
            actual_trkMETphi.append(temp_metphi)

            temp_met,temp_metphi = predictMET(batch['abs_trk_word_pT'],batch['trk_gtt_phi'],FHassocnoFake,threshold=0.5,quality_func=linear_res_function(batch['trk_gtt_eta'],return_bool=True))
            predictedMET_FHnoFake.append(temp_met)
            predictedMETphi_FHnoFake.append(temp_metphi)

            temp_met,temp_metphi = predictMET(batch['abs_trk_word_pT'],batch['trk_gtt_phi'],FHassocMVA,threshold=0.5,quality_func=MVA_res_function(batch['trk_word_MVAquality'],return_bool=True))
            predictedMET_FHMVA.append(temp_met)
            predictedMETphi_FHMVA.append(temp_metphi)

            temp_met,temp_metphi = predictMET(batch['abs_trk_word_pT'],batch['trk_gtt_phi'],FHassocres,
                                              threshold=0.5, quality_func=chi_res_function(batch['trk_word_chi2rphi'], batch['trk_word_chi2rz'], batch['trk_word_bendchi2'],return_bool=True))
            predictedMET_FHres.append(temp_met)
            predictedMETphi_FHres.append(temp_metphi)

            temp_met,temp_metphi = predictMET(batch['abs_trk_word_pT'],batch['trk_gtt_phi'],FHassoc,
                                              threshold=0.5, quality_func=chi_res_function(batch['trk_word_chi2rphi'], batch['trk_word_chi2rz'], batch['trk_word_bendchi2'],return_bool=True))
            predictedMET_FH.append(temp_met)
            predictedMETphi_FH.append(temp_metphi)

            if met:
                for i in range(0,num_threshold):
                    temp_met,temp_metphi = predictMET(batch['abs_trk_word_pT'],batch['trk_gtt_phi'],predictedAssoc_QNN_temp.numpy().squeeze(),threshold=i/num_threshold, quality_func=chi_res_function(batch['trk_word_chi2rphi'], batch['trk_word_chi2rz'], batch['trk_word_bendchi2'],return_bool=True))
                    predictedMET_QNN[str(i/num_threshold)].append(temp_met)
                    predictedMETphi_QNN[str(i/num_threshold)].append(temp_metphi)

                    temp_met,temp_metphi = predictMET(batch['abs_trk_word_pT'],batch['trk_gtt_phi'],predictedAssoc_QPNN_temp.numpy().squeeze(),threshold=i/num_threshold,  quality_func=chi_res_function(batch['trk_word_chi2rphi'], batch['trk_word_chi2rz'], batch['trk_word_bendchi2'],return_bool=True))
                    predictedMET_QPNN[str(i/num_threshold)].append(temp_met)
                    predictedMETphi_QPNN[str(i/num_threshold)].append(temp_metphi)

                    temp_met,temp_metphi = predictMET(batch['abs_trk_word_pT'],batch['trk_gtt_phi'],predictedAssoc_DANN_temp.numpy().squeeze(),threshold=i/num_threshold,  quality_func=chi_res_function(batch['trk_word_chi2rphi'], batch['trk_word_chi2rz'], batch['trk_word_bendchi2'],return_bool=True))
                    predictedMET_DANN[str(i/num_threshold)].append(temp_met)
                    predictedMETphi_DANN[str(i/num_threshold)].append(temp_metphi)

        z0_QNN_array = np.concatenate(predictedZ0_QNN).ravel()
        z0_QPNN_array = np.concatenate(predictedZ0_QPNN).ravel()
        z0_DANN_array = np.concatenate(predictedZ0_DANN).ravel()

        z0_FH_array = np.concatenate(predictedZ0_FH).ravel()
        z0_FHres_array = np.concatenate(predictedZ0_FHz0res).ravel()
        z0_FHMVA_array = np.concatenate(predictedZ0_FHz0MVA).ravel()
        z0_FHnoFake_array = np.concatenate(predictedZ0_FHnoFake).ravel()
        z0_PV_array = np.concatenate(actual_PV).ravel()

        predictedQWeightsarray = np.concatenate(predictedQWeights).ravel()
        predictedQPWeightsarray = np.concatenate(predictedQPWeights).ravel()
        predictedDAWeightsarray = np.concatenate(predictedDAWeights).ravel()

        trk_z0_array = np.concatenate(trk_z0).ravel()
        trk_mva_array = np.concatenate(trk_MVA).ravel()
        trk_gtt_pt_array = np.concatenate(trk_gtt_pt).ravel()
        trk_gtt_eta_array = np.concatenate(trk_gtt_eta).ravel()
        trk_z0_res_array = np.concatenate(trk_z0_res).ravel()
        trk_gtt_phi_array = np.concatenate(trk_gtt_phi).ravel()

        trk_chi2rphi_array = np.concatenate(trk_chi2rphi).ravel()
        trk_chi2rz_array = np.concatenate(trk_chi2rz).ravel()
        trk_bendchi2_array = np.concatenate(trk_bendchi2).ravel()

        assoc_QNN_array = np.concatenate(predictedAssoc_QNN).ravel()
        assoc_QPNN_array = np.concatenate(predictedAssoc_QPNN).ravel()
        assoc_DANN_array = np.concatenate(predictedAssoc_DANN).ravel()

        assoc_FH_array = np.concatenate(predictedAssoc_FH).ravel()
        assoc_FHres_array = np.concatenate(predictedAssoc_FHres).ravel()
        assoc_FHMVA_array = np.concatenate(predictedAssoc_FHMVA).ravel()
        assoc_FHnoFake_array = np.concatenate(predictedAssoc_FHnoFake).ravel()
        assoc_PV_array = np.concatenate(actual_Assoc).ravel()

        if met:

            actual_MET_array = np.concatenate(actual_MET).ravel()
            actual_METphi_array = np.concatenate(actual_METphi).ravel()
            actual_trkMET_array = np.concatenate(actual_trkMET).ravel()
            actual_trkMETphi_array = np.concatenate(actual_trkMETphi).ravel()
            MET_FHnoFake_array = np.concatenate(predictedMET_FHnoFake).ravel()
            METphi_FHnoFake_array = np.concatenate(predictedMETphi_FHnoFake).ravel()
            MET_FHMVA_array = np.concatenate(predictedMET_FHMVA).ravel()
            METphi_FHMVA_array = np.concatenate(predictedMETphi_FHMVA).ravel()
            MET_FHres_array = np.concatenate(predictedMET_FHres).ravel()
            METphi_FHres_array = np.concatenate(predictedMETphi_FHres).ravel()
            MET_FH_array = np.concatenate(predictedMET_FH).ravel()
            METphi_FH_array = np.concatenate(predictedMETphi_FH).ravel()

            MET_QNN_RMS_array = np.zeros([num_threshold])
            MET_QNN_Quartile_array = np.zeros([num_threshold])
            MET_QNN_Centre_array = np.zeros([num_threshold])
            METphi_QNN_RMS_array = np.zeros([num_threshold])
            METphi_QNN_Quartile_array = np.zeros([num_threshold])
            METphi_QNN_Centre_array = np.zeros([num_threshold])

            MET_DANN_RMS_array = np.zeros([num_threshold])
            MET_DANN_Quartile_array = np.zeros([num_threshold])
            MET_DANN_Centre_array = np.zeros([num_threshold])
            METphi_DANN_RMS_array = np.zeros([num_threshold])
            METphi_DANN_Quartile_array = np.zeros([num_threshold])
            METphi_DANN_Centre_array = np.zeros([num_threshold])

            for i in range(0,num_threshold):
                MET_QNN_array = np.concatenate(predictedMET_QNN[str(i/num_threshold)]).ravel()
                METphi_QNN_array = np.concatenate(predictedMETphi_QNN[str(i/num_threshold)]).ravel()

                Diff = MET_QNN_array - actual_MET_array
                PhiDiff = METphi_QNN_array - actual_METphi_array

                MET_QNN_RMS_array[i] = np.sqrt(np.mean(Diff**2))
                METphi_QNN_RMS_array[i] = np.sqrt(np.mean(PhiDiff**2))

                qMET = np.percentile(Diff,[32,50,68])
                qMETphi = np.percentile(PhiDiff,[32,50,68])

                MET_QNN_Quartile_array[i] = qMET[2] - qMET[0]
                METphi_QNN_Quartile_array[i] = qMETphi[2] - qMETphi[0]

                MET_QNN_Centre_array[i] = qMET[1]
                METphi_QNN_Centre_array[i] = qMETphi[1]

                MET_DANN_array = np.concatenate(predictedMET_DANN[str(i/num_threshold)]).ravel()
                METphi_DANN_array = np.concatenate(predictedMETphi_DANN[str(i/num_threshold)]).ravel()

                Diff = MET_DANN_array - actual_MET_array
                PhiDiff = METphi_DANN_array - actual_METphi_array

                MET_DANN_RMS_array[i] = np.sqrt(np.mean(Diff**2))
                METphi_DANN_RMS_array[i] = np.sqrt(np.mean(PhiDiff**2))

                qMET = np.percentile(Diff,[32,50,68])
                qMETphi = np.percentile(PhiDiff,[32,50,68])

                MET_DANN_Quartile_array[i] = qMET[2] - qMET[0]
                METphi_DANN_Quartile_array[i] = qMETphi[2] - qMETphi[0]

                MET_DANN_Centre_array[i] = qMET[1]
                METphi_DANN_Centre_array[i] = qMETphi[1]

            Quartilethreshold_choice = '0.7'#str(np.argmin(MET_QNN_Quartile_array)/num_threshold)
            RMSthreshold_choice= '0.6'#str(np.argmin(MET_QNN_RMS_array)/num_threshold)
            Quartilethreshold_choice_DNN = '0.7' #str(np.argmin(MET_DANN_Quartile_array)/num_threshold)
            RMSthreshold_choice_DNN = '0.6' #str(np.argmin(MET_DANN_RMS_array)/num_threshold)
            
            MET_QNN_bestQ_array = np.concatenate(predictedMET_QNN[Quartilethreshold_choice]).ravel()
            METphi_QNN_bestQ_array = np.concatenate(predictedMETphi_QNN[Quartilethreshold_choice]).ravel()

            MET_QNN_bestRMS_array = np.concatenate(predictedMET_QNN[RMSthreshold_choice]).ravel()
            METphi_QNN_bestRMS_array = np.concatenate(predictedMETphi_QNN[RMSthreshold_choice]).ravel()

            MET_DANN_bestQ_array = np.concatenate(predictedMET_DANN[Quartilethreshold_choice_DNN]).ravel()
            METphi_DANN_bestQ_array = np.concatenate(predictedMETphi_DANN[Quartilethreshold_choice_DNN]).ravel()

            MET_DANN_bestRMS_array = np.concatenate(predictedMET_DANN[RMSthreshold_choice_DNN]).ravel()
            METphi_DANN_bestRMS_array = np.concatenate(predictedMETphi_DANN[RMSthreshold_choice_DNN]).ravel()

        np.save(savingfolder+"z0_QNN_array",z0_QNN_array)
        np.save(savingfolder+"z0_QPNN_array",z0_QPNN_array)
        np.save(savingfolder+"z0_DANN_array",z0_DANN_array)
        np.save(savingfolder+"z0_FH_array",z0_FH_array)
        np.save(savingfolder+"z0_FHres_array",z0_FHres_array)
        np.save(savingfolder+"z0_FHMVA_array",z0_FHMVA_array)
        np.save(savingfolder+"z0_FHnoFake_array",z0_FHnoFake_array)
        np.save(savingfolder+"z0_PV_array",z0_PV_array)
        np.save(savingfolder+"predictedQWeightsarray",predictedQWeightsarray)
        np.save(savingfolder+"predictedQPWeightsarray",predictedQPWeightsarray)
        np.save(savingfolder+"predictedDAWeightsarray",predictedDAWeightsarray)
        np.save(savingfolder+"trk_z0_array",trk_z0_array)
        np.save(savingfolder+"trk_mva_array",trk_mva_array)
        np.save(savingfolder+"trk_gtt_pt_array",trk_gtt_pt_array)
        np.save(savingfolder+"trk_gtt_eta_array",trk_gtt_eta_array)
        np.save(savingfolder+"trk_z0_res_array",trk_z0_res_array)
        np.save(savingfolder+"trk_gtt_phi_array",trk_gtt_phi_array)
        np.save(savingfolder+"trk_chi2rphi_array",trk_chi2rphi_array)
        np.save(savingfolder+"trk_chi2rz_array",trk_chi2rz_array)
        np.save(savingfolder+"trk_bendchi2_array",trk_bendchi2_array)
        np.save(savingfolder+"assoc_QNN_array",assoc_QNN_array)
        np.save(savingfolder+"assoc_QPNN_array",assoc_QPNN_array)
        np.save(savingfolder+"assoc_DANN_array",assoc_DANN_array)
        np.save(savingfolder+"assoc_FH_array",assoc_FH_array)
        np.save(savingfolder+"assoc_FHres_array",assoc_FHres_array)
        np.save(savingfolder+"assoc_FHMVA_array",assoc_FHMVA_array)
        np.save(savingfolder+"assoc_FHnoFake_array",assoc_FHnoFake_array)
        np.save(savingfolder+"assoc_PV_array",assoc_PV_array)
        if met:
            np.save(savingfolder+"actual_MET_array",actual_MET_array)
            np.save(savingfolder+"actual_METphi_array",actual_METphi_array)
            np.save(savingfolder+"actual_trkMET_array",actual_trkMET_array)
            np.save(savingfolder+"actual_trkMETphi_array",actual_trkMETphi_array)
            np.save(savingfolder+"MET_FHnoFake_array",MET_FHnoFake_array)
            np.save(savingfolder+"METphi_FHnoFake_array",METphi_FHnoFake_array)
            np.save(savingfolder+"MET_FHMVA_array",MET_FHMVA_array)
            np.save(savingfolder+"METphi_FHMVA_array",METphi_FHMVA_array)
            np.save(savingfolder+"MET_FHres_array",MET_FHres_array)
            np.save(savingfolder+"METphi_FHres_array",METphi_FHres_array)
            np.save(savingfolder+"MET_FH_array",MET_FH_array)
            np.save(savingfolder+"METphi_FH_array",METphi_FH_array)
            np.save(savingfolder+"MET_QNN_RMS_array",MET_QNN_RMS_array)
            np.save(savingfolder+"MET_QNN_Quartile_array",MET_QNN_Quartile_array)
            np.save(savingfolder+"MET_QNN_Centre_array",MET_QNN_Centre_array)
            np.save(savingfolder+"METphi_QNN_RMS_array",METphi_QNN_RMS_array)
            np.save(savingfolder+"METphi_QNN_Quartile_array",METphi_QNN_Quartile_array)
            np.save(savingfolder+"METphi_QNN_Centre_array",METphi_QNN_Centre_array)
            np.save(savingfolder+"MET_DANN_RMS_arrayy",MET_DANN_RMS_array)
            np.save(savingfolder+"MET_DANN_Quartile_array",MET_DANN_Quartile_array)
            np.save(savingfolder+"MET_QNN_Centre_array",MET_QNN_Centre_array)
            np.save(savingfolder+"METphi_QNN_RMS_array",METphi_QNN_RMS_array)
            np.save(savingfolder+"METphi_QNN_Quartile_array",METphi_QNN_Quartile_array)
            np.save(savingfolder+"METphi_QNN_Centre_array",METphi_QNN_Centre_array)
            np.save(savingfolder+"MET_DANN_RMS_array",MET_DANN_RMS_array)
            np.save(savingfolder+"MET_DANN_Quartile_array",MET_DANN_Quartile_array)
            np.save(savingfolder+"MET_DANN_Centre_array",MET_DANN_Centre_array)
            np.save(savingfolder+"METphi_DANN_RMS_array",METphi_DANN_RMS_array)
            np.save(savingfolder+"METphi_DANN_Quartile_array",METphi_DANN_Quartile_array)
            np.save(savingfolder+"METphi_DANN_Centre_array",METphi_DANN_Centre_array)
            np.save(savingfolder+"MET_QNN_bestQ_array",MET_QNN_bestQ_array)
            np.save(savingfolder+"METphi_QNN_bestQ_array",METphi_QNN_bestQ_array)
            np.save(savingfolder+"MET_QNN_bestRMS_array",MET_QNN_bestRMS_array)
            np.save(savingfolder+"METphi_QNN_bestRMS_array",METphi_QNN_bestRMS_array)
            np.save(savingfolder+"MET_DANN_bestQ_array",MET_DANN_bestQ_array)
            np.save(savingfolder+"METphi_DANN_bestQ_array",METphi_DANN_bestQ_array)
            np.save(savingfolder+"MET_DANN_bestRMS_array",MET_DANN_bestRMS_array)
            np.save(savingfolder+"METphi_DANN_bestRMS_array",METphi_DANN_bestRMS_array)

    else:
        z0_QNN_array = np.load(savingfolder+"z0_QNN_array.npy")
        z0_QPNN_array = np.load(savingfolder+"z0_QPNN_array.npy")
        z0_DANN_array = np.load(savingfolder+"z0_DANN_array.npy")
        z0_FH_array = np.load(savingfolder+"z0_FH_array.npy")
        z0_FHres_array = np.load(savingfolder+"z0_FHres_array.npy")
        z0_FHMVA_array = np.load(savingfolder+"z0_FHMVA_array.npy")
        z0_FHnoFake_array = np.load(savingfolder+"z0_FHnoFake_array.npy")
        z0_PV_array = np.load(savingfolder+"z0_PV_array.npy")
        predictedQWeightsarray = np.load(savingfolder+"predictedQWeightsarray.npy")
        predictedQPWeightsarray = np.load(savingfolder+"predictedQPWeightsarray.npy")
        predictedDAWeightsarray = np.load(savingfolder+"predictedDAWeightsarray.npy")
        trk_z0_array = np.load(savingfolder+"trk_z0_array.npy")
        trk_mva_array = np.load(savingfolder+"trk_mva_array.npy")
        trk_gtt_pt_array = np.load(savingfolder+"trk_gtt_pt_array.npy")
        trk_gtt_eta_array = np.load(savingfolder+"trk_gtt_eta_array.npy")
        trk_z0_res_array = np.load(savingfolder+"trk_z0_res_array.npy")
        trk_gtt_phi_array = np.load(savingfolder+"trk_gtt_phi_array.npy")
        trk_chi2rphi_array = np.load(savingfolder+"trk_chi2rphi_array.npy")
        trk_chi2rz_array = np.load(savingfolder+"trk_chi2rz_array.npy")
        trk_bendchi2_array = np.load(savingfolder+"trk_bendchi2_array.npy")
        assoc_QNN_array = np.load(savingfolder+"assoc_QNN_array.npy")
        assoc_QPNN_array = np.load(savingfolder+"assoc_QPNN_array.npy")
        assoc_DANN_array = np.load(savingfolder+"assoc_DANN_array.npy")
        assoc_FH_array = np.load(savingfolder+"assoc_FH_array.npy")
        assoc_FHres_array = np.load(savingfolder+"assoc_FHres_array.npy")
        assoc_FHMVA_array = np.load(savingfolder+"assoc_FHMVA_array.npy")
        assoc_FHnoFake_array = np.load(savingfolder+"assoc_FHnoFake_array.npy")
        assoc_PV_array = np.load(savingfolder+"assoc_PV_array.npy")
        if met:
            actual_MET_array = np.load(savingfolder+"actual_MET_array.npy")
            actual_METphi_array = np.load(savingfolder+"actual_METphi_array.npy")
            actual_trkMET_array = np.load(savingfolder+"actual_trkMET_array.npy")
            actual_trkMETphi_array = np.load(savingfolder+"actual_trkMETphi_array.npy")
            MET_FHnoFake_array = np.load(savingfolder+"MET_FHnoFake_array.npy")
            METphi_FHnoFake_array = np.load(savingfolder+"METphi_FHnoFake_array.npy")
            MET_FHMVA_array = np.load(savingfolder+"MET_FHMVA_array.npy")
            METphi_FHMVA_array = np.load(savingfolder+"METphi_FHMVA_array.npy")
            MET_FHres_array = np.load(savingfolder+"MET_FHres_array.npy")
            METphi_FHres_array = np.load(savingfolder+"METphi_FHres_array.npy")
            MET_FH_array = np.load(savingfolder+"MET_FH_array.npy")
            METphi_FH_array = np.load(savingfolder+"METphi_FH_array.npy")
            MET_QNN_RMS_array = np.load(savingfolder+"MET_QNN_RMS_array.npy")
            MET_QNN_Quartile_array = np.load(savingfolder+"MET_QNN_Quartile_array.npy")
            MET_QNN_Centre_array = np.load(savingfolder+"MET_QNN_Centre_array.npy")
            METphi_QNN_RMS_array = np.load(savingfolder+"METphi_QNN_RMS_array.npy")
            METphi_QNN_Quartile_array = np.load(savingfolder+"METphi_QNN_Quartile_array.npy")
            METphi_QNN_Centre_array = np.load(savingfolder+"METphi_QNN_Centre_array.npy")
            MET_DANN_RMS_array  = np.load(savingfolder+"MET_DANN_RMS_array.npy")
            MET_DANN_Quartile_array = np.load(savingfolder+"MET_DANN_Quartile_array.npy")
            MET_QNN_Centre_array = np.load(savingfolder+"MET_QNN_Centre_array.npy")
            METphi_QNN_RMS_array = np.load(savingfolder+"METphi_QNN_RMS_array.npy")
            METphi_QNN_Quartile_array = np.load(savingfolder+"METphi_QNN_Quartile_array.npy")
            METphi_QNN_Centre_array = np.load(savingfolder+"METphi_QNN_Centre_array.npy")
            MET_DANN_RMS_array = np.load(savingfolder+"MET_DANN_RMS_array.npy")
            MET_DANN_Quartile_array = np.load(savingfolder+"MET_DANN_Quartile_array.npy")
            MET_DANN_Centre_array = np.load(savingfolder+"MET_DANN_Centre_array.npy")
            METphi_DANN_RMS_array = np.load(savingfolder+"METphi_DANN_RMS_array.npy")
            METphi_DANN_Quartile_array = np.load(savingfolder+"METphi_DANN_Quartile_array.npy")
            METphi_DANN_Centre_array = np.load(savingfolder+"METphi_DANN_Centre_array.npy")
            MET_QNN_bestQ_array = np.load(savingfolder+"MET_QNN_bestQ_array.npy")
            METphi_QNN_bestQ_array = np.load(savingfolder+"METphi_QNN_bestQ_array.npy")
            MET_QNN_bestRMS_array = np.load(savingfolder+"MET_QNN_bestRMS_array.npy")
            METphi_QNN_bestRMS_array = np.load(savingfolder+"METphi_QNN_bestRMS_array.npy")
            MET_DANN_bestQ_array = np.load(savingfolder+"MET_DANN_bestQ_array.npy")
            METphi_DANN_bestQ_array = np.load(savingfolder+"METphi_DANN_bestQ_array.npy")
            MET_DANN_bestRMS_array = np.load(savingfolder+"MET_DANN_bestRMS_array.npy")
            METphi_DANN_bestRMS_array = np.load(savingfolder+"METphi_DANN_bestRMS_array.npy")

    pv_track_sel = assoc_PV_array == 1
    pu_track_sel = assoc_PV_array == 0

    Qweightmax = np.max(predictedQWeightsarray)
    Qweightmin = np.min(predictedQWeightsarray)

    QPweightmax = np.max(predictedQPWeightsarray)
    QPweightmin = np.min(predictedQPWeightsarray)
    nonzero_Qweights = trk_gtt_pt_array != 0

    #########################################################################################
    #                                                                                       #
    #                                   Parameter Plots                                     #  
    #                                                                                       #
    #########################################################################################
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    ax.hist(trk_bendchi2_array[pv_track_sel],range=(0,8), bins=8, label="PV tracks", density=True,histtype="stepfilled",color='g',alpha=0.7,linewidth=LINEWIDTH)
    ax.hist(trk_bendchi2_array[pu_track_sel],range=(0,8), bins=8, label="PU tracks", density=True,histtype="stepfilled",color='r',alpha=0.7,linewidth=LINEWIDTH)
    ax.set_xlabel("Track $\\chi^2_{bend}$", horizontalalignment='right', x=1.0)
    ax.set_ylabel("# Tracks", horizontalalignment='right', y=1.0)
    ax.set_yscale("log")
    ax.legend()
    ax.tick_params(axis='x', which='minor', bottom=False,top=False)
    plt.tight_layout()
    plt.savefig("%s/bendchi2hist.png" % outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    ax.hist(trk_chi2rz_array[pv_track_sel],range=(0,16), bins=16, label="PV tracks", density=True,histtype="stepfilled",color='g',alpha=0.7,linewidth=LINEWIDTH)
    ax.hist(trk_chi2rz_array[pu_track_sel],range=(0,16), bins=16, label="PU tracks", density=True,histtype="stepfilled",color='r',alpha=0.7,linewidth=LINEWIDTH)
    ax.set_xlabel("Track $\\chi^2_{rz}$", horizontalalignment='right', x=1.0)
    ax.set_ylabel("# Tracks", horizontalalignment='right', y=1.0)
    ax.set_yscale("log")
    ax.legend()
    ax.tick_params(axis='x', which='minor', bottom=False,top=False)
    plt.tight_layout()
    plt.savefig("%s/chi2rzhist.png" % outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    ax.hist(trk_chi2rphi_array[pv_track_sel],range=(0,16), bins=16, label="PV tracks", density=True,histtype="stepfilled",color='g',alpha=0.7,linewidth=LINEWIDTH)
    ax.hist(trk_chi2rphi_array[pu_track_sel],range=(0,16), bins=16, label="PU tracks", density=True,histtype="stepfilled",color='r',alpha=0.7,linewidth=LINEWIDTH)
    ax.set_xlabel("Track $\\chi^2_{r\\phi}$", horizontalalignment='right', x=1.0)
    ax.set_ylabel("# Tracks", horizontalalignment='right', y=1.0)
    ax.set_yscale("log")
    ax.legend()
    ax.tick_params(axis='x', which='minor', bottom=False,top=False)
    plt.tight_layout()
    plt.savefig("%s/chi2rphihist.png" % outputFolder)
    plt.close()


    plt.clf()
    fig,ax = plt.subplots(2,1,figsize=(20,20))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])
    
    ax[0].hist(predictedQWeightsarray[pv_track_sel],range=(Qweightmin,Qweightmax), bins=50, label="PV tracks", density=True,histtype="stepfilled",color='g',alpha=0.7,linewidth=LINEWIDTH)
    ax[0].hist(predictedQWeightsarray[pu_track_sel],range=(Qweightmin,Qweightmax), bins=50, label="PU tracks", density=True,histtype="stepfilled",color='r',alpha=0.7,linewidth=LINEWIDTH)
    ax[1].hist(predictedQPWeightsarray[pv_track_sel],range=(Qweightmin,Qweightmax), bins=50, label="PV tracks", density=True,histtype="stepfilled",color='g',alpha=0.7,linewidth=LINEWIDTH)
    ax[1].hist(predictedQPWeightsarray[pu_track_sel],range=(Qweightmin,Qweightmax), bins=50, label="PU tracks", density=True,histtype="stepfilled",color='r',alpha=0.7,linewidth=LINEWIDTH)
    ax[0].set_xlabel("Quantised Weights", horizontalalignment='right', x=1.0)
    ax[1].set_xlabel("Pruned Weights", horizontalalignment='right', x=1.0)
    ax[0].set_ylabel("# Tracks", horizontalalignment='right', y=1.0)
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    #ax.set_title("Histogram weights for PU and PV tracks")
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("%s/corr-assoc-1d.png" % outputFolder)
    plt.close()

    pv_track_no = np.sum(pv_track_sel)
    pu_track_no = np.sum(pu_track_sel)

    assoc_scale = (pv_track_no / pu_track_no)
    
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    ax.hist(predictedQWeightsarray[pv_track_sel],range=(Qweightmin,Qweightmax), bins=50, label="PV tracks", weights=np.ones_like(predictedQWeightsarray[pv_track_sel]) / assoc_scale, density=True,histtype="stepfilled",color='g',alpha=0.7,linewidth=LINEWIDTH)
    ax.hist(predictedQWeightsarray[pu_track_sel],range=(Qweightmin,Qweightmax), bins=50, label="PU tracks", density=True,histtype="stepfilled",color='r',alpha=0.7,linewidth=LINEWIDTH)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("a.u.", horizontalalignment='right', y=1.0)
    #ax.set_title("Histogram weights for PU and PV tracks (normalised)")
    ax.set_yscale("log")

    ax.legend()
    plt.tight_layout()
    plt.savefig("%s/Qcorr-weights-1d-norm.png" % outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    ax.hist(assoc_QPNN_array[pv_track_sel],range=(0,1), bins=50, label="PV tracks", density=True,histtype="stepfilled",color='g',alpha=0.7,linewidth=LINEWIDTH)
    ax.hist(assoc_QPNN_array[pu_track_sel],range=(0,1), bins=50, label="PU tracks", density=True,histtype="stepfilled",color='r',alpha=0.7,linewidth=LINEWIDTH)
    ax.set_xlabel("Association Flag", horizontalalignment='right', x=1.0)
    ax.set_ylabel("a.u.", horizontalalignment='right', y=1.0)
    #ax.set_title("Histogram weights for PU and PV tracks (normalised)")
    ax.set_yscale("log")

    ax.legend()
    plt.tight_layout()
    plt.savefig("%s/Qcorr-assoc-1d-norm.png" % outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(2,1,figsize=(20,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])
    
    hist2d = ax[0].hist2d(predictedQWeightsarray, assoc_QNN_array, range=((Qweightmin,Qweightmax),(0,1)),bins=50,norm=matplotlib.colors.LogNorm(),cmap=colormap)
    hist2dp = ax[1].hist2d(predictedQPWeightsarray, assoc_QPNN_array, range=((QPweightmin,QPweightmax),(0,1)),bins=50,norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax[0].set_xlabel("Quantised Weights", horizontalalignment='right', x=1.0)
    ax[0].set_ylabel("Quantised Track-to-Vertex Association Flag", horizontalalignment='right', y=1.0)
    ax[0].set_xlabel("Pruned Weights", horizontalalignment='right', x=1.0)
    ax[0].set_ylabel("Pruned Track-to-Vertex Association Flag", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] ,ax=ax[0])
    cbar.set_label('# Tracks')
    cbar = plt.colorbar(hist2d[3] ,ax=ax[1])
    cbar.set_label('# Tracks')
    ax[0].vlines(0,0,1,linewidth=3,linestyle='dashed',color='k')
    ax[1].vlines(0,0,1,linewidth=3,linestyle='dashed',color='k')
    plt.tight_layout()
    plt.savefig("%s/Qcorr-assoc.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(predictedQWeightsarray[nonzero_Qweights], trk_z0_array[nonzero_Qweights], range=((Qweightmin,Qweightmax),(-20,20)), bins=50, norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track $z_0$ [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    ax.vlines(0,-20,20,linewidth=3,linestyle='dashed',color='k')
    plt.tight_layout()
    plt.savefig("%s/Qcorr-z0.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(predictedQWeightsarray[nonzero_Qweights], trk_mva_array[nonzero_Qweights], range=((Qweightmin,Qweightmax),(0,7)), bins=8, norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track MVA", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    ax.vlines(0,0,7,linewidth=3,linestyle='dashed',color='k')
    plt.tight_layout()
    plt.savefig("%s/Qcorr-mva.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    hidst2d = ax.hist2d(predictedQWeightsarray[nonzero_Qweights], trk_gtt_pt_array[nonzero_Qweights], range=((Qweightmin,Qweightmax),(0,512)), bins=50, norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track $p_T$ [GeV]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    ax.vlines(0,0,512,linewidth=3,linestyle='dashed',color='k')
    plt.tight_layout()
    plt.savefig("%s/Qcorr-pt.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    hidst2d = ax.hist2d(predictedQWeightsarray[nonzero_Qweights], trk_z0_res_array[nonzero_Qweights], range=((Qweightmin,Qweightmax),(0,7)), bins=(50,127), norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track $z_0$ resolution [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    ax.vlines(0,0,7,linewidth=3,linestyle='dashed',color='k')
    plt.tight_layout()
    plt.savefig("%s/Qcorr-z0res.png" %  outputFolder)
    plt.close()


    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(predictedQWeightsarray[nonzero_Qweights], np.abs(trk_gtt_eta_array[nonzero_Qweights]), range=((Qweightmin,Qweightmax),(0,2.4)), bins=50, norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track $|\\eta|$", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    ax.vlines(0,0,2.4,linewidth=3,linestyle='dashed',color='k')
    plt.tight_layout()
    plt.savefig("%s/Qcorr-abs-eta.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(predictedQWeightsarray[nonzero_Qweights], trk_gtt_eta_array[nonzero_Qweights], range=((Qweightmin,Qweightmax),(-2.4,2.4)), bins=50, norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track $\\eta$", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    ax.vlines(0,-2.4,2.4,linewidth=3,linestyle='dashed',color='k')
    plt.tight_layout()
    plt.savefig("%s/Qcorr-eta.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(predictedQWeightsarray[nonzero_Qweights], trk_chi2rphi_array[nonzero_Qweights], range=((Qweightmin,Qweightmax),(0,16)), bins=(50,16), norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track $\\chi^2_{r\\phi}$", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    ax.vlines(0,0,16,linewidth=3,linestyle='dashed',color='k')
    plt.tight_layout()
    plt.savefig("%s/Qcorr-chi2rphi.png" % outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(predictedQWeightsarray[nonzero_Qweights], trk_chi2rz_array[nonzero_Qweights], range=((Qweightmin,Qweightmax),(0,16)), bins=(50,16), norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track $\\chi^2_{rz}$", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    ax.vlines(0,0,16,linewidth=3,linestyle='dashed',color='k')
    plt.tight_layout()
    plt.savefig("%s/Qcorr-chi2rz.png" % outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(predictedQWeightsarray[nonzero_Qweights], trk_bendchi2_array[nonzero_Qweights] , range=((Qweightmin,Qweightmax),(0,8)), bins=(50,8), norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track $\\chi^2_{bend}$", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    ax.vlines(0,0,8,linewidth=3,linestyle='dashed',color='k')
    plt.tight_layout()
    plt.savefig("%s/Qcorr-chi2bend.png" % outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    #hist2d = ax.hist2d(predictedQWeightsarray, predictedDAWeightsarray, bins=50,range=((0,1),(0,1)), norm=matplotlib.colors.LogNorm(),cmap=colormap)
    #ax.set_xlabel("Quantised weights", horizontalalignment='right', x=1.0)
    #ax.set_ylabel("Weights", horizontalalignment='right', y=1.0)
    #cbar = plt.colorbar(hist2d[3] , ax=ax)
    #cbar.set_label('# Tracks')
    #ax.vlines(0,0,1,linewidth=3,linestyle='dashed',color='k')
    #plt.tight_layout()
    #plt.savefig("%s/Qweightvsweight.png" % outputFolder)
    #plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(assoc_QNN_array, assoc_DANN_array,range=((0,1),(0,1)), bins=50, norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Quantised Track-to-Vertex Association flag", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track-to-Vertex Association Flag", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    #ax.vlines(0,0,1,linewidth=3,linestyle='dashed',color='k')
    plt.tight_layout()
    plt.savefig("%s/Qassocvsassoc.png" % outputFolder)
    plt.close()

    do_scatter = False
    if (do_scatter):
        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        
        ax.scatter(predictedQWeightsarray, trk_z0_array, label="z0")
        ax.set_xlabel("weight")
        ax.set_ylabel("variable")
        ax.set_title("Correlation between predicted weight and track variables")
        ax.legend()
        plt.savefig("%s/scatter-z0.png" %  outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        
        ax.scatter(predictedQWeightsarray, trk_gtt_pt_array, label="pt")
        ax.set_xlabel("weight")
        ax.set_ylabel("variable")
        ax.set_title("Correlation between predicted weight and track variables")
        ax.legend()
        plt.savefig("%s/scatter-pt.png" % outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        
        ax.scatter(predictedQWeightsarray, trk_gtt_eta_array, label="eta")
        ax.set_xlabel("weight")
        ax.set_ylabel("variable")
        ax.set_title("Correlation between predicted weight and track variables")
        ax.legend()
        plt.savefig("%s/scatter-eta.png" %  outputFolder)
        plt.close()

    #########################################################################################
    #                                                                                       #
    #                                    Z0 Residual Plots                                  #  
    #                                                                                       #
    #########################################################################################

    #plt.clf()
    figure=plotz0_residual([(z0_PV_array-z0_QPNN_array)],
                          [(z0_PV_array-z0_FH_array),(z0_PV_array-z0_FHMVA_array),(z0_PV_array-z0_FHnoFake_array)],
                          ["QPNN"],
                          ["Base","MVA Cut","No Fakes"])
    plt.savefig("%s/Z0Residual.png" % outputFolder)
    plt.close()

    if PVROCs:

        plt.clf()
        figure=plotPV_roc([assoc_PV_array,assoc_PV_array,assoc_PV_array],
                         [assoc_QPNN_array],
                         [assoc_FH_array,assoc_FHMVA_array,assoc_FHnoFake_array],
                         ["QPNN"],
                         ["Base","BDT Cut","No Fakes"])
        plt.savefig("%s/PVROC.png" % outputFolder)

        plt.clf()
        figure=plotPV_roc([assoc_PV_array,assoc_PV_array,assoc_PV_array],
                          [assoc_DANN_array,assoc_QNN_array,assoc_QPNN_array],
                          [assoc_FH_array],
                          ["NN","QNN","QPNN"],
                          ["Baseline"])
        plt.savefig("%s/QcompPVROC.png" % outputFolder)
        plt.close()

    plt.clf()
    figure=plotz0_percentile([(z0_PV_array-z0_QPNN_array)],
                             [(z0_PV_array-z0_FH_array),(z0_PV_array-z0_FHMVA_array),(z0_PV_array-z0_FHnoFake_array)],
                             ["QPNN"],
                             ["Base","BDT Cut","No Fakes"])
    plt.savefig("%s/Z0percentile.png" % outputFolder)
    plt.close()

    plt.clf()
    figure=plotz0_residual([(z0_PV_array-z0_DANN_array),(z0_PV_array-z0_QNN_array),(z0_PV_array-z0_QPNN_array)],
                          [(z0_PV_array-z0_FH_array)],
                          ["NN               ","QNN             ","QPNN            "],
                          ["Baseline         "])
    plt.savefig("%s/QcompZ0Residual.png" % outputFolder)
    plt.close()


    if met:

        #########################################################################################
        #                                                                                       #
        #                                   MET Residual Plots                                  #  
        #                                                                                       #
        #########################################################################################

        plt.clf()
        figure=plotMET_residual([(MET_QNN_bestQ_array )],
                                [(MET_FH_array ),(MET_FHMVA_array ),(MET_FHnoFake_array ),(actual_trkMET_array )],
                                ["ArgMax thresh=" + Quartilethreshold_choice],
                                ["Base","BDT Cut","No Fakes","PV Tracks"],range=(-100,100),logrange=(-300,300),actual=actual_MET_array)
        plt.savefig("%s/METbestQresidual.png" % outputFolder)
        plt.close()

        plt.clf()
        figure=plotMETphi_residual([(METphi_QNN_bestQ_array )],
                                [(METphi_FH_array ),(METphi_FHMVA_array ),(METphi_FHnoFake_array ),(actual_trkMETphi_array)],
                                ["ArgMax thresh=" + Quartilethreshold_choice],
                                ["Base","BDT Cut","No Fakes","PV Tracks"],actual=actual_METphi_array)
        plt.savefig("%s/METphibestQresidual.png" % outputFolder)
        plt.close()

        plt.clf()
        figure=plotMET_residual([(MET_QNN_bestRMS_array )],
                                [(MET_FH_array ),(MET_FHMVA_array ),(MET_FHnoFake_array ),(actual_trkMET_array )],
                                ["ArgMax thresh=" + RMSthreshold_choice],
                                ["Base","BDT Cut","No Fakes","PV Tracks"],range=(-100,100),logrange=(-300,300),actual=actual_MET_array)
        plt.savefig("%s/METbestRMSresidual.png" % outputFolder)
        plt.close()

        plt.clf()
        figure=plotMETphi_residual([(METphi_QNN_bestRMS_array )],
                                [(METphi_FH_array ),(METphi_FHMVA_array ),(METphi_FHnoFake_array ),(actual_trkMETphi_array )],
                                ["ArgMax thresh=" + RMSthreshold_choice],
                                ["Base","BDT Cut","No Fakes","PV Tracks"],actual=actual_METphi_array)
        plt.savefig("%s/METphibestRMSresidual.png" % outputFolder)
        plt.close()

        plt.clf()
        figure=plotMET_residual([(MET_DANN_bestQ_array ),(MET_QNN_bestQ_array )],
                                [(MET_FH_array )],
                                ["NN thresh =  " + Quartilethreshold_choice_DNN+"    ","QNN thresh = " + Quartilethreshold_choice+"    "],
                                ["Baseline            "],range=(-100,100),logrange=(-300,300),actual=actual_MET_array)
        plt.savefig("%s/QcompMETbestQresidual.png" % outputFolder)
        plt.close()

        plt.clf()
        figure=plotMET_residual([(MET_DANN_bestRMS_array ),(MET_QNN_bestRMS_array )],
                                [(MET_FH_array )],
                                ["NN thresh =  " + RMSthreshold_choice_DNN+"    ","QNN thresh = " + RMSthreshold_choice+"    "],
                                ["Baseline            "],range=(-100,100),logrange=(-300,300),actual=actual_MET_array)
        plt.savefig("%s/QcompMETbestRMSresidual.png" % outputFolder)
        plt.close()


        #########################################################################################
        #                                                                                       #
        #                          Relative MET Residual Plots                                  #  
        #                                                                                       #
        #########################################################################################

        plt.clf()
        figure=plotMET_residual([(MET_QNN_bestQ_array )],
                                [(MET_FH_array ),(MET_FHMVA_array ),(MET_FHnoFake_array ),(actual_trkMET_array )],
                                ["ArgMax thresh=" + Quartilethreshold_choice],
                                ["Base","BDT Cut","No Fakes","PV Tracks"],range=(-1,2),logrange=(-1,30),relative=True,actual=actual_MET_array)
        plt.savefig("%s/relMETbestQresidual.png" % outputFolder)
        plt.close()

        plt.clf()
        figure=plotMET_residual([(MET_QNN_bestRMS_array )],
                                [(MET_FH_array ),(MET_FHMVA_array ),(MET_FHnoFake_array ),(actual_trkMET_array )],
                                ["ArgMax thresh=" + RMSthreshold_choice],
                                ["Base","BDT Cut","No Fakes","PV Tracks"],range=(-1,2),logrange=(-1,30),relative=True,actual=actual_MET_array)
        plt.savefig("%s/relMETbestRMSresidual.png" % outputFolder)
        plt.close()

        plt.clf()
        figure=plotMET_residual([(MET_DANN_bestQ_array ),(MET_QNN_bestQ_array )],
                                [(MET_FH_array )],
                                ["NN thresh =  " + Quartilethreshold_choice_DNN+"         ","QNN thresh = " + Quartilethreshold_choice+"         "],
                                ["Baseline            "],range=(-1,2),logrange=(-1,30),relative=True,actual=actual_MET_array)
        plt.savefig("%s/QcomprelMETbestQresidual.png" % outputFolder)
        plt.close()

        plt.clf()
        figure=plotMET_residual([(MET_DANN_bestRMS_array ),(MET_QNN_bestRMS_array )],
                                [(MET_FH_array )],
                                ["NN thresh =  " + RMSthreshold_choice_DNN+"         ","QNN thresh = " + RMSthreshold_choice+"         "],
                                ["Baseline            "],range=(-1,2),logrange=(-1,30),relative=True,actual=actual_MET_array)
        plt.savefig("%s/QcomprelMETbestRMSresidual.png" % outputFolder)
        plt.close()

        plt.clf()
        figure=plotMET_residual([(MET_DANN_bestQ_array ),(MET_QNN_bestQ_array )],
                                [(MET_FH_array )],
                                ["NN thresh =  " + Quartilethreshold_choice_DNN+"         ","QNN thresh = " + Quartilethreshold_choice+"         "],
                                ["Baseline            "],range=(-1,2),logrange=(-1,2),relative=True,actual=actual_MET_array,logbins=True)
        plt.savefig("%s/QcomprelMETbestQresidual_logbins.png" % outputFolder)
        plt.close()

        plt.clf()
        figure=plotMET_residual([(MET_DANN_bestRMS_array ),(MET_QNN_bestRMS_array )],
                                [(MET_FH_array )],
                                ["NN thresh =  " + RMSthreshold_choice_DNN+"         ","QNN thresh = " + RMSthreshold_choice+"         "],
                                ["Baseline            "],range=(-1,2),logrange=(-1,2),relative=True,actual=actual_MET_array,logbins=True)
        plt.savefig("%s/QcomprelMETbestRMSresidual_logbins.png" % outputFolder)
        plt.close()

        plt.clf()
        plotMET_resolution([(MET_QNN_bestRMS_array )],
                        [(MET_FH_array ),(MET_FHMVA_array ),(MET_FHnoFake_array ),(actual_trkMET_array )],
                        ["QNN"],["Baseline","BDT Cut","No Fakes","PV Tracks"],
                        actual=actual_MET_array,Et_bins = [0,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,240,280,300])
        plt.savefig("%s/relMETbestRMSresolution.png" % outputFolder)
        plt.close()

        plt.clf()
        plotMET_resolution([(MET_QNN_bestQ_array )],
                        [(MET_FH_array ),(MET_FHMVA_array ),(MET_FHnoFake_array ),(actual_trkMET_array )],
                        ["QNN"],["Baseline","BDT Cut","No Fakes","PV Tracks"],
                        actual=actual_MET_array,Et_bins = [0,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,240,280,300])
        plt.savefig("%s/relMETbestQresolution.png" % outputFolder)
        plt.close()

        plt.clf()
        plotMET_resolution([(MET_DANN_bestQ_array ),(MET_QNN_bestQ_array )],
                        [(MET_FH_array )],
                        ["NN","QNN"],["Baseline"],
                        actual=actual_MET_array,Et_bins = [0,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,240,280,300])
        plt.savefig("%s/QcomprelMETbestQresolution.png" % outputFolder)
        plt.close()

        plt.clf()
        plotMET_resolution([(MET_DANN_bestRMS_array ),(MET_QNN_bestRMS_array )],
                        [(MET_FH_array )],
                        ["NN","QNN"],["Baseline"],
                        actual=actual_MET_array,Et_bins = [0,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,240,280,300])
        plt.savefig("%s/QcomprelMETbestRMSresolution.png" % outputFolder)
        plt.close()

    #########################################################################################
    #                                                                                       #
    #                                   Z0 pred vs True Plots                               #  
    #                                                                                       #
    #########################################################################################
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    ax.hist(z0_FH_array,range=(-1*max_z0,max_z0),bins=120,density=True,color='r',histtype="step",label="FastHisto Base")
    ax.hist(z0_FHres_array,range=(-1*max_z0,max_z0),bins=120,density=True,color='g',histtype="step",label="FastHisto with z0 res")
    ax.hist(z0_QNN_array,range=(-1*max_z0,max_z0),bins=120,density=True,color='b',histtype="step",label="CNN")
    ax.hist(z0_PV_array,range=(-1*max_z0,max_z0),bins=120,density=True,color='y',histtype="step",label="Truth")
    ax.grid(True)
    ax.set_xlabel('$z_0$ [cm]',ha="right",x=1)
    ax.set_ylabel('Events',ha="right",y=1)
    ax.legend() 
    plt.tight_layout()
    plt.savefig("%s/Qz0hist.png" % outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_PV_array, (z0_PV_array-z0_FH_array), bins=60,range=((-1*max_z0,max_z0),(-2*max_z0,2*max_z0)) ,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("True PV $z_0$ [cm]", horizontalalignment='right', x=1.0)
    ax.set_ylabel("True PV $z_0$ - Reco PV $z_0$ Baseline [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/FHerr_vs_z0.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_PV_array, (z0_PV_array-z0_FHnoFake_array), bins=60,range=((-1*max_z0,max_z0),(-2*max_z0,2*max_z0)) ,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("True PV $z_0$ [cm]", horizontalalignment='right', x=1.0)
    ax.set_ylabel("True PV $z_0$ - Reco PV $z_0$ Baseline [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/FHnoFakeerr_vs_z0.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_PV_array, (z0_PV_array-z0_FHMVA_array), bins=60,range=((-1*max_z0,max_z0),(-2*max_z0,2*max_z0)) ,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("True PV $z_0$ [cm]", horizontalalignment='right', x=1.0)
    ax.set_ylabel("True PV $z_0$ - Reco PV $z_0$ Baseline [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/FHMVAerr_vs_z0.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_PV_array, (z0_PV_array-z0_QNN_array), bins=60,range=((-1*max_z0,max_z0),(-2*max_z0,2*max_z0)), norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("True PV $z_0$ [cm]", horizontalalignment='right', x=1.0)
    ax.set_ylabel("True PV $z_0$ - Reco PV $z_0$ QNN [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/QNNerr_vs_z0.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_PV_array, z0_FH_array, bins=60,range=((-1*max_z0,max_z0),(-1*max_z0,max_z0)) ,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("True PV $z_0$ [cm]", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Reco PV $z_0$ Baseline [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/FH_vs_z0.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_PV_array, z0_FHMVA_array, bins=60,range=((-1*max_z0,max_z0),(-1*max_z0,max_z0)) ,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("True PV $z_0$ [cm]", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Reco PV $z_0$ Baseline [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/FHMVA_vs_z0.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_PV_array, z0_FHnoFake_array, bins=60,range=((-1*max_z0,max_z0),(-1*max_z0,max_z0)) ,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("True PV $z_0$ [cm]", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Reco PV $z_0$ Baseline [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/FHnoFake_vs_z0.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_PV_array, z0_QNN_array, bins=60,range=((-1*max_z0,max_z0),(-1*max_z0,max_z0)), norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("True PV $z_0$ [cm]", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Reco PV $z_0$ QNN [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/QNN_vs_z0.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_DANN_array, z0_QNN_array, bins=60,range=((-1*max_z0,max_z0),(-1*max_z0,max_z0)), norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("Reco PV $z_0$ NN [cm]", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Reco PV $z_0$ QNN [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/QNN_vs_NN.png" %  outputFolder)
    plt.close()

    #########################################################################################
    #                                                                                       #
    #                                  MET pred vs True Plots                               #  
    #                                                                                       #
    #########################################################################################

    if met:

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(12,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        hist2d = ax.hist2d(actual_MET_array, MET_QNN_bestQ_array, bins=60,range=((0,300),(0,300)), norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
        ax.set_xlabel("$E_{T}^{miss,True}$", horizontalalignment='right', x=1.0)
        ax.set_ylabel("$E_{T}^{miss,QNN}$", horizontalalignment='right', y=1.0)
        cbar = plt.colorbar(hist2d[3] , ax=ax)
        cbar.set_label('# Events')
        plt.tight_layout()
        plt.savefig("%s/QNNbestQ_vs_MET.png" %  outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(12,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        hist2d = ax.hist2d(actual_MET_array, MET_QNN_bestRMS_array, bins=60,range=((0,300),(0,300)), norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
        ax.set_xlabel("$E_{T}^{miss,True}$", horizontalalignment='right', x=1.0)
        ax.set_ylabel("$E_{T}^{miss,QNN}$", horizontalalignment='right', y=1.0)
        cbar = plt.colorbar(hist2d[3] , ax=ax)
        cbar.set_label('# Events')
        plt.tight_layout()
        plt.savefig("%s/QNNbestRMS_vs_MET.png" %  outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(12,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        hist2d = ax.hist2d(actual_MET_array, MET_FH_array, bins=60,range=((0,300),(0,300)), norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
        ax.set_xlabel("$E_{T}^{miss,True}$", horizontalalignment='right', x=1.0)
        ax.set_ylabel("$E_{T}^{miss,FH}$", horizontalalignment='right', y=1.0)
        cbar = plt.colorbar(hist2d[3] , ax=ax)
        cbar.set_label('# Events')
        plt.tight_layout()
        plt.savefig("%s/FH_vs_MET.png" %  outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(12,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        hist2d = ax.hist2d(actual_METphi_array, METphi_QNN_bestQ_array, bins=60,range=((-np.pi,np.pi),(-np.pi,np.pi)), norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
        ax.set_xlabel("$E_{T,\\phi}^{miss,True}$", horizontalalignment='right', x=1.0)
        ax.set_ylabel("$E_{T,\\phi}^{miss,QNN}$", horizontalalignment='right', y=1.0)
        cbar = plt.colorbar(hist2d[3] , ax=ax)
        cbar.set_label('# Events')
        plt.tight_layout()
        plt.savefig("%s/QNNbestQ_vs_METphi.png" %  outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(12,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        hist2d = ax.hist2d(actual_METphi_array, METphi_QNN_bestRMS_array, bins=60,range=((-np.pi,np.pi),(-np.pi,np.pi)), norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
        ax.set_xlabel("$E_{T,\\phi}^{miss,True}$", horizontalalignment='right', x=1.0)
        ax.set_ylabel("$E_{T,\\phi}^{miss,QNN}$", horizontalalignment='right', y=1.0)
        cbar = plt.colorbar(hist2d[3] , ax=ax)
        cbar.set_label('# Events')
        plt.tight_layout()
        plt.savefig("%s/QNNbestRMS_vs_METphi.png" %  outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(12,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        hist2d = ax.hist2d(actual_METphi_array, METphi_FH_array, bins=60,range=((-np.pi,np.pi),(-np.pi,np.pi)), norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
        ax.set_xlabel("$E_{T,\\phi}^{miss,True}$", horizontalalignment='right', x=1.0)
        ax.set_ylabel("$E_{T,\\phi}^{miss,FH}$", horizontalalignment='right', y=1.0)
        cbar = plt.colorbar(hist2d[3] , ax=ax)
        cbar.set_label('# Events')
        plt.tight_layout()
        plt.savefig("%s/FH_vs_METphi.png" %  outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(12,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        hist2d = ax.hist2d(MET_DANN_bestQ_array, MET_QNN_bestQ_array, bins=60,range=((0,300),(0,300)), norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
        ax.set_xlabel("$E_{T}^{miss,NN}$", horizontalalignment='right', x=1.0)
        ax.set_ylabel("$E_{T}^{miss,QNN}$", horizontalalignment='right', y=1.0)
        cbar = plt.colorbar(hist2d[3] , ax=ax)
        cbar.set_label('# Events')
        plt.tight_layout()
        plt.savefig("%s/QNNbestQ_vs_NNbestQ.png" %  outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(12,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        hist2d = ax.hist2d(MET_DANN_bestRMS_array, MET_QNN_bestRMS_array, bins=60,range=((0,300),(0,300)), norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
        ax.set_xlabel("$E_{T}^{miss,NN}$", horizontalalignment='right', x=1.0)
        ax.set_ylabel("$E_{T}^{miss,QNN}$", horizontalalignment='right', y=1.0)
        cbar = plt.colorbar(hist2d[3] , ax=ax)
        cbar.set_label('# Events')
        plt.tight_layout()
        plt.savefig("%s/QNNbestRMS_vs_NNbestRMS.png" %  outputFolder)
        plt.close()


        #########################################################################################
        #                                                                                       #
        #                         Relative MET pred vs True Plots                               #  
        #                                                                                       #
        #########################################################################################

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(12,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        hist2d = ax.hist2d(actual_MET_array, (MET_QNN_bestQ_array-actual_MET_array)/actual_MET_array, bins=60,range=((0,300),(-1,30)), norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
        ax.set_xlabel("$E_{T}^{miss,True}$", horizontalalignment='right', x=1.0)
        ax.set_ylabel("$(E_{T}^{miss,NN} - E_{T}^{miss,True}) / E_{T}^{miss,True}$", horizontalalignment='right', y=1.0)
        cbar = plt.colorbar(hist2d[3] , ax=ax)
        cbar.set_label('# Events')
        plt.tight_layout()
        plt.savefig("%s/QrelNNbestQ_vs_MET.png" %  outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(12,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        hist2d = ax.hist2d(actual_MET_array, (MET_QNN_bestRMS_array-actual_MET_array)/actual_MET_array, bins=60,range=((0,300),(-1,30)), norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
        ax.set_xlabel("$E_{T}^{miss,True}$", horizontalalignment='right', x=1.0)
        ax.set_ylabel("$(E_{T}^{miss,NN} - E_{T}^{miss,True}) / E_{T}^{miss,True}$", horizontalalignment='right', y=1.0)
        cbar = plt.colorbar(hist2d[3] , ax=ax)
        cbar.set_label('# Events')
        plt.tight_layout()
        plt.savefig("%s/QrelNNbestRMS_vs_MET.png" %  outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(12,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        hist2d = ax.hist2d(actual_MET_array, (MET_FH_array-actual_MET_array)/actual_MET_array, bins=60,range=((0,300),(-1,30)), norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
        ax.set_xlabel("$E_{T}^{miss,True}$", horizontalalignment='right', x=1.0)
        ax.set_ylabel("$(E_{T}^{miss,FH} - E_{T}^{miss,True}) / E_{T}^{miss,True}$", horizontalalignment='right', y=1.0)
        cbar = plt.colorbar(hist2d[3] , ax=ax)
        cbar.set_label('# Events')
        plt.tight_layout()
        plt.savefig("%s/relFH_vs_MET.png" %  outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(12,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        hist2d = ax.hist2d((MET_DANN_bestQ_array-actual_MET_array)/actual_MET_array, (MET_QNN_bestQ_array-actual_MET_array)/actual_MET_array, bins=60,range=((-1,30),(-1,30)), norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
        ax.set_xlabel("$(E_{T}^{miss,NN} - E_{T}^{miss,True}) / E_{T}^{miss,True}$", horizontalalignment='right', x=1.0)
        ax.set_ylabel("$(E_{T}^{miss,QNN} - E_{T}^{miss,True}) / E_{T}^{miss,True}$", horizontalalignment='right', y=1.0)
        cbar = plt.colorbar(hist2d[3] , ax=ax)
        cbar.set_label('# Events')
        plt.tight_layout()
        plt.savefig("%s/QrelNNbestQ_vs_relNNbestQ.png" %  outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(12,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        hist2d = ax.hist2d((MET_DANN_bestRMS_array-actual_MET_array)/actual_MET_array, (MET_QNN_bestRMS_array-actual_MET_array)/actual_MET_array, bins=60,range=((-1,30),(-1,30)), norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
        ax.set_xlabel("$(E_{T}^{miss,QNN} - E_{T}^{miss,True}) / E_{T}^{miss,True}$", horizontalalignment='right', x=1.0)
        ax.set_ylabel("$(E_{T}^{miss,NN} - E_{T}^{miss,True}) / E_{T}^{miss,True}$", horizontalalignment='right', y=1.0)
        cbar = plt.colorbar(hist2d[3] , ax=ax)
        cbar.set_label('# Events')
        plt.tight_layout()
        plt.savefig("%s/QrelNNbestRMS_vs_relNNbestRMS.png" %  outputFolder)
        plt.close()


        #########################################################################################
        #                                                                                       #
        #                                  MET Threshold Plots                                  #  
        #                                                                                       #
        #########################################################################################

        def calc_widths(actual,predicted):
            diff = (actual-predicted)
            RMS = np.sqrt(np.mean(diff**2))
            qs = np.percentile(diff,[32,50,68])
            qwidth = qs[2] - qs[0]
            qcentre = qs[1]

            return [RMS,qwidth,qcentre]

        FHwidths = calc_widths(actual_MET_array,MET_FH_array)
        FHMVAwidths = calc_widths(actual_MET_array,MET_FHMVA_array)
        FHNoFakeWidths = calc_widths(actual_MET_array,MET_FHnoFake_array)
        TrkMETWidths = calc_widths(actual_MET_array,actual_trkMET_array)

        FHphiwidths = calc_widths(actual_METphi_array,METphi_FH_array)
        FHMVAphiwidths = calc_widths(actual_METphi_array,METphi_FHMVA_array)
        FHNoFakephiWidths = calc_widths(actual_METphi_array,METphi_FHnoFake_array)
        TrkMETphiWidths = calc_widths(actual_METphi_array,actual_trkMETphi_array)


        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        
        ax.plot(thresholds,MET_QNN_RMS_array,label="Argmax NN",markersize=10,linewidth=LINEWIDTH,marker='o')
        ax.plot(thresholds,np.full(len(thresholds),FHwidths[0]),label="Base FH",linestyle='--',linewidth=LINEWIDTH)
        ax.plot(thresholds,np.full(len(thresholds),FHMVAwidths[0]),label="BDT Cut FH",linestyle='--',linewidth=LINEWIDTH)
        ax.plot(thresholds,np.full(len(thresholds),FHNoFakeWidths[0]),label="No Fakes FH",linestyle='--',linewidth=LINEWIDTH)
        ax.plot(thresholds,np.full(len(thresholds),TrkMETWidths[0]),label="PV Tracks FH",linestyle='--',linewidth=LINEWIDTH)
        ax.set_ylabel("$E_{T}^{miss}$ Residual RMS", horizontalalignment='right', y=1.0)
        ax.set_xlabel("Track-to-vertex association threshold", horizontalalignment='right', x=1.0)
        ax.legend()
        plt.tight_layout()
        plt.savefig("%s/METRMSvsThreshold.png" %  outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        
        ax.plot(thresholds,MET_QNN_Quartile_array,label="Argmax NN",markersize=10,linewidth=LINEWIDTH,marker='o')
        ax.plot(thresholds,np.full(len(thresholds),FHwidths[1]),label="Base FH",linestyle='--',linewidth=LINEWIDTH)
        ax.plot(thresholds,np.full(len(thresholds),FHMVAwidths[1]),label="BDT Cut FH",linestyle='--',linewidth=LINEWIDTH)
        ax.plot(thresholds,np.full(len(thresholds),FHNoFakeWidths[1]),label="No Fakes FH",linestyle='--',linewidth=LINEWIDTH)
        ax.plot(thresholds,np.full(len(thresholds),TrkMETWidths[1]),label="PV Tracks FH",linestyle='--',linewidth=LINEWIDTH)
        ax.set_ylabel("$E_{T}^{miss}$ Residual Quartile Width", horizontalalignment='right', y=1.0)
        ax.set_xlabel("Track-to-vertex association threshold", horizontalalignment='right', x=1.0)
        ax.legend()
        plt.tight_layout()
        plt.savefig("%s/METQsvsThreshold.png" %  outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        
        ax.plot(thresholds,MET_QNN_Centre_array,label="Argmax NN",markersize=10,linewidth=LINEWIDTH,marker='o')
        ax.plot(thresholds,np.full(len(thresholds),FHwidths[2]),label="Base FH",linestyle='--',linewidth=LINEWIDTH)
        ax.plot(thresholds,np.full(len(thresholds),FHMVAwidths[2]),label="BDT Cut FH",linestyle='--',linewidth=LINEWIDTH)
        ax.plot(thresholds,np.full(len(thresholds),FHNoFakeWidths[2]),label="No Fakes FH",linestyle='--',linewidth=LINEWIDTH)
        ax.plot(thresholds,np.full(len(thresholds),TrkMETWidths[2]),label="PV Tracks FH",linestyle='--',linewidth=LINEWIDTH)
        ax.set_ylabel("$E_{T}^{miss}$ Residual Centre", horizontalalignment='right', y=1.0)
        ax.set_xlabel("Track-to-vertex association threshold", horizontalalignment='right', x=1.0)
        ax.legend()
        plt.tight_layout()
        plt.savefig("%s/METCentrevsThreshold.png" %  outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        
        ax.plot(thresholds,METphi_QNN_RMS_array,label="Argmax NN",markersize=10,linewidth=LINEWIDTH,marker='o')
        ax.plot(thresholds,np.full(len(thresholds),FHphiwidths[0]),label="Base FH",linestyle='--',linewidth=LINEWIDTH)
        ax.plot(thresholds,np.full(len(thresholds),FHMVAphiwidths[0]),label="BDT Cut FH",linestyle='--',linewidth=LINEWIDTH)
        ax.plot(thresholds,np.full(len(thresholds),FHNoFakephiWidths[0]),label="No Fakes FH",linestyle='--',linewidth=LINEWIDTH)
        ax.plot(thresholds,np.full(len(thresholds),TrkMETphiWidths[0]),label="PV Tracks FH",linestyle='--',linewidth=LINEWIDTH)
        ax.set_ylabel("$E_{T,\\phi}^{miss}$ Residual RMS", horizontalalignment='right', y=1.0)
        ax.set_xlabel("Track-to-vertex association threshold", horizontalalignment='right', x=1.0)
        ax.legend()
        plt.tight_layout()
        plt.savefig("%s/METphiRMSvsThreshold.png" %  outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        
        ax.plot(thresholds,METphi_QNN_Quartile_array,label="Argmax NN",markersize=10,linewidth=LINEWIDTH,marker='o')
        ax.plot(thresholds,np.full(len(thresholds),FHphiwidths[1]),label="Base FH",linestyle='--',linewidth=LINEWIDTH)
        ax.plot(thresholds,np.full(len(thresholds),FHMVAphiwidths[1]),label="BDT Cut FH",linestyle='--',linewidth=LINEWIDTH)
        ax.plot(thresholds,np.full(len(thresholds),FHNoFakephiWidths[1]),label="No Fakes FH",linestyle='--',linewidth=LINEWIDTH)
        ax.plot(thresholds,np.full(len(thresholds),TrkMETphiWidths[1]),label="PV Tracks FH",linestyle='--',linewidth=LINEWIDTH)
        ax.set_ylabel("$E_{T,\\phi}^{miss}$ Residual Quartile Width", horizontalalignment='right', y=1.0)
        ax.set_xlabel("Track-to-vertex association threshold", horizontalalignment='right', x=1.0)
        ax.legend()
        plt.tight_layout()
        plt.savefig("%s/METphiQsvsThreshold.png" %  outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        
        ax.plot(thresholds,METphi_QNN_Centre_array,label="Argmax NN",markersize=10,linewidth=LINEWIDTH,marker='o')
        ax.plot(thresholds,np.full(len(thresholds),FHphiwidths[2]),label="Base FH",linestyle='--',linewidth=LINEWIDTH)
        ax.plot(thresholds,np.full(len(thresholds),FHMVAphiwidths[2]),label="BDT Cut FH",linestyle='--',linewidth=LINEWIDTH)
        ax.plot(thresholds,np.full(len(thresholds),FHNoFakephiWidths[2]),label="No Fakes FH",linestyle='--',linewidth=LINEWIDTH)
        ax.plot(thresholds,np.full(len(thresholds),TrkMETphiWidths[2]),label="PV Tracks FH",linestyle='--',linewidth=LINEWIDTH)
        ax.set_ylabel("$E_{T,\\phi}^{miss}$ Residual Centre", horizontalalignment='right', y=1.0)
        ax.set_xlabel("Track-to-vertex association threshold", horizontalalignment='right', x=1.0)
        ax.legend()
        plt.tight_layout()
        plt.savefig("%s/METphiCentrevsThreshold.png" %  outputFolder)
        plt.close()
    
    image_list = os.listdir(outputFolder) # returns list
    for image in image_list:
        experiment.log_image(image, name=image)
