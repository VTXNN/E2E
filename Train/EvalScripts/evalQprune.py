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

    save = True
    savingfolder = kf+"SavedArrays/"
    PVROCs = True 


    with open(kf+'experimentkey.txt') as f:
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


    outputFolder = kf+config['eval_folder']
    trainable = config["trainable"]
    trackfeat = config["track_features"] 
    weightfeat = config["weight_features"] 


    features = {
            "pvz0": tf.io.FixedLenFeature([1], tf.float32),
            "trk_fromPV":tf.io.FixedLenFeature([nMaxTracks], tf.float32),
            "trk_hitpattern": tf.io.FixedLenFeature([nMaxTracks*11], tf.float32), 
            "PV_hist"  :tf.io.FixedLenFeature([256,1], tf.float32),
            "tp_met_pt" : tf.io.FixedLenFeature([1], tf.float32),
            "tp_met_phi" : tf.io.FixedLenFeature([1], tf.float32)

    }

    def decode_data(raw_data):
        decoded_data = tf.io.parse_example(raw_data,features)
        decoded_data['trk_hitpattern'] = tf.reshape(decoded_data['trk_hitpattern'],[-1,nMaxTracks,11])
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

    trackFeatures = [
            'trk_z0',
            'normed_trk_pt',
            'normed_trk_eta', 
            'normed_trk_invR',
            'binned_trk_chi2rphi', 
            'binned_trk_chi2rz', 
            'binned_trk_bendchi2',
            'trk_z0_res',
            'trk_pt',
            'trk_eta',
            'trk_phi',
            'trk_MVA1',
            'trk_fake',
            'corrected_trk_z0',
            'normed_trk_over_eta',
            'normed_trk_over_eta_squared',
            'trk_over_eta_squared'

        ]

    for trackFeature in trackFeatures:
        features[trackFeature] = tf.io.FixedLenFeature([nMaxTracks], tf.float32)

    nlatent = 2

    qnetwork = vtx.nn.E2EQKerasDiffArgMax(
            nbins=256,
            start=0,
            end=4095,
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
    qmodel.load_weights(kf+"best_weights.tf")

    DAnetwork = vtx.nn.E2EDiffArgMax(
            nbins=256,
            start=0,
            end=4095,
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

    plt.clf()
    fig,ax = plt.subplots(1,2,figsize=(20,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax[0])
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax[1])

    prune_level = []
    for i,layer in enumerate(qmodel.layers):
        get_weights = layer.get_weights()
        if len(get_weights) > 0:
            if "Bin_weight" not in layer.name:  
                weights = get_weights[0].flatten()[get_weights[0].flatten() != 0]
                if len(get_weights) > 1:
                    biases = get_weights[1].flatten()[get_weights[1].flatten() != 0]
                prune_level.append(weights.shape[0]/get_weights[0].flatten().shape[0])

                ax[0].hist(weights,alpha=1,label=layer.name,histtype="step",linewidth=2,bins=50,range=(-2,2))
                ax[1].hist(biases,alpha=1,label=layer.name,histtype="step",linewidth=2,bins=50,range=(-1,1))


    prune = 100-np.mean(prune_level)*100


    ax[0].set_title("QKeras Weights with " + "%.1f"%prune + "% Prune",loc='left')
    ax[0].grid(True)
    ax[0].set_xlabel("Weight Magnitude",ha="right",x=1)
    ax[0].set_ylabel("# Parameters",ha="right",y=1)
    ax[0].set_yscale("log")
    ax[0].legend(loc=2) 

    ax[1].grid(True)
    ax[1].set_xlabel("Bias Magnitude",ha="right",x=1)
    ax[1].set_ylabel("# Parameters",ha="right",y=1)
    ax[1].legend(loc=2) 

    plt.tight_layout()
    plt.savefig("%s/Qweights_biases.png" %  outputFolder)
    plt.close()

    predictedZ0_FH = []
    predictedZ0_FHz0res = []
    predictedZ0_FHz0MVA = []
    predictedZ0_FHnoFake = []
    predictedZ0_QNN = []
    predictedZ0_DANN = []

    predictedQWeights = []
    predictedDAWeights = []

    predictedAssoc_QNN = []
    predictedAssoc_DANN = []
    predictedAssoc_FH = []
    predictedAssoc_FHres = []
    predictedAssoc_FHMVA = []
    predictedAssoc_FHnoFake = []

    num_threshold = 10
    thresholds = [str(i/num_threshold) for i in range(0,num_threshold)]

    predictedMET_QNN = {key: value for key, value in zip(thresholds, [[] for i in range(0,num_threshold)])}
    predictedMETphi_QNN = {key: value for key, value in zip(thresholds, [[] for i in range(0,num_threshold)])}
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
    trk_phi = []
    trk_pt = []
    trk_eta = []

    trk_chi2rphi = []
    trk_chi2rz = []
    trk_bendchi2 = []

    threshold = -1

    if save:
        for step,batch in enumerate(setup_pipeline(test_files)):

            #if step > 0:
            #    break

            trackFeatures = np.stack([batch[feature] for feature in trackfeat],axis=2)
            WeightFeatures = np.stack([batch[feature] for feature in weightfeat],axis=2)
            nBatch = batch['pvz0'].shape[0]
            FH = predictFastHisto(batch[z0],batch['trk_pt'])
            predictedZ0_FH.append(FH)
            predictedZ0_FHz0res.append(predictFastHistoZ0res(batch[z0],batch['trk_pt'],batch['trk_eta']))
            predictedZ0_FHz0MVA.append(predictFastHistoMVAcut(batch[z0],batch['trk_pt'],batch['trk_MVA1']))
            predictedZ0_FHnoFake.append(predictFastHistoNoFakes(batch[z0],batch['trk_pt'],batch['trk_fake']))

            trk_z0.append(batch[z0])
            trk_MVA.append(batch["trk_MVA1"])
            trk_pt.append(batch['normed_trk_pt'])
            trk_eta.append(batch['normed_trk_eta'])
            trk_phi.append(batch['trk_phi'])

            trk_chi2rphi.append(batch['binned_trk_chi2rphi'])
            trk_chi2rz.append(batch['binned_trk_chi2rz'])
            trk_bendchi2.append(batch['binned_trk_bendchi2'])

            actual_Assoc.append(batch["trk_fromPV"])
            actual_PV.append(batch['pvz0'])
            FHassoc = FastHistoAssoc(predictFastHisto(batch[z0],batch['trk_pt']),batch[z0],batch['trk_eta'],kf)
            predictedAssoc_FH.append(FHassoc)
            FHassocres = FastHistoAssoc(predictFastHistoZ0res(batch[z0],batch['trk_pt'],batch['trk_eta']),batch[z0],batch['trk_eta'],kf)
            predictedAssoc_FHres.append(FHassocres)

            FHassocMVA = FastHistoAssocMVAcut(predictFastHistoMVAcut(batch[z0],batch['trk_pt'],batch['trk_MVA1']),batch[z0],batch['trk_eta'],batch['trk_MVA1'],kf)
            predictedAssoc_FHMVA.append(FHassocMVA)

            FHassocnoFake = FastHistoAssocNoFakes(predictFastHistoNoFakes(batch[z0],batch['trk_pt'],batch['trk_fake']),batch[z0],batch['trk_eta'],batch['trk_fake'],kf)
            predictedAssoc_FHnoFake.append(FHassocnoFake)

            for i,event in enumerate(batch[z0]):
                if abs(FH[i] - batch['pvz0'][i]) > 0:
                    figure = plotKDEandTracks(batch['trk_z0'][i],batch['trk_fake'][i],batch['pvz0'][i],FH[i],batch['trk_pt'][i],weight_label="Baseline",threshold=0.5)
                    plt.savefig("%s/event.png" % outputFolder)
                    break


            #### Q NETWORK #########################################################################################################################
            XX = qmodel.input 
            YY = qmodel.layers[5].output
            new_model = Model(XX, YY)

            if bit:
                predictedQWeights_QNN = new_model.predict_on_batch(
                            [batch[bit_z0],WeightFeatures,trackFeatures])

                predictedZ0_QNN_temp, predictedAssoc_QNN_temp, QWeights_QNN = qmodel.predict_on_batch(
                            [batch[bit_z0],WeightFeatures,trackFeatures]
                        )

            else:
                predictedQWeights_QNN = new_model.predict_on_batch(
                            [batch[z0],WeightFeatures,trackFeatures])

                predictedZ0_QNN_temp, predictedAssoc_QNN_temp, QWeights_QNN = qmodel.predict_on_batch(
                            [batch[z0],WeightFeatures,trackFeatures]
                        )


            
            predictedAssoc_QNN_temp = tf.math.divide( tf.math.subtract( predictedAssoc_QNN_temp,tf.reduce_min(predictedAssoc_QNN_temp)), 
                                                    tf.math.subtract( tf.reduce_max(predictedAssoc_QNN_temp), tf.reduce_min(predictedAssoc_QNN_temp) ))

            predictedZ0_QNN.append(predictedZ0_QNN_temp)
            predictedAssoc_QNN.append(predictedAssoc_QNN_temp)
            predictedQWeights.append(predictedQWeights_QNN)

            

            #### DA NETWORK #########################################################################################################################
            XX = DAmodel.input 
            YY = DAmodel.layers[5].output
            new_model = Model(XX, YY)

            if bit:
                predictedDAWeights_DANN = new_model.predict_on_batch(
                                [batch[bit_z0],WeightFeatures,trackFeatures])


                predictedZ0_DANN_temp, predictedAssoc_DANN_temp, DAWeights_DANN = DAmodel.predict_on_batch(
                                [batch[bit_z0],WeightFeatures,trackFeatures]
                            )

            else:

                predictedDAWeights_DANN = new_model.predict_on_batch(
                                [batch[z0],WeightFeatures,trackFeatures])


                predictedZ0_DANN_temp, predictedAssoc_DANN_temp, DAWeights_DANN = DAmodel.predict_on_batch(
                                [batch[z0],WeightFeatures,trackFeatures]
                            )
            predictedAssoc_DANN_temp = tf.math.divide( tf.math.subtract( predictedAssoc_DANN_temp,tf.reduce_min(predictedAssoc_DANN_temp)), 
                                                    tf.math.subtract( tf.reduce_max(predictedAssoc_DANN_temp), tf.reduce_min(predictedAssoc_DANN_temp) ))

            predictedZ0_DANN.append(predictedZ0_DANN_temp)
            predictedAssoc_DANN.append(predictedAssoc_DANN_temp)
            predictedDAWeights.append(predictedDAWeights_DANN)

            actual_MET.append(batch['tp_met_pt'])
            actual_METphi.append(batch['tp_met_phi'])

            temp_met,temp_metphi = predictMET(batch['trk_pt'],batch['trk_phi'],batch['trk_fromPV'],threshold=0.5)
            actual_trkMET.append(temp_met)
            actual_trkMETphi.append(temp_metphi)

            temp_met,temp_metphi = predictMET(batch['trk_pt'],batch['trk_phi'],FHassocnoFake,threshold=0.5)
            predictedMET_FHnoFake.append(temp_met)
            predictedMETphi_FHnoFake.append(temp_metphi)

            temp_met,temp_metphi = predictMET(batch['trk_pt'],batch['trk_phi'],FHassocMVA,threshold=0.5)
            predictedMET_FHMVA.append(temp_met)
            predictedMETphi_FHMVA.append(temp_metphi)

            temp_met,temp_metphi = predictMET(batch['trk_pt'],batch['trk_phi'],FHassocres,threshold=0.5)
            predictedMET_FHres.append(temp_met)
            predictedMETphi_FHres.append(temp_metphi)

            temp_met,temp_metphi = predictMET(batch['trk_pt'],batch['trk_phi'],FHassoc,threshold=0.5,quality=True,
                                              chi2rphi = batch['binned_trk_chi2rphi'],chi2rz = batch['binned_trk_chi2rz'],bendchi2 = batch['binned_trk_bendchi2'])
            predictedMET_FH.append(temp_met)
            predictedMETphi_FH.append(temp_metphi)

            for i in range(0,num_threshold):
                temp_met,temp_metphi = predictMET(batch['trk_pt'],batch['trk_phi'],predictedAssoc_QNN_temp.numpy().squeeze(),threshold=i/num_threshold,quality=True,
                                              chi2rphi = batch['binned_trk_chi2rphi'],chi2rz = batch['binned_trk_chi2rz'],bendchi2 = batch['binned_trk_bendchi2'])
                predictedMET_QNN[str(i/num_threshold)].append(temp_met)
                predictedMETphi_QNN[str(i/num_threshold)].append(temp_metphi)

                temp_met,temp_metphi = predictMET(batch['trk_pt'],batch['trk_phi'],predictedAssoc_DANN_temp.numpy().squeeze(),threshold=i/num_threshold,quality=True,
                                              chi2rphi = batch['binned_trk_chi2rphi'],chi2rz = batch['binned_trk_chi2rz'],bendchi2 = batch['binned_trk_bendchi2'])
                predictedMET_DANN[str(i/num_threshold)].append(temp_met)
                predictedMETphi_DANN[str(i/num_threshold)].append(temp_metphi)

        z0_QNN_array = np.concatenate(predictedZ0_QNN).ravel()
        z0_DANN_array = np.concatenate(predictedZ0_DANN).ravel()

        z0_FH_array = np.concatenate(predictedZ0_FH).ravel()
        z0_FHres_array = np.concatenate(predictedZ0_FHz0res).ravel()
        z0_FHMVA_array = np.concatenate(predictedZ0_FHz0MVA).ravel()
        z0_FHnoFake_array = np.concatenate(predictedZ0_FHnoFake).ravel()
        z0_PV_array = np.concatenate(actual_PV).ravel()

        predictedQWeightsarray = np.concatenate(predictedQWeights).ravel()
        predictedDAWeightsarray = np.concatenate(predictedDAWeights).ravel()

        trk_z0_array = np.concatenate(trk_z0).ravel()
        trk_mva_array = np.concatenate(trk_MVA).ravel()
        trk_pt_array = np.concatenate(trk_pt).ravel()
        trk_eta_array = np.concatenate(trk_eta).ravel()
        trk_phi_array = np.concatenate(trk_phi).ravel()

        trk_chi2rphi_array = np.concatenate(trk_chi2rphi).ravel()
        trk_chi2rz_array = np.concatenate(trk_chi2rz).ravel()
        trk_bendchi2_array = np.concatenate(trk_bendchi2).ravel()


        assoc_QNN_array = np.concatenate(predictedAssoc_QNN).ravel()
        assoc_DANN_array = np.concatenate(predictedAssoc_DANN).ravel()

        assoc_FH_array = np.concatenate(predictedAssoc_FH).ravel()
        assoc_FHres_array = np.concatenate(predictedAssoc_FHres).ravel()
        assoc_FHMVA_array = np.concatenate(predictedAssoc_FHMVA).ravel()
        assoc_FHnoFake_array = np.concatenate(predictedAssoc_FHnoFake).ravel()
        assoc_PV_array = np.concatenate(actual_Assoc).ravel()


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


        MET_QNN_bestQ_array = np.concatenate(predictedMET_QNN[str(np.argmin(MET_QNN_Quartile_array)/num_threshold)]).ravel()
        METphi_QNN_bestQ_array = np.concatenate(predictedMETphi_QNN[str(np.argmin(MET_QNN_Quartile_array)/num_threshold)]).ravel()

        MET_QNN_bestRMS_array = np.concatenate(predictedMET_QNN[str(np.argmin(MET_QNN_RMS_array)/num_threshold)]).ravel()
        METphi_QNN_bestRMS_array = np.concatenate(predictedMETphi_QNN[str(np.argmin(MET_QNN_RMS_array)/num_threshold)]).ravel()

        MET_DANN_bestQ_array = np.concatenate(predictedMET_DANN[str(np.argmin(MET_DANN_Quartile_array)/num_threshold)]).ravel()
        METphi_DANN_bestQ_array = np.concatenate(predictedMETphi_DANN[str(np.argmin(MET_DANN_Quartile_array)/num_threshold)]).ravel()

        MET_DANN_bestRMS_array = np.concatenate(predictedMET_DANN[str(np.argmin(MET_DANN_RMS_array)/num_threshold)]).ravel()
        METphi_DANN_bestRMS_array = np.concatenate(predictedMETphi_DANN[str(np.argmin(MET_DANN_RMS_array)/num_threshold)]).ravel()

        np.save(savingfolder+"z0_QNN_array",z0_QNN_array)
        np.save(savingfolder+"z0_DANN_array",z0_DANN_array)
        np.save(savingfolder+"z0_FH_array",z0_FH_array)
        np.save(savingfolder+"z0_FHres_array",z0_FHres_array)
        np.save(savingfolder+"z0_FHMVA_array",z0_FHMVA_array)
        np.save(savingfolder+"z0_FHnoFake_array",z0_FHnoFake_array)
        np.save(savingfolder+"z0_PV_array",z0_PV_array)
        np.save(savingfolder+"predictedQWeightsarray",predictedQWeightsarray)
        np.save(savingfolder+"predictedDAWeightsarray",predictedDAWeightsarray)
        np.save(savingfolder+"trk_z0_array",trk_z0_array)
        np.save(savingfolder+"trk_mva_array",trk_mva_array)
        np.save(savingfolder+"trk_pt_array",trk_pt_array)
        np.save(savingfolder+"trk_eta_array",trk_eta_array)
        np.save(savingfolder+"trk_phi_array",trk_phi_array)
        np.save(savingfolder+"trk_chi2rphi_array",trk_chi2rphi_array)
        np.save(savingfolder+"trk_chi2rz_array",trk_chi2rz_array)
        np.save(savingfolder+"trk_bendchi2_array",trk_bendchi2_array)
        np.save(savingfolder+"assoc_QNN_array",assoc_QNN_array)
        np.save(savingfolder+"assoc_DANN_array",assoc_DANN_array)
        np.save(savingfolder+"assoc_FH_array",assoc_FH_array)
        np.save(savingfolder+"assoc_FHres_array",assoc_FHres_array)
        np.save(savingfolder+"assoc_FHMVA_array",assoc_FHMVA_array)
        np.save(savingfolder+"assoc_FHnoFake_array",assoc_FHnoFake_array)
        np.save(savingfolder+"assoc_PV_array",assoc_PV_array)
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
        z0_DANN_array = np.load(savingfolder+"z0_DANN_array.npy")
        z0_FH_array = np.load(savingfolder+"z0_FH_array.npy")
        z0_FHres_array = np.load(savingfolder+"z0_FHres_array.npy")
        z0_FHMVA_array = np.load(savingfolder+"z0_FHMVA_array.npy")
        z0_FHnoFake_array = np.load(savingfolder+"z0_FHnoFake_array.npy")
        z0_PV_array = np.load(savingfolder+"z0_PV_array.npy")
        predictedQWeightsarray = np.load(savingfolder+"predictedQWeightsarray.npy")
        predictedDAWeightsarray = np.load(savingfolder+"predictedDAWeightsarray.npy")
        trk_z0_array = np.load(savingfolder+"trk_z0_array.npy")
        trk_mva_array = np.load(savingfolder+"trk_mva_array.npy")
        trk_pt_array = np.load(savingfolder+"trk_pt_array.npy")
        trk_eta_array = np.load(savingfolder+"trk_eta_array.npy")
        trk_phi_array = np.load(savingfolder+"trk_phi_array.npy")
        trk_chi2rphi_array = np.load(savingfolder+"trk_chi2rphi_array.npy")
        trk_chi2rz_array = np.load(savingfolder+"trk_chi2rz_array.npy")
        trk_bendchi2_array = np.load(savingfolder+"trk_bendchi2_array.npy")
        assoc_QNN_array = np.load(savingfolder+"assoc_QNN_array.npy")
        assoc_DANN_array = np.load(savingfolder+"assoc_DANN_array.npy")
        assoc_FH_array = np.load(savingfolder+"assoc_FH_array.npy")
        assoc_FHres_array = np.load(savingfolder+"assoc_FHres_array.npy")
        assoc_FHMVA_array = np.load(savingfolder+"assoc_FHMVA_array.npy")
        assoc_FHnoFake_array = np.load(savingfolder+"assoc_FHnoFake_array.npy")
        assoc_PV_array = np.load(savingfolder+"assoc_PV_array.npy")
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

    #########################################################################################
    #                                                                                       #
    #                                   Parameter Plots                                     #  
    #                                                                                       #
    #########################################################################################
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
    ax.hist(trk_bendchi2_array[pv_track_sel],range=(0,1), bins=50, label="PV tracks", alpha=0.5, density=True)
    ax.hist(trk_bendchi2_array[pu_track_sel],range=(0,1), bins=50, label="PU tracks", alpha=0.5, density=True)
    ax.set_xlabel("Track $\\chi^2_{bend}$", horizontalalignment='right', x=1.0)
    # ax.set_ylabel("tracks/counts", horizontalalignment='right', y=1.0)
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    plt.savefig("%s/bendchi2hist.png" % outputFolder)
    plt.close()


    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
    ax.hist(predictedQWeightsarray[pv_track_sel],range=(Qweightmin,Qweightmax), bins=50, label="PV tracks", alpha=0.5, density=True)
    ax.hist(predictedQWeightsarray[pu_track_sel],range=(Qweightmin,Qweightmax), bins=50, label="PU tracks", alpha=0.5, density=True)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    # ax.set_ylabel("tracks/counts", horizontalalignment='right', y=1.0)
    ax.set_yscale("log")
    #ax.set_title("Histogram weights for PU and PV tracks")
    ax.legend()
    plt.tight_layout()
    plt.savefig("%s/corr-assoc-1d.png" % outputFolder)
    plt.close()

    pv_track_no = np.sum(pv_track_sel)
    pu_track_no = np.sum(pu_track_sel)


    assoc_scale = (pv_track_no / pu_track_no)
    # plt.bar(b[:-1], h, width=bw, label="PV tracks", alpha=0.5)

    
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
    ax.hist(predictedQWeightsarray[pv_track_sel],range=(Qweightmin,Qweightmax), bins=50, label="PV tracks", alpha=0.5, weights=np.ones_like(predictedQWeightsarray[pv_track_sel]) / assoc_scale)
    ax.hist(predictedQWeightsarray[pu_track_sel],range=(Qweightmin,Qweightmax), bins=50, label="PU tracks", alpha=0.5)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("a.u.", horizontalalignment='right', y=1.0)
    #ax.set_title("Histogram weights for PU and PV tracks (normalised)")
    ax.set_yscale("log")

    ax.legend()
    plt.tight_layout()
    plt.savefig("%s/Qcorr-assoc-1d-norm.png" % outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(predictedQWeightsarray, assoc_QNN_array, range=((Qweightmin,Qweightmax),(0,1)),bins=50, norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track-to-Vertex Association Flag", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] ,ax=ax)
    cbar.set_label('# Tracks')
    plt.tight_layout()
    plt.savefig("%s/Qcorr-assoc.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(predictedQWeightsarray, trk_z0_array, range=((Qweightmin,Qweightmax),(0,1)), bins=50, norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track $z_0$ [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    plt.tight_layout()
    plt.savefig("%s/Qcorr-z0.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(predictedQWeightsarray, trk_mva_array, range=((Qweightmin,Qweightmax),(0,1)), bins=50, norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track MVA", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    plt.tight_layout()
    plt.savefig("%s/Qcorr-mva.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
    hidst2d = ax.hist2d(predictedQWeightsarray, trk_pt_array, range=((Qweightmin,Qweightmax),(0,1)), bins=50, norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track $p_T$ [GeV]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')

    plt.tight_layout()
    plt.savefig("%s/Qcorr-pt.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(predictedQWeightsarray, np.abs(trk_eta_array), range=((Qweightmin,Qweightmax),(0,1)), bins=50, norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track $|\\eta|$", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    plt.tight_layout()
    plt.savefig("%s/Qcorr-abs-eta.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(predictedQWeightsarray, trk_eta_array, range=((Qweightmin,Qweightmax),(0,1)), bins=50, norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track $\\eta$", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    plt.tight_layout()
    plt.savefig("%s/Qcorr-eta.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(predictedQWeightsarray, trk_chi2rphi_array, range=((Qweightmin,Qweightmax),(0,1)), bins=50, norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track $\\chi^2_{r\\phi}$", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    plt.tight_layout()
    plt.savefig("%s/Qcorr-chi2rphi.png" % outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(predictedQWeightsarray, trk_chi2rz_array, range=((Qweightmin,Qweightmax),(0,1)), bins=50, norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track $\\chi^2_{rz}$", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    plt.tight_layout()
    plt.savefig("%s/Qcorr-chi2rz.png" % outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(predictedQWeightsarray, trk_bendchi2_array , range=((Qweightmin,Qweightmax),(0,1)), bins=50, norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track $\\chi^2_{bend}$", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    plt.tight_layout()
    plt.savefig("%s/Qcorr-chi2bend.png" % outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(predictedQWeightsarray, predictedDAWeightsarray, bins=50,range=((Qweightmin,Qweightmax),(Qweightmin,Qweightmax)), norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Quantised weights", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Weights", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    plt.tight_layout()
    plt.savefig("%s/Qweightvsweight.png" % outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(assoc_QNN_array, assoc_DANN_array,range=((0,1),(0,1)), bins=50, norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Quantised Track-to-Vertex Association flag", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Track-to-Vertex Association Flag", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    plt.tight_layout()
    plt.savefig("%s/Qassocvsassoc.png" % outputFolder)
    plt.close()

    do_scatter = False
    if (do_scatter):
        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
        
        ax.scatter(predictedQWeightsarray, trk_z0_array, label="z0")
        ax.set_xlabel("weight")
        ax.set_ylabel("variable")
        ax.set_title("Correlation between predicted weight and track variables")
        ax.legend()
        plt.savefig("%s/scatter-z0.png" %  outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
        
        ax.scatter(predictedQWeightsarray, trk_pt_array, label="pt")
        ax.set_xlabel("weight")
        ax.set_ylabel("variable")
        ax.set_title("Correlation between predicted weight and track variables")
        ax.legend()
        plt.savefig("%s/scatter-pt.png" % outputFolder)
        plt.close()

        plt.clf()
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
        
        ax.scatter(predictedQWeightsarray, trk_eta_array, label="eta")
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
    figure=plotz0_residual([(z0_PV_array-z0_QNN_array)],
                          [(z0_PV_array-z0_FH_array),(z0_PV_array-z0_FHMVA_array),(z0_PV_array-z0_FHnoFake_array)],
                          ["ArgMax"],
                          ["Base","MVA Cut","No Fakes"])
    plt.savefig("%s/Z0Residual.png" % outputFolder)
    plt.close()

    if PVROCs:

        plt.clf()
        figure=plotPV_roc(assoc_PV_array,[assoc_QNN_array],
                         [assoc_FH_array,assoc_FHMVA_array,assoc_FHnoFake_array],
                         ["ArgMax"],
                         ["Base","BDT Cut","No Fakes"])
        plt.savefig("%s/PVROC.png" % outputFolder)

        plt.clf()
        figure=plotPV_roc(assoc_PV_array,[assoc_DANN_array,assoc_QNN_array],
                        [assoc_FH_array],
                        ["NN","QNN"],
                        ["Baseline"])
        plt.savefig("%s/QcompPVROC.png" % outputFolder)
        plt.close()

    plt.clf()
    figure=plotz0_percentile([(z0_PV_array-z0_QNN_array)],
                             [(z0_PV_array-z0_FH_array),(z0_PV_array-z0_FHMVA_array),(z0_PV_array-z0_FHnoFake_array)],
                             ["ArgMax"],
                             ["Base","BDT Cut","No Fakes"])
    plt.savefig("%s/Z0percentile.png" % outputFolder)
    plt.close()

    plt.clf()
    figure=plotz0_residual([(z0_PV_array-z0_DANN_array),(z0_PV_array-z0_QNN_array)],
                          [(z0_PV_array-z0_FH_array)],
                          ["NN               ","QNN             "],
                          ["Baseline         "])
    plt.savefig("%s/QcompZ0Residual.png" % outputFolder)
    plt.close()




    #########################################################################################
    #                                                                                       #
    #                                   MET Residual Plots                                  #  
    #                                                                                       #
    #########################################################################################

    plt.clf()
    figure=plotMET_residual([(MET_QNN_bestQ_array )],
                             [(MET_FH_array ),(MET_FHMVA_array ),(MET_FHnoFake_array ),(actual_trkMET_array )],
                             ["ArgMax thresh=" + str(np.argmin(MET_QNN_Quartile_array)/num_threshold)],
                             ["Base","BDT Cut","No Fakes","PV Tracks"],range=(-100,100),logrange=(-300,300),actual=actual_MET_array)
    plt.savefig("%s/METbestQresidual.png" % outputFolder)
    plt.close()

    plt.clf()
    figure=plotMETphi_residual([(METphi_QNN_bestQ_array )],
                             [(METphi_FH_array ),(METphi_FHMVA_array ),(METphi_FHnoFake_array ),(actual_trkMETphi_array)],
                             ["ArgMax thresh=" + str(np.argmin(MET_QNN_Quartile_array)/num_threshold)],
                             ["Base","BDT Cut","No Fakes","PV Tracks"],actual=actual_METphi_array)
    plt.savefig("%s/METphibestQresidual.png" % outputFolder)
    plt.close()

    plt.clf()
    figure=plotMET_residual([(MET_QNN_bestRMS_array )],
                             [(MET_FH_array ),(MET_FHMVA_array ),(MET_FHnoFake_array ),(actual_trkMET_array )],
                             ["ArgMax thresh=" + str(np.argmin(MET_QNN_RMS_array)/num_threshold)],
                             ["Base","BDT Cut","No Fakes","PV Tracks"],range=(-100,100),logrange=(-300,300),actual=actual_MET_array)
    plt.savefig("%s/METbestRMSresidual.png" % outputFolder)
    plt.close()

    plt.clf()
    figure=plotMETphi_residual([(METphi_QNN_bestRMS_array )],
                             [(METphi_FH_array ),(METphi_FHMVA_array ),(METphi_FHnoFake_array ),(actual_trkMETphi_array )],
                            ["ArgMax thresh=" + str(np.argmin(MET_QNN_RMS_array)/num_threshold)],
                             ["Base","BDT Cut","No Fakes","PV Tracks"],actual=actual_METphi_array)
    plt.savefig("%s/METphibestRMSresidual.png" % outputFolder)
    plt.close()

    plt.clf()
    figure=plotMET_residual([(MET_DANN_bestQ_array ),(MET_QNN_bestQ_array )],
                             [(MET_FH_array )],
                             ["NN thresh =  " + str(np.argmin(MET_DANN_Quartile_array)/num_threshold)+"    ","QNN thresh = " + str(np.argmin(MET_QNN_Quartile_array)/num_threshold)+"    "],
                             ["Baseline            "],range=(-100,100),logrange=(-300,300),actual=actual_MET_array)
    plt.savefig("%s/QcompMETbestQresidual.png" % outputFolder)
    plt.close()

    plt.clf()
    figure=plotMET_residual([(MET_DANN_bestRMS_array ),(MET_QNN_bestRMS_array )],
                             [(MET_FH_array )],
                             ["NN thresh =  " + str(np.argmin(MET_DANN_RMS_array)/num_threshold)+"    ","QNN thresh = " + str(np.argmin(MET_QNN_RMS_array)/num_threshold)+"    "],
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
                             ["ArgMax thresh=" + str(np.argmin(MET_QNN_Quartile_array)/num_threshold)],
                             ["Base","BDT Cut","No Fakes","PV Tracks"],range=(-1,2),logrange=(-1,30),relative=True,actual=actual_MET_array)
    plt.savefig("%s/relMETbestQresidual.png" % outputFolder)
    plt.close()

    plt.clf()
    figure=plotMET_residual([(MET_QNN_bestRMS_array )],
                             [(MET_FH_array ),(MET_FHMVA_array ),(MET_FHnoFake_array ),(actual_trkMET_array )],
                             ["ArgMax thresh=" + str(np.argmin(MET_QNN_RMS_array)/num_threshold)],
                             ["Base","BDT Cut","No Fakes","PV Tracks"],range=(-1,2),logrange=(-1,30),relative=True,actual=actual_MET_array)
    plt.savefig("%s/relMETbestRMSresidual.png" % outputFolder)
    plt.close()

    plt.clf()
    figure=plotMET_residual([(MET_DANN_bestQ_array ),(MET_QNN_bestQ_array )],
                             [(MET_FH_array )],
                             ["NN thresh =  " + str(np.argmin(MET_DANN_Quartile_array)/num_threshold)+"         ","QNN thresh = " + str(np.argmin(MET_QNN_Quartile_array)/num_threshold)+"         "],
                             ["Baseline            "],range=(-1,2),logrange=(-1,30),relative=True,actual=actual_MET_array)
    plt.savefig("%s/QcomprelMETbestQresidual.png" % outputFolder)
    plt.close()

    plt.clf()
    figure=plotMET_residual([(MET_DANN_bestRMS_array ),(MET_QNN_bestRMS_array )],
                             [(MET_FH_array )],
                             ["NN thresh =  " + str(np.argmin(MET_DANN_RMS_array)/num_threshold)+"         ","QNN thresh = " + str(np.argmin(MET_QNN_RMS_array)/num_threshold)+"         "],
                             ["Baseline            "],range=(-1,2),logrange=(-1,30),relative=True,actual=actual_MET_array)
    plt.savefig("%s/QcomprelMETbestRMSresidual.png" % outputFolder)
    plt.close()

    plt.clf()
    figure=plotMET_residual([(MET_DANN_bestQ_array ),(MET_QNN_bestQ_array )],
                             [(MET_FH_array )],
                             ["NN thresh =  " + str(np.argmin(MET_DANN_Quartile_array)/num_threshold)+"         ","QNN thresh = " + str(np.argmin(MET_QNN_Quartile_array)/num_threshold)+"         "],
                             ["Baseline            "],range=(-1,2),logrange=(-1,2),relative=True,actual=actual_MET_array,logbins=True)
    plt.savefig("%s/QcomprelMETbestQresidual_logbins.png" % outputFolder)
    plt.close()

    plt.clf()
    figure=plotMET_residual([(MET_DANN_bestRMS_array ),(MET_QNN_bestRMS_array )],
                             [(MET_FH_array )],
                             ["NN thresh =  " + str(np.argmin(MET_DANN_RMS_array)/num_threshold)+"         ","QNN thresh = " + str(np.argmin(MET_QNN_RMS_array)/num_threshold)+"         "],
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
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
    ax.hist(z0_FH_array,range=(-15,15),bins=120,density=True,color='r',histtype="step",label="FastHisto Base")
    ax.hist(z0_FHres_array,range=(-15,15),bins=120,density=True,color='g',histtype="step",label="FastHisto with z0 res")
    ax.hist(z0_QNN_array,range=(-15,15),bins=120,density=True,color='b',histtype="step",label="CNN")
    ax.hist(z0_PV_array,range=(-15,15),bins=120,density=True,color='y',histtype="step",label="Truth")
    ax.grid(True)
    ax.set_xlabel('$z_0$ [cm]',ha="right",x=1)
    ax.set_ylabel('Events',ha="right",y=1)
    ax.legend() 
    plt.tight_layout()
    plt.savefig("%s/Qz0hist.png" % outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_PV_array, (z0_PV_array-z0_FH_array), bins=60,range=((-15,15),(-30,30)) ,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("True PV $z_0$ [cm]", horizontalalignment='right', x=1.0)
    ax.set_ylabel("True PV $z_0$ - Reco PV $z_0$ Baseline [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/FHerr_vs_z0.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_PV_array, (z0_PV_array-z0_FHnoFake_array), bins=60,range=((-15,15),(-30,30)) ,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("True PV $z_0$ [cm]", horizontalalignment='right', x=1.0)
    ax.set_ylabel("True PV $z_0$ - Reco PV $z_0$ Baseline [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/FHnoFakeerr_vs_z0.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_PV_array, (z0_PV_array-z0_FHMVA_array), bins=60,range=((-15,15),(-30,30)) ,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("True PV $z_0$ [cm]", horizontalalignment='right', x=1.0)
    ax.set_ylabel("True PV $z_0$ - Reco PV $z_0$ Baseline [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/FHMVAerr_vs_z0.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_PV_array, (z0_PV_array-z0_QNN_array), bins=60,range=((-15,15),(-30,30)), norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("True PV $z_0$ [cm]", horizontalalignment='right', x=1.0)
    ax.set_ylabel("True PV $z_0$ - Reco PV $z_0$ QNN [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/QNNerr_vs_z0.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_PV_array, z0_FH_array, bins=60,range=((-15,15),(-15,15)) ,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("True PV $z_0$ [cm]", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Reco PV $z_0$ Baseline [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/FH_vs_z0.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_PV_array, z0_FHMVA_array, bins=60,range=((-15,15),(-15,15)) ,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("True PV $z_0$ [cm]", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Reco PV $z_0$ Baseline [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/FHMVA_vs_z0.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_PV_array, z0_FHnoFake_array, bins=60,range=((-15,15),(-15,15)) ,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("True PV $z_0$ [cm]", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Reco PV $z_0$ Baseline [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/FHnoFake_vs_z0.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_PV_array, z0_QNN_array, bins=60,range=((-15,15),(-15,15)), norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("True PV $z_0$ [cm]", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Reco PV $z_0$ QNN [cm]", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/QNN_vs_z0.png" %  outputFolder)
    plt.close()

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_DANN_array, z0_QNN_array, bins=60,range=((-15,15),(-15,15)), norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
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

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
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
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
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
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
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
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
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
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
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
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
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
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
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
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
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
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
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
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
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
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
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
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
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
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
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
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
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
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
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
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
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
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
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
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
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
    hep.cms.label(llabel="Phase-2 Simulation",rlabel="14 TeV, 200 PU",ax=ax)
    
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
    
    experiment.log_asset_folder(outputFolder, step=None, log_file_name=True)
    experiment.log_asset(sys.argv[2]+'.yaml')
