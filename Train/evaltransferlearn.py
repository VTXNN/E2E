import comet_ml
import tensorflow as tf
import numpy as np

import glob
import sklearn.metrics as metrics
import vtx
import yaml
import sys

from train import *

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
#hep.set_style("CMSTex")
#hep.cms.label()
#hep.cms.text("Simulation")
plt.style.use(hep.style.CMS)

SMALL_SIZE = 12
MEDIUM_SIZE = 13
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

kf = sys.argv[1]

if kf == "NewKF":
    test_files = glob.glob("NewKFData/Test/*.tfrecord")
    z0 = 'trk_z0'
elif kf == "OldKF":
    test_files = glob.glob("OldKFData/Test/*.tfrecord")
    z0 = 'corrected_trk_z0'


with open(kf+'experimentkey.txt') as f:
    first_line = f.readline()

EXPERIMENT_KEY = first_line

if (EXPERIMENT_KEY is not None):
    # There is one, but the experiment might not exist yet:
    api = comet_ml.API() # Assumes API key is set in config/env
    try:
        api_experiment = api.get_experiment_by_id(EXPERIMENT_KEY)
    except Exception:
        api_experiment = None

experiment = comet_ml.ExistingExperiment(
        previous_experiment=EXPERIMENT_KEY,
        log_env_details=True, # to continue env logging
        log_env_gpu=True,     # to continue GPU logging
        log_env_cpu=True,     # to continue CPU logging
    )

nMaxTracks = 250

def predictFastHisto(value,weight):
    z0List = []
    halfBinWidth = 0.5*30./256.
    for ibatch in range(value.shape[0]):
        hist,bin_edges = np.histogram(value[ibatch],256,range=(-15,15),weights=weight[ibatch])
        hist = np.convolve(hist,[1,1,1],mode='same')
        z0Index= np.argmax(hist)
        z0 = -15.+30.*z0Index/256.+halfBinWidth
        z0List.append([z0])
    return np.array(z0List,dtype=np.float32)

def predictFastHistoZ0res(value,weight,eta):
    eta_bins = np.array([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2,2.2,2.4,np.inf])
    #res_bins = np.array([0.0,0.14,0.14,0.14,0.14,0.16,0.18,0.23,0.23,0.3,0.35,0.38,0.42,0.5,1])

    res_bins = np.array([0.0,0.014,0.014,0.014,0.014,0.032,0.036,0.092,0.092,0.15,0.2,0.38,0.42,0.5,1])

    def res_function(eta):
        res = 0.1 + 0.2*eta**2
        return res
    
    z0List = []
    halfBinWidth = 0.5*30./256.
    for ibatch in range(value.shape[0]):
        #res = res_bins[np.digitize(abs(eta[ibatch]),eta_bins)]
        res = res_function(eta[ibatch])
        hist,bin_edges = np.histogram(value[ibatch],256,range=(-15,15),weights=weight[ibatch]/res)
        hist = np.convolve(hist,[1,1,1],mode='same')
        z0Index= np.argmax(hist)
        z0 = -15.+30.*z0Index/256.+halfBinWidth
        z0List.append([z0])
    return np.array(z0List,dtype=np.float32)

def predictMET(pt,phi,predictedAssoc,threshold):

    met_px_list = []
    met_py_list = []
    met_pt_list = []
    met_phi_list = []


    

    for ibatch in range(pt.shape[0]):
        if type(predictedAssoc) == np.ndarray:
            assoc = np.expand_dims(predictedAssoc[ibatch],1)
        else:
            assoc = predictedAssoc[ibatch].numpy()
        newpt = pt[ibatch].numpy()
        newphi = phi[ibatch].numpy()

        assoc[assoc > threshold] = 1
        NN_track_sel = assoc == 1


        met_px = np.sum(newpt[NN_track_sel[:,0]]*np.cos(newphi[NN_track_sel[:,0]]))
        met_py = np.sum(newpt[NN_track_sel[:,0]]*np.sin(newphi[NN_track_sel[:,0]]))
        met_px_list.append([met_px])
        met_py_list.append([met_py])
        met_pt_list.append([math.sqrt(met_px**2+met_py**2)])
        met_phi_list.append([math.atan2(met_py,met_px)])
    return  [np.array(met_px_list,dtype=np.float32),
            np.array(met_py_list,dtype=np.float32),
            np.array(met_pt_list,dtype=np.float32),
            np.array(met_phi_list,dtype=np.float32)]

def FastHistoAssoc(PV,trk_z0,trk_eta,threshold=1):
    eta_bins = np.array([0.0,0.7,1.0,1.2,1.6,2.0,2.4])
    deltaz_bins = threshold*np.array([0.0,0.4,0.6,0.76,1.0,1.7,2.2,0.0])
    

    eta_bin = np.digitize(abs(trk_eta),eta_bins)
    deltaz = abs(trk_z0 - PV)

    assoc = (deltaz < deltaz_bins[eta_bin])

    return np.array(assoc,dtype=np.float32)

def plotz0_residual(bmNNdiff,rmNNdiff,FHdiff):
    plt.clf()
    figure = plt.figure(figsize=(10,10))
    qz0_bmNN = np.percentile(bmNNdiff,[32,50,68])
    qz0_rmNN = np.percentile(rmNNdiff,[32,50,68])
    qz0_FH = np.percentile(FHdiff,[32,50,68])

    plt.hist(bmNNdiff,bins=50,range=(-1,1),histtype="step",label=f"Teacher NN: = (%.4f,%.4f), Width = %.4f, Centre = %.4f" %(qz0_bmNN[0],qz0_bmNN[2],qz0_bmNN[2]-qz0_bmNN[0], qz0_bmNN[1]))
    plt.hist(rmNNdiff,bins=50,range=(-1,1),histtype="step",label=f"Learner NN: = (%.4f,%.4f), Width = %.4f, Centre = %.4f" %(qz0_rmNN[0],qz0_rmNN[2],qz0_rmNN[2]-qz0_rmNN[0], qz0_rmNN[1]))
    plt.hist(FHdiff,bins=50,range=(-1,1),histtype="step",label=f"FH: = (%.4f,%.4f), Width = %.4f, Centre = %.4f" %(qz0_FH[0],qz0_FH[2],qz0_FH[2]-qz0_FH[0], qz0_FH[1]))
    plt.grid(True)
    plt.xlabel('$z_0$ Residual [cm]',ha="right",x=1)
    plt.ylabel('Events',ha="right",y=1)
    plt.legend() 
    plt.tight_layout()
    return figure

def plotPV_roc(actual,bmNNpred,rmNNpred,FHpred):
    precisionbmNN = []
    precisionrmNN = []

    recallbmNN = []
    FPRbmNN = []

    recallrmNN = []
    FPRrmNN = []

    bmNNpred = (bmNNpred - min(bmNNpred))/(max(bmNNpred) - min(bmNNpred))
    rmNNpred = (rmNNpred - min(rmNNpred))/(max(rmNNpred) - min(rmNNpred))

    thresholds = np.linspace(0,1,100)

    for i,threshold in enumerate(thresholds):
        print("Testing ROC threshold: "+str(i) + " out of "+str(len(thresholds)))
        tnbmNN, fpbmNN, fnbmNN, tpbmNN = metrics.confusion_matrix(actual, NNbmpred>threshold).ravel()
        precisionbmNN.append( tpbmNN / (tpbmNN + fpbmNN) )
        recallNN.append(tpbmNN / (tpbmNN + fnbmNN) )
        FPRNN.append(fpbmNN / (fpbmNN + tnbmNN) )

        tnrmNN, fprmNN, fnrmNN, tprmNN = metrics.confusion_matrix(actual, NNrmpred>threshold).ravel()
        precisionrmNN.append( tprmNN / (tprmNN + fprmNN) )
        recallNN.append(tprmNN / (tprmNN + fnrmNN) )
        FPRNN.append(fprmNN / (fprmNN + tnrmNN) )



    tnFH, fpFH, fnFH, tpFH = metrics.confusion_matrix(actual, FHpred).ravel()

    precisionFH = tpFH / (tpFH + fpFH) 
    recallFH = tpFH / (tpFH + fnFH) 

    TPRFH = recallFH
    FPRFH = fpFH / (fpFH + tnFH) 

    plt.clf()
    fig,ax = plt.subplots(1,2,figsize=(20,10))

    ax[0].set_title("Purity Efficiency Plot" ,loc='left')
    ax[0].plot(recallbmNN,precisionbmNN,label="Teacher NN")
    ax[0].plot(recallrmNN,precisionrmNN,label="Learner NN")
    ax[0].scatter(recallFH,precisionFH,color='orange',label="FH")
    ax[0].grid(True)
    ax[0].set_xlabel('Efficiency',ha="right",x=1)
    ax[0].set_ylabel('Purity',ha="right",y=1)
    ax[0].set_xlim([0.01,1])
    ax[0].legend()

    ax[1].set_title("Reciever Operator Characteristic Plot" ,loc='left')
    ax[1].plot(recallbmNN,FPRbmNN,label=f"Teacher NN AUC: %.4f" %(metrics.roc_auc_score(actual,NNbmpred)))
    ax[1].plot(recallrmNN,FPRrmNN,label=f"Learner NN AUC: %.4f" %(metrics.roc_auc_score(actual,NNrmpred)))
    ax[1].scatter(TPRFH,FPRFH,color='orange',label=f"FH AUC: %.4f" %(metrics.roc_auc_score(actual,FHpred)))
    ax[1].grid(True)
    ax[1].set_yscale("log")
    ax[1].set_xlabel('True Positive Rate',ha="right",x=1)
    ax[1].set_ylabel('False Positive Rate',ha="right",y=1)
    ax[0].set_ylim([1e-4,1])
    ax[1].legend()
    plt.tight_layout()
    return fig

def plotz0_percentile(bmNNdiff,rmNNdiff,FHdiff):
    plt.clf()
    figure = plt.figure(figsize=(10,10))

    percentiles = np.linspace(0,100,100)
    bmNNpercentiles = np.percentile(bmNNdiff,percentiles)
    rmNNpercentiles = np.percentile(rmNNdiff,percentiles)
    FHpercentiles = np.percentile(FHdiff,percentiles)
    
    plt.plot(percentiles,abs(bmNNpercentiles),label=f"Teacher NN minimum: %.4f at : %.2f " %(min(abs(bmNNpercentiles)),np.argmin(abs(bmNNpercentiles))))
    plt.plot(percentiles,abs(rmNNpercentiles),label=f"Learner NN minimum: %.4f at : %.2f " %(min(abs(rmNNpercentiles)),np.argmin(abs(rmNNpercentiles))))
    plt.plot(percentiles,abs(FHpercentiles),label=f"FH minimum: %.4f at : %.2f " %(min(abs(FHpercentiles)),np.argmin(abs(FHpercentiles))))
    plt.grid(True)
    plt.xlabel('Percentile',ha="right",x=1)
    plt.ylabel('$|\\delta z_{0}| [cm]$',ha="right",y=1)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    

    return figure

def plotPV_classdistribution(actual,bmNNpred,rmNNpred,FHpred):
    #NNpred = (NNpred - min(NNpred))/(max(NNpred) - min(NNpred))
    bmgenuine = []
    bmfake = []

    rmgenuine = []
    rmfake = []

    FHgenuine = []
    FHfake = []

    for i in range(len(actual)):
        if actual[i] == 1:
            bmgenuine.append(bmNNpred[i])
            rmgenuine.append(rmNNpred[i])
            FHgenuine.append(FHpred[i])
        else:
            bmfake.append(bmNNpred[i])
            rmfake.append(rmNNpred[i])
            FHfake.append(FHpred[i])


    plt.clf()  
    fig, ax = plt.subplots(1,3, figsize=(30,10)) 

    ax[0].set_title("Teacher NN Normalised Class Distribution" ,loc='left')
    ax[0].hist(bmgenuine,color='g',bins=20,range=(-10,10),histtype="step",label="Genuine",density=True,linewidth=2)
    ax[0].hist(bmfake,color='r',bins=20,range=(-10,10),histtype="step",label="Fake",density=True,linewidth=2)
    ax[0].grid()
    #ax[0].set_yscale("log")
    ax[0].set_xlabel("Teacher NN Output",ha="right",x=1)
    ax[0].set_ylabel("a.u.",ha="right",y=1)
    ax[0].legend(loc="upper center")

    ax[1].set_title("Learner NN Normalised Class Distribution" ,loc='left')
    ax[1].hist(rmgenuine,color='g',bins=20,range=(-10,10),histtype="step",label="Genuine",density=True,linewidth=2)
    ax[1].hist(rmfake,color='r',bins=20,range=(-10,10),histtype="step",label="Fake",density=True,linewidth=2)
    ax[1].grid()
    #ax[1].set_yscale("log")
    ax[1].set_xlabel("Learner NN Output",ha="right",x=1)
    ax[1].set_ylabel("a.u.",ha="right",y=1)
    ax[1].legend(loc="upper center")


    ax[1].set_title("FH Normalised Class Distribution" ,loc='left')
    ax[1].hist(FHgenuine,color='g',bins=20,range=(0,1),histtype="step",label="Genuine",density=True,linewidth=2)
    ax[1].hist(FHfake,color='r',bins=20,range=(0,1),histtype="step",label="Fake",density=True,linewidth=2)
    ax[1].grid()
    #ax[1].set_yscale("log")
    ax[1].set_xlabel("FH Output",ha="right",x=1)
    ax[1].set_ylabel("a.u.",ha="right",y=1)
    ax[1].legend(loc="upper center")
    plt.tight_layout()
    return fig

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

if __name__=="__main__":
    with open(sys.argv[2]+'.yaml', 'r') as f:
        config = yaml.load(f)

    outputFolder = kf+config['eval_folder']

    features = {
            "pvz0": tf.io.FixedLenFeature([1], tf.float32),
            #"pv2z0": tf.io.FixedLenFeature([1], tf.float32),
            "tp_met_pt":tf.io.FixedLenFeature([1], tf.float32),
            "pv_trk_met_pt":tf.io.FixedLenFeature([1], tf.float32),
            "true_met_pt":tf.io.FixedLenFeature([1], tf.float32),

            "tp_met_px":tf.io.FixedLenFeature([1], tf.float32),
            "pv_trk_met_px":tf.io.FixedLenFeature([1], tf.float32),
            "true_met_px":tf.io.FixedLenFeature([1], tf.float32),

            "tp_met_py":tf.io.FixedLenFeature([1], tf.float32),
            "pv_trk_met_py":tf.io.FixedLenFeature([1], tf.float32),
            "true_met_py":tf.io.FixedLenFeature([1], tf.float32),

            "tp_met_phi":tf.io.FixedLenFeature([1], tf.float32),
            "pv_trk_met_phi":tf.io.FixedLenFeature([1], tf.float32),
            "true_met_phi":tf.io.FixedLenFeature([1], tf.float32),

            "trk_fromPV":tf.io.FixedLenFeature([nMaxTracks], tf.float32),
            "trk_hitpattern": tf.io.FixedLenFeature([nMaxTracks*11], tf.float32), 
            "PV_hist"  :tf.io.FixedLenFeature([256,1], tf.float32),
    }

    trackFeatures = [
            'trk_z0',
            'normed_trk_pt',
            'normed_trk_eta', 
            'normed_trk_invR',
            'binned_trk_chi2rphi', 
            'binned_trk_chi2rz', 
            'binned_trk_bendchi2',
            'normed_trk_overeta',
            'trk_z0_res',
            'trk_pt',
            'log_pt',
            'trk_eta',
            'trk_phi',
            'trk_MVA1',
            'corrected_trk_z0'
        ]

    for trackFeature in trackFeatures:
        features[trackFeature] = tf.io.FixedLenFeature([nMaxTracks], tf.float32)

            
    bignetwork = vtx.nn.E2Ecomparekernel(
            nbins=256,
            ntracks=nMaxTracks, 
            nweightfeatures=6, 
            nfeatures=6, 
            nweights=1, 
            nlatent=2, 
            activation='relu',
            regloss=1e-10
        )


    bigmodel = network.createE2EModel()
    bigoptimizer = tf.keras.optimizers.Adam(lr=0.01)
    bigmodel.compile(
            bigoptimizer,
            loss=[
                tf.keras.losses.MeanAbsoluteError(),
                #tf.keras.losses.MeanSquaredError(),
                tf.keras.losses.BinaryCrossentropy(from_logits=True),
                lambda y,x: 0.,
                tf.keras.losses.MeanSquaredError()
                
            ],
            metrics=[
                tf.keras.metrics.BinaryAccuracy(threshold=0.,name='assoc_acc') #use thres=0 here since logits are used
            ],
            loss_weights=[1.,1.,0.,0.]
        )
    bigmodel.summary()
    bigmodel.load_weights(kf+"bmweights_"+str( config['epochs'] - 1)+".tf")

    rednetwork = vtx.nn.E2EReduced(
            nbins=256,
            ntracks=nMaxTracks, 
            nweightfeatures=6, 
            nfeatures=6, 
            nweights=1, 
            nlatent=2, 
            activation='relu',
            regloss=1e-10
        )


    redmodel = network.createE2EModel()
    redoptimizer = tf.keras.optimizers.Adam(lr=0.01)
    redmodel.compile(
            redoptimizer,
            loss=[
                tf.keras.losses.MeanAbsoluteError(),
                #tf.keras.losses.MeanSquaredError(),
                tf.keras.losses.BinaryCrossentropy(from_logits=True),
                lambda y,x: 0.,
                tf.keras.losses.MeanSquaredError()
                
            ],
            metrics=[
                tf.keras.metrics.BinaryAccuracy(threshold=0.,name='assoc_acc') #use thres=0 here since logits are used
            ],
            loss_weights=[1.,1.,0.,0.]
        )
    redmodel.summary()
    redmodel.load_weights(kf+"rmweights_"+str( config['epochs'] - 1)+".tf")

    predictedZ0_FH = []
    predictedZ0_bmNN = []
    predictedZ0_rmNN = []

    predictedAssoc_bmNN = []
    predictedAssoc_rmNN = []
    predictedAssoc_FH = []

    actual_Assoc = []
    actual_PV = []

    threshold = -1

    for step,batch in enumerate(setup_pipeline(test_files)):

        trackFeatures = np.stack([batch[feature] for feature in [
                    'normed_trk_pt','normed_trk_eta','trk_MVA1','binned_trk_chi2rphi', 'binned_trk_chi2rz', 'binned_trk_bendchi2'
            ]],axis=2)
        WeightFeatures = np.stack([batch[feature] for feature in [
                 'normed_trk_pt','normed_trk_eta','trk_MVA1','binned_trk_chi2rphi', 'binned_trk_chi2rz', 'binned_trk_bendchi2'
            ]],axis=2)
            #trackFeatures = np.concatenate([trackFeatures,batch['trk_hitpattern']],axis=2)
            #trackFeatures = np.concatenate([trackFeatures,batch['trk_z0_res']],axis=2)
        nBatch = batch['pvz0'].shape[0]
        FH = predictFastHisto(batch[z0],batch['trk_pt'])
        predictedZ0_FH.append(FH)
        predictedZ0_FHz0res.append(predictFastHistoZ0res(batch[z0],batch['trk_pt'],batch['trk_eta']))

        actual_Assoc.append(batch["trk_fromPV"])
        actual_PV.append(batch['pvz0'])
        FHassoc = FastHistoAssoc(predictFastHisto(batch[z0],batch['trk_pt']),batch[z0],batch['trk_eta'])
        predictedAssoc_FH.append(FHassoc)
                
        predictedZ0_bmNN_temp, predictedAssoc_bmNN_temp, predictedWeights_bmNN, predictedHists_bmNN = bigmodel.predict_on_batch(
                        [batch[z0],WeightFeatures,trackFeatures]
                    )

        predictedZ0_rmNN_temp, predictedAssoc_rmNN_temp, predictedWeights_rmNN, predictedHists_rmNN = reducedmodel.predict_on_batch(
                        [batch[z0],WeightFeatures,trackFeatures]
                    )

        predictedZ0_bmNN.append(predictedZ0_bmNN_temp)
        predictedAssoc_bmNN.append(predictedAssoc_bmNN_temp)

        predictedZ0_rmNN.append(predictedZ0_rmNN_temp)
        predictedAssoc_rmNN.append(predictedAssoc_rmNN_temp)


    z0_bmNN_array = np.concatenate(predictedZ0_bmNN).ravel()
    z0_rmNN_array = np.concatenate(predictedZ0_rmNN).ravel()

    z0_FH_array = np.concatenate(predictedZ0_FH).ravel()
    z0_PV_array = np.concatenate(actual_PV).ravel()

    assoc_bmNN_array = np.concatenate(predictedAssoc_bmNN).ravel()
    assoc_rmNN_array = np.concatenate(predictedAssoc_rmNN).ravel()

    assoc_FH_array = np.concatenate(predictedAssoc_FH).ravel()
    assoc_PV_array = np.concatenate(actual_Assoc).ravel()


    plt.clf()
    figure=plotz0_residual((z0_PV_array-z0_bmNN_array),(z0_PV_array-z0_rmNN_array),(z0_PV_array-z0_FH_array))
    plt.savefig("%s/Z0Residual.png" % outputFolder)

    plt.clf()
    figure=plotPV_roc(assoc_PV_array,assoc_bmNN_array,assoc_rmNN_array,assoc_FH_array)
    plt.savefig("%s/PVROC.png" % outputFolder)

    plt.clf()
    figure=plotPV_classdistribution(assoc_PV_array,assoc_bmNN_array,assoc_rmNN_array,assoc_FH_array)
    plt.savefig("%s/ClassDistribution.png" % outputFolder)

    plt.clf()
    figure=plotz0_percentile((z0_PV_array-z0_bmNN_array),(z0_PV_array-z0_rmNN_array),(z0_PV_array-z0_FH_array))
    plt.savefig("%s/Z0percentile.png" % outputFolder)

    plt.clf()
    plt.hist(z0_FH_array,range=(-15,15),bins=120,density=True,color='r',histtype="step",label="FastHisto Base")
    plt.hist(z0_bmNN_array,range=(-15,15),bins=120,density=True,color='b',histtype="step",label="Teacher NN")
    plt.hist(z0_rmNN_array,range=(-15,15),bins=120,density=True,color='orange',histtype="step",label="Learner NN")
    plt.hist(z0_PV_array,range=(-15,15),bins=120,density=True,color='y',histtype="step",label="Truth")
    plt.grid(True)
    plt.xlabel('$z_0$ [cm]',ha="right",x=1)
    plt.ylabel('Events',ha="right",y=1)
    plt.legend() 
    plt.tight_layout()
    plt.savefig("%s/z0hist.png" % outputFolder)

    
    experiment.log_asset_folder(outputFolder, step=None, log_file_name=True)





