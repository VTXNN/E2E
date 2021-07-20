import comet_ml
import tensorflow as tf
import numpy as np
import yaml
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

def plotz0_residual(NNdiff,FHdiff):
    plt.clf()
    figure = plt.figure(figsize=(10,10))
    qz0_NN = np.percentile(NNdiff,[32,50,68])
    qz0_FH = np.percentile(FHdiff,[32,50,68])
    plt.hist(NNdiff,bins=50,range=(-1,1),histtype="step",label=f"NN: = (%.4f,%.4f), Width = %.4f, Centre = %.4f" %(qz0_NN[0],qz0_NN[2],qz0_NN[2]-qz0_NN[0], qz0_NN[1]))
    plt.hist(FHdiff,bins=50,range=(-1,1),histtype="step",label=f"FH: = (%.4f,%.4f), Width = %.4f, Centre = %.4f" %(qz0_FH[0],qz0_FH[2],qz0_FH[2]-qz0_FH[0], qz0_FH[1]))
    plt.grid(True)
    plt.xlabel('$z_0$ Residual [cm]',ha="right",x=1)
    plt.ylabel('Events',ha="right",y=1)
    plt.legend() 
    plt.tight_layout()
    return figure

def plotPV_roc(actual,NNpred,FHpred):
    precisionNN = []
    recallNN = []
    FPRNN = []

    NNpred = (NNpred - min(NNpred))/(max(NNpred) - min(NNpred))

    thresholds = np.linspace(0,1,100)

    for i,threshold in enumerate(thresholds):
        print("Testing ROC threshold: "+str(i) + " out of "+str(len(thresholds)))
        tnNN, fpNN, fnNN, tpNN = metrics.confusion_matrix(actual, NNpred>threshold).ravel()
        precisionNN.append( tpNN / (tpNN + fpNN) )
        recallNN.append(tpNN / (tpNN + fnNN) )
        FPRNN.append(fpNN / (fpNN + tnNN) )


    tnFH, fpFH, fnFH, tpFH = metrics.confusion_matrix(actual, FHpred).ravel()

    precisionFH = tpFH / (tpFH + fpFH) 
    recallFH = tpFH / (tpFH + fnFH) 

    TPRFH = recallFH
    FPRFH = fpFH / (fpFH + tnFH) 

    plt.clf()
    fig,ax = plt.subplots(1,2,figsize=(20,10))

    ax[0].set_title("Purity Efficiency Plot" ,loc='left')
    ax[0].plot(recallNN,precisionNN,label="NN")
    ax[0].scatter(recallFH,precisionFH,color='orange',label="FH")
    ax[0].grid(True)
    ax[0].set_xlabel('Efficiency',ha="right",x=1)
    ax[0].set_ylabel('Purity',ha="right",y=1)
    ax[0].set_xlim([0.01,1])
    ax[0].legend()

    ax[1].set_title("Reciever Operator Characteristic Plot" ,loc='left')
    ax[1].plot(recallNN,FPRNN,label=f"NN AUC: %.4f" %(metrics.roc_auc_score(actual,NNpred)))
    ax[1].scatter(TPRFH,FPRFH,color='orange',label=f"FH AUC: %.4f" %(metrics.roc_auc_score(actual,FHpred)))
    ax[1].grid(True)
    ax[1].set_yscale("log")
    ax[1].set_xlabel('True Positive Rate',ha="right",x=1)
    ax[1].set_ylabel('False Positive Rate',ha="right",y=1)
    ax[0].set_ylim([1e-4,1])
    ax[1].legend()
    plt.tight_layout()
    return fig

def plotz0_percentile(NNdiff,FHdiff):
    plt.clf()
    figure = plt.figure(figsize=(10,10))

    percentiles = np.linspace(0,100,100)
    NNpercentiles = np.percentile(NNdiff,percentiles)
    FHpercentiles = np.percentile(FHdiff,percentiles)
    
    plt.plot(percentiles,abs(NNpercentiles),label=f"NN minimum: %.4f at : %.2f " %(min(abs(NNpercentiles)),np.argmin(abs(NNpercentiles))))
    plt.plot(percentiles,abs(FHpercentiles),label=f"FH minimum: %.4f at : %.2f " %(min(abs(FHpercentiles)),np.argmin(abs(FHpercentiles))))
    plt.grid(True)
    plt.xlabel('Percentile',ha="right",x=1)
    plt.ylabel('$|\\delta z_{0}| [cm]$',ha="right",y=1)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    

    return figure

def plotPV_classdistribution(actual,NNpred,FHpred):
    #NNpred = (NNpred - min(NNpred))/(max(NNpred) - min(NNpred))
    genuine = []
    fake = []

    FHgenuine = []
    FHfake = []

    for i in range(len(actual)):
        if actual[i] == 1:
            genuine.append(NNpred[i])
            FHgenuine.append(FHpred[i])
        else:
            fake.append(NNpred[i])
            FHfake.append(FHpred[i])


    plt.clf()  
    fig, ax = plt.subplots(1,2, figsize=(20,10)) 

    ax[0].set_title("CNN Normalised Class Distribution" ,loc='left')
    ax[0].hist(genuine,color='g',bins=20,range=(-10,10),histtype="step",label="Genuine",density=True,linewidth=2)
    ax[0].hist(fake,color='r',bins=20,range=(-10,10),histtype="step",label="Fake",density=True,linewidth=2)
    ax[0].grid()
    #ax[0].set_yscale("log")
    ax[0].set_xlabel("CNN Output",ha="right",x=1)
    ax[0].set_ylabel("a.u.",ha="right",y=1)
    ax[0].legend(loc="upper center")


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

def plotMET_residual(NNdiff,trkdiff,tpdiff,threshold,relative=False,true=None):
    plt.clf()
    figure = plt.figure(figsize=(10,10))


    if relative:
        NNdiff = NNdiff/true
        trkdiff = trkdiff/true
        tpdiff = tpdiff/true

    qmet_NN = np.percentile(NNdiff,[32,50,68])
    qmet_trk = np.percentile(trkdiff,[32,50,68])
    qmet_tp = np.percentile(tpdiff,[32,50,68])

    if relative:
        plt.hist(NNdiff,bins=50,range=(-10,1),histtype="step",label=f"NN: = (%.4f,%.4f), Width = %.4f, Centre = %.4f, at PV Assoc Threshold of: %.4f" %(qmet_NN[0],qmet_NN[2],qmet_NN[2]-qmet_NN[0],qmet_NN[1],threshold))
        plt.hist(trkdiff,bins=50,range=(-10,1),histtype="step",label=f"PV Tracks: = (%.4f,%.4f), Width = %.4f, Centre = %.4f" %(qmet_trk[0],qmet_trk[2],qmet_trk[2]-qmet_trk[0],qmet_trk[1]))
        plt.hist(tpdiff,bins=50,range=(-10,1),histtype="step",label=f"FH Tracks: = (%.4f,%.4f), Width = %.4f, Centre = %.4f" %(qmet_tp[0],qmet_tp[2],qmet_tp[2]-qmet_tp[0],qmet_tp[1]))
        plt.xlabel('Relative $E_{miss}^{T}$ Residual',ha="right",x=1)

    else:
        plt.hist(NNdiff,bins=50,range=(-300,300),histtype="step",label=f"NN: = (%.4f,%.4f), Width = %.4f, Centre = %.4f, at PV Assoc Threshold of: %.4f" %(qmet_NN[0],qmet_NN[2],qmet_NN[2]-qmet_NN[0],qmet_NN[1],threshold))
        plt.hist(trkdiff,bins=50,range=(-300,300),histtype="step",label=f"PV Tracks: = (%.4f,%.4f), Width = %.4f, Centre = %.4f" %(qmet_trk[0],qmet_trk[2],qmet_trk[2]-qmet_trk[0],qmet_trk[1]))
        plt.hist(tpdiff,bins=50,range=(-300,300),histtype="step",label=f"FH Tracks: = (%.4f,%.4f), Width = %.4f, Centre = %.4f" %(qmet_tp[0],qmet_tp[2],qmet_tp[2]-qmet_tp[0],qmet_tp[1]))
        plt.xlabel('$E_{miss}^{T}$ Residual [GeV]',ha="right",x=1)

    plt.ylabel('Events',ha="right",y=1)
    plt.grid(True)
    plt.legend() 
    plt.tight_layout()
    return figure

def plotKDEandTracks(tracks,assoc,genPV,predictedPV,weights,weight_label="KDE",threshold=-1):
  plt.clf()
  hist,bin_edges = np.histogram(tracks,256,range=(-15,15),weights=weights)
  plt.bar(bin_edges[:-1],hist,width=30/256,color='grey',alpha=0.5, label=weight_label)

  assoc[assoc > threshold] = 1
  assoc[assoc < threshold] = 0
  pv_track_sel = assoc == 1


  pu_track_sel = assoc == 0

  plt.plot(tracks[pv_track_sel], [0] * len(tracks[pv_track_sel]), '+b', label='PV Trk')
  plt.plot(tracks[pu_track_sel], [0] * len(tracks[pu_track_sel]), '+r', label='PU Trk')
   

  plt.plot([predictedPV, predictedPV], [0, max(hist)], '--k', label='Reco Vx')

  plt.plot([genPV, genPV], [0,max(hist)], '--g', label='True Vx')

  plt.xlabel('z / cm')
  plt.ylabel('density')
  plt.xlim()
  plt.ylim()
  plt.legend()
  plt.show()

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
            'corrected_trk_z0',
            'normed_trk_overeta_squared'
        ]

    for trackFeature in trackFeatures:
        features[trackFeature] = tf.io.FixedLenFeature([nMaxTracks], tf.float32)

            
    network = vtx.nn.E2Ecomparekernel(
            nbins=256,
            ntracks=nMaxTracks, 
            nweightfeatures=7, 
            nfeatures=6, 
            nweights=1, 
            nlatent=2, 
            activation='relu',
            regloss=1e-10
        )


    model = network.createE2EModel()
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(
            optimizer,
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
    model.summary()
    model.load_weights(kf+"weights_"+str( config['epochs'] - 1)+".tf")

    predictedZ0_FH = []
    predictedZ0_FHz0res = []
    predictedZ0_NN = []

    predictedWeights = []

    predictedAssoc_NN = []
    predictedAssoc_FH = []

    actual_Assoc = []
    actual_PV = []

    tp_met_px  = []
    pv_trk_met_px = []
    true_met_px = []
    predicted_met_px  = []

    tp_met_py  = []
    pv_trk_met_py = []
    true_met_py = []
    predicted_met_py = []

    tp_met_phi  = []
    pv_trk_met_phi = []
    true_met_phi = []
    predicted_met_phi = []

    actual_numtracks = []
    actual_pt_total = []

    predictedFH_met_px = []
    predictedFH_met_py = []
    predictedFH_met_phi = []
    predictedFH_met = []
 
    tp_met  = []
    pv_trk_met= []
    true_met = []
    predicted_met = []

    trk_z0 = []
    trk_MVA = []
    trk_phi = []
    trk_pt = []
    trk_z0res = []
    trk_eta = []

    trk_chi2rphi = []
    trk_chi2rz = []
    trk_bendchi2 = []

    threshold = -1

    for step,batch in enumerate(setup_pipeline(test_files)):

        trackFeatures = np.stack([batch[feature] for feature in [
                    'normed_trk_pt',
                    'normed_trk_eta',
                    'trk_MVA1',
                    'binned_trk_chi2rphi',
                    'binned_trk_chi2rz',
                    'binned_trk_bendchi2'
            ]],axis=2)
        WeightFeatures = np.stack([batch[feature] for feature in [
                'normed_trk_pt',
                'normed_trk_eta',
                'trk_MVA1',
                'binned_trk_chi2rphi',
                'binned_trk_chi2rz',
                'binned_trk_bendchi2',
                'normed_trk_overeta_squared'
            ]],axis=2)
            #trackFeatures = np.concatenate([trackFeatures,batch['trk_hitpattern']],axis=2)
            #trackFeatures = np.concatenate([trackFeatures,batch['trk_z0_res']],axis=2)
        nBatch = batch['pvz0'].shape[0]
        FH = predictFastHisto(batch[z0],batch['trk_pt'])
        predictedZ0_FH.append(FH)
        predictedZ0_FHz0res.append(predictFastHistoZ0res(batch[z0],batch['trk_pt'],batch['trk_eta']))

        trk_z0.append(batch[z0])
        trk_MVA.append(batch["trk_MVA1"])
        trk_pt.append(batch['normed_trk_pt'])
        trk_z0res.append(batch['trk_z0_res'])
        trk_eta.append(batch['normed_trk_eta'])
        trk_phi.append(batch['trk_phi'])

        trk_chi2rphi.append(batch['binned_trk_chi2rphi'])
        trk_chi2rz.append(batch['binned_trk_chi2rz'])
        trk_bendchi2.append(batch['binned_trk_bendchi2'])

        tp_met_px .append(batch['tp_met_px'])
        pv_trk_met_px.append(batch['pv_trk_met_px'])
        true_met_px.append(batch['true_met_px'])

        tp_met_py .append(batch['tp_met_py'])
        pv_trk_met_py.append(batch['pv_trk_met_py'])
        true_met_py.append(batch['true_met_py'])

        tp_met_phi .append(batch['tp_met_phi'])
        pv_trk_met_phi.append(batch['pv_trk_met_phi'])
        true_met_phi.append(batch['true_met_phi'])

        tp_met .append(batch['tp_met_pt'])
        pv_trk_met.append(batch['pv_trk_met_pt'])
        true_met.append(batch['true_met_pt'])


        actual_Assoc.append(batch["trk_fromPV"])
        actual_PV.append(batch['pvz0'])
        FHassoc = FastHistoAssoc(predictFastHisto(batch[z0],batch['trk_pt']),batch[z0],batch['trk_eta'])
        predictedAssoc_FH.append(FHassoc)
                
        predictedZ0_NN_temp, predictedAssoc_NN_temp, predictedWeights_NN, predictedHists_NN = model.predict_on_batch(
                        [batch[z0],WeightFeatures,trackFeatures]
                    )

        #print(batch["PV_hist"])
        #print("---------------")
        #print(predictedHists_NN)

        for i,event in enumerate(batch[z0]):
            actual_numtracks.append(np.sum(batch["trk_fromPV"].numpy()[i]))
            actual_pt_total.append(np.sum(batch["trk_fromPV"].numpy()[i]*batch["trk_pt"].numpy()[i]))
            if abs(predictedZ0_NN_temp.numpy()[i] - batch['pvz0'][i]) > 100:
                plotKDEandTracks(batch['trk_z0'][i],predictedAssoc_NN_temp[i].numpy()[:,0],batch['pvz0'][i],predictedZ0_NN_temp.numpy()[i],weights=predictedWeights_NN[i].numpy()[:,0],weight_label="KDE",threshold=-1)
                plotKDEandTracks(batch['trk_z0'][i],FHassoc[i],batch['pvz0'][i],FH[i],batch['trk_pt'][i],weight_label="FH",threshold=0.5)
                plotKDEandTracks(batch['trk_z0'][i],batch["trk_fromPV"].numpy()[i],batch['pvz0'][i],batch['pvz0'][i],batch["trk_fromPV"][i],weight_label="True",threshold=0.5)

        predicted_trk_met_px,predicted_trk_met_py,predicted_trk_met_pt,predicted_trk_met_phi = predictMET(batch['trk_pt'],batch['trk_phi'],predictedAssoc_NN_temp,threshold)
        predictedFH_trk_met_px,predictedFH_trk_met_py,predictedFH_trk_met_pt,predictedFH_trk_met_phi = predictMET(batch['trk_pt'],batch['trk_phi'],
                                                                  FastHistoAssoc(predictFastHisto(batch['trk_z0'],batch['trk_pt']),batch['trk_z0'],batch['trk_eta']),0.5)
        
        predicted_met_px.append(predicted_trk_met_px)
        predicted_met_py.append(predicted_trk_met_py)
        predicted_met_phi.append(predicted_trk_met_phi)
        predicted_met.append(predicted_trk_met_pt)

        predictedFH_met_px.append(predictedFH_trk_met_px)
        predictedFH_met_py.append(predictedFH_trk_met_py)
        predictedFH_met_phi.append(predictedFH_trk_met_phi)
        predictedFH_met.append(predictedFH_trk_met_pt)

        predictedZ0_NN.append(predictedZ0_NN_temp)
        predictedAssoc_NN.append(predictedAssoc_NN_temp)
        predictedWeights.append(predictedWeights_NN)

    z0_NN_array = np.concatenate(predictedZ0_NN).ravel()
    z0_FH_array = np.concatenate(predictedZ0_FH).ravel()
    z0_FHzres_array = np.concatenate(predictedZ0_FHz0res).ravel()
    z0_PV_array = np.concatenate(actual_PV).ravel()


    predictedWeightsarray = np.concatenate(predictedWeights).ravel()

    trk_z0_array = np.concatenate(trk_z0).ravel()
    trk_mva_array = np.concatenate(trk_MVA).ravel()
    trk_pt_array = np.concatenate(trk_pt).ravel()
    trk_z0res_array = np.concatenate(trk_z0res).ravel()
    trk_eta_array = np.concatenate(trk_eta).ravel()
    trk_phi_array = np.concatenate(trk_phi).ravel()

    pv_numtracks = np.array(actual_numtracks)
    pv_pttotal = np.array(actual_pt_total)

    trk_chi2rphi_array = np.concatenate(trk_chi2rphi).ravel()
    trk_chi2rz_array = np.concatenate(trk_chi2rz).ravel()
    trk_bendchi2_array = np.concatenate(trk_bendchi2).ravel()


    assoc_NN_array = np.concatenate(predictedAssoc_NN).ravel()
    assoc_FH_array = np.concatenate(predictedAssoc_FH).ravel()
    assoc_PV_array = np.concatenate(actual_Assoc).ravel()

    pv_track_sel = assoc_PV_array == 1
    pu_track_sel = assoc_PV_array == 0

    weightmax = np.max(predictedWeightsarray)


    plt.clf()
    plt.hist(trk_z0res_array[pv_track_sel],range=(0,1), bins=50, label="PV tracks", alpha=0.5, density=True)
    plt.hist(trk_z0res_array[pu_track_sel],range=(0,1), bins=50, label="PU tracks", alpha=0.5, density=True)
    plt.xlabel("Track $z_0$ res", horizontalalignment='right', x=1.0)
    # plt.ylabel("tracks/counts", horizontalalignment='right', y=1.0)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/reshist.png" % outputFolder)
    plt.clf()

    plt.clf()
    plt.hist(trk_bendchi2_array[pv_track_sel],range=(0,1), bins=50, label="PV tracks", alpha=0.5, density=True)
    plt.hist(trk_bendchi2_array[pu_track_sel],range=(0,1), bins=50, label="PU tracks", alpha=0.5, density=True)
    plt.xlabel("Track $\\chi^2_{bend}$", horizontalalignment='right', x=1.0)
    # plt.ylabel("tracks/counts", horizontalalignment='right', y=1.0)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/bendchi2hist.png" % outputFolder)
    plt.clf()


    plt.clf()
    plt.hist(predictedWeightsarray[pv_track_sel],range=(0,weightmax), bins=50, label="PV tracks", alpha=0.5, density=True)
    plt.hist(predictedWeightsarray[pu_track_sel],range=(0,weightmax), bins=50, label="PU tracks", alpha=0.5, density=True)
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    # plt.ylabel("tracks/counts", horizontalalignment='right', y=1.0)
    plt.yscale("log")
    plt.title("Histogram weights for PU and PV tracks")
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/corr-assoc-1d.png" % outputFolder)
    plt.clf()

    pv_track_no = np.sum(pv_track_sel)
    pu_track_no = np.sum(pu_track_sel)


    assoc_scale = (pv_track_no / pu_track_no)
    # plt.bar(b[:-1], h, width=bw, label="PV tracks", alpha=0.5)

    plt.hist(predictedWeightsarray[pv_track_sel],range=(0,weightmax), bins=50, label="PV tracks", alpha=0.5, weights=np.ones_like(predictedWeightsarray[pv_track_sel]) / assoc_scale)
    plt.hist(predictedWeightsarray[pu_track_sel],range=(0,weightmax), bins=50, label="PU tracks", alpha=0.5)
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("a.u.", horizontalalignment='right', y=1.0)
    plt.title("Histogram weights for PU and PV tracks (normalised)")
    plt.yscale("log")

    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/corr-assoc-1d-norm.png" % outputFolder)

    plt.clf()
    plt.hist2d(predictedWeightsarray, assoc_NN_array, range=((0,weightmax),(-12,12)),bins=50, norm=matplotlib.colors.LogNorm());
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("track-to-vertex association flag", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/corr-assoc.png" %  outputFolder)

    plt.clf()
    plt.hist2d(predictedWeightsarray, trk_z0_array, bins=50, norm=matplotlib.colors.LogNorm());
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("track $z_0$ [cm]", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/corr-z0.png" %  outputFolder)

    plt.clf()
    plt.hist2d(predictedWeightsarray, trk_mva_array, bins=50, norm=matplotlib.colors.LogNorm());
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("track MVA", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/corr-mva.png" %  outputFolder)

    plt.clf()
    plt.hist2d(predictedWeightsarray, trk_pt_array, bins=50, norm=matplotlib.colors.LogNorm());
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("track $p_T$ [GeV]", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/corr-pt.png" %  outputFolder)

    plt.clf()
    plt.hist2d(predictedWeightsarray, np.abs(trk_eta_array), bins=50, norm=matplotlib.colors.LogNorm());
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("track $|\\eta|$", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/corr-abs-eta.png" %  outputFolder)

    plt.clf()
    plt.hist2d(predictedWeightsarray, trk_eta_array, bins=50, norm=matplotlib.colors.LogNorm());
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("track $\\eta$", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/corr-eta.png" %  outputFolder)

    plt.clf()
    plt.hist2d(predictedWeightsarray, trk_z0res_array, bins=50, norm=matplotlib.colors.LogNorm());
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("track z0 res [cm]", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/corr-z0res.png" % outputFolder)

    plt.clf()
    plt.hist2d(predictedWeightsarray, trk_chi2rphi_array, bins=50, norm=matplotlib.colors.LogNorm());
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("track $\\chi^2_{r\\phi}$", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/corr-chi2rphi.png" % outputFolder)

    plt.clf()
    plt.hist2d(predictedWeightsarray, trk_chi2rz_array, bins=50, norm=matplotlib.colors.LogNorm());
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("track $\\chi^2_{rz}$", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/corr-chi2rz.png" % outputFolder)

    plt.clf()
    plt.hist2d(predictedWeightsarray, trk_bendchi2_array, bins=50, norm=matplotlib.colors.LogNorm());
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("track $\\chi^2_{bend}$", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/corr-chi2bend.png" % outputFolder)

    do_scatter = True
    if (do_scatter):
        plt.clf()
        plt.scatter(predictedWeightsarray, trk_z0_array, label="z0")
        plt.xlabel("weight")
        plt.ylabel("variable")
        plt.title("Correlation between predicted weight and track variables")
        plt.legend()
        plt.savefig("%s/scatter-z0.png" %  outputFolder)
        # plt.show()

        plt.clf()
        plt.scatter(predictedWeightsarray, trk_pt_array, label="pt")
        plt.xlabel("weight")
        plt.ylabel("variable")
        plt.title("Correlation between predicted weight and track variables")
        plt.legend()
        plt.savefig("%s/scatter-pt.png" % outputFolder)

        plt.clf()
        plt.scatter(predictedWeightsarray, trk_eta_array, label="eta")
        plt.xlabel("weight")
        plt.ylabel("variable")
        plt.title("Correlation between predicted weight and track variables")
        plt.legend()
        plt.savefig("%s/scatter-eta.png" %  outputFolder)

        plt.clf()
        plt.scatter(predictedWeightsarray, trk_z0res_array, label="z0res")
        plt.xlabel("weight")
        plt.ylabel("variable")
        plt.title("Correlation between predicted weight and track variables")
        plt.legend()
        plt.savefig("%s/scatter-z0res.png" %  outputFolder)


    tp_met_px_array  = np.concatenate(tp_met_px).ravel()
    pv_trk_met_px_array = np.concatenate(pv_trk_met_px).ravel()
    true_met_px_array = np.concatenate(true_met_px).ravel()

    tp_met_py_array  = np.concatenate(tp_met_py).ravel()
    pv_trk_met_py_array = np.concatenate(pv_trk_met_py).ravel()
    true_met_py_array = np.concatenate(true_met_py).ravel()

    tp_met_phi_array = np.concatenate(tp_met_phi).ravel()
    pv_trk_met_phi_array = np.concatenate(pv_trk_met_phi).ravel()
    true_met_phi_array = np.concatenate(true_met_phi).ravel()

    tp_met_pt_array  = np.concatenate(tp_met).ravel()
    pv_trk_met_pt_array = np.concatenate(pv_trk_met).ravel()
    true_met_pt_array = np.concatenate(true_met).ravel()


    predicted_met_px_array = np.concatenate(predicted_met_px).ravel()
    predicted_met_py_array = np.concatenate(predicted_met_py).ravel()
    predicted_met_phi_array = np.concatenate(predicted_met_phi).ravel()
    predicted_met_array = np.concatenate(predicted_met).ravel()

    predictedFH_met_px_array = np.concatenate(predictedFH_met_px).ravel()
    predictedFH_met_py_array = np.concatenate(predictedFH_met_py).ravel()
    predictedFH_met_phi_array = np.concatenate(predictedFH_met_phi).ravel()
    predictedFH_met_array = np.concatenate(predictedFH_met).ravel()

    '''
    plt.clf()
    plt.hist(tp_met_px_array,range=(-150,150),bins=50,histtype="step",color="g",label="Tracking Particle MET px")
    plt.hist(pv_trk_met_px_array,range=(-150,150),bins=50,histtype="step",color="b",label="True PV Track MET px")
    plt.hist(true_met_px_array,range=(-150,150),bins=50,histtype="step",color="y",label="True MET px")
    plt.hist(predicted_met_px_array,range=(-150,150),bins=50,histtype="step",color="r",label="Predicted Track MET px, threshold = "+str(threshold))
    plt.hist(predictedFH_met_px_array,range=(-150,150),bins=50,histtype="step",color="orange",label="Predicted FH Track MET px")
    plt.xlabel("MET px [GeV]")
    plt.ylabel("Events")
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/METpx.png" % outputFolder)

    plt.clf()
    plt.hist(tp_met_py_array,range=(-150,150),bins=50,histtype="step",color="g",label="Tracking Particle MET py")
    plt.hist(pv_trk_met_py_array,range=(-150,150),bins=50,histtype="step",color="b",label="True PV Track MET py")
    plt.hist(true_met_py_array,range=(-150,150),bins=50,histtype="step",color="y",label="True MET py")
    plt.hist(predicted_met_py_array,range=(-150,150),bins=50,histtype="step",color="r",label="Predicted Track MET py, threshold = "+str(threshold))
    plt.hist(predictedFH_met_py_array,range=(-150,150),bins=50,histtype="step",color="orange",label="Predicted FH Track MET py")
    plt.xlabel("MET py [GeV]")
    plt.ylabel("Events")
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/METpy.png" % outputFolder)

    plt.clf()
    plt.hist(tp_met_pt_array,range=(0,300),bins=50,histtype="step",color="g",label="Tracking Particle MET pt")
    plt.hist(pv_trk_met_pt_array,range=(0,300),bins=50,histtype="step",color="b",label="True PV Track MET pt")
    plt.hist(true_met_pt_array,range=(0,300),bins=50,histtype="step",color="y",label="True MET pt")
    plt.hist(predicted_met_array,range=(0,300),bins=50,histtype="step",color="r",label="Predicted Track MET pt, threshold = "+str(threshold))
    plt.hist(predictedFH_met_array,range=(0,300),bins=50,histtype="step",color="orange",label="Predicted FH Track MET pt")
    plt.xlabel("MET pt [GeV]")
    plt.ylabel("Events")
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/METpt.png" % outputFolder)


    plt.clf()
    plt.hist(tp_met_phi_array,range=(-np.pi,np.pi),bins=50,histtype="step",color="g",label="Tracking Particle MET phi")
    plt.hist(pv_trk_met_phi_array,range=(-np.pi,np.pi),bins=50,histtype="step",color="b",label="True PV Track MET phi")
    plt.hist(true_met_phi_array,range=(-np.pi,np.pi),bins=50,histtype="step",color="y",label="True MET phi")
    plt.hist(predicted_met_phi_array,range=(-np.pi,np.pi),bins=50,histtype="step",color="r",label="Predicted Track MET phi, threshold = "+str(threshold))
    plt.hist(predictedFH_met_phi_array,range=(-np.pi,np.pi),bins=50,histtype="step",color="orange",label="Predicted FH Track MET phi")
    plt.xlabel("MET phi [GeV]")
    plt.xlim(-np.pi,np.pi)
    plt.ylabel("Events")
    plt.legend(loc="lower center")
    plt.tight_layout()
    plt.savefig("%s/METphi.png" % outputFolder)
    '''

    plt.clf()
    figure=plotz0_residual((z0_PV_array-z0_NN_array),(z0_PV_array-z0_FH_array))
    plt.savefig("%s/Z0Residual.png" % outputFolder)

    plt.clf()
    figure=plotz0_residual((z0_PV_array-z0_NN_array),(z0_PV_array-z0_FHzres_array))
    plt.savefig("%s/Z0Residualzres.png" % outputFolder)

    plt.clf()
    figure=plotPV_roc(assoc_PV_array,assoc_NN_array,assoc_FH_array)
    plt.savefig("%s/PVROC.png" % outputFolder)

    plt.clf()
    figure=plotPV_classdistribution(assoc_PV_array,assoc_NN_array,assoc_FH_array)
    plt.savefig("%s/ClassDistribution.png" % outputFolder)

    plt.clf()
    figure=plotz0_percentile((z0_PV_array-z0_NN_array),(z0_PV_array-z0_FH_array))
    plt.savefig("%s/Z0percentile.png" % outputFolder)

    plt.clf()
    plt.hist(z0_FH_array,range=(-15,15),bins=120,density=True,color='r',histtype="step",label="FastHisto Base")
    plt.hist(z0_FHzres_array,range=(-15,15),bins=120,density=True,color='g',histtype="step",label="FastHisto with z0 res")
    plt.hist(z0_NN_array,range=(-15,15),bins=120,density=True,color='b',histtype="step",label="CNN")
    plt.hist(z0_PV_array,range=(-15,15),bins=120,density=True,color='y',histtype="step",label="Truth")
    plt.grid(True)
    plt.xlabel('$z_0$ [cm]',ha="right",x=1)
    plt.ylabel('Events',ha="right",y=1)
    plt.legend() 
    plt.tight_layout()
    plt.savefig("%s/z0hist.png" % outputFolder)

    plt.clf()
    plt.hist2d(z0_PV_array, (z0_PV_array-z0_FH_array), bins=60,range=((-15,15),(-30,30)) ,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("PV", horizontalalignment='right', x=1.0)
    plt.ylabel("PV - $z_0^{FH}$", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/FHerr_vs_z0.png" %  outputFolder)

    plt.clf()
    plt.hist2d(z0_PV_array, (z0_PV_array-z0_NN_array), bins=60,range=((-15,15),(-30,30)), norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("PV", horizontalalignment='right', x=1.0)
    plt.ylabel("PV - $z_0^{NN}$", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/NNerr_vs_z0.png" %  outputFolder)

    plt.clf()
    plt.hist2d(z0_PV_array, z0_FH_array, bins=60,range=((-15,15),(-15,15)) ,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("PV", horizontalalignment='right', x=1.0)
    plt.ylabel("$z_0^{FH}$", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/FH_vs_z0.png" %  outputFolder)

    plt.clf()
    plt.hist2d(z0_PV_array, z0_NN_array, bins=60,range=((-15,15),(-15,15)), norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("PV", horizontalalignment='right', x=1.0)
    plt.ylabel("$z_0^{NN}$", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/NN_vs_z0.png" %  outputFolder)


    plt.clf()
    figure=plotz0_percentile((z0_PV_array-z0_NN_array),(z0_PV_array-z0_FHzres_array))
    plt.savefig("%s/Z0withrespercentile.png" % outputFolder)

    plt.clf()
    plt.scatter(pv_numtracks,z0_PV_array/z0_NN_array,color='r',alpha=0.5)
    plt.xlabel("# PV tracks per event ", horizontalalignment='right', x=1.0)
    plt.ylabel("$z_0^{gen}$/$z_0^{NN}$ ", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/z0_vs_numtracks.png" %  outputFolder)

    plt.clf()
    plt.scatter(pv_numtracks,z0_PV_array-z0_NN_array,color='r',alpha=0.5)
    plt.xlabel("# PV tracks per event ", horizontalalignment='right', x=1.0)
    plt.ylabel("$z_0^{gen}$ - $z_0^{NN}$ ", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/z0err_vs_numtracks.png" %  outputFolder)

    plt.clf()
    plt.scatter(pv_pttotal,z0_PV_array/z0_NN_array,color='r',alpha=0.5)
    plt.xlabel("Total PV $p_{T}$ per event [GeV] ", horizontalalignment='right', x=1.0)
    plt.ylabel("$z_0^{gen}$/$z_0^{NN}$ ", horizontalalignment='right', y=1.0)
    plt.xscale("log")
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/z0_vs_pv_pttotal.png" %  outputFolder)

    plt.clf()
    plt.scatter(pv_pttotal,z0_PV_array-z0_NN_array,color='r',alpha=0.5)
    plt.xlabel("Total PV $p_{T}$ per event [GeV] ", horizontalalignment='right', x=1.0)
    plt.ylabel("$z_0^{gen}$ - $z_0^{NN}$ ", horizontalalignment='right', y=1.0)
    plt.xscale("log")
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/zerr0_vs_pv_pttotal.png" %  outputFolder)

    plt.clf()
    plt.scatter(pv_pttotal/pv_numtracks,z0_PV_array/z0_NN_array,color='r',alpha=0.5,label="50 % Quartile")
    plt.xlabel("PV $p_{T}$ density per event [GeV] ", horizontalalignment='right', x=1.0)
    plt.ylabel("$z_0^{gen}$/$z_0^{NN}$ ", horizontalalignment='right', y=1.0)
    plt.xscale("log")
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/z0_vs_pv_ptdensity.png" %  outputFolder)

    plt.clf()
    plt.scatter(pv_pttotal/pv_numtracks,z0_PV_array-z0_NN_array,color='r',alpha=0.5,label="50 % Quartile")
    plt.xlabel("PV $p_{T}$ density per event [GeV] ", horizontalalignment='right', x=1.0)
    plt.ylabel("$z_0^{gen}$ - $z_0^{NN}$ ", horizontalalignment='right', y=1.0)
    plt.xscale("log")
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/z0diff_vs_pv_ptdensity.png" %  outputFolder)

    '''
    plt.clf()
    figure=plotMET_residual((true_met_pt_array-predicted_met_array),
                            (true_met_pt_array-pv_trk_met_pt_array),
                            (true_met_pt_array-predictedFH_met_array),
                            threshold = threshold,
                            )
    plt.savefig("%s/METresidual.png" % outputFolder)

    plt.clf()
    figure=plotMET_residual((true_met_pt_array-predicted_met_array),
                            (true_met_pt_array-pv_trk_met_pt_array),
                            (true_met_pt_array-predictedFH_met_array),
                            threshold = threshold,
                            relative=True,
                            true=true_met_pt_array
                            )
    plt.savefig("%s/METrelresidual.png" % outputFolder)
    '''
    experiment.log_asset_folder(outputFolder, step=None, log_file_name=True)





