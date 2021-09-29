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

import vtx
from TrainingScripts.train import *
from EvalScripts.evalDA import *

#hep.set_style("CMSTex")
#hep.cms.label()
#hep.cms.text("Simulation")
plt.style.use(hep.style.CMS)

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

nMaxTracks = 250

def predictFastHistoNoFakesResEta(value,weight,Fakes,eta):

    def res_function1(fakes):
        res = fakes != 0
        return res

    def res_function2(eta):
        res = 0.1 + 0.2*eta**2
        return res

    z0List = []
    halfBinWidth = 0.5*30./256.
    for ibatch in range(value.shape[0]):
        #res = res_bins[np.digitize(abs(eta[ibatch]),eta_bins)]
        
        res1 = res_function1(Fakes[ibatch])
        res2 = res_function2(eta[ibatch][res1])
        hist,bin_edges = np.histogram(value[ibatch][res1],256,range=(-15,15),weights=(weight[ibatch][res1]/res2))
        #hist,bin_edges = np.histogram(value[ibatch][MVA[ibatch][MVA[ibatch] > 0.08]],256,range=(-15,15),weights=(weight[ibatch][MVA[ibatch][MVA[ibatch] > 0.08]]))
        hist = np.convolve(hist,[1,1,1],mode='same')
        z0Index= np.argmax(hist)
        z0 = -15.+30.*z0Index/256.+halfBinWidth
        z0List.append([z0])
    return np.array(z0List,dtype=np.float32)


if __name__=="__main__":

    kf = sys.argv[1]

    with open(sys.argv[2]+'.yaml', 'r') as f:
            config = yaml.load(f)

    if kf == "NewKF":
        test_files = glob.glob(config["data_folder"]+"NewKFData/Test/*.tfrecord")
        z0 = 'trk_z0'
    elif kf == "OldKF":
        test_files = glob.glob(config["data_folder"]+"OldKFData/Test/*.tfrecord")
        z0 = 'corrected_trk_z0'
    

    outputFolder = kf+"ComparePlots/"

    trackfeat_1 =['normed_trk_pt',
                  'trk_MVA1',
                  'normed_trk_eta',
                  'trk_z0_res',
                  'binned_trk_chi2rphi',
                  'binned_trk_chi2rz',
                  'binned_trk_bendchi2']
    weightfeat_1 = ['normed_trk_pt',
                    'trk_MVA1',
                    'trk_over_eta_squared',
                    'binned_trk_chi2rphi',
                    'binned_trk_chi2rz',
                    'binned_trk_bendchi2']

    trackfeat_2 =['normed_trk_pt',
                  'normed_trk_eta',
                  'trk_MVA1',
                  'binned_trk_chi2rphi',
                  'binned_trk_chi2rz',
                  'binned_trk_bendchi2']
    weightfeat_2 = ['normed_trk_pt',
                    'normed_trk_eta',
                    'trk_MVA1',
                    'binned_trk_chi2rphi',
                    'binned_trk_chi2rz',
                    'binned_trk_bendchi2']

    features = {
            "pvz0": tf.io.FixedLenFeature([1], tf.float32),
            "trk_fromPV":tf.io.FixedLenFeature([nMaxTracks], tf.float32),
            "trk_hitpattern": tf.io.FixedLenFeature([nMaxTracks*11], tf.float32), 
            "PV_hist"  :tf.io.FixedLenFeature([256,1], tf.float32),
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
            'corrected_trk_z0',
            'normed_trk_over_eta',
            'normed_trk_over_eta_squared',
            'trk_over_eta_squared',
            'trk_fake'

        ]

    for trackFeature in trackFeatures:
        features[trackFeature] = tf.io.FixedLenFeature([nMaxTracks], tf.float32)

    network1 = vtx.nn.E2EDiffArgMax(
            nbins=256,
            ntracks=max_ntracks, 
            nweightfeatures=len(weightfeat_1), 
            nfeatures=len(trackfeat_1), 
            nweights=1, 
            nlatent = 2,
            activation='relu',
            regloss=1e-10
        )

    network2 = vtx.nn.E2Ecomparekernel(
            nbins=256,
            ntracks=max_ntracks, 
            nweightfeatures=len(weightfeat_2), 
            nfeatures=len(trackfeat_2), 
            nweights=1, 
            nlatent = 2,
            activation='relu',
            regloss=1e-10
        )

    model1 = network1.createE2EModel()
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    model1.compile(
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

    model2 = network2.createE2EModel()
    model2.compile(
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

    model1.summary()
    model1.load_weights("../PretrainedModels/"+kf+"weightsReduced_ExtraConv.tf")

    model2.summary()
    model2.load_weights("../PretrainedModels/"+kf+"weights_pretrained.tf")

    NNnames = ["ArgMax","Full CNN"]
    FHnames = ["Base","1/$\eta^2$","MVA Cut","No Fakes"]

    predictedZ0_FH = []
    predictedZ0_FHz0res = []
    predictedZ0_FHz0MVA = []
    predictedZ0_FHnoFake = []
    predictedZ0_FHnoFakeeta = []
    predictedZ0_NN_1 = []
    predictedZ0_NN_2 = []

    predictedWeights_1 = []
    predictedWeights_2 = []

    predictedAssoc_NN_1 = []
    predictedAssoc_NN_2 = []
    predictedAssoc_FH = []
    predictedAssoc_FHres = []
    predictedAssoc_FHMVA = []
    predictedAssoc_FHnoFake = []

    actual_Assoc = []
    actual_PV = []

    trk_z0 = []
    trk_MVA = []
    trk_phi = []
    trk_pt = []
    trk_eta = []

    trk_chi2rphi = []
    trk_chi2rz = []
    trk_bendchi2 = []

    threshold = -1

    for step,batch in enumerate(setup_pipeline(test_files)):
        nBatch = batch['pvz0'].shape[0]
        FH = predictFastHisto(batch[z0],batch['trk_pt'])
        predictedZ0_FH.append(FH)
        predictedZ0_FHz0res.append(predictFastHistoZ0res(batch[z0],batch['trk_pt'],batch['trk_eta']))
        predictedZ0_FHz0MVA.append(predictFastHistoMVAcut(batch[z0],batch['trk_pt'],batch['trk_MVA1']))
        predictedZ0_FHnoFake.append(predictFastHistoNoFakes(batch[z0],batch['trk_pt'],batch['trk_fake']))
        predictedZ0_FHnoFakeeta.append(predictFastHistoNoFakesResEta(batch[z0],batch['trk_pt'],batch['trk_fake'],batch['trk_eta']))

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

        trackFeatures = np.stack([batch[feature] for feature in trackfeat_1],axis=2)
        WeightFeatures = np.stack([batch[feature] for feature in weightfeat_1],axis=2)
                
        predictedZ0_NN1_temp, predictedAssoc_NN1_temp, predictedWeights_NN1 = model1.predict_on_batch(
                        [batch[z0],WeightFeatures,trackFeatures]
                    )

        predictedZ0_NN_1.append(predictedZ0_NN1_temp)
        predictedAssoc_NN_1.append(predictedAssoc_NN1_temp)
        predictedWeights_1.append(predictedWeights_NN1)

        #=========================================================================#

        trackFeatures = np.stack([batch[feature] for feature in trackfeat_2],axis=2)
        WeightFeatures = np.stack([batch[feature] for feature in weightfeat_2],axis=2)
                
        predictedZ0_NN2_temp, predictedAssoc_NN2_temp, predictedWeights_NN2 = model2.predict_on_batch(
                        [batch[z0],WeightFeatures,trackFeatures]
                    )

        predictedZ0_NN_2.append(predictedZ0_NN2_temp)
        predictedAssoc_NN_2.append(predictedAssoc_NN2_temp)
        predictedWeights_2.append(predictedWeights_NN2)

    z0_NN_array_1 = np.concatenate(predictedZ0_NN_1).ravel()
    z0_NN_array_2 = np.concatenate(predictedZ0_NN_2).ravel()
    z0_FH_array = np.concatenate(predictedZ0_FH).ravel()
    z0_FHres_array = np.concatenate(predictedZ0_FHz0res).ravel()
    z0_FHMVA_array = np.concatenate(predictedZ0_FHz0MVA).ravel()
    z0_FHnoFake_array = np.concatenate(predictedZ0_FHnoFake).ravel()
    z0_FHnoFakeeta_array = np.concatenate(predictedZ0_FHnoFakeeta).ravel()
    z0_PV_array = np.concatenate(actual_PV).ravel()


    predictedWeightsarray_1 = np.concatenate(predictedWeights_1).ravel()
    predictedWeightsarray_2 = np.concatenate(predictedWeights_2).ravel()

    trk_z0_array = np.concatenate(trk_z0).ravel()
    trk_mva_array = np.concatenate(trk_MVA).ravel()
    trk_pt_array = np.concatenate(trk_pt).ravel()
    trk_eta_array = np.concatenate(trk_eta).ravel()
    trk_phi_array = np.concatenate(trk_phi).ravel()

    trk_chi2rphi_array = np.concatenate(trk_chi2rphi).ravel()
    trk_chi2rz_array = np.concatenate(trk_chi2rz).ravel()
    trk_bendchi2_array = np.concatenate(trk_bendchi2).ravel()


    assoc_NN_array_1 = np.concatenate(predictedAssoc_NN_1).ravel()
    assoc_NN_array_2 = np.concatenate(predictedAssoc_NN_2).ravel()
    assoc_FH_array = np.concatenate(predictedAssoc_FH).ravel()
    assoc_FHres_array = np.concatenate(predictedAssoc_FHres).ravel()
    assoc_FHMVA_array = np.concatenate(predictedAssoc_FHMVA).ravel()
    assoc_FHnoFake_array = np.concatenate(predictedAssoc_FHnoFake).ravel()
    assoc_PV_array = np.concatenate(actual_Assoc).ravel()

    pv_track_sel = assoc_PV_array == 1
    pu_track_sel = assoc_PV_array == 0

    

    plt.clf()
    plt.hist2d(predictedWeightsarray_1, predictedWeightsarray_2, bins=60,range=((0,0.6),(0,0.6)), norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel(NNnames[0] + " NN Weights", horizontalalignment='right', x=1.0)
    plt.ylabel(NNnames[1] + " NN Weights", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/NNweight_vs_NNweight.png" %  outputFolder)

    
    
    plt.clf()
    figure=plotz0_residual([(z0_PV_array-z0_NN_array_1),(z0_PV_array-z0_NN_array_2)],
                          [(z0_PV_array-z0_FH_array)],
                          NNnames,
                          ["Base"],colours=["red","blue","green","orange","purple","yellow"])
    plt.savefig("%s/CombZ0Residual.png" % outputFolder)

    plt.clf()
    figure=plotz0_residual([],
                          [(z0_PV_array-z0_FH_array),(z0_PV_array-z0_FHMVA_array),(z0_PV_array-z0_FHnoFake_array)],
                          [],
                          ["Base","MVA","No Fakes"])
    plt.savefig("%s/Z0FHResidual.png" % outputFolder)

    plt.clf()
    figure=plotz0_residual([],
                          [(z0_PV_array-z0_FH_array),(z0_PV_array-z0_FHres_array),(z0_PV_array-z0_FHnoFakeeta_array),(z0_PV_array-z0_FHnoFake_array)],
                          [],
                          ["Base","1/$\eta^2$","No Fakes + 1/$\eta^2$","No Fakes"])
    plt.savefig("%s/Z0FHfakesResidual.png" % outputFolder)

    plt.clf()
    figure=plotz0_residual([(z0_PV_array-z0_NN_array_1)],
                          [(z0_PV_array-z0_FH_array)],
                          ["ArgMax"],
                          ["Base"])
    plt.savefig("%s/AMaxZ0Residual.png" % outputFolder)

    plt.clf()
    figure=plotz0_residual([(z0_PV_array-z0_NN_array_2)],
                          [(z0_PV_array-z0_FH_array)],
                          ["Full CNN"],
                          ["Base"])
    plt.savefig("%s/UltimateZ0Residual.png" % outputFolder)

    plt.clf()
    figure=plotPV_roc(assoc_PV_array,
                     [assoc_NN_array_1,assoc_NN_array_2],
                     [assoc_FH_array,assoc_FHres_array,assoc_FHMVA_array,assoc_FHnoFake_array],
                     NNnames,
                     FHnames,colours=["red","blue","green","orange","purple","yellow"])
    plt.savefig("%s/PVROC.png" % outputFolder)

    #plt.clf()
    #figure=plotPV_roc(assoc_PV_array,
    #                 [assoc_NN_array_1],
    #                 [(assoc_FH_array)],
    #                 ["ArgMax"],
    #                 ["Base"])
    #plt.savefig("%s/AMaxPVROC.png" % outputFolder)

    #plt.clf()
    #figure=plotPV_roc(assoc_PV_array,
    #                 [assoc_NN_array_2],
    #                 [(assoc_FH_array)],
    #                 ["Full CNN"],
    #                 ["Base"])
    #plt.savefig("%s/UltimatePVROC.png" % outputFolder)

    #plt.clf()
    #figure=plotz0_percentile([(z0_PV_array-z0_NN_array_1),(z0_PV_array-z0_NN_array_2)],
    #                         [(z0_PV_array-z0_FH_array)],
    #                         NNnames,
    #                         ["Base"],colours=["red","blue","green","orange","purple","yellow"])
    #plt.savefig("%s/Z0percentile.png" % outputFolder)

    plt.figure(figsize=(10,8))

    plt.clf()
    plt.hist(z0_FH_array,range=(-15,15),bins=120,density=True,color='r',histtype="step",label="FastHisto Base")
    plt.hist(z0_FHres_array,range=(-15,15),bins=120,density=True,color='g',histtype="step",label="FastHisto with z0 res")
    plt.hist(z0_NN_array_1,range=(-15,15),bins=120,density=True,color='b',histtype="step",label="ArgMax NN")
    plt.hist(z0_NN_array_2,range=(-15,15),bins=120,density=True,color='b',histtype="step",label="Ultimate NN")
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
    plt.hist2d(z0_PV_array, (z0_PV_array-z0_FHMVA_array), bins=60,range=((-15,15),(-30,30)) ,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("PV", horizontalalignment='right', x=1.0)
    plt.ylabel("PV - $z_0^{FH}$", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/FHMVAerr_vs_z0.png" %  outputFolder)

    plt.clf()
    plt.hist2d(z0_PV_array, (z0_PV_array-z0_NN_array_1), bins=60,range=((-15,15),(-30,30)), norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("PV", horizontalalignment='right', x=1.0)
    plt.ylabel("PV - $z_0^{NN,ArgMax}$", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/AmaxNNerr_vs_z0.png" %  outputFolder)

    plt.clf()
    plt.hist2d(z0_PV_array, (z0_PV_array-z0_NN_array_2), bins=60,range=((-15,15),(-30,30)), norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("PV", horizontalalignment='right', x=1.0)
    plt.ylabel("PV - $z_0^{NN,Ultimate}$", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/UltNNerr_vs_z0.png" %  outputFolder)

    plt.clf()
    plt.hist2d(z0_PV_array, z0_FH_array, bins=60,range=((-15,15),(-15,15)) ,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("PV", horizontalalignment='right', x=1.0)
    plt.ylabel("$z_0^{FH}$", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/FH_vs_z0.png" %  outputFolder)

    plt.clf()
    plt.hist2d(z0_PV_array, z0_NN_array_1, bins=60,range=((-15,15),(-15,15)), norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("PV", horizontalalignment='right', x=1.0)
    plt.ylabel("$z_0^{NN}$", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/AmaxNN_vs_z0.png" %  outputFolder)

    plt.clf()
    plt.hist2d(z0_PV_array, z0_NN_array_2, bins=60,range=((-15,15),(-15,15)), norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("PV", horizontalalignment='right', x=1.0)
    plt.ylabel("$z_0^{NN,Ultimate}$", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/UltNN_vs_z0.png" %  outputFolder)

    





