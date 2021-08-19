from comet_ml import Experiment
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import scipy
import h5py
import os
import sys
import glob
import math
import re
import csv
import sklearn.metrics as metrics
import vtx
import pandas as pd
from Callbacks import OwnReduceLROnPlateau
from eval import*

import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
#hep.set_style("CMSTex")
hep.cms.label()
hep.cms.text("Simulation")

plt.style.use(hep.style.ROOT)

tf.config.threading.set_inter_op_parallelism_threads(
    8
)

kf = sys.argv[1]

if kf == "NewKF":
    z0 = 'trk_z0'
elif kf == "OldKF":
    z0 = 'corrected_trk_z0'

SMALL_SIZE = 15
MEDIUM_SIZE = 15
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

max_ntracks = 250

def decode_data(raw_data):
    decoded_data = tf.io.parse_example(raw_data,features)
    decoded_data['trk_hitpattern'] = tf.reshape(decoded_data['trk_hitpattern'],[-1,max_ntracks,11])
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

def train_model(bigmodel,reducedmodel,experiment,train_files,val_files,epochs=50,callbacks=None,nlatent=0):

    total_steps = 0
    callbacks[0].on_train_begin()
    callbacks[1].on_train_begin()
    bmold_lr = bigmodel.optimizer.learning_rate.numpy()
    rmold_lr = reducedmodel.optimizer.learning_rate.numpy()

    for epoch in range(epochs):

        bmnew_lr = bmold_lr
        rmnew_lr = rmold_lr

        tf.keras.backend.set_value(bigmodel.optimizer.learning_rate, bmnew_lr)
        tf.keras.backend.set_value(reducedmodel.optimizer.learning_rate, rmnew_lr)

        experiment.log_metric("biglearning_rate",bigmodel.optimizer.learning_rate,step=total_steps,epoch=epoch)
        experiment.log_metric("reducedlearning_rate",reducedmodel.optimizer.learning_rate,step=total_steps,epoch=epoch)
        
        print ("Epoch %i"%epoch)
        
        if epoch>0:
            reducedmodel.load_weights(kf+"bmweights_%i.tf"%(epoch-1))
            bigmodel.load_weights(kf+"rmweights_%i.tf"%(epoch-1))
        
        for step,batch in enumerate(setup_pipeline(train_files)):
            
            trackFeatures = np.stack([batch[feature] for feature in [
                'normed_trk_pt','normed_trk_eta','trk_MVA1','binned_trk_chi2rphi', 'binned_trk_chi2rz', 'binned_trk_bendchi2'
            ]],axis=2)

            WeightFeatures = np.stack([batch[feature] for feature in [
                'normed_trk_pt','normed_trk_eta','trk_MVA1','binned_trk_chi2rphi', 'binned_trk_chi2rz', 'binned_trk_bendchi2'
            ]],axis=2)

            ### Train Larger Model

            nBatch = batch['pvz0'].shape[0]
            bmresult = bigmodel.train_on_batch(
            [batch[z0],WeightFeatures,trackFeatures],
            [batch['pvz0'],batch['trk_fromPV'],np.zeros((nBatch,max_ntracks,1)),batch['PV_hist']]
        )    ##True Z0      ##True PV            ##True weights                     ##True Hists
            bmresult = dict(zip(bigmodel.metrics_names,bmresult))

            ### Predictions for Larger Model to be passed to smaller model

            bm_predictedZ0_NN, bm_predictedAssoc_NN, bm_predictedWeights_NN,bm_predictedHist_NN = bigmodel.predict_on_batch(
                    [batch[z0],WeightFeatures,trackFeatures]
            )


            rmresult = reducedmodel.train_on_batch(
            [batch[z0],WeightFeatures,trackFeatures],
            [batch['pvz0'],batch['trk_fromPV'],bm_predictedWeights_NN,bm_predictedHist_NN ]
            )    ##True Z0      ##True PV            ##Big Model weights  ##Big Model Hists
            rmresult = dict(zip(reducedmodel.metrics_names,rmresult))

            if nlatent == 0:
                experiment.log_metric("big_loss",bmresult['loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("big_z0_loss",bmresult['position_final_loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("big_assoc_loss",bmresult['association_final_loss'],step=total_steps,epoch=epoch)

                experiment.log_metric("red_loss",rmresult['loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("red_z0_loss",rmresult['position_final_loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("red_assoc_loss",rmresult['association_final_loss'],step=total_steps,epoch=epoch)
            else:
                experiment.log_metric("big_loss",bmresult['loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("big_z0_loss",bmresult['split_latent_loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("big_assoc_loss",bmresult['association_final_loss'],step=total_steps,epoch=epoch)

                experiment.log_metric("red_loss",rmresult['loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("red_z0_loss",rmresult['split_latent_loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("red_assoc_loss",rmresult['association_final_loss'],step=total_steps,epoch=epoch)


            
            if step%10==0:
                predictedZ0_FH = predictFastHisto(batch[z0],batch['trk_pt'])
 
                predictedZ0_bmNN, predictedAssoc_bmNN, predictedWeights_bmNN,predictedHist_bmNN = bigmodel.predict_on_batch(
                    [batch[z0],WeightFeatures,trackFeatures]
                )

                predictedZ0_rmNN, predictedAssoc_rmNN, predictedWeights_rmNN,predictedHist_rmNN = reducedmodel.predict_on_batch(
                    [batch[z0],WeightFeatures,trackFeatures]
                )


                qz0_bmNN = np.percentile(predictedZ0_bmNN-batch['pvz0'],[5,32,50,68,95])
                qz0_rmNN = np.percentile(predictedZ0_rmNN-batch['pvz0'],[5,32,50,68,95])
                qz0_FH = np.percentile(predictedZ0_FH-batch['pvz0'],[5,32,50,68,95])

                if nlatent == 0:
                    print ("Step %02i-%02i: big loss=%.3f (big z0=%.3f, big assoc=%.3f),big q68=(%.4f,%.4f), [FH: q68=(%.4f,%.4f)]"%(
                        epoch,step,
                        bmresult['loss'],bmresult['position_final_loss'],bmresult['association_final_loss'],
                        qz0_bmNN[1],qz0_bmNN[3],qz0_FH[1],qz0_FH[3]
                    ))
                    print ("red loss=%.3f (red z0=%.3f, red assoc=%.3f),red q68=(%.4f,%.4f)"%(
                        rmresult['position_final_loss'],rmresult['association_final_loss'],
                        qz0_rmNN[1],qz0_rmNN[3]
                    ))
                else:
                    print ("Step %02i-%02i: loss=%.3f (split latent z0=%.3f, assoc=%.3f), q68=(%.4f,%.4f), [FH: q68=(%.4f,%.4f)]"%(
                        epoch,step,
                        result['loss'],result['split_latent_loss'],result['association_final_loss'],
                        qz0_NN[1],qz0_NN[3],qz0_FH[1],qz0_FH[3]
                    ))

                    print ("Step %02i-%02i: big loss=%.3f (big split latent z0=%.3f, big assoc=%.3f),big q68=(%.4f,%.4f), [FH: q68=(%.4f,%.4f)]"%(
                        epoch,step,
                        bmresult['loss'],bmresult['split_latent_loss'],bmresult['association_final_loss'],
                        qz0_bmNN[1],qz0_bmNN[3],qz0_FH[1],qz0_FH[3]
                    ))
                    print ("red loss=%.3f (red split latent z0=%.3f, red assoc=%.3f),red q68=(%.4f,%.4f)"%(
                        rmresult['position_final_loss'],rmresult['association_final_loss'],
                        qz0_rmNN[1],qz0_rmNN[3]
                    ))


                print ("Train_bmNN_z0_MSE: "+str(metrics.mean_squared_error(batch['pvz0'],predictedZ0_bmNN))+" Train_FH_z0_MSE: "+str(metrics.mean_squared_error(batch['pvz0'],predictedZ0_FH)))
                print ("Train_rmNN_z0_MSE: "+str(metrics.mean_squared_error(batch['pvz0'],predictedZ0_rmNN)))

            total_steps += 1

        val_actual_PV = []

        val_predictedZ0_FH = []
        val_predictedZ0_bmNN = []
        val_predictedZ0_rmNN = []        

        val_actual_assoc = []

        val_predictedAssoc_FH = []
        val_predictedAssoc_bmNN = []
        val_predictedAssoc_rmNN = []        

        for val_step,val_batch in enumerate(setup_pipeline(val_files)):
            val_predictedZ0_FH.append(predictFastHisto(val_batch[z0],val_batch['trk_pt']).flatten())

            val_trackFeatures = np.stack([val_batch[feature] for feature in [
            'normed_trk_pt','normed_trk_eta','trk_MVA1','binned_trk_chi2rphi', 'binned_trk_chi2rz', 'binned_trk_bendchi2'
            ]],axis=2)

            val_WeightFeatures = np.stack([val_batch[feature] for feature in [
                'normed_trk_pt','normed_trk_eta','trk_MVA1','binned_trk_chi2rphi', 'binned_trk_chi2rz', 'binned_trk_bendchi2'
            ]],axis=2)
            #val_trackFeatures = np.concatenate([val_trackFeatures,val_batch['trk_hitpattern']],axis=2)
            #val_trackFeatures = np.concatenate([val_trackFeatures,val_batch['trk_z0_res']],axis=2)
            
            temp_predictedZ0_bmNN, temp_predictedAssoc_bmNN, predictedWeights_bmNN,predictedHist_bmNN = bigmodel.predict_on_batch(
                    [val_batch[z0],val_WeightFeatures,val_trackFeatures]
            )

            temp_predictedZ0_rmNN, temp_predictedAssoc_rmNN, predictedWeights_rmNN,predictedHist_rmNN = reducedmodel.predict_on_batch(
                    [val_batch[z0],val_WeightFeatures,val_trackFeatures]
            )
                
            val_predictedZ0_bmNN.append(temp_predictedZ0_bmNN.numpy().flatten())
            val_predictedZ0_rmNN.append(temp_predictedZ0_rmNN.numpy().flatten())
            val_predictedAssoc_bmNN.append(temp_predictedAssoc_bmNN.numpy().flatten())
            val_predictedAssoc_rmNN.append(temp_predictedAssoc_rmNN.numpy().flatten())

            val_actual_PV.append(val_batch['pvz0'].numpy().flatten()) 
            val_actual_assoc.append(val_batch["trk_fromPV"].numpy().flatten())

            val_predictedAssoc_FH.append(FastHistoAssoc(val_batch['pvz0'],val_batch[z0],val_batch['trk_eta']).flatten())

        val_z0_bmNN_array = np.concatenate(val_predictedZ0_bmNN).ravel()
        val_z0_rmNN_array = np.concatenate(val_predictedZ0_rmNN).ravel()
        val_z0_FH_array = np.concatenate(val_predictedZ0_FH).ravel()
        val_z0_PV_array = np.concatenate(val_actual_PV).ravel()

        val_assoc_bmNN_array = np.concatenate(val_predictedAssoc_bmNN).ravel()
        val_assoc_rmNN_array = np.concatenate(val_predictedAssoc_rmNN).ravel()
        val_assoc_FH_array = np.concatenate(val_predictedAssoc_FH).ravel()
        val_assoc_PV_array = np.concatenate(val_actual_assoc).ravel()

        experiment.log_histogram_3d(val_z0_bmNN_array-val_z0_PV_array, name="Validation_Big_CNN" , epoch=epoch)
        experiment.log_histogram_3d(val_z0_rmNN_array-val_z0_PV_array, name="Validation_Reduced_CNN" , epoch=epoch)
        experiment.log_histogram_3d(val_z0_FH_array-val_z0_PV_array, name="Validation_FH", epoch=epoch)

        experiment.log_metric("Validation_bmNN_z0_MSE",metrics.mean_squared_error(val_z0_PV_array,val_z0_bmNN_array),epoch=epoch)
        experiment.log_metric("Validation_bmNN_z0_AE",metrics.mean_absolute_error(val_z0_PV_array,val_z0_bmNN_array),epoch=epoch)
        experiment.log_metric("Validation_rmNN_z0_MSE",metrics.mean_squared_error(val_z0_PV_array,val_z0_rmNN_array),epoch=epoch)
        experiment.log_metric("Validation_rmNN_z0_AE",metrics.mean_absolute_error(val_z0_PV_array,val_z0_rmNN_array),epoch=epoch)

        experiment.log_metric("Validation_FH_z0_MSE",metrics.mean_squared_error(val_z0_PV_array,val_z0_FH_array),epoch=epoch)
        experiment.log_metric("Validation_FH_z0_AE",metrics.mean_absolute_error(val_z0_PV_array,val_z0_FH_array),epoch=epoch)

        experiment.log_metric("Validation_bmNN_PV_ROC",metrics.roc_auc_score(val_assoc_PV_array,val_assoc_bmNN_array))
        experiment.log_metric("Validation_bmNN_PV_ACC",metrics.balanced_accuracy_score(val_assoc_PV_array,(val_assoc_bmNN_array>0.0)))
        experiment.log_metric("Validation_rmNN_PV_ROC",metrics.roc_auc_score(val_assoc_PV_array,val_assoc_rmNN_array))
        experiment.log_metric("Validation_rmNN_PV_ACC",metrics.balanced_accuracy_score(val_assoc_PV_array,(val_assoc_rmNN_array>0.0)))

        experiment.log_metric("Validation_FH_PV_ROC",metrics.roc_auc_score(val_assoc_PV_array,val_assoc_FH_array))
        experiment.log_metric("Validation_FH_PV_ACC",metrics.balanced_accuracy_score(val_assoc_PV_array,val_assoc_FH_array))

        bigmodel.save_weights(kf+"bmweights_%i.tf"%(epoch))
        reducedmodel.save_weights(kf+"rmweights_%i.tf"%(epoch))
        bmold_lr = callbacks[0].on_epoch_end(epoch=epoch,logs=bmresult,lr=bmnew_lr)
        rmold_lr = callbacks[1].on_epoch_end(epoch=epoch,logs=rmresult,lr=rmnew_lr)
        
def test_model(bigmodel,reducedmodel,experiment,test_files):

    predictedZ0_FH = []
    predictedZ0_bmNN = []
    predictedZ0_rmNN = []

    predictedAssoc_bmNN = []
    predictedAssoc_rmNN = []
    predictedAssoc_FH = []
    
    actual_Assoc = []
    actual_PV = []

    for step,batch in enumerate(setup_pipeline(test_files)):

        trackFeatures = np.stack([batch[feature] for feature in [
                'normed_trk_pt','normed_trk_eta','trk_MVA1','binned_trk_chi2rphi', 'binned_trk_chi2rz', 'binned_trk_bendchi2'
            ]],axis=2)

        WeightFeatures = np.stack([batch[feature] for feature in [
                'normed_trk_pt','normed_trk_eta','trk_MVA1','binned_trk_chi2rphi', 'binned_trk_chi2rz', 'binned_trk_bendchi2'
            ]],axis=2)

        nBatch = batch['pvz0'].shape[0]
        predictedZ0_FH.append(predictFastHisto(batch[z0],batch['trk_pt']))

        actual_Assoc.append(batch["trk_fromPV"])
        actual_PV.append(batch['pvz0'])

        predictedAssoc_FH.append(FastHistoAssoc(batch['pvz0'],batch[z0],batch['trk_eta']))
            
        predictedZ0_bmNN_temp, predictedAssoc_bmNN_temp, predictedWeights_bmNN,predictedHist_bmNN = bigmodel.predict_on_batch(
                    [batch[z0],WeightFeatures,trackFeatures]
                )

        predictedZ0_rmNN_temp, predictedAssoc_rmNN_temp, predictedWeights_rmNN,predictedHist_rmNN = reducedmodel.predict_on_batch(
                    [batch[z0],WeightFeatures,trackFeatures]
                )

        
        threshold = 0.0

        predictedZ0_rmNN.append(predictedZ0_rmNN_temp)
        predictedAssoc_rmNN.append(predictedAssoc_rmNN_temp)
        predictedZ0_bmNN.append(predictedZ0_bmNN_temp)
        predictedAssoc_bmNN.append(predictedAssoc_bmNN_temp)

    z0_rmNN_array = np.concatenate(predictedZ0_rmNN).ravel()
    z0_bmNN_array = np.concatenate(predictedZ0_bmNN).ravel()
    z0_FH_array = np.concatenate(predictedZ0_FH).ravel()
    z0_PV_array = np.concatenate(actual_PV).ravel()

    assoc_rmNN_array = np.concatenate(predictedAssoc_rmNN).ravel()
    assoc_bmNN_array = np.concatenate(predictedAssoc_bmNN).ravel()
    assoc_FH_array = np.concatenate(predictedAssoc_FH).ravel()
    assoc_PV_array = np.concatenate(actual_Assoc).ravel()

    qz0_bmNN = np.percentile(z0_bmNN_array-z0_PV_array,[5,15,50,85,95])
    qz0_rmNN = np.percentile(z0_rmNN_array-z0_PV_array,[5,15,50,85,95])
    qz0_FH = np.percentile(z0_FH_array-z0_PV_array,[5,15,50,85,95])

    experiment.log_asset(kf+"bmweights_"+str(epochs-1)+".tf.index")
    experiment.log_asset(kf+"bmweights_"+str(epochs-1)+".tf.data-00000-of-00001")

    experiment.log_asset(kf+"rmweights_"+str(epochs-1)+".tf.index")
    experiment.log_asset(kf+"rmweights_"+str(epochs-1)+".tf.data-00000-of-00001")

    experiment.log_metric("Test_bmNN_z0_MSE",metrics.mean_squared_error(z0_PV_array,z0_bmNN_array))
    experiment.log_metric("Test_bmNN_PV_ROC",metrics.roc_auc_score(assoc_PV_array,assoc_bmNN_array))
    experiment.log_metric("Test_rmNN_z0_MSE",metrics.mean_squared_error(z0_PV_array,z0_rmNN_array))
    experiment.log_metric("Test_rmNN_PV_ROC",metrics.roc_auc_score(assoc_PV_array,assoc_rmNN_array))


    experiment.log_metric("Test_FH_z0_MSE",metrics.mean_squared_error(z0_PV_array,z0_FH_array))
    experiment.log_metric("Test_FH_PV_ROC",metrics.roc_auc_score(assoc_PV_array,assoc_FH_array))

    
if __name__=="__main__":
    with open(sys.argv[2]+'.yaml', 'r') as f:
        config = yaml.load(f)
    retrain = config["retrain"]

    if kf == "NewKF":
        train_files = glob.glob("NewKFData/Train/*.tfrecord")
        test_files = glob.glob("NewKFData/Test/*.tfrecord")
        val_files = glob.glob("NewKFData/Val/*.tfrecord")
        
    elif kf == "OldKF":
        train_files = glob.glob("OldKFData/Train/*.tfrecord")
        test_files = glob.glob("OldKFData/Test/*.tfrecord")
        val_files = glob.glob("OldKFData/Val/*.tfrecord")
       

    print ("Input Train files: ",len(train_files))
    print ("Input Validation files: ",len(val_files))
    print ("Input Test files: ",len(test_files))

    features = {
        "pvz0": tf.io.FixedLenFeature([1], tf.float32),
        #"pv2z0": tf.io.FixedLenFeature([1], tf.float32),
        "tp_met_pt":tf.io.FixedLenFeature([1], tf.float32),
        "pv_trk_met_pt":tf.io.FixedLenFeature([1], tf.float32),
        "true_met_pt":tf.io.FixedLenFeature([1], tf.float32),
        "trk_fromPV":tf.io.FixedLenFeature([max_ntracks], tf.float32),
        "PV_hist"  :tf.io.FixedLenFeature([256,1], tf.float32),
        #"PVpt_hist"  :tf.io.FixedLenFeature([256,1], tf.float32),
        "trk_hitpattern": tf.io.FixedLenFeature([max_ntracks*11], tf.float32)
        
    }

    trackFeatures = [
        'trk_z0',
        'trk_MVA1',
        'normed_trk_pt',
        'normed_trk_eta', 
        'normed_trk_invR',
        'binned_trk_chi2rphi', 
        'binned_trk_chi2rz', 
        'binned_trk_bendchi2',
        'normed_trk_overeta',
        'trk_z0_res',
        'log_pt',
        'trk_pt',
        'trk_eta',
        'trk_phi',
        'corrected_trk_z0'

    ]

    for trackFeature in trackFeatures:
        features[trackFeature] = tf.io.FixedLenFeature([max_ntracks], tf.float32)


    experiment = Experiment(
        project_name=config["comet_project_name"],
        auto_metric_logging=True,
        auto_param_logging=True,
        auto_histogram_weight_logging=True,
        auto_histogram_gradient_logging=True,
        auto_histogram_activation_logging=True,
    )

    experiment.set_name(kf+config['comet_experiment_name'])
    experiment.log_other("description",kf + config["description"])
    with open(kf+'experimentkey.txt', 'w') as fh:
      fh.write(experiment.get_key())  

    bignetwork = vtx.nn.E2Ecomparekernel(
        nbins=256,
        ntracks=max_ntracks, 
        nweightfeatures=6, 
        nfeatures=6, 
        nweights=1, 
        nlatent=2, 
        activation='relu',
        regloss=1e-10
    )

    reducednetwork = vtx.nn.E2EReduced(
        nbins=256,
        ntracks=max_ntracks, 
        nweightfeatures=6, 
        nfeatures=6, 
        nweights=1, 
        nlatent=2, 
        activation='relu',
        regloss=1e-10
    )

    startingLR = config['starting_lr']
    epochs = config['epochs']

    experiment.log_parameter("nbins",256)
    experiment.log_parameter("ntracks",max_ntracks)
    experiment.log_parameter("nfeatures",3)
    experiment.log_parameter("nlatent",2)
    experiment.log_parameter("activation",'relu')
    experiment.log_parameter("regloss",1e-10)
    experiment.log_parameter("Start LR",startingLR)
    experiment.log_parameter("Epochs",epochs)

    bigmodel = network.createE2EModel()
    bigoptimizer = tf.keras.optimizers.Adam(lr=startingLR)
    bigmodel.compile(
        bigoptimizer,
        loss=[
            tf.keras.losses.MeanAbsoluteError(),
            tf.keras.losses.BinaryCrossentropy(from_logits=True),
            lambda y,x: 0.,
            tf.keras.losses.MeanSquaredError()
        ],
        metrics=[
            tf.keras.metrics.BinaryAccuracy(threshold=0.,name='assoc_acc') #use thres=0 here since logits are used
        ],
        loss_weights=[config['z0_loss_weight'],
                      config['crossentropy_loss_weight'],
                      0.,
                      config['kernel_compare_loss_weight']]
    )
    bigmodel.summary()

    reducedmodel = network.createE2EModel()
    reducedoptimizer = tf.keras.optimizers.Adam(lr=startingLR)
    reducedmodel.compile(
        reducedoptimizer,
        loss=[
            tf.keras.losses.MeanAbsoluteError(),
            tf.keras.losses.BinaryCrossentropy(from_logits=True),
            tf.keras.losses.MeanSquaredError(),
            tf.keras.losses.MeanSquaredError(),
        ],
        metrics=[
            tf.keras.metrics.BinaryAccuracy(threshold=0.,name='assoc_acc') #use thres=0 here since logits are used
        ],
        loss_weights=[config['z0_loss_weight'],
                      config['crossentropy_loss_weight'],
                      config['weight_loss_weight'],
                      config['kernel_compare_loss_weight']]
    )
    model.summary()

    bigreduceLR = OwnReduceLROnPlateau(
    monitor='loss', factor=0.1, patience=6, verbose=1,
    mode='auto', min_delta=0.005, cooldown=0, min_lr=0
    )

    redreduceLR = OwnReduceLROnPlateau(
    monitor='loss', factor=0.1, patience=6, verbose=1,
    mode='auto', min_delta=0.005, cooldown=0, min_lr=0
    )

    if retrain:
        with experiment.train():
           train_model(bigmodel,reducedmodel,experiment,train_files,val_files,epochs=epochs,callbacks=[bigreduceLR,redreduceLR],nlatent=2),
    else:
        bigmodel.load_weights(kf+"bmweights_"+str( config['epochs'] - 1)+".tf")
        reducedmodel.load_weights(kf+"rmweights_"+str( config['epochs'] - 1)+".tf")

    with experiment.test():
        test_model(bigmodel,reducedmodel,experiment,test_files)
