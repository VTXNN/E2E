from comet_ml import Experiment
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
from TrainingScripts.Callbacks import OwnReduceLROnPlateau
from EvalScripts.eval import*

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
#hep.set_style("CMSTex")
hep.cms.label()
hep.cms.text("Simulation")

plt.style.use(hep.style.ROOT)

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

def train_model(model,experiment,train_files,val_files,epochs=50,callbacks=None,nlatent=0):

    total_steps = 0

    
    callbacks.on_train_begin()
    old_lr = model.optimizer.learning_rate.numpy()


    for epoch in range(epochs):
        
        #lr = starting_LR/(1+0.1*max(0,epoch-10)**1.5)
        #tf.keras.backend.set_value(model.optimizer.learning_rate, lr)
        new_lr = old_lr
        tf.keras.backend.set_value(model.optimizer.learning_rate, new_lr)
        experiment.log_metric("learning_rate",model.optimizer.learning_rate,step=total_steps,epoch=epoch)
        
        print ("Epoch %i"%epoch)
        
        if epoch>0:
            model.load_weights("weights_%i.tf"%(epoch-1))
        
        for step,batch in enumerate(setup_pipeline(train_files)):

            
            
            trackFeatures = np.stack([batch[feature] for feature in [
                'normed_trk_pt','normed_trk_eta', 'binned_trk_chi2rphi', 'binned_trk_chi2rz', 'binned_trk_bendchi2','trk_z0_res'
            ]],axis=2)
            #trackFeatures = np.concatenate([trackFeatures,batch['trk_hitpattern']],axis=2)
            #trackFeatures = np.concatenate([trackFeatures,batch['trk_z0_res']],axis=2)
            nBatch = batch['pvz0'].shape[0]
            result = model.train_on_batch(
            [batch['trk_z0'],trackFeatures],
            [batch['pvz0'],batch['trk_fromPV'],np.zeros((nBatch,max_ntracks,1))]
        )
            result = dict(zip(model.metrics_names,result))

            if nlatent == 0:
                experiment.log_metric("loss",result['loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("z0_loss",result['position_final_loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("assoc_loss",result['association_final_loss'],step=total_steps,epoch=epoch)
            else:
                experiment.log_metric("loss",result['loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("z0_loss",result['split_latent_loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("assoc_loss",result['association_final_loss'],step=total_steps,epoch=epoch)


            
            if step%10==0:
                predictedZ0_FH = predictFastHisto(batch['trk_z0'],batch['trk_pt'])
 
                predictedZ0_NN, predictedAssoc_NN, predictedWeights_NN = model.predict_on_batch(
                    [batch['trk_z0'],trackFeatures]
                )
                qz0_NN = np.percentile(predictedZ0_NN-batch['pvz0'],[5,32,50,68,95])
                qz0_FH = np.percentile(predictedZ0_FH-batch['pvz0'],[5,32,50,68,95])

                if nlatent == 0:
                    print ("Step %02i-%02i: loss=%.3f (z0=%.3f, assoc=%.3f), q68=(%.4f,%.4f), [FH: q68=(%.4f,%.4f)]"%(
                        epoch,step,
                        result['loss'],result['position_final_loss'],result['association_final_loss'],
                        qz0_NN[1],qz0_NN[3],qz0_FH[1],qz0_FH[3]
                    ))
                else:
                    print ("Step %02i-%02i: loss=%.3f (split latent z0=%.3f, assoc=%.3f), q68=(%.4f,%.4f), [FH: q68=(%.4f,%.4f)]"%(
                        epoch,step,
                        result['loss'],result['split_latent_loss'],result['association_final_loss'],
                        qz0_NN[1],qz0_NN[3],qz0_FH[1],qz0_FH[3]
                    ))


                print ("Train_NN_z0_MSE: "+str(metrics.mean_squared_error(batch['pvz0'],predictedZ0_NN))+" Train_FH_z0_MSE: "+str(metrics.mean_squared_error(batch['pvz0'],predictedZ0_FH)))

            total_steps += 1

        val_actual_PV = []
        val_predictedZ0_FH = []
        val_predictedZ0_NN = []
        val_predictedAssoc_NN = []
        val_predictedAssoc_FH = []
        val_actual_assoc = []

        for val_step,val_batch in enumerate(setup_pipeline(val_files)):
            val_predictedZ0_FH.append(predictFastHisto(val_batch['trk_z0'],val_batch['trk_pt']).flatten())

            val_trackFeatures = np.stack([val_batch[feature] for feature in [
            'normed_trk_pt','normed_trk_eta', 'binned_trk_chi2rphi', 'binned_trk_chi2rz', 'binned_trk_bendchi2','trk_z0_res'
            ]],axis=2)
            #val_trackFeatures = np.concatenate([val_trackFeatures,val_batch['trk_hitpattern']],axis=2)
            #val_trackFeatures = np.concatenate([val_trackFeatures,val_batch['trk_z0_res']],axis=2)
            
            temp_predictedZ0_NN, temp_predictedAssoc_NN, predictedWeights_NN = model.predict_on_batch(
                    [val_batch['trk_z0'],val_trackFeatures]
            )
                
            val_predictedZ0_NN.append(temp_predictedZ0_NN.numpy().flatten())
            val_predictedAssoc_NN.append(temp_predictedAssoc_NN.numpy().flatten())
            val_actual_PV.append(val_batch['pvz0'].numpy().flatten()) 
            val_actual_assoc.append(val_batch["trk_fromPV"].numpy().flatten())

            val_predictedAssoc_FH.append(FastHistoAssoc(val_batch['pvz0'],val_batch['trk_z0'],val_batch['trk_eta']).flatten())

        val_z0_NN_array = np.concatenate(val_predictedZ0_NN).ravel()
        val_z0_FH_array = np.concatenate(val_predictedZ0_FH).ravel()
        val_z0_PV_array = np.concatenate(val_actual_PV).ravel()

        val_assoc_NN_array = np.concatenate(val_predictedAssoc_NN).ravel()
        val_assoc_FH_array = np.concatenate(val_predictedAssoc_FH).ravel()
        val_assoc_PV_array = np.concatenate(val_actual_assoc).ravel()

        experiment.log_histogram_3d(val_z0_NN_array-val_z0_PV_array, name="Validation_CNN" , epoch=epoch)
        experiment.log_histogram_3d(val_z0_FH_array-val_z0_PV_array, name="Validation_FH", epoch=epoch)

        experiment.log_metric("Validation_NN_z0_MSE",metrics.mean_squared_error(val_z0_PV_array,val_z0_NN_array),epoch=epoch)
        experiment.log_metric("Validation_NN_z0_AE",metrics.mean_absolute_error(val_z0_PV_array,val_z0_NN_array),epoch=epoch)
        experiment.log_metric("Validation_FH_z0_MSE",metrics.mean_squared_error(val_z0_PV_array,val_z0_FH_array),epoch=epoch)
        experiment.log_metric("Validation_FH_z0_AE",metrics.mean_absolute_error(val_z0_PV_array,val_z0_FH_array),epoch=epoch)

        experiment.log_metric("Validation_NN_PV_ROC",metrics.roc_auc_score(val_assoc_PV_array,val_assoc_NN_array))
        experiment.log_metric("Validation_NN_PV_ACC",metrics.balanced_accuracy_score(val_assoc_PV_array,(val_assoc_NN_array>0.0)))

        experiment.log_metric("Validation_FH_PV_ROC",metrics.roc_auc_score(val_assoc_PV_array,val_assoc_FH_array))
        experiment.log_metric("Validation_FH_PV_ACC",metrics.balanced_accuracy_score(val_assoc_PV_array,val_assoc_FH_array))


        model.save_weights("weights_%i.tf"%(epoch))
        old_lr = callbacks.on_epoch_end(epoch=epoch,logs=result,lr=new_lr)
        
def test_model(model,experiment,test_files):

    predictedZ0_FH = []
    predictedZ0_NN = []

    predictedAssoc_NN = []
    predictedAssoc_FH = []
    
    actual_Assoc = []
    actual_PV = []

    predictedMET_pt = []
    tp_met  = []
    pv_trk_met= []
    true_met = []

    for step,batch in enumerate(setup_pipeline(test_files)):

        trackFeatures = np.stack([batch[feature] for feature in [
                'normed_trk_pt','normed_trk_eta', 'binned_trk_chi2rphi', 'binned_trk_chi2rz', 'binned_trk_bendchi2','trk_z0_res'
            ]],axis=2)
        #trackFeatures = np.concatenate([trackFeatures,batch['trk_hitpattern']],axis=2)
        #trackFeatures = np.concatenate([trackFeatures,batch['trk_z0_res']],axis=2)
        nBatch = batch['pvz0'].shape[0]
        predictedZ0_FH.append(predictFastHisto(batch['trk_z0'],batch['trk_pt']))

        actual_Assoc.append(batch["trk_fromPV"])
        actual_PV.append(batch['pvz0'])

        predictedAssoc_FH.append(FastHistoAssoc(batch['pvz0'],batch['trk_z0'],batch['trk_eta']))
            
        predictedZ0_NN_temp, predictedAssoc_NN_temp, predictedWeights_NN = model.predict_on_batch(
                    [batch['trk_z0'],trackFeatures,np.zeros((nBatch,max_ntracks,1))]
                )

        threshold = 0.0

        predicted_trk_met_px,predicted_trk_met_py,predicted_trk_met_pt,predicted_trk_met_phi = predictMET(batch['trk_pt'],batch['trk_phi'],predictedAssoc_NN_temp,threshold)

        predictedMET_pt.append(predicted_trk_met_pt)
        tp_met .append(batch['tp_met_pt'])
        pv_trk_met.append(batch['pv_trk_met_pt'])
        true_met.append(batch['true_met_pt'])

        predictedZ0_NN.append(predictedZ0_NN_temp)
        predictedAssoc_NN.append(predictedAssoc_NN_temp)

    z0_NN_array = np.concatenate(predictedZ0_NN).ravel()
    z0_FH_array = np.concatenate(predictedZ0_FH).ravel()
    z0_PV_array = np.concatenate(actual_PV).ravel()

    assoc_NN_array = np.concatenate(predictedAssoc_NN).ravel()
    assoc_FH_array = np.concatenate(predictedAssoc_FH).ravel()
    assoc_PV_array = np.concatenate(actual_Assoc).ravel()


    predictedMET_pt_array = np.concatenate(predictedMET_pt).ravel()
    tp_met_array = np.concatenate(tp_met).ravel()
    pv_trk_met_array = np.concatenate(pv_trk_met).ravel()
    true_met_array = np.concatenate(true_met).ravel()

    
    qz0_NN = np.percentile(z0_NN_array-z0_PV_array,[5,15,50,85,95])
    qz0_FH = np.percentile(z0_FH_array-z0_PV_array,[5,15,50,85,95])

    experiment.log_asset("weights_"+str(epochs-1)+".tf.index")

    experiment.log_figure("Test_z0_residual",figure=plotz0_residual((z0_PV_array-z0_NN_array),(z0_PV_array-z0_FH_array))) 
    experiment.log_figure("Test_PV_purityefficiency",figure=plotPV_roc(assoc_PV_array,assoc_NN_array,assoc_FH_array)) 
    experiment.log_figure("Test_PV_classdistributions",figure=plotPV_classdistribution(assoc_PV_array,assoc_NN_array,assoc_FH_array))
    experiment.log_figure("Test_z0_percentiles",figure=plotz0_percentile((z0_PV_array-z0_NN_array),(z0_PV_array-z0_FH_array)))

    experiment.log_figure("Test_MET_residual",figure=plotMET_residual((true_met_array-predictedMET_pt_array),
                                                                      (true_met_array-pv_trk_met_array),
                                                                      (true_met_array-tp_met_array),
                                                                      threshold = threshold,
                                                                     ))
                                                                    
    experiment.log_figure("Test_MET_rel_residual",figure=plotMET_residual((true_met_array-predictedMET_pt_array),
                                                                          (true_met_array-pv_trk_met_array),
                                                                          (true_met_array-tp_met_array),
                                                                          threshold = threshold,
                                                                          relative=True,
                                                                          true=true_met_array
                                                                         ))

    experiment.log_metric("Test_NN_z0_MSE",metrics.mean_squared_error(z0_PV_array,z0_NN_array))
    experiment.log_metric("Test_NN_z0_AE",metrics.mean_absolute_error(z0_PV_array,z0_NN_array))

    experiment.log_metric("Test_FH_z0_MSE",metrics.mean_squared_error(z0_PV_array,z0_FH_array))
    experiment.log_metric("Test_FH_z0_AE",metrics.mean_absolute_error(z0_PV_array,z0_FH_array))

    experiment.log_metric("Test_NN_PV_ROC",metrics.roc_auc_score(assoc_PV_array,assoc_NN_array))
    experiment.log_metric("Test_NN_PV_ACC",metrics.balanced_accuracy_score(assoc_PV_array,(assoc_NN_array>0.0)))

    experiment.log_metric("Test_FH_PV_ROC",metrics.roc_auc_score(assoc_PV_array,assoc_FH_array))
    experiment.log_metric("Test_FH_PV_ACC",metrics.balanced_accuracy_score(assoc_PV_array,assoc_FH_array))

    experiment.log_metric("Test_NN_z0_quartiles_5",qz0_NN[0])
    experiment.log_metric("Test_NN_z0_quartiles_15",qz0_NN[1])
    experiment.log_metric("Test_NN_z0_quartiles_50",qz0_NN[2])
    experiment.log_metric("Test_NN_z0_quartiles_85",qz0_NN[3])
    experiment.log_metric("Test_NN_z0_quartiles_95",qz0_NN[4])
    experiment.log_metric("Test_FH_z0_quartiles_5",qz0_FH[0])
    experiment.log_metric("Test_FH_z0_quartiles_15",qz0_FH[1])
    experiment.log_metric("Test_FH_z0_quartiles_50",qz0_FH[2])
    experiment.log_metric("Test_FH_z0_quartiles_85",qz0_FH[3])
    experiment.log_metric("Test_FH_z0_quartiles_95",qz0_FH[4])
    
if __name__=="__main__":
    retrain = True

    train_files = glob.glob("/home/cb719/Documents/L1Trigger/GTT/Vertexing/E2E/Train/Data/Train/*.tfrecord")
    test_files = glob.glob("/home/cb719/Documents/L1Trigger/GTT/Vertexing/E2E/Train/Data/Test/*.tfrecord")
    val_files = glob.glob("/home/cb719/Documents/L1Trigger/GTT/Vertexing/E2E/Train/Data/Val/*.tfrecord")
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
        "trk_hitpattern": tf.io.FixedLenFeature([max_ntracks*11], tf.float32)
        
    }

    trackFeatures = [
        'trk_z0',
        'normed_trk_pt',
        'normed_trk_pt',
        'normed_trk_eta', 
        'normed_trk_invR',
        'binned_trk_chi2rphi', 
        'binned_trk_chi2rz', 
        'binned_trk_bendchi2',
        'trk_z0_res',
        'log_pt',
        'trk_pt',
        'trk_eta',
        'trk_phi',

    ]

    for trackFeature in trackFeatures:
        features[trackFeature] = tf.io.FixedLenFeature([max_ntracks], tf.float32)


    experiment = Experiment(
        project_name="Vertex_CNN",
        auto_metric_logging=True,
        auto_param_logging=True,
        auto_histogram_weight_logging=True,
        auto_histogram_gradient_logging=True,
        auto_histogram_activation_logging=True,
    )

    experiment.set_name("z0CNN")
    experiment.log_other("description","NoHitpattern,maloss,z0_res,normedandbinnedfeatures,NewKFTracks")
        
    network = vtx.nn.E2ERef(
        nbins=256,
        ntracks=max_ntracks, 
        nfeatures=6, 
        nweights=1, 
        nlatent=0, 
        activation='relu',
        regloss=1e-10
    )

    startingLR = 0.001
    epochs = 75

    experiment.log_parameter("nbins",256)
    experiment.log_parameter("ntracks",max_ntracks)
    experiment.log_parameter("nfeatures",6)
    experiment.log_parameter("nlatent",0)
    experiment.log_parameter("activation",'relu')
    experiment.log_parameter("regloss",1e-10)
    experiment.log_parameter("Start LR",startingLR)
    experiment.log_parameter("Epochs",epochs)

    model = network.createE2EModel()
    optimizer = tf.keras.optimizers.Adam(lr=startingLR)
    model.compile(
        optimizer,
        loss=[
            tf.keras.losses.MeanAbsoluteError(),
            #tf.keras.losses.MeanSquaredError(),
            tf.keras.losses.BinaryCrossentropy(from_logits=True),
            lambda y,x: 0.
        ],
        metrics=[
            tf.keras.metrics.BinaryAccuracy(threshold=0.,name='assoc_acc') #use thres=0 here since logits are used
        ],
        loss_weights=[1.,1.,0.]
    )
    model.summary()

    reduceLR = OwnReduceLROnPlateau(
    monitor='loss', factor=0.1, patience=6, verbose=1,
    mode='auto', min_delta=0.01, cooldown=0, min_lr=0
    )

    if retrain:
        with experiment.train():
           train_model(model,experiment,train_files,val_files,epochs=epochs,callbacks=reduceLR,nlatent=0),
    else:
        model.load_weights("weights_74.tf")

    with experiment.test():
        test_model(model,experiment,test_files)
