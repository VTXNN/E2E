from comet_ml import Experiment
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

import sys
import glob

import sklearn.metrics as metrics
import vtx
import TrainingScripts
import EvalScripts.eval as eval
import pandas as pd

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
    bit_z0 = 'bit_trk_z0'
elif kf == "OldKF":
    z0 = 'corrected_trk_z0'
    bit_z0 = 'bit_corrected_trk_z0'

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

def train_model(model,experiment,train_files,val_files,trackfeat,weightfeat,epochs=50,callbacks=None,nlatent=0,trainable=False,bit=False,model_name=None):

    total_steps = 0
    early_stop_patience = 100
    wait = 0
    best_score = 100000
    callbacks.on_train_begin()
    old_lr = model.optimizer.learning_rate.numpy()

    for epoch in range(epochs):

        new_lr = old_lr
        tf.keras.backend.set_value(model.optimizer.learning_rate, new_lr)
        experiment.log_metric("learning_rate",model.optimizer.learning_rate,step=total_steps,epoch=epoch)
        
        print ("Epoch %i"%epoch)
        
        if epoch>0:
            model.load_weights(model_name+".tf")
        
        for step,batch in enumerate(setup_pipeline(train_files)):
            z0Shift = np.random.normal(0.0,1.0,size=batch['pvz0'].shape)
            z0Flip = 2.*np.random.randint(2,size=batch['pvz0'].shape)-1.
            batch[z0]=batch[z0]*z0Flip + z0Shift
            batch['pvz0']=batch['pvz0']*z0Flip + z0Shift

            trackFeatures = np.stack([batch[feature] for feature in trackfeat],axis=2)

            WeightFeatures = np.stack([batch[feature] for feature in weightfeat ],axis=2)

            nBatch = batch['pvz0'].shape[0]

            if bit:
                result = model.train_on_batch([batch[bit_z0],WeightFeatures,trackFeatures], [batch['pvz0'],batch['trk_fromPV'],np.zeros([nBatch,max_ntracks,1])])   
                                              # convs                          #softmax     #binweight                                           
            else:
                result = model.train_on_batch([batch[z0],WeightFeatures,trackFeatures], [batch['pvz0'],batch['trk_fromPV'],np.zeros([nBatch,max_ntracks,1])])   

            result = dict(zip(model.metrics_names,result))

            if trainable == "FH":
                experiment.log_metric("loss",result['loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("z0_loss",result['binto_vertex_loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("assoc_loss",result['association_final_loss'],step=total_steps,epoch=epoch)
            elif (trainable == "DiffArgMax"):
                if nlatent > 0:
                    experiment.log_metric("z0_loss",result['split_latent_loss'],step=total_steps,epoch=epoch)
                else:
                    experiment.log_metric("z0_loss",result['position_final_loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("loss",result['loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("assoc_loss",result['association_final_loss'],step=total_steps,epoch=epoch)

            elif (trainable == "QDiffArgMax"):
                if nlatent > 0:
                    experiment.log_metric("z0_loss",result['split_latent_loss'],step=total_steps,epoch=epoch)
                else:
                    experiment.log_metric("z0_loss",result['position_final_loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("loss",result['loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("assoc_loss",result['association_final_loss'],step=total_steps,epoch=epoch)

            elif trainable == "FullNetwork":
                experiment.log_metric("loss",result['loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("z0_loss",result['split_latent_loss'],step=total_steps,epoch=epoch)
                experiment.log_metric("assoc_loss",result['association_final_loss'],step=total_steps,epoch=epoch)

            if step%10==0:
                predictedZ0_FH = eval.predictFastHisto(batch[z0],batch['trk_pt'])
                if bit:
                    predictedZ0_NN, predictedAssoc_NN,predicted_weights = model.predict_on_batch( [batch[bit_z0],WeightFeatures,trackFeatures] )
                else:
                    predictedZ0_NN, predictedAssoc_NN,predicted_weights = model.predict_on_batch( [batch[z0],WeightFeatures,trackFeatures] )

                qz0_NN = np.percentile(predictedZ0_NN-batch['pvz0'],[5,32,50,68,95])
                qz0_FH = np.percentile(predictedZ0_FH-batch['pvz0'],[5,32,50,68,95])

                if trainable == "FH":
                    print ("Step %02i-%02i: loss=%.3f (z0=%.3f, assoc=%.3f), q68=(%.4f,%.4f), [FH: q68=(%.4f,%.4f)]"%(
                            epoch,step,
                            result['loss'],result['binto_vertex_loss'],result['association_final_loss'],
                            qz0_NN[1],qz0_NN[3],qz0_FH[1],qz0_FH[3]
                        ))
                    print ("Train_NN_z0_MSE: "+str(metrics.mean_squared_error(batch['pvz0'],predictedZ0_NN))+" Train_FH_z0_MSE: "+str(metrics.mean_squared_error(batch['pvz0'],predictedZ0_FH)))          
                elif ((trainable == "DiffArgMax") | (trainable == "QDiffArgMax")) :
                    if nlatent > 0:
                        print ("Step %02i-%02i: loss=%.3f (z0=%.3f, assoc=%.3f), q68=(%.4f,%.4f), [FH: q68=(%.4f,%.4f)]"%(
                                epoch,step,
                                result['loss'],result['split_latent_loss'],result['association_final_loss'],
                                qz0_NN[1],qz0_NN[3],qz0_FH[1],qz0_FH[3]
                            ))
                    else:
                        print ("Step %02i-%02i: loss=%.3f (z0=%.3f, assoc=%.3f), q68=(%.4f,%.4f), [FH: q68=(%.4f,%.4f)]"%(
                                epoch,step,
                                result['loss'],result['position_final_loss'],result['association_final_loss'],
                                qz0_NN[1],qz0_NN[3],qz0_FH[1],qz0_FH[3]
                            ))
                    print ("Train_NN_z0_MSE: "+str(metrics.mean_squared_error(batch['pvz0'],predictedZ0_NN))+" Train_FH_z0_MSE: "+str(metrics.mean_squared_error(batch['pvz0'],predictedZ0_FH)))   
                elif trainable == "FullNetwork":
                    print ("Step %02i-%02i: loss=%.3f (split latent z0=%.3f, assoc=%.3f), q68=(%.4f,%.4f), [FH: q68=(%.4f,%.4f)]"%(
                        epoch,step,
                        result['loss'],result['split_latent_loss'],result['association_final_loss'],
                        qz0_NN[1],qz0_NN[3],qz0_FH[1],qz0_FH[3]
                    ))


            total_steps += 1

        val_actual_PV = []
        val_predictedZ0_FH = []
        val_predictedZ0_NN = []
        val_predictedAssoc_NN = []
        val_predictedAssoc_FH = []
        val_actual_assoc = []

        for val_step,val_batch in enumerate(setup_pipeline(val_files)):
            val_predictedZ0_FH.append(eval.predictFastHisto(val_batch[z0],val_batch['trk_pt']).flatten())

            val_trackFeatures = np.stack([val_batch[feature] for feature in trackfeat],axis=2)

            val_WeightFeatures = np.stack([val_batch[feature] for feature in weightfeat],axis=2)

            if bit:
                temp_predictedZ0_NN, temp_predictedAssoc_NN,predicted_weights  = model.predict_on_batch(
                        [val_batch[bit_z0],val_WeightFeatures,val_trackFeatures]
                )
            else:
                temp_predictedZ0_NN, temp_predictedAssoc_NN,predicted_weights  = model.predict_on_batch(
                        [val_batch[z0],val_WeightFeatures,val_trackFeatures]
                )
  
            val_predictedZ0_NN.append(temp_predictedZ0_NN.flatten())
            val_predictedAssoc_NN.append(temp_predictedAssoc_NN.flatten())
            val_actual_PV.append(val_batch['pvz0'].numpy().flatten()) 
            val_actual_assoc.append(val_batch["trk_fromPV"].numpy().flatten())

            val_predictedAssoc_FH.append(eval.FastHistoAssoc(val_batch['pvz0'],val_batch[z0],val_batch['trk_eta'],kf=kf).flatten())
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

        old_lr = callbacks.on_epoch_end(epoch=epoch,logs=result,lr=new_lr)

        val_loss = metrics.mean_squared_error(val_z0_PV_array,val_z0_NN_array)

        print ("Val_NN_z0_MSE: "+str(val_loss)+" Best_NN_z0_MSE: "+str(best_score)) 
        

        wait += 1
        if val_loss < best_score:
            best_score = val_loss
            model.save_weights(model_name+".tf")
            wait = 0
        if wait >= early_stop_patience:
            break
        
def test_model(model,experiment,test_files,trackfeat,weightfeat,model_name=None):

    predictedZ0_FH = []
    predictedZ0_NN = []

    predictedAssoc_NN = []
    predictedAssoc_FH = []
    
    actual_Assoc = []
    actual_PV = []

    for step,batch in enumerate(setup_pipeline(test_files)):

        trackFeatures = np.stack([batch[feature] for feature in trackfeat],axis=2)
        WeightFeatures = np.stack([batch[feature] for feature in weightfeat],axis=2)

        predictedZ0_FH.append(eval.predictFastHisto(batch[z0],batch['trk_pt']))

        actual_Assoc.append(batch["trk_fromPV"])
        actual_PV.append(batch['pvz0'])

        predictedAssoc_FH.append(eval.FastHistoAssoc(batch['pvz0'],batch[z0],batch['trk_eta'],kf=kf))

        if bit:   
            predictedZ0_NN_temp, predictedAssoc_NN_temp,predicted_weights = model.predict_on_batch( [batch[bit_z0],WeightFeatures,trackFeatures] )
        else:
            predictedZ0_NN_temp, predictedAssoc_NN_temp,predicted_weights = model.predict_on_batch( [batch[z0],WeightFeatures,trackFeatures] )

        predictedZ0_NN.append(predictedZ0_NN_temp)
        predictedAssoc_NN.append(predictedAssoc_NN_temp)

    z0_NN_array = np.concatenate(predictedZ0_NN).ravel()
    z0_FH_array = np.concatenate(predictedZ0_FH).ravel()
    z0_PV_array = np.concatenate(actual_PV).ravel()

    assoc_NN_array = np.concatenate(predictedAssoc_NN).ravel()
    assoc_FH_array = np.concatenate(predictedAssoc_FH).ravel()
    assoc_PV_array = np.concatenate(actual_Assoc).ravel()

    qz0_NN = np.percentile(z0_NN_array-z0_PV_array,[5,15,50,85,95])
    qz0_FH = np.percentile(z0_FH_array-z0_PV_array,[5,15,50,85,95])

    experiment.log_asset(model_name+".tf.index")
    experiment.log_asset(model_name+".tf.data-00000-of-00001")

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
    with open(sys.argv[2]+'.yaml', 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    retrain = config["retrain"]

    trainable = config["trainable"]
    trackfeat = config["track_features"] 
    weightfeat = config["weight_features"] 
    bit = config["bit_inputs"]
    nlatent = config["Nlatent"]
    PretrainedModelName = config["PretrainedModelName"] 
    UnQuantisedModelName = config["UnquantisedModelName"] 
    pretrain_DA = config["pretrain_DA"]

    if (trainable == "DiffArgMax") | (trainable == "QDiffArgMax"):
        
        network = vtx.nn.E2EDiffArgMax(
            nbins=256,
            start=-15,
            end = 15,
            ntracks=max_ntracks, 
            nweightfeatures=len(weightfeat), 
            nfeatures=len(trackfeat), 
            nweights=1, 
            nlatent = nlatent,
            activation='relu',
            l2regloss=1e-10,
            temperature=1e-2,
            nweightnodes = config['nweightnodes'],
            nweightlayers = config['nweightlayers'],
            nassocnodes = config['nassocnodes'],
            nassoclayers = config['nassoclayers'],
        )

    elif trainable == "FH":

        network = vtx.nn.E2EFH(
            nbins=256,
            ntracks=max_ntracks, 
            nweightfeatures=len(weightfeat), 
            nfeatures=len(trackfeat), 
            nweights=1, 
            activation='relu',
            regloss=1e-10
        )

    elif trainable == "FullNetwork":

        network = vtx.nn.E2Ecomparekernel(
            nbins=256,
            ntracks=max_ntracks, 
            nweightfeatures=len(weightfeat), 
            nfeatures=len(trackfeat), 
            nweights=1, 
            nlatent = nlatent,
            activation='relu',
            regloss=1e-10
        )

    if kf == "NewKF":
        train_files = glob.glob(config["data_folder"]+"/Train/*.tfrecord")
        test_files = glob.glob(config["data_folder"]+"/Test/*.tfrecord")
        val_files = glob.glob(config["data_folder"]+"/Val/*.tfrecord")

        trackFeatures = [
            'trk_z0',
            'normed_trk_pt',
            'normed_trk_eta',
            'trk_pt',
            'trk_eta',
            'trk_MVA1',
            'trk_z0_res',
            'corrected_trk_z0',
            'normed_trk_over_eta',
    ]
        
    elif kf == "OldKF":
        train_files = glob.glob(config["data_folder"]+"/Train/*.tfrecord")
        test_files = glob.glob(config["data_folder"]+"/Test/*.tfrecord")
        val_files = glob.glob(config["data_folder"]+"/Val/*.tfrecord")

        trackFeatures = [
            'trk_z0',
            'normed_trk_pt',
            'normed_trk_eta',
            'trk_pt',
            'trk_eta',
            'trk_MVA1',
            'trk_z0_res',
            'corrected_trk_z0',
            'normed_trk_over_eta',
            'abs_trk_word_pT',
            'rescaled_trk_word_MVAquality',
            'abs_trk_word_eta',
    ]
       

    print ("Input Train files: ",len(train_files))
    print ("Input Validation files: ",len(val_files))
    print ("Input Test files: ",len(test_files))

    features = {
        "pvz0": tf.io.FixedLenFeature([1], tf.float32),    
        "trk_fromPV":tf.io.FixedLenFeature([max_ntracks], tf.float32) ,
        "PV_hist"  :tf.io.FixedLenFeature([256,1], tf.float32),
    }

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

    model = network.createE2EModel()
    optimizer = tf.keras.optimizers.Adam(lr=startingLR)
    model.compile(
        optimizer,
        loss=[
            #tf.keras.losses.Huber(config['Huber_delta']),
            tf.keras.losses.MeanAbsoluteError(),
            tf.keras.losses.BinaryCrossentropy(from_logits=True),
            lambda y,x: 0.,
        ],
        metrics=[
            tf.keras.metrics.BinaryAccuracy(threshold=0.,name='assoc_acc') #use thres=0 here since logits are used
        ],
        loss_weights=[config['z0_loss_weight'],
                      config['crossentropy_loss_weight'],
                      0
                      ]
    )
    model.summary()

    if trainable == "FH":

        model.layers[2].set_weights([np.array([[1]], dtype=np.float32)])
        model.layers[4].set_weights([np.array([[[1]],[[1]],[[1]]], dtype=np.float32)])
        model.layers[7].set_weights([np.expand_dims(np.arange(256),axis=0)])
    elif trainable == "DiffArgMax":
        model.layers[17].set_weights([np.expand_dims(np.arange(256),axis=0)]) #Set to bin index 
        model.layers[19].set_weights([np.array([[1]], dtype=np.float32), np.array([0], dtype=np.float32)])

    elif trainable == "QDiffArgMax":
        model.layers[11].set_weights([np.expand_dims(np.arange(256),axis=0)]) #Set to bin index 

    if pretrain_DA:
        model.load_weights(PretrainedModelName+".tf")
        

    reduceLR = TrainingScripts.Callbacks.OwnReduceLROnPlateau(
    monitor='loss', factor=0.1, patience=6, verbose=1,
    mode='auto', min_delta=0.005, cooldown=0, min_lr=0
    )
    with experiment.train():
        train_model(model,experiment,train_files,val_files,trackfeat,weightfeat,epochs=epochs,callbacks=reduceLR,nlatent=nlatent,trainable=trainable,bit=bit,model_name=UnQuantisedModelName),
    with experiment.test():
        test_model(model,experiment,test_files,trackfeat,weightfeat,model_name=UnQuantisedModelName)
