from comet_ml import Experiment
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

import sys
import glob

import sklearn.metrics as metrics
import vtx
import TrainingScripts
import EvalScripts.eval_funcs as eval_funcs
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

max_z0 = 20.46912512

if kf == "NewKF":
    z0 = 'trk_z0'
    FH_z0 = 'trk_z0'
elif kf == "OldKF":
    z0 = 'corrected_trk_z0'
    FH_z0 = 'corrected_trk_z0'
elif kf == "OldKF_intZ":
    z0 = 'corrected_int_z0'
    FH_z0 = 'corrected_trk_z0'
elif kf == "NewKF_intZ":
    z0 = 'int_z0'
    FH_z0 = 'trk_z0'


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

def train_model(model,experiment,train_files,val_files,trackfeat,weightfeat,epochs=50,callbacks=None,nlatent=0,bit=False,model_name=[None,None]):

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
            model.load_weights(model_name[0]+model_name[1]+".tf")
        
        for step,batch in enumerate(setup_pipeline(train_files)):
            z0Shift = np.random.randint(-10,10,size=batch['pvz0'].shape)

            #Zflip = np.random.randint(2,size=batch['pvz0'].shape)
            #z0Flip = 2.*Zflip-1.
            #flipz0Flip = 1 - Zflip

            batch[z0] = batch[z0] + z0Shift
            batch['pvz0']= batch['pvz0'] + (z0Shift*((max_z0*2)/nbins))
            batch[FH_z0]= batch[FH_z0] + (z0Shift*((max_z0*2)/nbins))

            trackFeatures = np.stack([batch[feature] for feature in trackfeat],axis=2)

            WeightFeatures = np.stack([batch[feature] for feature in weightfeat ],axis=2)

            nBatch = batch['pvz0'].shape[0]
                                  
            result = model.train_on_batch([batch[z0],WeightFeatures,trackFeatures], [batch['pvz0'],batch['trk_fromPV'],np.zeros([nBatch,max_ntracks,1])])   

            result = dict(zip(model.metrics_names,result))

            if nlatent > 0:
                experiment.log_metric("z0_loss",result['split_latent_loss'],step=total_steps,epoch=epoch)
            else:
                experiment.log_metric("z0_loss",result['position_final_loss'],step=total_steps,epoch=epoch)
            experiment.log_metric("loss",result['loss'],step=total_steps,epoch=epoch)
            experiment.log_metric("assoc_loss",result['association_final_loss'],step=total_steps,epoch=epoch)

            if step%10==0:
                predictedZ0_FH = eval_funcs.predictFastHisto(batch[FH_z0],batch['trk_pt'],res_func=eval_funcs.linear_res_function(batch['trk_pt']))
  
                predictedZ0_NN, predictedAssoc_NN,predicted_weights = model.predict_on_batch( [batch[z0],WeightFeatures,trackFeatures] )

                qz0_NN = np.percentile(predictedZ0_NN-batch['pvz0'],[5,32,50,68,95])
                qz0_FH = np.percentile(predictedZ0_FH-batch['pvz0'],[5,32,50,68,95])
         
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

            total_steps += 1

        prune_level = []
        for i,layer in enumerate(model.layers):
            get_weights = layer.get_weights()
            if len(get_weights) > 0:
                if "Bin_weight" not in layer.name:
                    weights = get_weights[0].flatten()[get_weights[0].flatten() != 0]
                    prune_level.append(weights.shape[0]/get_weights[0].flatten().shape[0])
                    experiment.log_histogram_3d(get_weights[0].flatten()[get_weights[0].flatten() != 0],name="Weight"+layer.name,epoch=epoch)
                    if len(get_weights) > 1:
                        experiment.log_histogram_3d(get_weights[1].flatten()[get_weights[1].flatten() != 0],name="Bias"+layer.name,epoch=epoch)
                    

        val_actual_PV = []
        val_predictedZ0_FH = []
        val_predictedZ0_NN = []
        val_predictedAssoc_NN = []
        val_predictedAssoc_FH = []
        val_actual_assoc = []

        for val_step,val_batch in enumerate(setup_pipeline(val_files)):
            val_predictedZ0_FH.append(eval_funcs.predictFastHisto(val_batch[FH_z0],val_batch['trk_pt'],res_func=eval_funcs.linear_res_function(val_batch['trk_pt'])).flatten())

            val_trackFeatures = np.stack([val_batch[feature] for feature in trackfeat],axis=2)

            val_WeightFeatures = np.stack([val_batch[feature] for feature in weightfeat],axis=2)

            temp_predictedZ0_NN, temp_predictedAssoc_NN,predicted_weights  = model.predict_on_batch(
                        [val_batch[z0],val_WeightFeatures,val_trackFeatures]
                )
  
            val_predictedZ0_NN.append(temp_predictedZ0_NN.flatten())
            val_predictedAssoc_NN.append(temp_predictedAssoc_NN.flatten())
            val_actual_PV.append(val_batch['pvz0'].numpy().flatten()) 
            val_actual_assoc.append(val_batch["trk_fromPV"].numpy().flatten())

            val_predictedAssoc_FH.append(eval_funcs.FastHistoAssoc(val_batch['pvz0'],val_batch[FH_z0],val_batch['trk_eta'],res_func=eval_funcs.linear_res_function(val_batch['trk_eta']),kf=kf).flatten())
        val_z0_NN_array = np.concatenate(val_predictedZ0_NN).ravel()
        val_z0_FH_array = np.concatenate(val_predictedZ0_FH).ravel()
        val_z0_PV_array = np.concatenate(val_actual_PV).ravel()

        val_assoc_NN_array = np.concatenate(val_predictedAssoc_NN).ravel()
        val_assoc_FH_array = np.concatenate(val_predictedAssoc_FH).ravel()
        val_assoc_PV_array = np.concatenate(val_actual_assoc).ravel()

        experiment.log_histogram_3d(val_z0_NN_array-val_z0_PV_array, name="Validation_CNN" , epoch=epoch)
        experiment.log_histogram_3d(val_z0_FH_array-val_z0_PV_array, name="Validation_FH", epoch=epoch)

        experiment.log_histogram_3d(val_z0_NN_array, name="CNN_Output" , epoch=epoch)
        experiment.log_histogram_3d(val_z0_FH_array, name="FH_Output", epoch=epoch)

        experiment.log_metric("Validation_NN_z0_MSE",metrics.mean_squared_error(val_z0_PV_array,val_z0_NN_array),epoch=epoch)
        experiment.log_metric("Validation_NN_z0_AE",metrics.mean_absolute_error(val_z0_PV_array,val_z0_NN_array),epoch=epoch)
        experiment.log_metric("Validation_FH_z0_MSE",metrics.mean_squared_error(val_z0_PV_array,val_z0_FH_array),epoch=epoch)
        experiment.log_metric("Validation_FH_z0_AE",metrics.mean_absolute_error(val_z0_PV_array,val_z0_FH_array),epoch=epoch)

        experiment.log_metric("Validation_NN_PV_ROC",metrics.roc_auc_score(val_assoc_PV_array,val_assoc_NN_array))
        experiment.log_metric("Validation_NN_PV_ACC",metrics.balanced_accuracy_score(val_assoc_PV_array,(val_assoc_NN_array>0.0)))

        experiment.log_metric("Validation_FH_PV_ROC",metrics.roc_auc_score(val_assoc_PV_array,val_assoc_FH_array))
        experiment.log_metric("Validation_FH_PV_ACC",metrics.balanced_accuracy_score(val_assoc_PV_array,val_assoc_FH_array))

        experiment.log_metric("Validation_Prune_Level",np.mean(prune_level))
        
        old_lr = callbacks.on_epoch_end(epoch=epoch,logs=result,lr=new_lr)

        val_loss = metrics.mean_squared_error(val_z0_PV_array,val_z0_NN_array)

        print ("Val_NN_z0_MSE: "+str(val_loss)+" Best_NN_z0_MSE: "+str(best_score)) 
        
        wait += 1
        if val_loss < best_score:
            best_score = val_loss
            model.save_weights(model_name[0]+model_name[1]+".tf")
            wait = 0
        if wait >= early_stop_patience:
            break
        
def test_model(model,experiment,test_files,trackfeat,weightfeat,model_name=[None,None]):

    predictedZ0_FH = []
    predictedZ0_NN = []

    predictedAssoc_NN = []
    predictedAssoc_FH = []
    
    actual_Assoc = []
    actual_PV = []

    for step,batch in enumerate(setup_pipeline(test_files)):

        trackFeatures = np.stack([batch[feature] for feature in trackfeat],axis=2)
        WeightFeatures = np.stack([batch[feature] for feature in weightfeat],axis=2)

        predictedZ0_FH.append(eval_funcs.predictFastHisto(batch[FH_z0],batch['trk_pt'],res_func=eval_funcs.linear_res_function(batch['trk_pt'])))

        actual_Assoc.append(batch["trk_fromPV"])
        actual_PV.append(batch['pvz0'])

        predictedAssoc_FH.append(eval_funcs.FastHistoAssoc(batch['pvz0'],batch[FH_z0],batch['trk_eta'],res_func=eval_funcs.linear_res_function(batch['trk_eta']),kf=kf))
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

    experiment.log_asset(model_name[0]+model_name[1]+".tf.index")
    experiment.log_asset(model_name[0]+model_name[1]+".tf.data-00000-of-00001")

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

    trainable = sys.argv[3]
    trackfeat = config["track_features"] 
    weightfeat = config["weight_features"] 

    nbins = config['nbins']

    train_cnn = config["train_cnn"]
    
    nlatent = config["Nlatent"]

    position_final_weights = []
    position_final_bias = []
    for i in range(nlatent+1):
        position_final_weights.append(1)
        position_final_bias.append(30/256)

    
    PretrainedModelName = config["PretrainedModelName"] 
    pretrain_DA = config["pretrain_DA"]

    if (kf == 'NewKF') | (kf == 'OldKF'):
        start = -1*max_z0
        end = max_z0
        bit = False

    elif (kf == 'OldKF_intZ') | (kf == 'NewKF_intZ'):
        start = 0
        end = nbins - 1
        bit = True

    if trainable == 'DA':

        network = vtx.nn.E2EDiffArgMax(
            nbins=nbins,
            start=start,
            end = end,
            max_z0 = max_z0,
            ntracks=max_ntracks, 
            nweightfeatures=len(weightfeat), 
            nfeatures=len(trackfeat), 
            nweights=1, 
            nlatent = nlatent,
            return_index = bit,
            train_cnn = train_cnn,
            activation='relu',
            l2regloss=1e-10,
            temperature=1e-4,
            nweightnodes = config['nweightnodes'],
            nweightlayers = config['nweightlayers'],
            nassocnodes = config['nassocnodes'],
            nassoclayers = config['nassoclayers'],
        )

        model_name = [config["UnquantisedModelName"] ,""]
        epochs = config['epochs']

    if trainable == 'QDA':

        network = vtx.nn.E2EQKerasDiffArgMax(
            nbins=nbins,
            start = start,
            end = end,
            max_z0 = max_z0,
            ntracks=max_ntracks, 
            nweightfeatures=len(weightfeat), 
            nfeatures=len(trackfeat), 
            return_index = bit,
            nweights=1, 
            nlatent = nlatent,
            train_cnn = train_cnn,
            activation='relu',
            l1regloss = (float)(config['l1regloss']),
            l2regloss = (float)(config['l2regloss']),
            nweightnodes = config['nweightnodes'],
            nweightlayers = config['nweightlayers'],
            nassocnodes = config['nassocnodes'],
            nassoclayers = config['nassoclayers'],
            temperature = 1e-4,
            qconfig = config['QConfig']
        )

        model_name = [config['QuantisedModelName'],"_prune_iteration_0"]
        epochs = config['qtrain_epochs']

    if trainable == 'QDA_prune':
        network = vtx.nn.E2EQKerasDiffArgMaxConstraint(
            nbins=nbins,
            start = start,
            end = end,
            max_z0 = max_z0,
            ntracks=max_ntracks, 
            return_index = bit,
            nweightfeatures=len(weightfeat), 
            nfeatures=len(trackfeat), 
            nweights=1, 
            nlatent = nlatent,
            train_cnn = train_cnn,
            activation='relu',
            l1regloss = (float)(config['l1regloss']),
            l2regloss = (float)(config['l2regloss']),
            nweightnodes = config['nweightnodes'],
            nweightlayers = config['nweightlayers'],
            nassocnodes = config['nassocnodes'],
            nassoclayers = config['nassoclayers'],
            temperature = 1e-4,
            qconfig = config['QConfig'],
            h5fName = config['QuantisedModelName']+'_drop_weights_iteration_'+sys.argv[4]+'.h5'
        )

        model_name = [config['QuantisedModelName'],"_prune_iteration_"+str(int(sys.argv[4])+1)]
        epochs = config['qtrain_epochs']

    if (kf == "NewKF")  | (kf == 'NewKF_intZ'):
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
                'normed_trk_over_eta',
                'int_z0',
                'trk_bendchi2',
                'trk_chi2rphi',
                'trk_chi2rz'
        ]
        
    elif (kf == "OldKF") | (kf == 'OldKF_intZ'):
        train_files = glob.glob(config["data_folder"]+"/Train/*.tfrecord")
        test_files = glob.glob(config["data_folder"]+"/Test/*.tfrecord")
        val_files = glob.glob(config["data_folder"]+"/Val/*.tfrecord")

        trackFeatures = [
                'trk_pt',
                'trk_eta',
                'trk_z0_res',
                'corrected_trk_z0',
                'corrected_int_z0',
                'abs_trk_word_pT',
                'rescaled_trk_word_MVAquality',
                'abs_trk_word_eta',
                'unscaled_trk_word_pT',
                'unscaled_trk_word_eta',
                'unscaled_trk_word_MVAquality',
                'unscaled_trk_z0_res',
                'binned_trk_bendchi2',
                'binned_trk_chi2rphi',
                'binned_trk_chi2rz'
        ]


    print ("Input Train files: ",len(train_files))
    print ("Input Validation files: ",len(val_files))
    print ("Input Test files: ",len(test_files))

    features = {
        "pvz0": tf.io.FixedLenFeature([1], tf.float32),  
        "trk_fromPV":tf.io.FixedLenFeature([max_ntracks], tf.float32) 
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

    
    experiment.log_other("description",kf + config["description"])
    with open(kf+'experimentkey.txt', 'w') as fh:
      fh.write(experiment.get_key())

    startingLR = config['starting_lr']
    if trainable == 'DA':
        loss_function = tf.keras.losses.Huber(config['Huber_delta'])
    else:
        #loss_function = tf.keras.losses.MeanSquaredError()
        loss_function = tf.keras.losses.Huber(0.2)
    
    print(loss_function)
    model = network.createE2EModel()
    optimizer = tf.keras.optimizers.Adam(lr=startingLR)
    model.compile(
        optimizer,
        loss=[
            loss_function,
            #tf.keras.losses.MeanAbsoluteError(),
            #TrainingScripts.Callbacks.ModifiedHuberDelta(config['Huber_delta']),
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

    if trainable == 'DA':
        if not train_cnn:
            model.get_layer('pattern_1').set_weights([np.array([[[1]],[[1]],[[1]]], dtype=np.float32)]) 
            model.get_layer('Bin_weight').set_weights([np.expand_dims(np.arange(nbins),axis=0)]) #Set to bin index 
            model.get_layer('position_final').set_weights([np.array([position_final_weights], dtype=np.float32).T, np.array(position_final_bias, dtype=np.float32)])
        else:
            model.get_layer('Bin_weight').set_weights([np.expand_dims(np.arange(nbins),axis=0)]) #Set to bin index 
            model.get_layer('position_final').set_weights([np.array([position_final_weights], dtype=np.float32), np.array(position_final_bias, dtype=np.float32)])
        experiment.set_name(kf+config['comet_experiment_name'])

        if pretrain_DA:
            loadedmodel = network.createE2EModel()
            loadedmodel.compile(
                optimizer,
                loss=[
                    loss_function,
                    #tf.keras.losses.MeanAbsoluteError(),
                    #TrainingScripts.Callbacks.ModifiedHuberDelta(config['Huber_delta']),
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

        
            model.load_weights(PretrainedModelName+".tf").expect_partial()

            # WeightModel = network.createWeightModel()
            # AssocModel = network.createAssociationModel()
        
            # WeightModel.load_weights(PretrainedModelName+"_weightModel_weights.hdf5")
            # AssocModel.load_weights(PretrainedModelName+"_associationModel_weights.hdf5")

            # model.get_layer('weight_1').set_weights    (WeightModel.get_layer('weight_1').get_weights())
            # model.get_layer('dropout').set_weights     (WeightModel.get_layer('dropout').get_weights())
            # model.get_layer('weight_2').set_weights    (WeightModel.get_layer('weight_2').get_weights())
            # model.get_layer('dropout_1').set_weights   (WeightModel.get_layer('dropout_1').get_weights())
            # model.get_layer('weight_final').set_weights(WeightModel.get_layer('weight_final').get_weights())

            # if not train_cnn:
            #     model.get_layer('pattern_1').set_weights        ([np.array([[[1]],[[1]],[[1]]], dtype=np.float32)]) 
            #     model.get_layer('Bin_weight').set_weights       ([np.expand_dims(np.arange(nbins),axis=0)]) #Set to bin index 
            #     model.get_layer('position_final').set_weights   ([np.array([position_final_weights], dtype=np.float32), np.array(position_final_bias, dtype=np.float32)])
            #     model.get_layer('association_0').set_weights    (AssocModel.get_layer('association_0').get_weights())
            #     model.get_layer('dropout_2').set_weights        (AssocModel.get_layer('dropout_2').get_weights())
            #     model.get_layer('association_1').set_weights    (AssocModel.get_layer('association_1').get_weights())
            #     model.get_layer('dropout_3').set_weights        (AssocModel.get_layer('dropout_3').get_weights())
            #     model.get_layer('association_final').set_weights(AssocModel.get_layer('association_final').get_weights())
            # else:
            #     model.get_layer('Bin_weight').set_weights       ([np.expand_dims(np.arange(nbins),axis=0)]) #Set to bin index 
            #     model.get_layer('position_final').set_weights   ([np.array([position_final_weights], dtype=np.float32), np.array(position_final_bias, dtype=np.float32)])
            #     model.get_layer('association_0').set_weights    (AssocModel.get_layer('association_0').get_weights())
            #     model.get_layer('dropout_2').set_weights        (AssocModel.get_layer('dropout_2').get_weights())
            #     model.get_layer('association_1').set_weights    (AssocModel.get_layer('association_1').get_weights())
            #     model.get_layer('dropout_3').set_weights        (AssocModel.get_layer('dropout_3').get_weights())
            #     model.get_layer('association_final').set_weights(AssocModel.get_layer('association_final').get_weights())



    elif trainable == 'QDA':
        experiment.set_name(kf+config['comet_experiment_name'])
        if config["pretrained"]:
            DAnetwork = vtx.nn.E2EDiffArgMax(
                nbins=nbins,
                ntracks=max_ntracks, 
                nweightfeatures=len(weightfeat), 
                nfeatures=len(trackfeat), 
                nweights=1, 
                nlatent = nlatent,
                activation='relu',
                l2regloss=1e-10,
                nweightnodes = config['nweightnodes'],
                nweightlayers = config['nweightlayers'],
                nassocnodes = config['nassocnodes'],
                nassoclayers = config['nassoclayers'],
            )
            
            DAmodel = DAnetwork.createE2EModel()
            optimizer = tf.keras.optimizers.Adam(lr=0.01)
            DAmodel.compile(
                optimizer,
                loss=[
                    loss_function,
                    #tf.keras.losses.MeanSquaredError(),
                    #TrainingScripts.Callbacks.ModifiedHuberDelta(config['Huber_delta']),
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
            DAmodel.load_weights(config["UnquantisedModelName"]+".tf")
            model.get_layer('weight_1').set_weights    (DAmodel.get_layer('weight_1').get_weights())
            model.get_layer('weight_2').set_weights    (DAmodel.get_layer('weight_2').get_weights()) 
            model.get_layer('weight_final').set_weights(DAmodel.get_layer('weight_final').get_weights()) 

            model.get_layer('pattern_1').set_weights(DAmodel.get_layer('pattern_1').get_weights()) 
            #model.get_layer('pattern_2').set_weights(DAmodel.get_layer('pattern_2').get_weights()) 

            if not train_cnn:
                model.get_layer('Bin_weight').set_weights       (DAmodel.get_layer('Bin_weight').get_weights()) 
                #model.get_layer('position_final').set_weights   (DAmodel.get_layer('position_final').get_weights()) 
                model.get_layer('position_final').set_weights   ([np.array([position_final_weights], dtype=np.float32).T, np.array(position_final_bias, dtype=np.float32)])
                model.get_layer('association_0').set_weights    (DAmodel.get_layer('association_0').get_weights()) 
                model.get_layer('association_1').set_weights    (DAmodel.get_layer('association_1').get_weights()) 
                model.get_layer('association_final').set_weights(DAmodel.get_layer('association_final').get_weights())
            else:
                model.get_layer('Bin_weight').set_weights       (DAmodel.get_layer('Bin_weight').get_weights()) 
                #model.get_layer('position_final').set_weights   (DAmodel.get_layer('position_final').get_weights()) 
                model.get_layer('position_final').set_weights   ([np.array([position_final_weights], dtype=np.float32).T, np.array(position_final_bias, dtype=np.float32)])
                model.get_layer('association_0').set_weights    (DAmodel.get_layer('association_0').get_weights()) 
                model.get_layer('association_1').set_weights    (DAmodel.get_layer('association_1').get_weights()) 
                model.get_layer('association_final').set_weights(DAmodel.get_layer('association_final').get_weights())


        else:
            model.get_layer('Bin_weight').set_weights([np.expand_dims(np.arange(nbins),axis=0)]) #Set to bin index 
            model.get_layer('position_final').set_weights([np.array([position_final_weights], dtype=np.float32).T, np.array(position_final_bias, dtype=np.float32)])


    elif trainable == 'QDA_prune':
        experiment.set_name(kf+config['comet_experiment_name']+str(sys.argv[4]))
        model.load_weights(config["QuantisedModelName"]+"_prune_iteration_"+sys.argv[4]+".tf").expect_partial()


    reduceLR = TrainingScripts.Callbacks.OwnReduceLROnPlateau(
        monitor='loss', factor=0.1, patience=6, verbose=1,
        mode='auto', min_delta=0.05, cooldown=0, min_lr=0
    )
    with experiment.train():
        train_model(model,experiment,train_files,val_files,trackfeat,weightfeat,epochs=epochs,callbacks=reduceLR,nlatent=nlatent,bit=bit,model_name=model_name),
    with experiment.test():
        test_model(model,experiment,test_files,trackfeat,weightfeat,model_name=model_name)
