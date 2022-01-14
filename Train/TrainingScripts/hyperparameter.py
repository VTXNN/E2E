from comet_ml import Optimizer
from comet_ml import Experiment
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

import tensorflow_model_optimization as tfmot

import qkeras

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
elif kf == "OldKF":
    z0 = 'corrected_trk_z0'


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

def train_model(model,experiment,train_files,val_files,trackfeat,weightfeat,epochs=50,callbacks=[None,None],nlatent=0,trainable=False,early_stop_patience=10):

    total_steps = 0
    wait = 0
    best_score = 100000
    callbacks[0].on_train_begin()
    callbacks[1].on_train_begin()
    
    old_lr = model.optimizer.learning_rate.numpy()
    
    for epoch in range(epochs):

        new_lr = old_lr
        tf.keras.backend.set_value(model.optimizer.learning_rate, new_lr)
        experiment.log_metric("learning_rate",model.optimizer.learning_rate,step=total_steps,epoch=epoch)
        callbacks[1].on_train_begin()
        print ("Epoch %i"%epoch)
        
        if epoch>0:
            model.load_weights(kf+"best_weights.tf")
        
        for step,batch in enumerate(setup_pipeline(train_files)):
            callbacks[1].on_train_batch_begin(epoch)
            z0Shift = np.random.normal(0.0,1.0,size=batch['pvz0'].shape)
            z0Flip = 2.*np.random.randint(2,size=batch['pvz0'].shape)-1.
            batch[z0]=batch[z0]*z0Flip + z0Shift
            batch['pvz0']=batch['pvz0']*z0Flip + z0Shift
            
            trackFeatures = np.stack([batch[feature] for feature in trackfeat],axis=2)

            WeightFeatures = np.stack([batch[feature] for feature in weightfeat ],axis=2)

            nBatch = batch['pvz0'].shape[0]

            result = model.train_on_batch([batch[z0],WeightFeatures,trackFeatures], [batch['pvz0'],batch['trk_fromPV'],np.zeros([nBatch,max_ntracks,1])])   
                                              # convs                          #softmax     #binweight                                           

            result = dict(zip(model.metrics_names,result))

            if nlatent > 0:
                experiment.log_metric("z0_loss",result['split_latent_loss'],step=total_steps,epoch=epoch)
            else:
                experiment.log_metric("z0_loss",result['position_final_loss'],step=total_steps,epoch=epoch)
            experiment.log_metric("loss",result['loss'],step=total_steps,epoch=epoch)
            experiment.log_metric("assoc_loss",result['association_final_loss'],step=total_steps,epoch=epoch)
 
            total_steps += 1
        prune_level = []
        for i,layer in enumerate(model.layers):
            get_weights = layer.get_weights()
            if len(get_weights) > 0:
                if "Bin_weight" not in layer.name:
                    weights = get_weights[0].flatten()[get_weights[0].flatten() != 0]
                    prune_level.append(weights.shape[0]/get_weights[0].flatten().shape[0])

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

            temp_predictedZ0_NN, temp_predictedAssoc_NN,predicted_weights  = model.predict_on_batch(
                    [val_batch[z0],val_WeightFeatures,val_trackFeatures]
            )
  
            val_predictedZ0_NN.append(temp_predictedZ0_NN.flatten())
            val_predictedAssoc_NN.append(temp_predictedAssoc_NN.flatten())
            val_actual_PV.append(val_batch['pvz0'].numpy().flatten()) 
            val_actual_assoc.append(val_batch["trk_fromPV"].numpy().flatten())

            val_predictedAssoc_FH.append(eval.FastHistoAssoc(val_batch['pvz0'],val_batch[z0],val_batch['trk_eta']).flatten())
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

        experiment.log_metric("Validation_Prune_Level",np.mean(prune_level))

        old_lr = callbacks[0].on_epoch_end(epoch=epoch,logs=result,lr=new_lr)
        callbacks[1].on_epoch_end(batch=epoch)

        val_loss = metrics.mean_squared_error(val_z0_PV_array,val_z0_NN_array)

        print ("Val_NN_z0_MSE: "+str(val_loss)+" Best_NN_z0_MSE: "+str(best_score)) 

        wait += 1
        if val_loss < best_score:
            best_score = val_loss
            model_for_export = tfmot.sparsity.keras.strip_pruning(model)
            model_for_export.save_weights(kf+"best_weights.tf")
            wait = 0
        if wait >= early_stop_patience:
            break
        
def test_model(model,experiment,test_files,trackfeat,weightfeat,network):

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

        predictedAssoc_FH.append(eval.FastHistoAssoc(batch['pvz0'],batch[z0],batch['trk_eta']))
            
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

    experiment.log_asset(kf+"best_weights.tf.index")
    experiment.log_asset(kf+"best_weights.tf.data-00000-of-00001")

    weightModel = network.createWeightModel()
    with open(kf + 'weightModel.json', 'w') as f:
        f.write(weightModel.to_json())
    weightModel.save_weights(kf + "weightModel_weights.hdf5")
    weightModel.save('weightModel')

    experiment.log_asset(kf+"weightModel.json")
    experiment.log_asset(kf+"weightModel_weights.hdf5")
    experiment.log_asset_folder("weightModel", log_file_name=True)

    patternModel = network.createPatternModel()
    with open(kf + 'patternModel.json', 'w') as f:
        f.write(patternModel.to_json())
    patternModel.save_weights(kf + "patternModel_weights.hdf5")
    patternModel.save('patternModel')

    experiment.log_asset(kf+"patternModel.json")
    experiment.log_asset(kf+"patternModel_weights.hdf5")
    experiment.log_asset_folder("patternModel", log_file_name=True)

    associationModel = network.createAssociationModel()
    with open(kf + 'asociationModel.json', 'w') as f:
        f.write(associationModel.to_json())
    associationModel.save_weights(kf + "asociationModel_weights.hdf5")
    associationModel.save("asociationModel5")

    experiment.log_asset(kf+"associationModel.json")
    experiment.log_asset(kf+"associationModel_weights.hdf5")
    experiment.log_asset_folder("associationModel", log_file_name=True)
   

if __name__=="__main__":
    with open(sys.argv[2]+'.yaml', 'r') as f:
        config = yaml.load(f)

    trackfeat = config["track_features"] 
    weightfeat = config["weight_features"]

    comet_config = {"algorithm": "bayes",
                "spec": {
                    "maxCombo": 0,
                    "objective": "maximize",
                    "metric": "Test_NN_PV_ROC",
                    "minSampleSize": 100,
                    "retryLimit": 20,
                    "retryAssignLimit": 0,
                },
                "parameters": {
                    "starting_lr": {"type": "float", "min": 0.00001, "max": 1.0, "scalingType": "loguniform"},
                    "z0_loss_weight": {"type": "discrete", "values": [0.1, 0.2, 0.5, 0.9]},
                    "Huber_delta": {"type": "integer", "min":0, "max":15, "scalingType": "uniform"},
                    #"l1regloss":{"type": "float", "min":1e-10, "max":1.0, "scalingType": "loguniform"},
                    "l2regloss":{"type": "float", "min":1e-10, "max":1.0, "scalingType": "loguniform"},
                    "nweightnodes":{"type": "discrete", "values": [5, 10, 15, 20, 25, 30]},
                    "nassocnodes": {"type": "discrete", "values": [5, 10, 15, 20, 25, 30]},
                    "nlatent": {"type": "discrete", "values": [1, 2, 3, 4]},
                    "temperature" : {"type": "float", "min":1e-10, "max":1e-3, "scalingType": "loguniform"},
                    "nbins" : {"type": "integer", "min":100, "max":500, "scalingType": "uniform"}

                },
                "name": "Vtx_HPS",
                "trials": 1,
            }

    opt = Optimizer(comet_config)

    for experiment in opt.get_experiments(project_name="Vtx_HPS"):

        nlatent = experiment.get_parameter("nlatent")

        network = vtx.nn.E2EDiffArgMax(
                nbins=experiment.get_parameter("nbins"),
                ntracks=max_ntracks, 
                nweightfeatures=len(weightfeat), 
                nfeatures=len(trackfeat), 
                nweights=1, 
                nlatent = experiment.get_parameter("nlatent"),
                activation='relu',
                #l1regloss = experiment.get_parameter("l1regloss"),
                l2regloss = experiment.get_parameter("l2regloss"),
                nweightnodes = experiment.get_parameter("nweightnodes"),
                nweightlayers = config['nweightlayers'],
                nassocnodes = experiment.get_parameter("nassocnodes"),
                nassoclayers = config['nassoclayers'],
                #qconfig = config['QConfig'],
                temperature = experiment.get_parameter("temperature"),
            )

        if kf == "NewKF":
            train_files = glob.glob(config["data_folder"]+"/Train/*.tfrecord")
            test_files = glob.glob(config["data_folder"]+"/Test/*.tfrecord")
            val_files = glob.glob(config["data_folder"]+"/Val/*.tfrecord")
            
        elif kf == "OldKF":
            train_files = glob.glob(config["data_folder"]+"/Train/*.tfrecord")
            test_files = glob.glob(config["data_folder"]+"/Test/*.tfrecord")
            val_files = glob.glob(config["data_folder"]+"/Val/*.tfrecord")

        features = {
            "pvz0": tf.io.FixedLenFeature([1], tf.float32),    
            "trk_fromPV":tf.io.FixedLenFeature([max_ntracks], tf.float32) ,
        }

        trackFeatures = [
            'trk_z0',
            'trk_pt',
            'trk_eta',
            'trk_MVA1',
            'trk_z0_res',
            'corrected_trk_z0',
            'normed_trk_pt',
            'normed_trk_eta',
            #'trk_word_pT',
            #'trk_word_MVAquality',
            #'trk_word_TanL',
            #'trk_word_Phi',
            #'trk_word_eta',
            #'trk_word_chi2rphi',
            #'trk_word_chi2rz',
            #'trk_word_bendchi2',
            #'abs_trk_word_pT',
            #'rescaled_trk_word_MVAquality',
            #'abs_trk_word_eta'
        ]

        for trackFeature in trackFeatures:
            features[trackFeature] = tf.io.FixedLenFeature([max_ntracks], tf.float32)

        experiment.set_name(kf+config['comet_experiment_name'])
        experiment.log_other("description",kf + config["description"])
        experiment.auto_histogram_weight_logging = True
        experiment.auto_histogram_activation_logging= True

        epochs = config['epochs']

        model = network.createE2EModel()
        optimizer = tf.keras.optimizers.Adam(lr=experiment.get_parameter("starting_lr"))
        model.compile(
            optimizer,
            loss=[
                tf.keras.losses.Huber(delta=experiment.get_parameter("Huber_delta")),
                tf.keras.losses.BinaryCrossentropy(from_logits=True),
                lambda y,x: 0.,
            ],
            metrics=[
                tf.keras.metrics.BinaryAccuracy(threshold=0.,name='assoc_acc') #use thres=0 here since logits are used
            ],
            loss_weights=[experiment.get_parameter('z0_loss_weight'),
                          1 - experiment.get_parameter('z0_loss_weight'),
                        0
                        ]
        )

        model.summary()

        model.layers[11].set_weights([np.array([[[1]],[[1]],[[1]]], dtype=np.float32)])
        model.layers[15].set_weights([np.expand_dims(np.arange(experiment.get_parameter("nbins")),axis=0)]) #Set to bin index 
        #model.layers[15].set_weights([np.array([[1]], dtype=np.float32), np.array([0], dtype=np.float32)])
        
        pruning_params = {"pruning_schedule" : tfmot.sparsity.keras.PolynomialDecay(0, config["Final_Sparsity"], config["Begin_step"], config["End_step"], power=3, frequency=1)}

            # Helper function uses `prune_low_magnitude` to make only the 
            # Dense layers train with pruning.

        prunable_layer_list = ["weight_1","weight_2","association_0","association_1"]

        def apply_pruning_to_dense(layer):
            if layer.name in prunable_layer_list:
                #if isinstance(layer, tf.keras.layers.Dense):
                return tfmot.sparsity.keras.prune_low_magnitude(layer)
            return layer

            # Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense` 
            # to the layers of the model.
        model_for_pruning = tf.keras.models.clone_model(
            model,
            clone_function=apply_pruning_to_dense,
        )

        reduceLR = TrainingScripts.Callbacks.OwnReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=6, verbose=1,
            mode='auto', min_delta=0.005, cooldown=0, min_lr=0
        )

        pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()
        pruning_callback.set_params(pruning_params)
        pruning_callback.set_model(model_for_pruning)

        optimizer = tf.keras.optimizers.Adam(lr=experiment.get_parameter("starting_lr"))
        model_for_pruning.compile(
            optimizer,
            loss=[
                tf.keras.losses.Huber(delta=experiment.get_parameter("Huber_delta")),
                #tf.keras.losses.MeanAbsoluteError(),
                tf.keras.losses.BinaryCrossentropy(from_logits=True),
                lambda y,x: 0.,
            ],
            metrics=[
                tf.keras.metrics.BinaryAccuracy(threshold=0.,name='assoc_acc') #use thres=0 here since logits are used
            ],
            loss_weights=[experiment.get_parameter('z0_loss_weight'),
                          1 - experiment.get_parameter('z0_loss_weight'),
                        0
                        ]
        )

        model_for_pruning.summary()
        
        with experiment.train():
            train_model(model,experiment,train_files,val_files,trackfeat,weightfeat,epochs=epochs,callbacks=[reduceLR,pruning_callback],nlatent=nlatent),
        with experiment.test():
            test_model(model,experiment,test_files,trackfeat,weightfeat,network)
        experiment.end()