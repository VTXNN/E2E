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
#from TrainingScripts.train import *
from EvalScripts.eval_funcs import *

from time import time
import pickle


if __name__=="__main__":
    with open(sys.argv[2]+'.yaml', 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    kf = sys.argv[1]

    if kf == "NewKF":
        test_files = glob.glob(config["data_folder"]+"/MET/*.tfrecord")
        z0 = 'trk_z0'
        FH_z0 = 'trk_z0'
        start = -15
        end = 15
        bit = False

    elif kf == "OldKF":
        test_files = glob.glob(config["data_folder"]+"/MET/*.tfrecord")
        z0 = 'corrected_trk_z0'
        start = -15
        end = 15
        bit = False

    elif kf == "OldKF_intZ":
        test_files = glob.glob(config["data_folder"]+"/MET/*.tfrecord")
        z0 = 'corrected_int_z0'
        FH_z0 = 'corrected_trk_z0'
        start = 0
        end = 255
        bit = True


    nMaxTracks = 250

    nlatent = config["Nlatent"]
    nbins = config['nbins']

    trainable = config["trainable"]
    trackfeat = config["track_features"] 
    weightfeat = config["weight_features"] 

    QuantisedPrunedModelName = config["QuantisedPrunedModelName"] 

    features = {
            "pvz0": tf.io.FixedLenFeature([1], tf.float32),
            "trk_fromPV":tf.io.FixedLenFeature([nMaxTracks], tf.float32)
    }

    def decode_data(raw_data):
        decoded_data = tf.io.parse_example(raw_data,features)
        #decoded_data['trk_hitpattern'] = tf.reshape(decoded_data['trk_hitpattern'],[-1,nMaxTracks,11])
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
            'corrected_trk_z0',
            'abs_trk_word_pT',
            'rescaled_trk_word_MVAquality',
            'abs_trk_word_eta',
            'corrected_int_z0'
        ]

    for trackFeature in trackFeatures:
        features[trackFeature] = tf.io.FixedLenFeature([nMaxTracks], tf.float32)

    QPnetwork = vtx.nn.E2EQKerasDiffArgMaxConstraint(
            nbins=nbins,
            ntracks=nMaxTracks, 
            start=start,
            end=end,
            return_index = bit,
            nweightfeatures=len(config["weight_features"]),  
            nfeatures=len(config["track_features"]), 
            nweights=1, 
            nlatent = nlatent,
            activation='relu',
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


    QPmodel.load_weights(QuantisedPrunedModelName+".tf").expect_partial()

    QPmodel.summary()

    for step,batch in enumerate(setup_pipeline(test_files)):
            print(step)
            
            trackFeatures = np.stack([batch[feature] for feature in trackfeat],axis=2)
            WeightFeatures = np.stack([batch[feature] for feature in weightfeat],axis=2)
            nBatch = batch['pvz0'].shape[0]

            #### QP NETWORK #########################################################################################################################
            model_layer_weight_activation_1 = Model(QPmodel.input, QPmodel.layers[2].output)
            model_layer_weight_activation_2 = Model(QPmodel.input, QPmodel.layers[4].output)
            model_layer_weight_activation_3 = Model(QPmodel.input, QPmodel.layers[7].output)
            model_layer_pattern_activation_1 = Model(QPmodel.input, QPmodel.layers[10].output)
            model_layer_association_activation_1 = Model(QPmodel.input, QPmodel.layers[20].output)
            model_layer_association_activation_2 = Model(QPmodel.input, QPmodel.layers[22].output)

            output_weight_activation_1 = model_layer_weight_activation_1.predict_on_batch([batch[z0],WeightFeatures,trackFeatures])
            output_weight_activation_2 = model_layer_weight_activation_2.predict_on_batch([batch[z0],WeightFeatures,trackFeatures])
            output_weight_activation_3 = model_layer_weight_activation_3.predict_on_batch([batch[z0],WeightFeatures,trackFeatures])
            output_pattern_activation_1 = model_layer_pattern_activation_1.predict_on_batch([batch[z0],WeightFeatures,trackFeatures])
            output_association_activation_1 = model_layer_association_activation_1.predict_on_batch([batch[z0],WeightFeatures,trackFeatures])
            output_association_activation_2 = model_layer_association_activation_2.predict_on_batch([batch[z0],WeightFeatures,trackFeatures])

            predictedZ0_QPNN_temp, predictedAssoc_QPNN_temp, QWeights_QPNN = QPmodel.predict_on_batch(
                            [batch[z0],WeightFeatures,trackFeatures]
                        )

            predictedAssoc_QPNN_temp = tf.math.divide( tf.math.subtract( predictedAssoc_QPNN_temp,-5),15)

            Events = []

            for i in range(len(batch['pvz0'])):
                df = pd.DataFrame(columns=['weight_1_0','weight_1_1','weight_1_2','weight_1_3','weight_1_4','weight_1_5','weight_1_6','weight_1_7','weight_1_8','weight_1_9',
                                           'weight_2_0','weight_2_1','weight_2_2','weight_2_3','weight_2_4','weight_2_5','weight_2_6','weight_2_7','weight_2_8','weight_2_9',
                                           'weight_3_0',
                                           'association_1_0','association_1_1','association_1_2','association_1_3','association_1_4','association_1_5','association_1_6','association_1_7','association_1_8','association_1_9',
                                           'association_1_10','association_1_11','association_1_12','association_1_13','association_1_14','association_1_15','association_1_16','association_1_17','association_1_18','association_1_19',
                                           'association_2_0','association_2_1','association_2_2','association_2_3','association_2_4','association_2_5','association_2_6','association_2_7','association_2_8','association_2_9',
                                           'association_2_10','association_2_11','association_2_12','association_2_13','association_2_14','association_2_15','association_2_16','association_2_17','association_2_18','association_2_19',
                                           'predicted_weight',
                                           'predicted_association',
                                           'trk_fromPV',
                                           'trk_z0',
                                           'predicted_z0',
                                           'true_z0'],
                                            index=[i for i in range(0,250)])
                
                for j in range(0,250):
                    dict_weight_1 = {'weight_1_'+str(k) : output_weight_activation_1[i][j][k] for k in range(0,10)}
                    dict_weight_2 = {'weight_2_'+str(k) : output_weight_activation_2[i][j][k] for k in range(0,10)}
                    dict_weight_3 = {'weight_3_0' : output_weight_activation_3[i][j][0]}
                    dict_association_1 = {'association_1_'+str(k) : output_association_activation_1[i][j][k] for k in range(0,20)}
                    dict_association_2 = {'association_2_'+str(k) : output_association_activation_2[i][j][k] for k in range(0,20)}
                    dict_predicted_weight = {'predicted_weight' : QWeights_QPNN[i][j][0]}

                    dict_predicted_assoc = {'predicted_association' : predictedAssoc_QPNN_temp[i][j][0]}
                    dict_trk_fake = {'trk_fromPV' : batch['trk_fromPV'][i][j].numpy()}
                    dict_trk_z0 = {'trk_z0' : batch['trk_z0'][i][j].numpy()}
                    dict_predicted_z0 = {'predicted_z0' : predictedZ0_QPNN_temp[i][0]}
                    dict_true_z0 = {'true_z0' : batch['pvz0'][i].numpy()[0]}

                    combined = {**dict_weight_1, **dict_weight_2, **dict_weight_3, **dict_association_1, **dict_association_2, **dict_predicted_weight,
                    **dict_predicted_assoc, **dict_trk_fake, **dict_trk_z0,**dict_predicted_z0,**dict_true_z0}

                    df.loc[j] = pd.Series(combined)

                Events.append(df)

            with open('SavedDFs/events_batch'+str(step)+'.pkl', 'wb') as outp:
                pickle.dump(Events, outp, pickle.HIGHEST_PROTOCOL)

            





