import uproot3 as uproot
import tensorflow as tf
import numpy as np
import math
from math import isnan
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vtx
import yaml
from tensorflow.keras.models import Model
import glob

nMaxTracks = 250
nbins = 256
max_z0 = 15

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
    #ds = ds.shuffle(5000,reshuffle_each_iteration=True)
    ds = ds.batch(2000)
    ds = ds.prefetch(5)
    
    return ds

def load_model(config):
    QuantisedModelName = config["QuantisedModelName"] 


    with open(config['QuantisedModelName']+'_prune_iteration_9_WeightQConfig.yaml', 'r') as f:
        weightqconfig = yaml.load(f,Loader=yaml.FullLoader)
    with open(config['QuantisedModelName']+'_prune_iteration_9_PatternQConfig.yaml', 'r') as f:
        patternqconfig = yaml.load(f,Loader=yaml.FullLoader)
    with open(config['QuantisedModelName']+'_prune_iteration_9_AssociationQConfig.yaml', 'r') as f:
        associationqconfig = yaml.load(f,Loader=yaml.FullLoader)

    network = vtx.nn.E2EQKerasDiffArgMaxConstraint(
                nbins=nbins,
                ntracks=nMaxTracks, 
                nweightfeatures=len(config["weight_features"]), 
                nfeatures=len(config["track_features"]), 
                nlatent = config['Nlatent'],
                l1regloss = (float)(config['l1regloss']),
                l2regloss = (float)(config['l2regloss']),
                nweightnodes = config['nweightnodes'],
                nweightlayers = config['nweightlayers'],
                nassocnodes = config['nassocnodes'],
                nassoclayers = config['nassoclayers'],
                weightqconfig = weightqconfig,
                patternqconfig = patternqconfig,
                associationqconfig = associationqconfig,
                h5fName = config['QuantisedModelName']+'_drop_weights_iteration_8.h5'
            )

    model = network.createE2EModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(
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


    model.load_weights(QuantisedModelName+"_prune_iteration_9.tf").expect_partial()
    network.load_weights(model)
    network.write_model_graph(QuantisedModelName+"_prune_iteration_9")

    # QuantisedModelName = config['QuantisedModelName']+"_prune_iteration_0"

    # with open(config["UnquantisedModelName"]+'_WeightQConfig.yaml', 'r') as f:
    #     weightqconfig = yaml.load(f,Loader=yaml.FullLoader)
    # with open(config["UnquantisedModelName"]+'_PatternQConfig.yaml', 'r') as f:
    #     patternqconfig = yaml.load(f,Loader=yaml.FullLoader)
    # with open(config["UnquantisedModelName"]+'_AssociationQConfig.yaml', 'r') as f:
    #     associationqconfig = yaml.load(f,Loader=yaml.FullLoader)

    # network = vtx.nn.E2EQKerasDiffArgMax(
    #     nbins = 256,
    #     start = 0,
    #     end = 255,
    #     max_z0 = max_z0,
    #     ntracks = nMaxTracks, 
    #     nweightfeatures = len(config["weight_features"]),
    #     nfeatures = len(config["track_features"]),  
    #     nlatent = config['Nlatent'],
    #     l1regloss = (float)(config['l1regloss']),
    #     l2regloss = (float)(config['l2regloss']),
    #     nweightnodes = config['nweightnodes'],
    #     nweightlayers = config['nweightlayers'],
    #     nassocnodes = config['nassocnodes'],
    #     nassoclayers = config['nassoclayers'],
    #     weightqconfig = weightqconfig,
    #     patternqconfig = patternqconfig,
    #     associationqconfig = associationqconfig
    # )

        
    # model = network.createE2EModel()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    # model.compile(
    #         optimizer,
    #         loss=[
    #             tf.keras.losses.MeanAbsoluteError(),
    #             tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #             lambda y,x: 0.
    #         ],
    #         metrics=[
    #             tf.keras.metrics.BinaryAccuracy(threshold=0.,name='assoc_acc') #use thres=0 here since logits are used
    #         ],
    #         loss_weights=[config['z0_loss_weight'],
    #                     config['crossentropy_loss_weight'],
    #                     0]
    # )

    # model.load_weights(QuantisedModelName+".tf").expect_partial()

    # UnQuantisedModelName = config["UnquantisedModelName"] 


    # network = vtx.nn.E2EDiffArgMax(
    #         nbins=config['nbins'],
    #         start=0,
    #         end=255,
    #         max_z0 = max_z0,
    #         ntracks=nMaxTracks, 
    #         nweightfeatures=len(config["weight_features"]),
    #         nfeatures=len(config["track_features"]), 
    #         nweights=1, 
    #         nlatent = config["Nlatent"],
    #         l2regloss=1e-10
    #     )
        
    # model = network.createE2EModel()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    # model.compile(
    #         optimizer,
    #         loss=[
    #             tf.keras.losses.MeanAbsoluteError(),
    #             tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #             lambda y,x: 0.
    #         ],
    #         metrics=[
    #             tf.keras.metrics.BinaryAccuracy(threshold=0.,name='assoc_acc') #use thres=0 here since logits are used
    #         ],
    #         loss_weights=[config['z0_loss_weight'],
    #                     config['crossentropy_loss_weight'],
    #                     0]
    # )

    # model.load_weights(UnQuantisedModelName+".tf").expect_partial()
    return model

with open('setup.yaml', 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

test_file = glob.glob("test/data0.tfrecord")

model = load_model(config)
trackfeat = config["track_features"] 
weightfeat = config["weight_features"] 

print(model.summary())

features = {
            "pvz0": tf.io.FixedLenFeature([1], tf.float32),
            "trk_fromPV":tf.io.FixedLenFeature([nMaxTracks], tf.float32)
    }

trackFeatures = [
            'trk_word_chi2rphi', 
            'trk_word_chi2rz', 
            'trk_word_bendchi2',
            'trk_z0_res',
            'trk_gtt_pt',
            'trk_gtt_eta',
            'trk_gtt_phi',
            'trk_fake',
            'trk_z0',
            'int_z0',
            'trk_class_weight',
            'trk_word_pT',
            'trk_word_eta',
            'trk_word_MVAquality',
        ]

for trackFeature in trackFeatures:
        features[trackFeature] = tf.io.FixedLenFeature([nMaxTracks], tf.float32)

for step,batch in enumerate(setup_pipeline(test_file)):    
        trackFeatures = np.stack([batch[feature] for feature in trackfeat],axis=2)
        WeightFeatures = np.stack([batch[feature] for feature in weightfeat],axis=2)
        XX = model.input 
        YY = model.layers[11].output

        ZZ = model.layers[8].output

        weight_model = Model(XX,ZZ)
        hist_model = Model(XX, YY)

        weight  = (weight_model.predict_on_batch(
                                [batch['int_z0'],WeightFeatures,trackFeatures]))

        hist = (hist_model.predict_on_batch(
                                [batch['int_z0'],WeightFeatures,trackFeatures]))

        for iev,event in enumerate(batch['int_z0']):
            print("Event: "+str(iev))
            #if iev > 0:
            #    break
            for i,track in enumerate(batch['int_z0'][iev]):
                print('Track['+str(i)+'] z0: ',batch['int_z0'][iev].numpy()[i] ,' pT: ',batch['trk_word_pT'][iev].numpy()[i], ' MVA: ',batch['trk_word_MVAquality'][iev].numpy()[i],  ' Eta: ', batch['trk_word_eta'][iev].numpy()[i])
                print('Track['+str(i)+'] weight: ',weight[iev][i][0])
            for i,entry in enumerate(hist[iev]):
                print('Histogram['+str(i)+']:',entry)



    
