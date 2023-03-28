import glob
import sys
import os
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
import hls4ml


from tensorflow.keras.models import Model

import vtx
from TrainingScripts.train import *
from EvalScripts.eval_funcs import *
from sklearn.metrics import mean_squared_error


nMaxTracks = 500
max_z0 = 20.46912512

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


if __name__=="__main__":
    from qkeras.qlayers import QDense, QActivation
    from qkeras.quantizers import quantized_bits, quantized_relu
    from qkeras.utils import _add_supported_quantized_objects
    co = {}
    _add_supported_quantized_objects(co)
    
    with open(sys.argv[1]+'.yaml', 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    trainable = sys.argv[2]

    test_files = glob.glob(config["data_folder"]+"/MET/*.tfrecord")
    z0 = 'int_z0' 
    trackfeat = config["track_features"] 
    weightfeat = config["weight_features"] 

    features = {
            "pvz0": tf.io.FixedLenFeature([1], tf.float32),
            "trk_fromPV":tf.io.FixedLenFeature([nMaxTracks], tf.float32),
    }

    trackFeatures = [
            'int_z0',
            'trk_word_pT',
            'trk_word_eta',
            'trk_z0_res',
            'trk_word_MVAquality'
        ]

    for trackFeature in trackFeatures:
        features[trackFeature] = tf.io.FixedLenFeature([nMaxTracks], tf.float32)

    if trainable == "DA":
        UnQuantisedModelName = config["UnquantisedModelName"] 


        network = vtx.nn.E2EDiffArgMax(
                nbins=config['nbins'],
                start=0,
                end=255,
                max_z0 = max_z0,
                ntracks=nMaxTracks, 
                nweightfeatures=len(weightfeat), 
                nfeatures=len(trackfeat), 
                nweights=1, 
                nlatent = config["Nlatent"],
                l2regloss=1e-10
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

        model.load_weights(UnQuantisedModelName+".tf").expect_partial()
        filename = UnQuantisedModelName+"_"

        lowerlim_cut = 16
        upperlim_cut = 6

        print("#=========================================#")
        print("|                                         |")
        print("|    Start of Association Quantisation    |")
        print("|                                         |")
        print("#=========================================#")

    elif trainable == "QDA":
        QuantisedModelName = config['QuantisedModelName']+"_prune_iteration_0"

        with open(config["UnquantisedModelName"]+'_WeightQConfig.yaml', 'r') as f:
            weightqconfig = yaml.load(f,Loader=yaml.FullLoader)
        with open(config["UnquantisedModelName"]+'_PatternQConfig.yaml', 'r') as f:
            patternqconfig = yaml.load(f,Loader=yaml.FullLoader)
        with open(config["UnquantisedModelName"]+'_AssociationQConfig.yaml', 'r') as f:
            associationqconfig = yaml.load(f,Loader=yaml.FullLoader)

        network = vtx.nn.E2EQKerasDiffArgMax(
            nbins = 256,
            start = 0,
            end = 255,
            max_z0 = max_z0,
            ntracks = max_ntracks, 
            nweightfeatures = len(weightfeat), 
            nfeatures = len(trackfeat), 
            nlatent = config['Nlatent'],
            l1regloss = (float)(config['l1regloss']),
            l2regloss = (float)(config['l2regloss']),
            nweightnodes = config['nweightnodes'],
            nweightlayers = config['nweightlayers'],
            nassocnodes = config['nassocnodes'],
            nassoclayers = config['nassoclayers'],
            weightqconfig = weightqconfig,
            patternqconfig = patternqconfig,
            associationqconfig = associationqconfig
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

        model.load_weights(QuantisedModelName+".tf").expect_partial()
        filename = QuantisedModelName+"_"

        lowerlim_cut = 12
        upperlim_cut = 6

        print("#=========================================#")
        print("|                                         |")
        print("|    Start of Association Quantisation 0  |")
        print("|                                         |")
        print("#=========================================#")

    elif trainable == "QDAPrune":
        QuantisedModelName = config["QuantisedModelName"] 


        with open(config['QuantisedModelName']+'_prune_iteration_'+str(int(sys.argv[3])-1)+'_WeightQConfig.yaml', 'r') as f:
            weightqconfig = yaml.load(f,Loader=yaml.FullLoader)
        with open(config['QuantisedModelName']+'_prune_iteration_'+str(int(sys.argv[3])-1)+'_PatternQConfig.yaml', 'r') as f:
            patternqconfig = yaml.load(f,Loader=yaml.FullLoader)
        with open(config['QuantisedModelName']+'_prune_iteration_'+str(int(sys.argv[3])-1)+'_AssociationQConfig.yaml', 'r') as f:
            associationqconfig = yaml.load(f,Loader=yaml.FullLoader)

        network = vtx.nn.E2EQKerasDiffArgMaxConstraint(
                    nbins=nbins,
                    ntracks=max_ntracks, 
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
                    h5fName = config['QuantisedModelName']+'_drop_weights_iteration_'+str(int(sys.argv[3]))+'.h5'
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


        model.load_weights(QuantisedModelName+"_prune_iteration_"+str(int(sys.argv[3]))+".tf").expect_partial()
        filename = QuantisedModelName+"_prune_iteration_"+str(int(sys.argv[3]))+"_"

        lowerlim_cut = 12
        upperlim_cut = 6

        print("#=========================================#")
        print("|                                         |")
        print("|    Start of Association Quantisation "+sys.argv[3]+"  |")
        print("|                                         |")
        print("#=========================================#")

    assocmodel = network.createAssociationModel()
    assocmodel.get_layer('association_1').set_weights    (model.get_layer('association_1').get_weights())
    assocmodel.get_layer('association_1_relu').set_weights    (model.get_layer('association_1_relu').get_weights())
    assocmodel.get_layer('association_2').set_weights     (model.get_layer('association_2').get_weights())
    assocmodel.get_layer('association_2_relu').set_weights     (model.get_layer('association_2_relu').get_weights())
    assocmodel.get_layer('association_final').set_weights(model.get_layer('association_final').get_weights())


    assocmodel.compile(
            optimizer,
            loss=[
                lambda y,x: 0.
            ]
    )

    assoc_input = [[],[],[],[]]

    for step,batch in enumerate(setup_pipeline(test_files)):
        if step > 0:
            break 
        trackFeatures = np.stack([batch[feature] for feature in trackfeat],axis=2)
        WeightFeatures = np.stack([batch[feature] for feature in weightfeat],axis=2)
        XX = model.input 
        YY = model.layers[19].output
        new_model = Model(XX, YY)

        assoc_in = new_model.predict_on_batch(
                                [batch[z0],WeightFeatures,trackFeatures])

        assoc_input[0].append(assoc_in[:,:,0])
        assoc_input[1].append(assoc_in[:,:,1])
        assoc_input[2].append(assoc_in[:,:,2])
        assoc_input[3].append(assoc_in[:,:,3])

    assoc_input_0 = np.concatenate(assoc_input[0]).ravel()
    assoc_input_1 = np.concatenate(assoc_input[1]).ravel()
    assoc_input_2 = np.concatenate(assoc_input[2]).ravel()
    assoc_input_3 = np.concatenate(assoc_input[3]).ravel()

    assoc_array = np.vstack([assoc_input_0,assoc_input_1,assoc_input_2,assoc_input_3]).T
    zeros = np.vstack([assoc_input_0,assoc_input_1,assoc_input_2]).T
    assoc_array = assoc_array[~np.all(zeros == 0, axis=1)]
        
    weight_summary = hls4ml.model.profiling.weights_keras(model=assocmodel,fmt="summary")

    weight_fixed = []

    for i in range(len(weight_summary)):
        print(weight_summary[i]['weight']," : ", weight_summary[i]['whislo'],weight_summary[i]['whishi']," log2: ", np.log2(weight_summary[i]['whislo']),np.log2(weight_summary[i]['whishi']))
        int_bits  = np.ceil(np.log2(weight_summary[i]['whishi'])) + 2 if np.rint(np.log2(weight_summary[i]['whishi'])) > 0 else 1
        total_bits = abs(np.rint(np.log2(weight_summary[i]['whislo']))) + int_bits + 1 if np.rint(np.log2(weight_summary[i]['whishi'])) > 0 else abs(np.rint(np.log2(weight_summary[i]['whislo']))) + int_bits + 2
        if total_bits > lowerlim_cut : total_bits = lowerlim_cut
        if int_bits > upperlim_cut : int_bits = upperlim_cut
        print("ap_range for: ",weight_summary[i]['weight'], " = <",total_bits,",",int_bits,">" )
        weight_fixed.append((int(total_bits),int(int_bits)))

    activations_fixed = []
    activation_summary = hls4ml.model.profiling.activations_keras(model=assocmodel,X=assoc_array,fmt="summary")
    for i in range(len(activation_summary)):
        print(activation_summary[i]['weight']," : ", activation_summary[i]['whislo'],activation_summary[i]['whishi']," log2: ", np.log2(activation_summary[i]['whislo']),np.log2(activation_summary[i]['whishi']))
        int_bits  = np.rint(np.log2(activation_summary[i]['whishi'])) + 2 if np.rint(np.log2(activation_summary[i]['whishi'])) > 0 else 1
        total_bits = abs(np.rint(np.log2(activation_summary[i]['whislo']))) + int_bits + 1 if np.rint(np.log2(activation_summary[i]['whishi'])) > 0 else abs(np.rint(np.log2(activation_summary[i]['whislo']))) + int_bits + 2
        if total_bits > lowerlim_cut : total_bits = lowerlim_cut
        if int_bits > upperlim_cut : int_bits = upperlim_cut
        print("ap_range for: ",activation_summary[i]['weight'], " = <",total_bits,",",int_bits,">" )
        activations_fixed.append((int(total_bits),int(int_bits)))


    hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(rounding_mode='AP_RND')
    hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(saturation_mode='AP_SAT')

    assocconfig = hls4ml.utils.config_from_keras_model(assocmodel, granularity='name')
    assocconfig['Model']['Strategy'] = 'Resource'
    assocconfig['Model']['TraceOutput'] = True
        
    assocconfig['LayerName']['assoc']['Trace'] = True
    assocconfig['LayerName']['association_1']['Trace'] = True
    assocconfig['LayerName']['association_2']['Trace'] = True
    assocconfig['LayerName']['association_final']['Trace'] = True

    assocconfig['Model']["Precision"] = 'fixed<'+str(config['overall_model_quantisation'][0])+','+str(config['overall_model_quantisation'][1])+'>'
    assocconfig['LayerName']['assoc']['Precision']['result'] = 'fixed<'+str(config['overall_model_quantisation'][0])+','+str(config['overall_model_quantisation'][1])+'>'

    assocconfig['LayerName']['association_1']['Precision']['weight'] = 'fixed<'+str(weight_fixed[0][0])+","+str(weight_fixed[0][1])+'>'
    assocconfig['LayerName']['association_1']['Precision']['bias']   = 'fixed<'+str(weight_fixed[1][0])+","+str(weight_fixed[1][1])+'>'

    assocconfig['LayerName']['association_1_relu']['Trace'] = True
    assocconfig['LayerName']['association_1']['Precision']['result']        =  'fixed<'+str(activations_fixed[0][0])+","+str(activations_fixed[0][1])+'>'
    assocconfig['LayerName']['association_1_relu']['Precision']['result']   =  'fixed<'+str(activations_fixed[0][0])+","+str(activations_fixed[0][1])+'>'
    assocconfig['LayerName']['association_1_linear']['Precision']['result']   =  'fixed<'+str(activations_fixed[1][0])+","+str(activations_fixed[1][1])+'>'

    assocconfig['LayerName']['association_2']['Precision']['weight'] =  'fixed<'+str(weight_fixed[2][0])+","+str(weight_fixed[2][1])+'>'
    assocconfig['LayerName']['association_2']['Precision']['bias']   =  'fixed<'+str(weight_fixed[3][0])+","+str(weight_fixed[3][1])+'>'

    assocconfig['LayerName']['association_2_relu']['Trace'] = True
    assocconfig['LayerName']['association_2']['Precision']['result']        =  'fixed<'+str(activations_fixed[2][0])+","+str(activations_fixed[2][1])+'>'
    assocconfig['LayerName']['association_2_relu']['Precision']['result']   =  'fixed<'+str(activations_fixed[2][0])+","+str(activations_fixed[2][1])+'>'
    assocconfig['LayerName']['association_2_linear']['Precision']['result']   =  'fixed<'+str(activations_fixed[3][0])+","+str(activations_fixed[3][1])+'>'


    assocconfig['LayerName']['association_final']['Precision']['weight'] =  'fixed<'+str(weight_fixed[4][0]+1)+","+str(weight_fixed[4][1])+'>'
    assocconfig['LayerName']['association_final']['Precision']['bias']   =  'fixed<'+str(weight_fixed[5][0]+1)+","+str(weight_fixed[5][1])+'>'

    assocconfig['LayerName']['association_final']['Precision']['result']          =  'fixed<'+str(activations_fixed[4][0])+","+str(activations_fixed[4][1])+'>'

    cfg = hls4ml.converters.create_config(backend='Vitis')
    #cfg['IOType']     = 'io_serial' # Must set this if using CNNs!
    cfg['HLSConfig']  = assocconfig
    cfg['KerasModel'] = assocmodel
    cfg['OutputDir']  = filename+'hls_assoc/'
    cfg['Part'] =  'xcvu13p-flga2577-2-e'
    cfg['ClockPeriod'] = 2.7

    hls_assoc_model = hls4ml.converters.keras_to_hls(cfg)

    hls_assoc_model.compile()

    wp, wph, ap, aph = hls4ml.model.profiling.numerical(model=assocmodel, hls_model=hls_assoc_model,X=assoc_array)

    ap.savefig(filename+"Association_model_activations_profile.png")
    wp.savefig(filename+"Association_model_weights_profile.png")
    aph.savefig(filename+"Association_model_activations_profile_opt.png")
    wph.savefig(filename+"Association_model_weights_profile_opt.png")

    fig = hls4ml.model.profiling.compare(keras_model=assocmodel, hls_model=hls_assoc_model,X=assoc_array)
    fig.savefig(filename+"output_Association_comparison.png")

    #print(hls4ml.model.profiling.weights_hlsmodel(model=hls_weight_model,fmt="summary"))
    #print(hls4ml.model.profiling.activations_hlsmodel(model=hls_weight_model,X=WeightFeatures,fmt="summary"))
    
    y_keras = assocmodel.predict(assoc_array)
    y_hls4ml   = hls_assoc_model.predict(np.ascontiguousarray(assoc_array))

    for i,value in enumerate(assoc_array):
        if i > 100:
            break
        if abs(y_keras[i][0]) - abs(y_hls4ml[i][0]) > 0.01: 
            print("Input: ",assoc_array[i]," keras: ",y_keras[i]," hls4ml: ",y_hls4ml[i])
    
    # "Accuracy" of hls4ml predictions vs keras
    rel_acc = mean_squared_error(y_keras,y_hls4ml)
    print('{} accuracy relative to keras: {} \n'.format("Association model",rel_acc))

    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    hist2d = ax.hist2d(np.concatenate(y_keras).ravel(), np.concatenate(y_hls4ml).ravel(), bins=50, norm=matplotlib.colors.LogNorm(),cmap=colormap)
    ax.set_xlabel("Keras Prediction", horizontalalignment='right', x=1.0)
    ax.set_ylabel("hls4ml Prediction", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Tracks')
    plt.tight_layout()
    plt.savefig(filename+'outputAssociation.png')
    plt.close()

    QConfig = { "association_1":{'kernel_quantizer': 'quantized_bits('+(str(weight_fixed[0][0])+","+str(weight_fixed[0][1]))+', alpha=1)',
                            'bias_quantizer': 'quantized_bits('+(str(weight_fixed[1][0])+","+str(weight_fixed[1][1]))+', alpha=1)', 
                            'activation': 'quantized_relu('+(str(activations_fixed[0][0])+","+str(activations_fixed[0][1]))+')', 
                            },
                "association_2":{'kernel_quantizer': 'quantized_bits('+(str(weight_fixed[2][0])+","+str(weight_fixed[2][1]))+', alpha=1)',
                            'bias_quantizer': 'quantized_bits('+(str(weight_fixed[3][0])+","+str(weight_fixed[3][1]))+', alpha=1)', 
                            'activation': 'quantized_relu('+(str(activations_fixed[2][0])+","+str(activations_fixed[2][1]))+')',  
                            },
                "association_final":{'kernel_quantizer': 'quantized_bits('+(str(weight_fixed[4][0])+","+str(weight_fixed[4][1]))+', alpha=1)',
                            'bias_quantizer': 'quantized_bits('+(str(weight_fixed[5][0])+","+str(weight_fixed[5][1]))+', alpha=1)', 
                            }
              }
    f = open(filename+'AssociationQConfig.yaml', 'w+')
    yaml.dump(QConfig, f, allow_unicode=True)

    if sys.argv[4] == "True":
        import cmsml

        cmsml.tensorflow.save_graph(filename+"associationModelgraph.pb", assocmodel, variables_to_constants=True)
        hls_assoc_model.build(synth=True,vsynth=True,cosim=True)