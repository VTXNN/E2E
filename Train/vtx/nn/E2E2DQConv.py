import tensorflow as tf
import tensorflow_probability as tfp
import vtx
import numpy

class E2E2DConv():
    def __init__(self,
        ntracks=200, 
        nbins=256,
        nfeatures=1, 
        nhistfeatures=9,
        activation='relu',
        nassocnodes = 20,
        nassoclayers = 2,
        1regloss=1e-3,
        l2regloss=1e-10,
        qconfig={}

    ):

        self.ntracks = ntracks

        self.nfeatures = nfeatures
        self.nhistfeatures = nhistfeatures
        self.nbins = nbins
        self.activation = activation
        self.l2regloss = l2regloss

        self.patternModel = None
        self.associationModel = None
        
        self.inputHistFeatures = tf.keras.layers.Input(shape=(self.ntracks,self.nhistfeatures,self.nbins),name='input_histogram_features')
        self.inputTrackFeatures = tf.keras.layers.Input(shape=(self.ntracks,self.nfeatures),name='input_PV_track_features')
        self.inputTrackZ0 = tf.keras.layers.Input(shape=(self.ntracks),name='input_track_z0')

        self.patternConvLayers = []
        for ilayer,(filterSize,kernelSize) in enumerate([
            [1,(9,self.nhistfeatures)],
            [1,(9,self.nhistfeatures)],
            [1,(5,self.nhistfeatures)],
            [1,(3,self.nhistfeatures)]
        ]):
            self.patternConvLayers.extend([
                QConv2D(
                    filterSize,
                    kernelSize,
                    padding='same',
                    activation='linear',
                    kernel_initializer='orthogonal',
                    trainable=True,
                    use_bias= True,
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1regloss,l2regloss),
                    kernel_quantizer=qconfig['pattern_'+str(ilayer+1)]['kernel_quantizer'],
                    bias_quantizer=qconfig['pattern_'+str(ilayer+1)]['bias_quantizer'],
                    name='pattern_'+str(ilayer+1)
                ) ,
                QActivation(qconfig['pattern_'+str(ilayer+1)]['activation']),
            ])
        

        self.pvDenseLayers = [
            QAveragePooling2D(),
            tf.keras.layers.Flatten(),
            QDense(
                128,
                activation='linear',
                trainable=True,
                use_bias= True,
                kernel_initializer='ones',
                bias_initializer='zeros',
                name='position_flatten',
                kernel_quantizer=qconfig['flatten']['kernel_quantizer'],
                bias_quantizer=qconfig['flatten']['bias_quantizer'],
                
            ),
            QActivation(qconfig['flatten']['activation']),
            QDense(
                1,
                activation='linear',
                trainable=True,
                use_bias= True,
                kernel_initializer='ones',
                name='position_final',
                kernel_quantizer='quantized_bits(10,1,alpha=1)',
                bias_quantizer='quantized_bits(10,1,alpha=1)',
            )
        ]
          
        self.assocLayers = []
        for ilayer,filterSize in enumerate(([nassocnodes]*nassoclayers)):
            self.assocLayers.extend([
                QDense(
                    filterSize,
                    kernel_initializer='orthogonal',
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1regloss,l2regloss),
                    kernel_quantizer=qconfig['association_'+str(ilayer)]['kernel_quantizer'],
                    bias_quantizer=qconfig['association_'+str(ilayer)]['bias_quantizer'],
                    activation=None,
                    name='association_'+str(ilayer)
                ),
                QActivation(qconfig['association_'+str(ilayer)]['activation']),
            ])
            
        self.assocLayers.extend([
            QDense(
                1,
                activation=None,
                kernel_initializer='orthogonal',
                kernel_regularizer=tf.keras.regularizers.l2(l2regloss),
                kernel_quantizer=qconfig['association_final']['kernel_quantizer'],
                bias_quantizer=qconfig['association_final']['bias_quantizer'],
                name='association_final'
            )
        ])

        self.tiledTrackDimLayer = tf.keras.layers.Lambda(lambda x: tf.reshape(tf.tile(x,[1,self.ntracks]),[-1,self.ntracks,x.shape[1]]),name='tiled_track_dim')

                
    def applyLayerList(self, inputs, layerList):
        outputs = inputs
        for layer in layerList:
            outputs = layer(outputs)
        return outputs

    
    def createPatternModel(self):
        histInput = tf.keras.layers.Input(shape=(self.nbins,self.nhistfeatures),name="hist")
        convs = self.applyLayerList(histInput,self.patternConvLayers)
        return tf.keras.Model(inputs=[histInput],outputs=[convs])
    
    def createAssociationModel(self):
        assocInput = tf.keras.layers.Input(shape=(self.nfeatures+1+self.nlatent),name="assoc")
        assocProbability = self.applyLayerList(assocInput,self.assocLayers)
        return tf.keras.Model(inputs=[assocInput],outputs=[assocProbability])
        
    def createE2EModel(self):
        convs = self.applyLayerList(self.inputHistFeatures,self.patternConvLayers)
        pvFeatures = self.applyLayerList(convs,self.pvDenseLayers)

        pvPosition = pvFeatures

        z0Diff = tf.keras.layers.Lambda(lambda x: tf.stop_gradient(tf.expand_dims(tf.abs(x[0]-x[1]),2)),name='z0_diff')([self.inputTrackZ0,pvPosition])

        assocFeatures = [self.inputTrackFeatures,z0Diff]   

        assocFeat = tf.keras.layers.Concatenate(axis=2,name='association_features')(assocFeatures)

        assocProbability = self.applyLayerList(assocFeat,self.assocLayers)
        
        model = tf.keras.Model(
            inputs=[self.inputTrackZ0,self.inputHistFeatures,self.inputTrackFeatures],
            outputs=[pvPosition,assocProbability]
        )

        return model

    def load_weights(self,largerModel):
        self.patternModel = self.createPatternModel()
        self.associationModel = self.createAssociationModel()

        self.patternModel.get_layer('pattern_1').set_weights(largerModel.get_layer('pattern_1').get_weights())
        self.patternModel.get_layer('pattern_2').set_weights(largerModel.get_layer('pattern_2').get_weights())
        self.patternModel.get_layer('pattern_3').set_weights(largerModel.get_layer('pattern_3').get_weights())
        self.patternModel.get_layer('pattern_4').set_weights(largerModel.get_layer('pattern_4').get_weights())

        self.associationModel.get_layer('association_0').set_weights    (largerModel.get_layer('association_0').get_weights())
        self.associationModel.get_layer('dropout_2').set_weights        (largerModel.get_layer('dropout_2').get_weights()) 
        self.associationModel.get_layer('association_1').set_weights    (largerModel.get_layer('association_1').get_weights()) 
        self.associationModel.get_layer('dropout_3').set_weights        (largerModel.get_layer('dropout_3').get_weights()) 
        self.associationModel.get_layer('association_final').set_weights(largerModel.get_layer('association_final').get_weights()) 


    def write_model_graph(self,modelName):
        import cmsml

        cmsml.tensorflow.save_graph(modelName+"_patternModelgraph.pb", self.patternModel, variables_to_constants=True)
        cmsml.tensorflow.save_graph(modelName+"_associationModelgraph.pb", self.associationModel, variables_to_constants=True)

    def export_individual_models(self,modelName):

        with open(modelName+"_patternModel.json", 'w') as f:
            f.write(self.patternModel.to_json())
        self.patternModel.save_weights(modelName+"_patternModel_weights.hdf5")
        self.patternModel.save(modelName+"_patternModel")

        with open(modelName+"_associationModel.json", 'w') as f:
            f.write(self.associationModel.to_json())
        self.associationModel.save_weights(modelName+"_associationModel_weights.hdf5")
        self.associationModel.save(modelName+"_associationModel")

    def export_hls_pattern_model(self,modelName):
        import hls4ml
        import numpy as np
        import matplotlib
        #matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        patternconfig = hls4ml.utils.config_from_keras_model(self.patternModel, granularity='name')
        patternconfig['Model']['Strategy'] = 'Resource'
        print("-----------------------------------")
        print("Configuration")
        #plotting.print_dict(config)
        print("-----------------------------------")
        random_pattern_data = np.random.rand(1000,256,1)
        hls_pattern_model = hls4ml.converters.convert_from_keras_model(self.patternModel,
                                                            hls_config=patternconfig,
                                                            output_dir=modelName+'_hls_pattern/hls4ml_prj',
                                                            fpga_part='xcvu9p-flga2104-2L-e',
                                                            #fpga_part='xcvu13p-flga2577-2-e',
                                                            clock_period=2.0)
        #hls4ml.utils.plot_model(hls_pattern_model, show_shapes=True, show_precision=True, to_file=modelName+"_pattern_model.png")
        #plt.clf()
        #ap,wp = hls4ml.model.profiling.numerical(model=self.patternModel, hls_model=hls_pattern_model, X=random_pattern_data)
        #wp.savefig(modelName+"_pattern_model_activations_profile.png")
        #ap.savefig(modelName+"_pattern_model_weights_profile.png")

        hls_pattern_model.compile()
        hls_pattern_model.build(csim=True,synth=True,vsynth=True)

    def export_hls_assoc_model(self,modelName):
        import hls4ml
        import numpy as np
        import matplotlib
        #matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        associationconfig = hls4ml.utils.config_from_keras_model(self.associationModel, granularity='name')
        print(associationconfig)
        print("-----------------------------------")
        print("Configuration")
        #plotting.print_dict(config)
        print("-----------------------------------")
        random_association_data = np.random.rand(1000,4+self.nlatent)
        hls_association_model = hls4ml.converters.convert_from_keras_model(self.associationModel,
                                                            hls_config=associationconfig,
                                                            output_dir=modelName+'_hls_association/hls4ml_prj',
                                                            part='xcvu9p-flga2104-2L-e',
                                                            #part='xcvu13p-flga2577-2-e',
                                                            clock_period=2.0)
        #hls4ml.utils.plot_model(hls_association_model, show_shapes=True, show_precision=True, to_file=modelName+"_association_model.png")
        #plt.clf()
        #ap,wp = hls4ml.model.profiling.numerical(model=self.associationModel, hls_model=hls_association_model, X=random_association_data)
        #wp.savefig(modelName+"_association_model_activations_profile.png")
        #ap.savefig(modelName+"_association_model_weights_profile.png")

        hls_association_model.compile()
        hls_association_model.build(csim=True,synth=True,vsynth=True)