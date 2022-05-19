import tensorflow_decision_forests as tfdf
import tensorflow as tf
import tensorflow_probability as tfp
import vtx
import numpy

class E2EDiffArgMax():
    def __init__(self,
        nbins=256,
        start=-15,
        end=15,
        max_z0 = 15,
        ntracks=200, 
        nweightfeatures=1,
        nfeatures=1, 
        nweights=1, 
        npattern=4,
        nlatent=0, 
        activation='relu',
        nweightnodes = 10,
        nweightlayers = 2,
        nassocnodes = 20,
        nassoclayers = 2,
        l2regloss=1e-10,
        temperature=1e-4,
        return_index = False,
        train_cnn = True,
    ):
        self.nbins = nbins
        self.start = start
        self.end = end
        self.ntracks = ntracks
        self.nweightfeatures = nweightfeatures
        self.nfeatures = nfeatures
        self.nweights = nweights
        self.npattern = npattern
        self.nlatent = nlatent
        self.activation = activation
        self.max_z0 = max_z0

        self.l2regloss = l2regloss

        self.temperature = temperature

        self.return_index = return_index

        self.train_cnn = train_cnn

        self.weightModel = None
        self.patternModel = None
        self.associationModel = None
        
        self.inputWeightFeatures = tf.keras.layers.Input(shape=(self.ntracks,self.nweightfeatures),name='input_weight_features')
        self.inputTrackFeatures = tf.keras.layers.Input(shape=(self.ntracks,self.nfeatures),name='input_PV_track_features')
        self.inputTrackZ0 = tf.keras.layers.Input(shape=(self.ntracks),name='input_track_z0')

        self.weightLayers = []
        for ilayer,nodes in enumerate([nweightnodes]*nweightlayers):
            self.weightLayers.extend([
                tf.keras.layers.Dense(
                    nodes,
                    activation=self.activation,
                    trainable=True,
                    kernel_initializer='orthogonal',
                    kernel_regularizer=tf.keras.regularizers.l2(self.l2regloss),
                    name='weight_'+str(ilayer+1)
                ),
                tf.keras.layers.Dropout(0.1),
                #tf.keras.layers.Activation(self.activation),
                #tf.keras.layers.BatchNormalization(),

                
            ])
            
        self.weightLayers.extend([
            tf.keras.layers.Dense(
                self.nweights,
                activation=self.activation, #need to use relu here to remove negative weights
                kernel_initializer='orthogonal',
                trainable=True,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2regloss),
                name='weight_final'
            ),
            #tf.keras.layers.Activation(self.activation)
        ])

        
        self.kdeLayer = vtx.nn.KDELayer(
            nbins=self.nbins,
            start=self.start,
            end=self.end,
            add_overflow=False
        )
        
        
        self.patternConvLayers = []
        for ilayer,(filterSize,kernelSize) in enumerate([
#            [1,5],
            [1,3]
        ]):
            self.patternConvLayers.extend([
                tf.keras.layers.Conv1D(
                    filterSize,
                    kernelSize,
                    padding='same',
                    activation='linear',#self.activation,
                    kernel_initializer='orthogonal',
                    trainable=train_cnn,
                    use_bias= False,
                    name='pattern_'+str(ilayer+1)
                ) ,
                tf.keras.layers.Activation(self.activation), 
            ])
        

        self.softMaxLayer = tf.keras.layers.Softmax(axis=1)
        
        self.binWeightLayer = tf.keras.layers.Dense(
                    self.nbins,
                    activation='linear',
                    trainable=False,
                    use_bias= False,
                    name='Bin_weight'
                )

        self.ArgMaxLayer = vtx.nn.BintoVertex(
            nbins=self.nbins,
            start=-1*self.max_z0,
            end=self.max_z0
        )

        self.pvDenseLayers = [
            tf.keras.layers.Dense(
                1+self.nlatent,
                activation='linear',
                trainable=True,
                use_bias= True,
                kernel_initializer='ones',
                bias_initializer='zeros',
                name='position_final'
            )
        ]
          
        self.first_vertex_BDT = tfdf.keras.RandomForestModel(num_trees=1000)
        self.first_vertex_BDT.task = task=tfdf.keras.Task.CLASSIFICATION
        self.first_vertex_BDT.input_spec=tf.keras.layers.InputSpec(shape=(None,250,4,1))

        self.tiledTrackDimLayer = tf.keras.layers.Lambda(lambda x: tf.reshape(tf.tile(x,[1,self.ntracks]),[-1,self.ntracks,x.shape[1]]),name='tiled_track_dim')

                
    def applyLayerList(self, inputs, layerList):
        outputs = inputs
        for layer in layerList:
            outputs = layer(outputs)
        return outputs

    def createWeightModel(self):
        weightInput = tf.keras.layers.Input(shape=(self.nweightfeatures),name="weight")
        weights = self.applyLayerList(weightInput,self.weightLayers)
        return tf.keras.Model(inputs=[weightInput],outputs=[weights])
    
    def createPatternModel(self):
        histInput = tf.keras.layers.Input(shape=(self.nbins,self.nweights),name="hist")
        convs = self.applyLayerList(histInput,self.patternConvLayers)
        return tf.keras.Model(inputs=[histInput],outputs=[convs])

    def createPositionModel(self):
        convsInput = tf.keras.layers.Input(shape=(self.nbins),name="conv")
        temp = tf.keras.layers.Lambda(lambda x: x / self.temperature)
        softmax = self.softMaxLayer(temp)
        binweight = self.binWeightLayer(softmax)
        argmax = self.ArgMaxLayer(binweight)
        return tf.keras.Model(inputs=[convsInput],outputs=[argmax])
    
    def createAssociationModel(self):
        assocInput = tf.keras.layers.Input(shape=(self.nfeatures+1),name="assoc")
        assocProbability = self.first_vertex_BDT(assocInput)
        return tf.keras.Model(inputs=[assocInput],outputs=[assocProbability])
        
    def createE2EModel(self):
        
        weights = self.applyLayerList(self.inputWeightFeatures,self.weightLayers)
        hists = self.kdeLayer([self.inputTrackZ0,weights])
        convs = self.applyLayerList(hists,self.patternConvLayers)
        temp = tf.keras.layers.Lambda(lambda x: x / self.temperature)(convs)
        softmax = self.softMaxLayer(temp)
        binweight = self.binWeightLayer(softmax)
        pv,argmax = self.ArgMaxLayer(binweight)

        pvFeatures = self.applyLayerList(pv,self.pvDenseLayers)
        pvFeatures_argmax = self.applyLayerList(argmax,self.pvDenseLayers)

        if self.nlatent>0:
            pvPosition,latentFeatures = tf.keras.layers.Lambda(lambda x: [x[:,0:1],x[:,1:]],name='split_latent')(pvFeatures)
        else:
            pvPosition = pvFeatures

        if self.nlatent>0:
            pvPosition_argmax,latentFeatures_argmax = tf.keras.layers.Lambda(lambda x: [x[:,0:1],x[:,1:]],name='split_latent_argmax')(pvFeatures_argmax)
        else:
            pvPosition_argmax = pvFeatures_argmax

        if self.return_index:
            z0Diff = tf.keras.layers.Lambda(lambda x: tf.stop_gradient(tf.expand_dims(tf.abs(x[0]-x[1])/8,2)),name='z0_diff_argmax')([self.inputTrackZ0,pvPosition_argmax])
        else:
            z0Diff = tf.keras.layers.Lambda(lambda x: tf.stop_gradient(tf.expand_dims(tf.abs(x[0]-x[1]),2)),name='z0_diff')([self.inputTrackZ0,pvPosition])
        
        assocFeatures = [self.inputTrackFeatures,z0Diff]   

        if self.nlatent>0:
            if self.return_index:
                assocFeatures.append(self.tiledTrackDimLayer(latentFeatures))  
            else:
                assocFeatures.append(self.tiledTrackDimLayer(latentFeatures_argmax))  
            
        assocFeat = tf.keras.layers.Concatenate(axis=2,name='association_features')(assocFeatures)

        print(assocFeat.shape)

        assocFeat = tf.expand_dims(assocFeat,3)

        print(assocFeat.shape)

        assocProbability =  self.first_vertex_BDT(assocFeat)

        print(assocProbability.shape)
        
        model = tf.keras.Model(
            inputs=[self.inputTrackZ0,self.inputWeightFeatures,self.inputTrackFeatures],
            outputs=[pvPosition,assocProbability,weights]
        )

        def q90loss(w):
            wq90 = tfp.stats.percentile(
                w,
                q=90.,
                axis=1,
                interpolation='nearest',
            )
            return tf.reduce_mean(0.1*tf.square(wq90-1.))
        
        model.add_loss(tf.keras.layers.Lambda(q90loss)(weights))
        return model

    def load_weights(self,largerModel):
        self.weightModel = self.createWeightModel()
        self.patternModel = self.createPatternModel()
        self.associationModel = self.createAssociationModel()

        self.weightModel.get_layer('weight_1').set_weights    (largerModel.get_layer('weight_1').get_weights())
        self.weightModel.get_layer('dropout').set_weights     (largerModel.get_layer('dropout').get_weights())
        self.weightModel.get_layer('weight_2').set_weights     (largerModel.get_layer('weight_2').get_weights())
        self.weightModel.get_layer('dropout_1').set_weights   (largerModel.get_layer('dropout_1').get_weights())
        self.weightModel.get_layer('weight_final').set_weights(largerModel.get_layer('weight_final').get_weights())

        self.patternModel.get_layer('pattern_1').set_weights(largerModel.get_layer('pattern_1').get_weights())

        self.associationModel.get_layer('association_0').set_weights    (largerModel.get_layer('association_0').get_weights())
        self.associationModel.get_layer('dropout_2').set_weights        (largerModel.get_layer('dropout_2').get_weights()) 
        self.associationModel.get_layer('association_1').set_weights    (largerModel.get_layer('association_1').get_weights()) 
        self.associationModel.get_layer('dropout_3').set_weights        (largerModel.get_layer('dropout_3').get_weights()) 
        self.associationModel.get_layer('association_final').set_weights(largerModel.get_layer('association_final').get_weights()) 


    def write_model_graph(self,modelName):
        import cmsml

        cmsml.tensorflow.save_graph(modelName+"_weightModelgraph.pb", self.weightModel, variables_to_constants=True)
        cmsml.tensorflow.save_graph(modelName+"_patternModelgraph.pb", self.patternModel, variables_to_constants=True)
        cmsml.tensorflow.save_graph(modelName+"_associationModelgraph.pb", self.associationModel, variables_to_constants=True)

    def export_individual_models(self,modelName):

        with open(modelName+"_weightModel.json", 'w') as f:
            f.write(self.weightModel.to_json())
        self.weightModel.save_weights(modelName+"_weightModel_weights.hdf5")
        self.weightModel.save(modelName+"_weightModel")

        with open(modelName+"_patternModel.json", 'w') as f:
            f.write(self.patternModel.to_json())
        self.patternModel.save_weights(modelName+"_patternModel_weights.hdf5")
        self.patternModel.save(modelName+"_patternModel")

        with open(modelName+"_associationModel.json", 'w') as f:
            f.write(self.associationModel.to_json())
        self.associationModel.save_weights(modelName+"_associationModel_weights.hdf5")
        self.associationModel.save(modelName+"_associationModel")

    def export_hls_weight_model(self,modelName):
        import hls4ml
        import numpy as np
        import matplotlib
        #matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        weightconfig = hls4ml.utils.config_from_keras_model(self.weightModel, granularity='name')
        print(weightconfig)
        print("-----------------------------------")
        print("Configuration")
        #plotting.print_dict(config)
        print("-----------------------------------")
        random_weight_data = np.random.rand(1000,3)
        hls_weight_model = hls4ml.converters.convert_from_keras_model(self.weightModel,
                                                            hls_config=weightconfig,
                                                            output_dir=modelName+'_hls_weight/hls4ml_prj',
                                                            part='xcvu9p-flga2104-2L-e',
                                                            #part='xcvu13p-flga2577-2-e',
                                                            clock_period=2.0)
        #hls4ml.utils.plot_model(hls_weight_model, show_shapes=True, show_precision=True, to_file=modelName+"_Weight_model.png")
        #plt.clf()
        #ap, wp = hls4ml.model.profiling.numerical(model=self.weightModel, hls_model=hls_weight_model, X=random_weight_data)
        #wp.savefig(modelName+"_Weight_model_activations_profile.png")
        #ap.savefig(modelName+"_Weight_model_weights_profile.png")

        hls_weight_model.compile()
        hls_weight_model.build(csim=True,synth=True,vsynth=True)



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