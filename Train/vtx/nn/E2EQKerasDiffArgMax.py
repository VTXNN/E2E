import tensorflow as tf
import tensorflow_probability as tfp
import vtx
from vtx.nn.constraints import *
from qkeras import QActivation
from qkeras import QDense, QConv1D, QBatchNormalization
from qkeras.quantizers import quantized_bits, quantized_relu
import numpy
import numpy as np
from sklearn.metrics import accuracy_score

class E2EQKerasDiffArgMax():
    def __init__(self,
        nbins=256,
        start=0,
        end=255,
        max_z0 = 15,
        ntracks=250, 
        nweightfeatures=1,
        nfeatures=1, 
        nweights=1, 
        nlatent=0, 
        nweightnodes = 5,
        nweightlayers = 2,
        nassocnodes = 10,
        nassoclayers = 2,
        l1regloss=1e-3,
        l2regloss=1e-10,
        temperature=1e-4,
        weightqconfig={},
        patternqconfig={},
        associationqconfig={},
    ):
        self.nbins = nbins
        self.start = start
        self.end = end
        self.ntracks = ntracks
        self.nweightfeatures = nweightfeatures
        self.nfeatures = nfeatures
        self.nweights = nweights
        self.nlatent = nlatent
        self.max_z0 = max_z0

        self.l1regloss = l1regloss
        self.l2regloss = l2regloss

        self.nweightnodes = nweightnodes
        self.nweightlayers = nweightlayers
        self.nassocnodes = nassocnodes
        self.nassoclayers = nassoclayers

        self.weightqconfig = weightqconfig
        self.patternqconfig = patternqconfig
        self.associationqconfig = associationqconfig

        self.temperature = temperature

        self.weightModel = None
        self.patternModel = None
        self.associationModel = None
        
        self.inputWeightFeatures = tf.keras.layers.Input(shape=(self.ntracks,self.nweightfeatures),name='input_weight_features')
        self.inputTrackFeatures = tf.keras.layers.Input(shape=(self.ntracks,self.nfeatures),name='input_PV_track_features')
        self.inputTrackZ0 = tf.keras.layers.Input(shape=(self.ntracks),name='input_track_z0')

        self.weightLayers = []
        for ilayer,nodes in enumerate([nweightnodes]*nweightlayers):
            self.weightLayers.extend([
                QDense(
                    nodes,
                    trainable=True,
                    kernel_initializer='orthogonal',
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1regloss,l2regloss),
                    kernel_quantizer=self.weightqconfig['weight_'+str(ilayer+1)]['kernel_quantizer'],
                    bias_quantizer=self.weightqconfig['weight_'+str(ilayer+1)]['bias_quantizer'],
                    activation=None,
                    name='weight_'+str(ilayer+1)
                ),
                QActivation(self.weightqconfig['weight_'+str(ilayer+1)]['activation'],name='weight_'+str(ilayer+1)+'_relu'),
            ])
            
        self.weightLayers.extend([
            QDense(
                self.nweights,
                kernel_initializer='orthogonal',
                trainable=True,
                kernel_quantizer=self.weightqconfig['weight_final']['kernel_quantizer'],
                bias_quantizer=self.weightqconfig['weight_final']['bias_quantizer'],
                activation=None,
                kernel_regularizer=tf.keras.regularizers.L1L2(l1regloss,l2regloss),
                name='weight_final'
            ),
            QActivation(self.weightqconfig['weight_final']['activation'],name='weight_final_relu')
        ])

        self.zerolayer = vtx.nn.ZeroWeighting()
        
        self.kdeLayer = vtx.nn.KDELayer(
            nbins=self.nbins,
            start=self.start,
            end=self.end,
            add_overflow=False
        )
        
        self.patternConvLayers = []
        for ilayer,(filterSize,kernelSize) in enumerate([
            [1,3]
        ]):
            self.patternConvLayers.extend([
                QConv1D(
                    filterSize,
                    kernelSize,
                    kernel_initializer=tf.keras.initializers.Constant(value=0.1),
                    padding='same',
                    trainable=True,
                    use_bias= False,
                    kernel_quantizer=self.patternqconfig['pattern_'+str(ilayer+1)]['kernel_quantizer'],
                    activation=None,
                    name='pattern_'+str(ilayer+1)
                ),
                QActivation(self.patternqconfig['pattern_'+str(ilayer+1)]['activation'],name='pattern_'+str(ilayer+1)+'_relu')
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
            QDense(
                1+self.nlatent,
                activation='linear',
                trainable=True,
                use_bias= True,
                kernel_initializer='ones',
                name='position_final',
                kernel_quantizer='quantized_bits(16,6)',
                bias_quantizer='quantized_bits(16,6)',
            )
        ]
          
        self.assocLayers = []
        for ilayer,filterSize in enumerate(([nassocnodes]*nassoclayers)):
            self.assocLayers.extend([
                QDense(
                    filterSize,
                    kernel_initializer='orthogonal',
                    bias_initializer='ones',
                    kernel_regularizer=tf.keras.regularizers.L1L2(l1regloss,l2regloss),
                    kernel_quantizer=self.associationqconfig['association_'+str(ilayer+1)]['kernel_quantizer'],
                    bias_quantizer=self.associationqconfig['association_'+str(ilayer+1)]['bias_quantizer'],
                    activation=None,
                    name='association_'+str(ilayer+1)
                ),
                QActivation(self.associationqconfig['association_'+str(ilayer+1)]['activation'],name='association_'+str(ilayer+1)+"_relu"),
            ])
            
        self.assocLayers.extend([
            QDense(
                1,
                activation=None,
                kernel_initializer='orthogonal',
                bias_initializer='ones',
                kernel_regularizer=tf.keras.regularizers.l2(l2regloss),
                kernel_quantizer=self.associationqconfig['association_final']['kernel_quantizer'],
                bias_quantizer=self.associationqconfig['association_final']['bias_quantizer'],
                name='association_final'
            )
        ])

        self.tiledTrackDimLayer = tf.keras.layers.Lambda(lambda x: tf.reshape(tf.tile(x,[1,self.ntracks]),[-1,self.ntracks,x.shape[1]]),name='tiled_track_dim')
                
    def applyLayerList(self, inputs, layerList):
        outputs = inputs
        for layer in layerList:
            outputs = layer(outputs)
        return outputs

    def createWeightModel(self):
        weightInput = tf.keras.layers.Input(shape=(self.nweightfeatures),name="input_weight")
        weightLayers = []
        for ilayer,nodes in enumerate([self.nweightnodes]*self.nweightlayers):
            weightLayers.extend([
                QDense(
                    nodes,
                    trainable=True,
                    kernel_initializer='orthogonal',
                    kernel_regularizer=tf.keras.regularizers.L1L2(self.l1regloss,self.l2regloss),
                    kernel_quantizer=self.weightqconfig['weight_'+str(ilayer+1)]['kernel_quantizer'],
                    bias_quantizer=self.weightqconfig['weight_'+str(ilayer+1)]['bias_quantizer'],
                    activation=None,
                    name='weight_'+str(ilayer+1)
                ),
                QActivation(self.weightqconfig['weight_'+str(ilayer+1)]['activation'],name='weight_'+str(ilayer+1)+'_relu'),
            ])
            
        weightLayers.extend([
            QDense(
                self.nweights,
                kernel_initializer='orthogonal',
                trainable=True,
                kernel_quantizer=self.weightqconfig['weight_final']['kernel_quantizer'],
                bias_quantizer=self.weightqconfig['weight_final']['bias_quantizer'],
                activation=None,
                kernel_regularizer=tf.keras.regularizers.L1L2(self.l1regloss,self.l2regloss),
                name='weight_final'
            ),
            QActivation(self.weightqconfig['weight_final']['activation'],name='weight_final_relu')
        ])

        outputs = self.applyLayerList(weightInput,weightLayers)

        return tf.keras.Model(inputs=weightInput,outputs=outputs)
    
    def createPatternModel(self):
        histInput = tf.keras.layers.Input(shape=(self.nbins,self.nweights),name="hist")
        patternConvLayers = []
        for ilayer,(filterSize,kernelSize) in enumerate([
            [1,3]
        ]):
            patternConvLayers.extend([
                QConv1D(
                    filterSize,
                    kernelSize,
                    kernel_initializer='orthogonal',
                    padding='same',
                    trainable=True,
                    use_bias= False,
                    kernel_quantizer=self.patternqconfig['pattern_'+str(ilayer+1)]['kernel_quantizer'],
                    activation=None,
                    name='pattern_'+str(ilayer+1)
                ),
                QActivation(self.patternqconfig['pattern_'+str(ilayer+1)]['activation'],name='pattern_'+str(ilayer+1)+'_relu')
            ])
        convs = self.applyLayerList(histInput,patternConvLayers)
        return tf.keras.Model(inputs=histInput,outputs=convs)
    
    def createAssociationModel(self):
        assocInput = tf.keras.layers.Input(shape=(self.nfeatures+1+self.nlatent),name="assoc")
        assocLayers = []
        for ilayer,filterSize in enumerate(([self.nassocnodes]*self.nassoclayers)):
            assocLayers.extend([
                QDense(
                    filterSize,
                    kernel_initializer='orthogonal',
                    bias_initializer='ones',
                    kernel_regularizer=tf.keras.regularizers.L1L2(self.l1regloss,self.l2regloss),
                    kernel_quantizer=self.associationqconfig['association_'+str(ilayer+1)]['kernel_quantizer'],
                    bias_quantizer=self.associationqconfig['association_'+str(ilayer+1)]['bias_quantizer'],
                    activation=None,
                    name='association_'+str(ilayer+1)
                ),
                QActivation(self.associationqconfig['association_'+str(ilayer+1)]['activation'],name='association_'+str(ilayer+1)+"_relu"),
            ])
            
        assocLayers.extend([
            QDense(
                1,
                activation=None,
                kernel_initializer='orthogonal',
                bias_initializer='ones',
                kernel_regularizer=tf.keras.regularizers.l2(self.l2regloss),
                kernel_quantizer=self.associationqconfig['association_final']['kernel_quantizer'],
                bias_quantizer=self.associationqconfig['association_final']['bias_quantizer'],
                name='association_final'
            )
        ])
        assocProbability = self.applyLayerList(assocInput,assocLayers)
        return tf.keras.Model(inputs=assocInput,outputs=assocProbability)
        
    def createE2EModel(self):
        
        weights = self.applyLayerList(self.inputWeightFeatures,self.weightLayers)
        weights = self.zerolayer(self.inputWeightFeatures,weights)
        hists = self.kdeLayer([self.inputTrackZ0,weights])
        convs = self.applyLayerList(hists,self.patternConvLayers)
        temp = tf.keras.layers.Lambda(lambda x: x / self.temperature)(convs)
        softmax = self.softMaxLayer(temp)
        binweight = self.binWeightLayer(softmax)
        pv,argmax = self.ArgMaxLayer(binweight)

        pvFeatures = self.applyLayerList(pv,self.pvDenseLayers)
        pvFeatures_argmax = self.applyLayerList(argmax,self.pvDenseLayers)

        if self.nlatent>0:
            pvPosition_argmax,latentFeatures_argmax = tf.keras.layers.Lambda(lambda x: [x[:,0:1],x[:,1:]],name='split_latent_argmax')(pvFeatures_argmax)
            pvPosition,latentFeatures = tf.keras.layers.Lambda(lambda x: [x[:,0:1],x[:,1:]],name='split_latent')(pvFeatures)

        else:
            pvPosition_argmax = pvFeatures_argmax
            pvPosition = pvFeatures

        z0Diff = tf.keras.layers.Lambda(lambda x: tf.stop_gradient(tf.expand_dims(abs(x[0]-x[1]),2)),name='z0_diff_argmax')([self.inputTrackZ0,pvPosition_argmax])

        
        assocFeatures = [self.inputTrackFeatures,z0Diff]   

        if self.nlatent>0:
            assocFeatures.append(self.tiledTrackDimLayer(latentFeatures_argmax))   
            
        assocFeat = tf.keras.layers.Concatenate(axis=2,name='association_features')(assocFeatures)

        assocProbability = self.applyLayerList(assocFeat,self.assocLayers)
        
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

        largerModel.summary()
        self.patternModel.summary()
        self.weightModel.summary()
        self.associationModel.summary()

        self.weightModel.get_layer('weight_1').set_weights       (largerModel.get_layer('weight_1').get_weights())
        self.weightModel.get_layer('weight_1_relu').set_weights   (largerModel.get_layer('weight_1_relu').get_weights())
        self.weightModel.get_layer('weight_2').set_weights       (largerModel.get_layer('weight_2').get_weights())
        self.weightModel.get_layer('weight_2_relu').set_weights (largerModel.get_layer('weight_2_relu').get_weights())
        self.weightModel.get_layer('weight_final').set_weights   (largerModel.get_layer('weight_final').get_weights())
        self.weightModel.get_layer('weight_final_relu').set_weights (largerModel.get_layer('weight_final_relu').get_weights())

        self.patternModel.get_layer('pattern_1').set_weights     (largerModel.get_layer('pattern_1').get_weights())
        self.patternModel.get_layer('pattern_1_relu').set_weights(largerModel.get_layer('pattern_1_relu').get_weights())

        self.associationModel.get_layer('association_1').set_weights    (largerModel.get_layer('association_1').get_weights())
        self.associationModel.get_layer('association_1_relu').set_weights   (largerModel.get_layer('association_1_relu').get_weights()) 
        self.associationModel.get_layer('association_2').set_weights    (largerModel.get_layer('association_2').get_weights()) 
        self.associationModel.get_layer('association_2_relu').set_weights   (largerModel.get_layer('association_2_relu').get_weights()) 
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
