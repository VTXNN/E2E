import tensorflow as tf
import tensorflow_probability as tfp
import vtx
from qkeras import QActivation
from qkeras import QDense, QConv1D
from qkeras.quantizers import quantized_bits, quantized_relu
import numpy

class E2EQKerasDiffArgMax():
    def __init__(self,
        nbins=256,
        ntracks=200, 
        nweightfeatures=1,
        nfeatures=1, 
        nweights=1, 
        npattern=4,
        nlatent=0, 
        activation=None,
        nweightnodes = 5,
        nweightlayers = 2,
        nassocnodes = 10,
        nassoclayers = 2,
        l1regloss=1e-3,
        l2regloss=1e-10,
        temperature=1e2,
        bits = 12,
        integer = 2,
        alpha = 1
    ):
        self.nbins = nbins
        self.ntracks = ntracks
        self.nweightfeatures = nweightfeatures
        self.nfeatures = nfeatures
        self.nweights = nweights
        self.npattern = npattern
        self.nlatent = nlatent

        self.bits = bits
        self.integer = integer
        self.alpha = alpha

        self.activation = 'relu'#quantized_relu(self.bits)

        self.temperature = temperature
        
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
                    kernel_initializer='lecun_normal',
                    kernel_regularizer=tf.keras.regularizers.l2(l2regloss),
                    #kernel_quantizer=quantized_bits(self.bits,self.integer,self.alpha),
                    #bias_quantizer=quantized_bits(self.bits,self.integer,self.alpha),
                    name='weight_'+str(ilayer+1)
                ),
                tf.keras.layers.Dropout(0.1)
            ])
            
        self.weightLayers.append(
            tf.keras.layers.Dense(
                self.nweights,
                activation=self.activation, #need to use relu here to remove negative weights
                kernel_initializer='lecun_normal',
                trainable=True,
                #kernel_quantizer=quantized_bits(self.bits,self.integer,self.alpha),
                #bias_quantizer=quantized_bits(self.bits,self.integer,self.alpha),
                kernel_regularizer=tf.keras.regularizers.l2(l2regloss),
                name='weight_final'
            ),
        )
        
        self.kdeLayer = vtx.nn.KDELayer(
            nbins=self.nbins,
            start=-15,
            end=15,
            add_overflow=False
        )
        
        
        self.patternConvLayers = []
        for ilayer,(filterSize,kernelSize) in enumerate([
            [1,3]
        ]):
            self.patternConvLayers.append(
                tf.keras.layers.Conv1D(
                    filterSize,
                    kernelSize,
                    padding='same',
                    activation=self.activation,
                    trainable=True,
                    use_bias= False,
                    #kernel_quantizer=quantized_bits(self.bits,self.integer,self.alpha),
                    #bias_quantizer=quantized_bits(self.bits,self.integer,self.alpha),
                    name='pattern_'+str(ilayer+1)
                )
            )

        

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
            start=-15,
            end=15
        )

        self.pvDenseLayers = [
            tf.keras.layers.Dense(
                1+self.nlatent,
                activation=None,
                kernel_initializer='lecun_normal',
                kernel_regularizer=tf.keras.regularizers.l2(l2regloss),
                name='position_final'
            )
        ]
          
        self.assocLayers = []
        for ilayer,filterSize in enumerate(([nassocnodes]*nassoclayers)):
            #self.assocLayers.extend([
            #    QDense(
            #        filterSize,
            #        activation=QActivation(self.activation),
            #        kernel_initializer='lecun_normal',
            #        kernel_regularizer=tf.keras.regularizers.L1L2(l1regloss,l2regloss),
            #        kernel_quantizer=quantized_bits(self.bits,self.integer,self.alpha),
            #        bias_quantizer=quantized_bits(self.bits,self.integer,self.alpha),
            #        name='association_'+str(ilayer)
            #    )
            #])

            self.assocLayers.extend([
                tf.keras.layers.Dense(
                    filterSize,
                    activation=self.activation,
                    kernel_initializer='lecun_normal',
                    kernel_regularizer=tf.keras.regularizers.l2(l2regloss),
                    name='association_'+str(ilayer)
                ),
                tf.keras.layers.Dropout(0.1)
            ])
            
        self.assocLayers.extend([
            tf.keras.layers.Dense(
                1,
                activation=None,
                kernel_initializer='lecun_normal',
                kernel_regularizer=tf.keras.regularizers.l2(l2regloss),
                #kernel_quantizer=quantized_bits(self.bits,self.integer,self.alpha),
                #bias_quantizer=quantized_bits(self.bits,self.integer,self.alpha),
                name='association_final'
            )
        ])

        self.tiledTrackDimLayer = tf.keras.layers.Lambda(lambda x: tf.reshape(tf.tile(x,[1,self.ntracks]),[-1,self.ntracks,x.shape[1]]),name='tiled_track_dim')

        #self.outputSoftmax = tf.keras.layers.Softmax(name='association_final')
                
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
        assocInput = tf.keras.layers.Input(shape=(self.nfeatures+1+self.nlatent),name="assoc")
        assocProbability = self.applyLayerList(assocInput,self.assocLayers)
        #assocProbability = self.outputSoftmax(assocQProbability)
        return tf.keras.Model(inputs=[assocInput],outputs=[assocProbability])
        
    def createE2EModel(self):
        
        weights = self.applyLayerList(self.inputWeightFeatures,self.weightLayers)
        hists = self.kdeLayer([self.inputTrackZ0,weights])
        convs = self.applyLayerList(hists,self.patternConvLayers)
        temp = tf.keras.layers.Lambda(lambda x: x / 1e-2)(convs)
        softmax = self.softMaxLayer(temp)
        binweight = self.binWeightLayer(softmax)
        argmax = self.ArgMaxLayer(binweight)

        pvFeatures = self.applyLayerList(argmax,self.pvDenseLayers)

        if self.nlatent>0:
            pvPosition,latentFeatures = tf.keras.layers.Lambda(lambda x: [x[:,0:1],x[:,1:]],name='split_latent')(pvFeatures)
        else:
            pvPosition = pvFeatures
        
        z0Diff = tf.keras.layers.Lambda(lambda x: tf.stop_gradient(tf.expand_dims(tf.abs(x[0]-x[1]),2)),name='z0_diff')([self.inputTrackZ0,pvPosition])
        
        assocFeatures = [self.inputTrackFeatures,z0Diff]   

        if self.nlatent>0:
            assocFeatures.append(self.tiledTrackDimLayer(latentFeatures))  
            
        assocFeat = tf.keras.layers.Concatenate(axis=2,name='association_features')(assocFeatures)

        assocProbability = self.applyLayerList(assocFeat,self.assocLayers)
        #assocProbability = self.outputSoftmax(assocQProbability)


        
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
        
        #model.add_loss(tf.keras.layers.Lambda(q90loss)(weights))
        return model