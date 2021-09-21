import tensorflow as tf
import tensorflow_probability as tfp
import vtx
import numpy

class E2EReduced():
    def __init__(self,
        nbins=256,
        ntracks=200, 
        nweightfeatures=2,
        nfeatures=10, 
        nweights=1, 
        npattern=4,
        nlatent=0, 
        activation='relu',
        regloss=1e-10,
        l1loss=0
    ):
        self.nbins = nbins
        self.ntracks = ntracks
        self.nweightfeatures = nweightfeatures
        self.nfeatures = nfeatures
        self.nweights = nweights
        self.npattern = npattern
        self.nlatent = nlatent
        self.activation = activation
        
        self.inputWeightFeatures = tf.keras.layers.Input(shape=(self.ntracks,self.nweightfeatures),name='input_weight_features')
        self.inputTrackFeatures = tf.keras.layers.Input(shape=(self.ntracks,self.nfeatures),name='input_PV_track_features')
        self.inputTrackZ0 = tf.keras.layers.Input(shape=(self.ntracks),name='input_track_z0')

        self.weightLayers = []
        for ilayer,nodes in enumerate([10,10]):
            self.weightLayers.extend([
                tf.keras.layers.Dense(
                    nodes,
                    activation=self.activation,
                    kernel_initializer='lecun_normal',
                    kernel_regularizer=tf.keras.regularizers.l2(regloss),
                    name='weight_'+str(ilayer+1)
                ),
                tf.keras.layers.Dropout(0.1),
            ])
            
        self.weightLayers.append(
            tf.keras.layers.Dense(
                self.nweights,
                activation='relu', #need to use relu here to remove negative weights
                kernel_initializer='lecun_normal',
                kernel_regularizer=tf.keras.regularizers.l2(regloss),
                name='weight_final'
            ),
        )

        
        self.kdeLayer = vtx.nn.KDELayer(
            nbins=self.nbins,
            start=-15,
            end=15
        )
        
        
        self.patternConvLayers = []
        for ilayer,(filterSize,kernelSize) in enumerate([
            [4,4],
        ]):
            self.patternConvLayers.append(
                tf.keras.layers.Conv1D(
                    filterSize,
                    kernelSize,
                    padding='same',
                    activation=self.activation,
                    kernel_initializer='lecun_normal',
                    kernel_regularizer=tf.keras.regularizers.l2(regloss),
                    name='pattern_'+str(ilayer+1)
                )
            )
            
            
        self.positionConvLayers = []
        for ilayer,(filterSize,kernelSize,strides) in enumerate([
            [4,1,1],
        ]):
            self.positionConvLayers.append(
                tf.keras.layers.Conv1D(
                    filterSize,
                    kernelSize,
                    strides=strides,
                    padding='same',
                    activation=self.activation,
                    kernel_initializer='lecun_normal',
                    kernel_regularizer=tf.keras.regularizers.l2(regloss),
                    name='position_'+str(ilayer+1)
                )
            )

        self.pvDenseLayers = [
            tf.keras.layers.Dense(
                1+self.nlatent,
                activation=None,
                kernel_initializer='lecun_normal',
                kernel_regularizer=tf.keras.regularizers.l2(regloss),
                name='position_final'
            )
        ]
          
        self.assocLayers = []
        for ilayer,filterSize in enumerate([20,20]):
            self.assocLayers.extend([
                tf.keras.layers.Dense(
                    filterSize,
                    activation=self.activation,
                    kernel_initializer='lecun_normal',
                    kernel_regularizer=tf.keras.regularizers.l2(regloss),
                    name='association_'+str(ilayer)
                ),
                tf.keras.layers.Dropout(0.1),
            ])
            
        self.assocLayers.extend([
            tf.keras.layers.Dense(
                1,
                activation=None,
                kernel_initializer='lecun_normal',
                kernel_regularizer=tf.keras.regularizers.l2(regloss),
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
        weightInput = tf.keras.layers.Input(shape=(self.nweightfeatures),name="weight")
        weights = self.applyLayerList(weightInput,self.weightLayers)
        return tf.keras.Model(inputs=[weightInput],outputs=[weights])
    
    def createPatternModel(self):
        histInput = tf.keras.layers.Input(shape=(self.nbins,self.nweights),name="hist")
        pattern = self.applyLayerList(histInput,self.patternConvLayers)
        return tf.keras.Model(inputs=[histInput],outputs=[pattern])
    
    def createPositionModel(self):
        patternInput = tf.keras.layers.Input(shape=(self.npattern,self.nbins),name="pattern")
        positionConv = self.applyLayerList(patternInput,self.positionConvLayers)
        flattened = tf.keras.layers.Flatten()(positionConv)
        pvFeatures = self.applyLayerList(flattened,self.pvDenseLayers)
        return tf.keras.Model(inputs=[patternInput],outputs=[pvFeatures])
    
    def createAssociationModel(self):
        assocInput = tf.keras.layers.Input(shape=(self.nfeatures+1+self.nlatent),name="assoc")
        assocProbability = self.applyLayerList(assocInput,self.assocLayers)
        return tf.keras.Model(inputs=[assocInput],outputs=[assocProbability])
        
    def createE2EModel(self):
        weights = self.applyLayerList(self.inputWeightFeatures,self.weightLayers)
        hists = self.kdeLayer([self.inputTrackZ0,weights])
        pattern = self.applyLayerList(hists,self.patternConvLayers)
        
        permuted = tf.keras.layers.Lambda(lambda x: tf.transpose(x,[0,2,1]))(pattern)
        positionConv = self.applyLayerList(permuted,self.positionConvLayers)
        flattened = tf.keras.layers.Flatten()(positionConv)
        
        pvFeatures = self.applyLayerList(flattened,self.pvDenseLayers)
        
        if self.nlatent>0:
            pvPosition,latentFeatures = tf.keras.layers.Lambda(lambda x: [x[:,0:1],x[:,1:]],name='split_latent')(pvFeatures)
        else:
            pvPosition = pvFeatures
        
        z0Diff = tf.keras.layers.Lambda(lambda x: tf.stop_gradient(tf.expand_dims(tf.abs(x[0]-x[1]),2)),name='z0_diff')([self.inputTrackZ0,pvPosition])
        
        assocFeatures = [self.inputTrackFeatures,z0Diff]
        if self.nlatent>0:
            assocFeatures.append(self.tiledTrackDimLayer(latentFeatures))
            
            
        assocFeatures = tf.keras.layers.Concatenate(axis=2,name='association_features')(assocFeatures)

        assocProbability = self.applyLayerList(assocFeatures,self.assocLayers)
        
        model = tf.keras.Model(
            inputs=[self.inputTrackZ0,self.inputWeightFeatures,self.inputTrackFeatures],
            outputs=[pvPosition,pvPosition,assocProbability,assocProbability,weights,hists]
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