import tensorflow as tf
import tensorflow_probability as tfp
import vtx
import numpy

class E2ERef():
    def __init__(self,
        nbins=256,
        ntracks=200, 
        nfeatures=10, 
        nweights=1, 
        nlatent=0, 
        activation='relu',
        regloss=1e-10
    ):
        self.nbins = nbins
        self.ntracks = ntracks
        self.nfeatures = nfeatures
        self.nweights = nweights
        self.nlatent = nlatent
        self.activation = activation
        
        self.inputTrackFeatures = tf.keras.layers.Input(shape=(self.ntracks,self.nfeatures),name='input_track_features')
        self.inputTrackZ0 = tf.keras.layers.Input(shape=(self.ntracks),name='input_track_z0')
        
        self.weightLayers = []
        for ilayer,nodes in enumerate([10]):
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
            [16,4],
            [16,4],
            [16,4],
            [16,4],
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
            [16,1,1],
            [16,1,1],
            [8,16,1],
            [8,1,1],
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
        
    def createE2EModel(self):
        weights = self.applyLayerList(self.inputTrackFeatures,self.weightLayers)
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
        
        z0Diff = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.abs(x[0]-x[1]),2),name='z0_diff')([self.inputTrackZ0,pvPosition])
        assocFeatures = [self.inputTrackFeatures,z0Diff]
        if self.nlatent>0:
            assocFeatures.append(self.tiledTrackDimLayer(latentFeatures))
            
            
        assocFeatures = tf.keras.layers.Concatenate(axis=2,name='association_features')(assocFeatures)

        assocProbability = self.applyLayerList(assocFeatures,self.assocLayers)
        
        model = tf.keras.Model(
            inputs=[self.inputTrackZ0,self.inputTrackFeatures],
            outputs=[pvPosition,assocProbability]
        )
        wq90 = tfp.stats.percentile(
            weights,
            q=90.,
            axis=1,
            interpolation='nearest',
        )
        model.add_loss(tf.reduce_mean(0.1*tf.square(wq90-1.)))
        return model
   
    

