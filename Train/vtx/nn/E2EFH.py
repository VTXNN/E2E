import tensorflow as tf
import tensorflow_probability as tfp
import vtx
import numpy

class E2EFH():
    def __init__(self,
        nbins=256,
        ntracks=200, 
        nweightfeatures=1,
        nfeatures=1, 
        nweights=1, 
        npattern=4,
        activation='relu',
        regloss=1e-10
    ):
        self.nbins = nbins
        self.ntracks = ntracks
        self.nweightfeatures = nweightfeatures
        self.nfeatures = nfeatures
        self.nweights = nweights
        self.npattern = npattern
        self.activation = activation
        
        self.inputWeightFeatures = tf.keras.layers.Input(shape=(self.ntracks,self.nweightfeatures),name='input_weight_features')
        self.inputTrackFeatures = tf.keras.layers.Input(shape=(self.ntracks,self.nfeatures),name='input_PV_track_features')
        self.inputTrackZ0 = tf.keras.layers.Input(shape=(self.ntracks),name='input_track_z0')

        self.InputDenseLayers = [
            tf.keras.layers.Dense(
                1,
                activation='linear',
                trainable = True,
                name='Input_weight',
                use_bias = False,
            )
        ]
        
        self.kdeLayer = vtx.nn.KDELayer(
            nbins=self.nbins,
            start=-15,
            end=15
        )
        
        
        self.patternConvLayers = []
        for ilayer,(filterSize,kernelSize) in enumerate([
            [1,3],
        ]):
            self.patternConvLayers.append(
                tf.keras.layers.Conv1D(
                    filterSize,
                    kernelSize,
                    padding='same',
                    activation='linear',
                    trainable=False,
                    use_bias= False,
                    name='pattern_'+str(ilayer+1)
                )
            )
            
        self.softMaxLayer = tf.keras.layers.Softmax(axis=1)

        self.binWeightLayer = tf.keras.layers.Dense(
                    256,
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
          
        self.assocLayers = []
        for ilayer,filterSize in enumerate([5,5]):
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
        
        weights = self.applyLayerList(self.inputWeightFeatures,self.InputDenseLayers)
        hists = self.kdeLayer([self.inputTrackZ0,weights])
        convs = self.applyLayerList(hists,self.patternConvLayers)
        temp = tf.keras.layers.Lambda(lambda x: x / 1e-2)(convs)
        softmax = self.softMaxLayer(temp)
        binweight = self.binWeightLayer(softmax)
        pvPosition = self.ArgMaxLayer(binweight)

        z0Diff = tf.keras.layers.Lambda(lambda x: tf.stop_gradient(tf.expand_dims(tf.abs(x[0]-x[1]),2)),name='z0_diff')([self.inputTrackZ0,pvPosition])
        
        assocFeatures = [self.inputTrackFeatures,z0Diff]     
            
        assocFeatures = tf.keras.layers.Concatenate(axis=2,name='association_features')(assocFeatures)

        assocProbability = self.applyLayerList(assocFeatures,self.assocLayers)
        
        model = tf.keras.Model(
            inputs=[self.inputTrackZ0,self.inputWeightFeatures,self.inputTrackFeatures],
            outputs=[pvPosition,assocProbability,weights]
        )
        return model