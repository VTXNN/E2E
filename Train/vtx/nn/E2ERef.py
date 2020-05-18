import tensorflow as tf
import keras
import vtx
import numpy

class E2ERef():
    def __init__(self,
        nbins=256,
        ntracks=250, 
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
        
        self.inputLayer = keras.layers.Input(shape=(self.ntracks,self.nfeatures))
        
        self.weightConvLayers = []
        for ilayer,filterSize in enumerate([10]):
            self.weightConvLayers.extend([
                keras.layers.Conv1D(
                    filterSize,
                    1,
                    padding='same',
                    activation=self.activation,
                    kernel_initializer='lecun_normal',
                    kernel_regularizer=keras.regularizers.l2(regloss),
                    name='weight_'+str(ilayer+1)
                ),
                keras.layers.Dropout(0.1),
            ])
            
        self.weightConvLayers.append(
            keras.layers.Conv1D(
                self.nweights,
                1,
                padding='same',
                activation='relu', #need to use relu here to remove negative weights
                kernel_initializer='lecun_normal',
                kernel_regularizer=keras.regularizers.l2(regloss),
                name='weight_final'
            ),
        )
        #self.weightConvLayers.append(keras.layers.Dropout(0.1))
        
        self.kdeLayer = vtx.nn.KDELayer(
            nbins=256,
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
                keras.layers.Conv1D(
                    filterSize,
                    kernelSize,
                    padding='same',
                    activation=self.activation,
                    kernel_initializer='lecun_normal',
                    kernel_regularizer=keras.regularizers.l2(regloss),
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
                keras.layers.Conv1D(
                    filterSize,
                    kernelSize,
                    strides=strides,
                    padding='same',
                    activation=self.activation,
                    kernel_initializer='lecun_normal',
                    kernel_regularizer=keras.regularizers.l2(regloss),
                    name='position_'+str(ilayer+1)
                )
            )

        self.positionDenseLayers = [
            keras.layers.Dense(
                1,
                activation=None,
                kernel_initializer='lecun_normal',
                kernel_regularizer=keras.regularizers.l2(regloss),
                name='position_final'
            )
        ]
        
        if self.nlatent>0:
            self.latentDenseLayers = [
                keras.layers.Dense(
                    self.nlatent,
                    activation=None,
                    kernel_initializer='lecun_normal',
                    kernel_regularizer=keras.regularizers.l2(regloss),
                )
            ]
        
        
               
        self.assocConvLayers = []
        for ilayer,filterSize in enumerate([20,20]):
            self.assocConvLayers.extend([
                keras.layers.Conv1D(
                    filterSize,
                    1,
                    padding='same',
                    activation=self.activation,
                    kernel_initializer='lecun_normal',
                    kernel_regularizer=keras.regularizers.l2(regloss),
                    name='association_'+str(ilayer)
                ),
                keras.layers.Dropout(0.1),
            ])
            
        self.assocConvLayers.extend([
            keras.layers.Conv1D(
                1,
                1,
                padding='same',
                activation='sigmoid',
                kernel_initializer='lecun_normal',
                kernel_regularizer=keras.regularizers.l2(regloss),
                name='association_final'
            )
        ])
        
    def applyLayerList(self, inputLayer, layerList):
        outputLayer = layerList[0](inputLayer)
        for layer in layerList[1:]:
            outputLayer = layer(outputLayer)
        return outputLayer
        
    def getInputs(self):
        return self.inputLayer

    def getTrackZ0(self,inputs):
        return keras.layers.Lambda(lambda x: x[:,:,0])(inputs)
        
    def getTrackFeatures(self,inputs):
        return keras.layers.Lambda(lambda x: x[:,:,1:])(inputs)
        
    def getHistWeights(self,inputs):
        weightFeatures = self.getTrackFeatures(inputs)
        weights = self.applyLayerList(weightFeatures,self.weightConvLayers)
        return weights         
        #return keras.layers.Lambda(lambda x: x[:,:,1:2])(inputs)
        
    def getHistValue(self,inputs):
        return self.getTrackZ0(inputs)
        
    def getHists(self,value,weights):
        return self.kdeLayer([value,weights])
        
    def getPVPosition(self,hists):
        #pattern [batch,bins,filters]
        pattern = self.applyLayerList(hists,self.patternConvLayers)
        '''
        def maxpos(x):
            #xT = tf.nn.relu(tf.transpose(x,[0,2,1])) #[batch,filters,nbins]
            halfBinWidth = 30./256.*0.5
            binCenter = tf.constant(numpy.linspace(-15+halfBinWidth,15-halfBinWidth,256),dtype=tf.float32)
            attention = tf.nn.softmax(x,axis=2)
            attentionT = tf.transpose(attention,[0,2,1])
            xT = tf.transpose(x,[0,2,1])            
            weighted = tf.reduce_sum(tf.multiply(xT,attentionT),axis=2)
            return weighted

        flattened = keras.layers.Lambda(maxpos)(pattern)
        '''
        permuted = keras.layers.Lambda(lambda x: tf.transpose(x,[0,2,1]))(pattern)
        positionConv = self.applyLayerList(permuted,self.positionConvLayers)
        flattened = keras.layers.Flatten()(positionConv)
        
        pvPosition = self.applyLayerList(flattened,self.positionDenseLayers)
        
        if self.nlatent>0:
            pvLatent = self.applyLayerList(flattened,self.latentDenseLayers)
            return pvPosition,pvLatent
        else:
            return pvPosition,None
        
            
    def addToFeatures(self,convFeatures,addFeatures):
        def tileFeatures(x):
            x = tf.reshape(
                tf.tile(x,[1,convFeatures.shape[1]]),
                [x.shape[0],convFeatures.shape[1],x.shape[1]]
            )
            return x
            
        tiled = keras.layers.Lambda(tileFeatures)(addFeatures)
        return keras.layers.Concatenate(axis=2)([conv,tiled])
            
    def getAssociationFeatures(self,inputs,pvPosition,pvLatent=None):
        z0 = self.getTrackZ0(inputs)
        features = self.getTrackFeatures(inputs)
        dz = keras.layers.Lambda(lambda x: tf.expand_dims(tf.abs(x[0]-x[1]),axis=2))([z0,pvPosition])
        if pvLatent!=None:
            #TODO
            pass
        
        associationFeatures = keras.layers.Concatenate(axis=2)([dz,features])
        return associationFeatures
        
    def getAssociation(self,associationFeatures):
        association = self.applyLayerList(associationFeatures, self.assocConvLayers)
        return keras.layers.Lambda(lambda x: x[:,:,0])(association)
    
    def createModel(self,optimizer):
        class Model():
            def __init__(self,network):
                self.network = network
                
                self.inputLayer = self.network.getInputs()
                self.histValueLayer = self.network.getHistValue(self.inputLayer)
                self.histWeightsLayer = self.network.getHistWeights(self.inputLayer)
                self.histsLayer = self.network.getHists(
                    self.histValueLayer,
                    self.histWeightsLayer
                )
                
                self.pvPositionLayer,self.pvlatentLayer = self.network.getPVPosition(self.histsLayer)
                
                
                self.histValueLayerFlipped = keras.layers.Lambda(lambda x: -x)(self.histValueLayer)
                self.histsLayerFlipped = self.network.getHists(
                    self.histValueLayerFlipped,
                    self.histWeightsLayer
                )
                self.pvPositionLayerFlipped,self.pvlatentLayerFlipped = self.network.getPVPosition(self.histsLayerFlipped)
                self.pvPositionLayerAveraged = keras.layers.Lambda(lambda x: 0.5*(x[0]-x[1]))([self.pvPositionLayer,self.pvPositionLayerFlipped])
                
                
                self.associationFeatureLayer = self.network.getAssociationFeatures(
                    self.inputLayer,
                    self.pvPositionLayerAveraged,
                    self.pvlatentLayer
                )
                self.associationLayer = self.network.getAssociation(self.associationFeatureLayer)
                
                
                self.weightModel = keras.models.Model(inputs=[self.inputLayer],outputs=[self.histWeightsLayer])
                self.histModel = keras.models.Model(inputs=[self.inputLayer],outputs=[self.histsLayer])
                self.pvPositionModel = keras.models.Model(inputs=[self.inputLayer],outputs=[self.pvPositionLayerAveraged])
                self.associationModel = keras.models.Model(inputs=[self.inputLayer],outputs=[self.pvPositionLayerAveraged,self.associationLayer])
                
                
                wq90 = tf.contrib.distributions.percentile(
                    self.histWeightsLayer,
                    q=90.,
                    axis=1,
                    interpolation='nearest',
                )
                self.associationModel.add_loss(0.01*tf.reduce_mean(tf.abs(wq90-1.)))
                
                def pseudohuber(x,d=0.05,s=1.,q=10.):
                    # d controls convexness at minimum (smaller = sharper)
                    # s is the linear slope for large values
                    # q is the percentage of values to cut out
                    
                    qLow = tf.contrib.distributions.percentile(
                        x,
                        q=0.5*q,
                        interpolation='nearest',
                    )
                    qHigh = tf.contrib.distributions.percentile(
                        x,
                        q=100.-0.5*q,
                        interpolation='nearest',
                    )
                    xmask = tf.math.logical_and(
                        tf.less(x,qHigh),
                        tf.greater(x,qLow)
                    )

                    xMasked = tf.boolean_mask(x,xmask)
                    
                    return s*d*(tf.reduce_mean(tf.sqrt(1+tf.square(xMasked/d)))-1)
                    
                def quantileLoss(x):
                    q5 = tf.contrib.distributions.percentile(
                        x,
                        q=5.,
                        interpolation='nearest',
                    )
                    q15 = tf.contrib.distributions.percentile(
                        x,
                        q=15.,
                        interpolation='nearest',
                    )
                    
                    q85 = tf.contrib.distributions.percentile(
                        x,
                        q=85.,
                        interpolation='nearest',
                    )
                    
                    q95 = tf.contrib.distributions.percentile(
                        x,
                        q=95.,
                        interpolation='nearest',
                    )
                    
                    return tf.square(q85-q15)+tf.abs(q95-q5)
                
                self.associationModel.compile(
                    optimizer,
                    loss=[
                        lambda x,y: pseudohuber(x-y),
                        keras.losses.binary_crossentropy
                    ],
                    metrics=[['mae'], ['acc']],
                    loss_weights = [1.,0.001]
                )
                
            def predictWeights(self,batch):
                pass
                
            def predictHists(self,batch):
                pass
                
            def predictPVPosition(self,batch):
                pass
                
            def predictAssociation(self,batch):
                pass
                
            def train_on_batch(self,batch):
                lossList = self.associationModel.train_on_batch([batch['X']],[batch['y_avg'],batch['assoc']])
                #print ("z0 loss: %.4f (mae: %.2f), assoc loss: %.4f (acc: %.2f%%)"%(lossList[1],lossList[3],lossList[2],100.*lossList[4]))
                return lossList[1]
                
            def test_on_batch(self,batch):
                lossList = self.associationModel.test_on_batch([batch['X']],[batch['y_avg'],batch['assoc']])
                #print ("z0 loss: %.4f (mae: %.2f), assoc loss: %.4f (acc: %.2f%%)"%(lossList[1],lossList[3],lossList[2],100.*lossList[4]))
                return lossList[1]
                
            def predict_on_batch(self,batch):
                return self.associationModel.predict_on_batch([batch['X']])
                
            def save_weights(self,path):
                self.associationModel.save_weights(path)
                
            def load_weights(self,path):
                self.associationModel.load_weights(path)
                
            def summary(self):
                self.associationModel.summary()
                
        return Model(self)

    

