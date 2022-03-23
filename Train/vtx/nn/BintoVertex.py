import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *

class BintoVertex(Layer):

    def __init__(self,nbins=256, start=-15, end=15, **kwargs):
        super(BintoVertex, self).__init__(**kwargs)
        self.nbins = nbins
        self.start = start
        self.end = end

    def call(self, x):
        start = tf.cast(self.start ,tf.float32)
        end =  tf.cast(self.end,tf.float32)
        nbins =  tf.cast(self.nbins,tf.float32)
        halfBinWidth = tf.cast((end-start)/(2*nbins),tf.float32)

        z0Index = tf.linalg.trace(x)

        z0Index = tf.expand_dims(z0Index,axis=1)
        z0 = start+(end-start)*z0Index/nbins 

        return z0,(tf.floor(z0Index))
