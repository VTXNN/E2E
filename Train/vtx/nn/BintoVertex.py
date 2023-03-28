from math import ceil
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *

class BintoVertex(Layer):

    def __init__(self,nbins=256, start=-20.46912512, end=20.46912512, **kwargs):
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

        return z0,z0Index

class ZeroWeighting(Layer):

    def __init__(self, **kwargs):
        super(ZeroWeighting, self).__init__(**kwargs)

    def call(self, input, weight):
        inputsum =tf.math.reduce_sum(input,axis=2)
        inputsumzeros = tf.math.not_equal(inputsum, 0)
        zeros = tf.zeros_like(inputsumzeros,dtype=tf.float32)
        newweight = tf.where(inputsumzeros,tf.squeeze(weight,axis=2),zeros )
        newweight = tf.expand_dims(newweight,axis=2)

        return newweight
