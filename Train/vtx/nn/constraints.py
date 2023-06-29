from tensorflow.keras.constraints import *
from tensorflow.keras.layers import multiply
from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow_probability as tfp

class ZeroSomeWeights(Constraint):
    """ZeroSomeWeights weight constraint.
    Constrains certain weights incident to each hidden unit
    to be zero.
    # Arguments
        binary_tensor: binary tensor of 0 or 1s corresponding to which weights to zero.
    """

    def __init__(self, binary_tensor=None):
        self.binary_tensor = binary_tensor

    def __call__(self, w):
        if self.binary_tensor is not None:
            w = w * self.binary_tensor
        return w

    def get_config(self):
        return {'binary_tensor': self.binary_tensor}

# Aliases.

zero_some_weights = ZeroSomeWeights

class Custom_quartile_Loss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
    def call(self, y_true, y_pred):       
        qz0 = tfp.stats.percentile(y_pred-y_true,[32,68])
        
        return abs(qz0[0] + qz0[1])


class Custom_huber_Loss(tf.keras.losses.Loss):
    def __init__(self,threshold=1.0,mean_weight=1.0):
        self.threshold = threshold
        self.mean_weight = mean_weight
        super().__init__()
    def call(self,y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2 + self.mean_weight*tf.abs(tf.reduce_mean(error))
        linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2 + self.mean_weight*tf.abs(tf.reduce_mean(error))
        return tf.where(is_small_error, squared_loss, linear_loss)