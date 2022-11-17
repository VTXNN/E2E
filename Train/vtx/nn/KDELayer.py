import tensorflow as tf
import vtxops

class KDELayer(tf.keras.layers.Layer):
    def __init__(self, nbins=256, start=-20.46912512, end=20.46912512, kernel='flat', bandwidth=1e-12, add_overflow=True, **kwargs):
        self.nbins = nbins
        self.start = start
        self.end = end
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.add_overflow = add_overflow
        super(KDELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape)==2
        
        value_shape, weights_shape = input_shape
        
        assert(len(value_shape)==2)
        assert(len(weights_shape)==3)
        
        super(KDELayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        assert len(x)==2
        
        value, weights = x
        
        assert(len(value.shape)==2)
        assert(len(weights.shape)==3)
        
        hists = []
        
        for weight in tf.unstack(weights,axis=2):
            hists.append(
                vtxops.KDEHistogram(
                    nbins=self.nbins,
                    start=self.start,
                    end=self.end,
                    kernel=self.kernel,
                    bandwidth_hist=self.bandwidth,
                    bandwidth_grad=self.bandwidth,
                    add_overflow = self.add_overflow
                )([value,weight])
            )
        
        if len(hists)==1:
            return tf.expand_dims(hists[0],axis=2)
        else:
            return tf.stack(hists,axis=2)
        
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape)==2
        
        value_shape, weights_shape = input_shape
        return (weights_shape[0], self.nbins, weights_shape[2])
        

