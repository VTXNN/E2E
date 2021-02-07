import tensorflow as tf
import os


kde_histogram_module = tf.load_op_library(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'libKDEHistogram.so'
    )
)

class KDEHistogram(tf.keras.layers.Layer):
    

    @tf.RegisterGradient("KDEHistogram")
    def _sub_grad(op, grad):
        grad_values, grad_weights = kde_histogram_module.kde_histogram_grad(
            op.inputs[0],
            op.inputs[1],
            grad,
            nbins=op.get_attr("nbins"),
            start=op.get_attr("start"),
            end=op.get_attr("end"),
            kernel=op.get_attr("kernel"),
            bandwidth_grad=op.get_attr("bandwidth_grad"),
            add_overflow=op.get_attr("add_overflow"),
        )
        return [grad_values, grad_weights]

    def __init__(self,
        nbins=100,
        start=-1.,
        end=1.,
        kernel="flat",
        #kernel="triangle",
        bandwidth_hist = 0.15,
        bandwidth_grad = None,
        add_overflow = False,
        **kwargs
    ):
        super(KDEHistogram, self).__init__(**kwargs)
        self.supports_masking = False

        self.nbins = nbins
        self.start = start
        self.end = end
        self.kernel = kernel
        self.bandwidth_hist = bandwidth_hist
        if bandwidth_grad==None:
            self.bandwidth_grad = self.bandwidth_hist
        else:
            self.bandwidth_grad = bandwidth_grad
        self.add_overflow = add_overflow

    def build(self, input_shape):
        super(KDEHistogram, self).build(input_shape)

    def call(self, inputs, mask=None):
        return kde_histogram_module.kde_histogram(
            inputs[0],
            inputs[1],
            self.nbins,
            self.start,
            self.end,
            self.kernel,
            self.bandwidth_hist,
            self.bandwidth_grad,
            self.add_overflow
        )
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0][0],self.nbins)

    '''
    def get_config(self):
        config = {}
        base_config = super(KdeHistogram, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    '''


