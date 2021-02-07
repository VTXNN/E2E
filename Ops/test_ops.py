import tensorflow as tf
import os
import math
import time
import numpy
import unittest

tf.compat.v1.disable_eager_execution()

from sklearn.neighbors import KernelDensity

from kde_histogram import KDEHistogram
from histogram_max import *


session = None
def get_session():
    global session
    if not session:
        session = tf.compat.v1.Session()
    return session


def makeModel(nelem,nbins,start,end,kernel,bandwidth):
    class TestModel():
        def __init__(self):
            self.nelem = nelem
            self.nbins = nbins
            self.start = start
            self.end = end
            self.kernel = kernel
            self.bandwidth = bandwidth

            self.values = tf.keras.layers.Input(shape=(self.nelem,))
            self.weights = tf.keras.layers.Input(shape=(self.nelem,))
            self.factors = tf.keras.layers.Input(shape=(self.nbins,))

            self.hist = KDEHistogram(
                nbins=self.nbins,
                start=self.start,
                end=self.end,
                kernel="flat",
                bandwidth_hist=self.bandwidth,
                bandwidth_grad=self.bandwidth,
                add_overflow = False
            )([self.values,self.weights])

            self.model = tf.keras.Model(inputs=[self.values,self.weights],outputs=[self.hist])

            score = tf.keras.layers.Lambda(lambda x: tf.multiply(x[0],x[1]))([self.hist,self.factors])
            self.score = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x))(score)

            self.model = tf.keras.Model(inputs=[self.values,self.weights],outputs=[self.hist])
            self.model.compile(loss='mse', optimizer='sgd') #dummy

            self.gradients = tf.gradients(self.score,[self.values,self.weights])
            
        def getHist(self,valuesArray,weightsArray):
            return self.model.predict_on_batch([valuesArray,weightsArray])
            
        def getScore(self,valuesArray,weightsArray,factorsArray):
            sess = get_session()
            scoreArray = sess.run(self.score, feed_dict = {
                self.values: valuesArray,
                self.weights: weightsArray,
                self.factors: factorsArray
            })
            return scoreArray
        def getGrad(self,valuesArray,weightsArray,factorsArray):
            sess = get_session()
            gradientsList = sess.run(self.gradients, feed_dict = {
                self.values: valuesArray,
                self.weights: weightsArray,
                self.factors: factorsArray
            })
            return gradientsList
            
    return TestModel()

class KDETest(unittest.TestCase):

    def testHist(self):
        for nelem in [1,23]:
            for nbins in [1,2,17]:
                for start in [-10,0,3]:
                    for d in [1,11]:
                        #nelem = 10
                        #nbins = 2 
                        #start = -10
                        end = start+d
                        kernel='flat'
                        bandwidth = 1e-12
        
                        testModel = makeModel(nelem,nbins,start,end,kernel,bandwidth)

                        for i in range(0,5):
                            valuesArray = numpy.zeros((1,nelem))
                            weightsArray = numpy.zeros((1,nelem))
                            factorsArray = numpy.zeros((1,nbins))
                            
                            for j in range(nelem):
                                valuesArray[0,j] = i*j+j*0.2-i*0.3+i*i+0.01
                                weightsArray[0,j] = i*i-10*j+i*j*j-0.25*i-2
                            for j in range(nbins):
                                factorsArray[0,j] = i*i*j-j*0.5+i*i*0.07-3
                                 
                            histArray = testModel.getHist(valuesArray,weightsArray)[0]
                            histArrayRef = numpy.histogram(
                                valuesArray[0,:], 
                                bins=nbins, 
                                range=(start,end),
                                weights=weightsArray[0,:]
                            )
                            for j in range(nbins):
                                self.assertEqual(histArray[j],histArrayRef[0][j])
                        
    
    def testGrad(self):
        for nelem in [1,11]:
            for nbins in [1,17]:
                for start in [-10,0,3]:
                    for d in [1,11]:
                        for bandwidth in [1e-12,0.1,2]:
                            #nelem = 10
                            #nbins = 2 
                            #start = -10
                            end = start+d
                            kernel='flat'
                            
                            testModel = makeModel(nelem,nbins,start,end,kernel,bandwidth)
                        
                            sess = get_session()

                            for i in range(3):
                                valuesArray = numpy.zeros((1,nelem))
                                weightsArray = numpy.zeros((1,nelem))
                                factorsArray = numpy.zeros((1,nbins))
                                
                                for j in range(nelem):
                                    valuesArray[0,j] = i*j+j*0.2-i*0.3+i*i+0.01
                                    weightsArray[0,j] = i*i-10*j+i*j*j-0.25*i-2
                                for j in range(nbins):
                                    factorsArray[0,j] = i*i*j-j*0.5+i*i*0.07-3
                                
                                gradientsList = testModel.getGrad(
                                    valuesArray,
                                    weightsArray,
                                    factorsArray
                                )
                                
                                
                                for j in range(nelem):
                                    hV = 1e-2*(end-start)/nbins
                                    hW = math.fabs(weightsArray[0,j]*1e-2)+1e-6
                                    diff = numpy.zeros(valuesArray.shape)
                                    diff[0,j]=1.
                                    scoreValueDiff = (testModel.getScore(
                                        valuesArray+diff*hV,
                                        weightsArray,
                                        factorsArray
                                    ) - testModel.getScore(
                                        valuesArray-diff*hV,
                                        weightsArray,
                                        factorsArray
                                    ))/(2*hV)
                                    scoreWeightDiff = (testModel.getScore(
                                        valuesArray,
                                        weightsArray+diff*hW,
                                        factorsArray
                                    ) - testModel.getScore(
                                        valuesArray,
                                        weightsArray-diff*hW,
                                        factorsArray
                                    ))/(2*hW)
                                    '''
                                    if bandwidth>hV:
                                        print (
                                            j,
                                            gradientsList[0][0,j],
                                            scoreValueDiff,
                                            gradientsList[0][0,j]-scoreValueDiff,
                                            hV
                                        )
                                    
                                        self.assertTrue(
                                            math.fabs(gradientsList[0][0,j]-scoreValueDiff)<(20*hV)
                                        )
                                    '''
                                    self.assertTrue(
                                        math.fabs(gradientsList[1][0,j]-scoreWeightDiff)<(2*hW)
                                    )
                
    
class HistogramMaxSampleTest(unittest.TestCase):
    def testHistSingle(self):
        sess = get_session()
        for n in range(2,200,10):
            hists = tf.compat.v1.placeholder(tf.float32, shape=(1, n,1))
            histMax = histogram_max_sample_module.histogram_max_sample(hists)
            for i in range(hists.shape[1]):
                val = numpy.zeros(hists.shape)
                val[0,i,0] = 1
                self.assertEqual(sess.run(histMax,feed_dict={
                    hists:val
                })[0,0],i)
                
            
    def testHistSample(self):
        sess = get_session()
        hists = tf.compat.v1.placeholder(tf.float32, shape=(100, 200,1))
        histMax = histogram_max_sample_module.histogram_max_sample(hists)
        
        val = numpy.zeros(hists.shape)
        
        for b in range(hists.shape[0]):
            for n in range(5):
                i = int(numpy.random.uniform(0,int(hists.shape[1])))
                val[b,i,0] = numpy.random.uniform(0.1,0.9)
        val/=numpy.sum(val,axis=1,keepdims=True)
        
        
        result = numpy.zeros(hists.shape)
        
        for t in range(10000):
            sampled = sess.run(histMax,feed_dict={hists:val})
            for b in range(hists.shape[0]):
                result[b,int(sampled[b,0]),0] += 1.
                
        result/=numpy.sum(result,axis=1,keepdims=True)
         
        p = 0
        f = 0      
        for b in range(hists.shape[0]):
            for i in range(hists.shape[1]):
                if val[b,i,0]>0.01:
                    if math.fabs(val[b,i,0]-result[b,i,0])/val[b,i,0]<0.1:
                        p += 1
                    else:
                        f += 1
        #require >90% to pass
        self.assertTrue(f<0.1*p)
        
        

if __name__ == '__main__':
    test_suite = unittest.TestSuite()
    test_suite.addTest(KDETest('testHist'))
    test_suite.addTest(KDETest('testGrad'))
    test_suite.addTest(HistogramMaxSampleTest('testHistSingle'))
    test_suite.addTest(HistogramMaxSampleTest('testHistSample'))
    unittest.runner.TextTestRunner(verbosity=2).run(test_suite)
    
    
