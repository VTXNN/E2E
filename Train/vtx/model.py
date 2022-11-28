import tensorflow as tf
import numpy as np
import sklearn.metrics as metrics
import math
import pickle

def linear_res_function(x,return_bool = False):
        if return_bool:
            return np.full_like(x,True)

        else:
            return np.ones_like(x)

def eta_res_function(eta):
        res = 0.1 + 0.2*eta**2
        return 1/res
    
def MVA_res_function(MVA,threshold=0.3,return_bool = False):
        res = MVA > threshold
        if return_bool:
            return res
        else:
            return tf.cast(res,tf.float32)

def chi_res_function(chi2rphi,chi2rz,bendchi2,return_bool = False):
        qrphi = chi2rphi < 12 
        qrz =  chi2rz < 9 
        qbend = bendchi2 < 4
        q = np.logical_and(qrphi,qrz)
        q = np.logical_and(q,qbend)
        if return_bool:
            return q
        else:
            return tf.cast(q,tf.float32)

def fake_res_function(fakes,return_bool = False):
        res = fakes != 0
        if return_bool:
            return res
        else:
            return tf.cast(res,tf.float32)


class GTTalgo(object):
    def __init__(self,kf,config):
        self.kf = kf
        self.config = config

        self.num_bins = 256
        self.range = (-15,15)

        self.predictedZ0 = []
        self.predictedAssoc = []
        self.predictedMET = []
        self.predictedMETphi = []

        self.predictedZ0_currentbatch = None
        self.predictedAssoc_currentbatch = None

        self.METquality_function = None

        self.result_dict = {}

        self.saveDirectory = ""
        self.name = ""

    def predictZ0():
        pass
    def predictAssoc():
         pass
    def predictMET(self,pt,phi,quality_var):
        met_pt_list = []
        met_phi_list = []

        def assoc_function(Assoc):
            res = Assoc > self.threshold
            return res

        for ibatch in range(pt.shape[0]):
            assoc = assoc_function(self.predictedAssoc_currentbatch[ibatch])
            selection = np.logical_and(assoc,self.METquality_function(quality_var))

            newpt = pt[ibatch][selection]
            newphi = phi[ibatch][selection]

            met_px = np.sum(newpt*np.cos(newphi))
            met_py = np.sum(newpt*np.sin(newphi))
            met_pt_list.append(math.sqrt(met_px**2+met_py**2))
            met_phi_list.append(math.atan2(met_py,met_px))

        self.predictedMET.append(np.array(met_pt_list,dtype=np.float32))
        self.predictedMETphi.append(np.array(met_phi_list,dtype=np.float32))

    def fillDict(self):
        self.result_dict['z0']     = np.concatenate(self.predictedZ0).ravel()
        self.result_dict['Assoc']  = np.concatenate(self.predictedAssoc).ravel()
        self.result_dict['MET']    = np.concatenate(self.predictedMET).ravel()
        self.result_dict['METphi'] = np.concatenate(self.predictedMETphi).ravel()

    def saveDict(self):
        np.save(self.saveDirectory+self.name+".npy",self.result_dict)

    def loadDict(self):
        self.result_dict = np.load(self.saveDirectory+self.name+".npy")

class NNGTTalgo(GTTalgo):
    def __init__(self,
                 kf,
                 model):
        super().__init__(kf)

        self.model = model
        self.NNweights = []

    def predictZ0(self,z0,weightfeatures,trackfeatures):
        NNpredictedZ0, NNpredictedAssoc, NNWeights = self.predict_on_batch([z0,weightfeatures,trackfeatures])
        self.predictedZ0_currentbatch = np.array(NNpredictedZ0,dtype=np.float32)
        self.predictedZ0.append(self.predictedZ0_currentbatch)

        NNpredictedAssoc_temp = tf.math.divide( tf.math.subtract( NNpredictedAssoc,tf.reduce_min(NNpredictedAssoc)), 
                                                    tf.math.subtract( tf.reduce_max(NNpredictedAssoc), tf.reduce_min(NNpredictedAssoc) ))

        self.predictedAssoc_currentbatch = np.array(NNpredictedAssoc_temp,dtype=np.float32)
        self.predictedAssoc.append(self.predictedAssoc_currentbatch)      

        self.NNweights.append(np.array(NNWeights,dtype=np.float32))

    def fillDict(self):
        self.result_dict['z0']     = np.concatenate(self.predictedZ0).ravel()
        self.result_dict['Assoc']  = np.concatenate(self.predictedAssoc).ravel()
        self.result_dict['MET']    = np.concatenate(self.predictedMET).ravel()
        self.result_dict['METphi'] = np.concatenate(self.predictedMETphi).ravel()
        self.result_dict['Weights'] = np.concatenate(self.NNweights).ravel()


class FHGTTalgo(GTTalgo):
    def __init__(self,kf,res_function):
        super().__init__(kf)

        if (kf == "NewKF") | (kf == "NewKF_intZ"):
            self.deltaz_bins = np.array([0.0,0.41,0.55,0.66,0.825,1.1,1.76,0.0])
        elif (kf == "OldKF") | (kf == "OldKF_intZ"):
            self.deltaz_bins = np.array([0.0,0.37,0.5,0.6,0.75,1.0,1.6,0.0])

        self.eta_bins = np.array([0.0,0.7,1.0,1.2,1.6,2.0,2.4])

        self.res_func = lambda : res_function


        def predictZ0(self,value,weight,res_var,return_index):
            z0List = []
            halfBinWidth = 0.5*(self.range[1]-self.range[0])/self.num_bins

            for ibatch in range(value.shape[0]):
                hist,bin_edges = np.histogram(value[ibatch],self.num_bins,range=self.range,weights=weight[ibatch]*self.res_func(res_var))
                hist = np.convolve(hist,[1,1,1],mode='same')
                z0Index = np.argmax(hist)
                if return_index:
                    z0List.append([z0Index])
                else:
                    z0 = self.range[0]+(self.range[1]-self.range[0])*z0Index/self.num_bins+halfBinWidth
                    z0List.append([z0])
            
            self.predictedZ0_currentbatch = np.array(z0List,dtype=np.float32)
            self.predictedZ0.append(self.predictedZ0_currentbatch)


        def predictAssoc(self,trk_z0,trk_eta,res_var):
            eta_bin = np.digitize(abs(trk_eta),self.eta_bins)
            deltaz = abs(trk_z0 - self.predictedZ0_currentbatch)

            assoc = (deltaz < self.deltaz_bins[eta_bin]) & self.res_func(res_var,return_bool=True)

            self.predictedAssoc_currentbatch = np.array(assoc,dtype=np.float32)
            self.predictedAssoc.append(self.predictedAssoc_currentbatch)


class TrueAlgo(GTTalgo):
    def __init__(self,kf):
        super().__init__(kf)

    def predictZ0(self,pv):
        self.predictedZ0_currentbatch = np.array(pv,dtype=np.float32)
        self.predictedZ0.append(self.predictedZ0_currentbatch)
    def predictAssoc(self,assoc):
        self.predictedAssoc_currentbatch = np.array(assoc,dtype=np.float32)
        self.predictedAssoc.append(self.predictedAssoc_currentbatch)
