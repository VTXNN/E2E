import uproot3 as uproot
import tensorflow as tf
import numpy as np
import math
from math import isnan
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

###### USAGE #####
# python convertTFRecords.py path-to-input-file/file.root path-to-output-folder

f = uproot.open(sys.argv[1])
skip_noPV_events = sys.argv[2]

branches = [ 
    'trk_gtt_pt',
    'trk_gtt_eta',
    'trk_gtt_phi',
    #'trk_gtt_z0',
    'trk_fake', 
    'trk_pt',
    'trk_z0',
    'trk_eta',
    'trk_nstub',
    'pv_MC',
    "trk_word_chi2rphi",
    "trk_word_chi2rz",
    "trk_word_bendchi2",
    "trk_word_MVAquality",
    "trk_MVA1",
    "tp_eventid",
    "tp_charge",
    "tp_d0",
    "tp_pt",
    "tp_phi",
    #"gen_pt",
    #"gen_phi",
    #"gen_pdgid"
]

trackFeatures = [
    'trk_z0',
    'trk_gtt_pt',
    'trk_eta',
    'trk_gtt_phi',
    'int_z0',
    'trk_fake',
    "trk_word_chi2rphi",
    "trk_word_chi2rz",
    "trk_word_bendchi2",
    "trk_word_MVAquality",
    "trk_MVA1",
    'trk_nstub',
]

max_z0 = 15

def unpackbits(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits]) # decode as little endian
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])[::-1] #flip for big endian
    
def padArray(x,n,num=0):
    arr = num*np.ones((n,),dtype=np.float32)
    for i in range(min(n,len(x))):
        arr[i] = x[i]
    return arr
    
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(
        value=np.nan_to_num(value.flatten(), nan=0.0, posinf=0.0, neginf=0.0)
    ))


eta_bins = np.linspace(0,2.5,127)
res_bins = np.floor(1/(0.1+0.2*(eta_bins**2)))

res_bins = np.append(res_bins,0)

chi2rz_bins   = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0, 20.0, 50.0,np.inf])
chi2rphi_bins = np.array([0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 10.0, 15.0, 20.0, 35.0, 60.0, 200.0, np.inf])
bendchi2_bins = np.array([0, 0.75, 1.0, 1.5, 2.25, 3.5, 5.0, 20.0,np.inf])

nMaxTracks = 250

chunkread = 5000


flip = 1

for ibatch,data in enumerate(f['L1TrackNtuple']['eventTree'].iterate(branches,entrysteps=chunkread)):
    data = {k.decode('utf-8'):v for k,v in data.items() }
    print ('processing batch:',ibatch+1,'/',math.ceil(1.*len(f['L1TrackNtuple']['eventTree'])/chunkread))
    
    tfwriter = tf.io.TFRecordWriter(
        '%s/data%i.tfrecord'%(sys.argv[2],ibatch),
        options=tf.io.TFRecordOptions(
            compression_type='GZIP',
            compression_level = 4,
            input_buffer_size=100,
            output_buffer_size=100,
            mem_level = 8,
        )
    )
    
    #### THIS NEEDS TO BE REMOVED AT SOME POINT #####
    #ad-hoc correction of track z0

    data['round_z0'] = round(((flip*data['trk_z0']+max_z0 )*256/(max_z0*2)),2)
    data['int_z0'] = np.floor(data['round_z0'] )

    #################################################
    
    tfData = {}
    

    for iev in range(len(data['trk_pt'])):
        
        selectPVTPs = ((data['tp_eventid'][iev]==0) & (data['tp_charge'][iev]!=0))
        #tp met
        tp_met_px = np.sum(data['tp_pt'][iev][selectPVTPs]*np.cos(data['tp_phi'][iev][selectPVTPs]))
        tp_met_py = np.sum(data['tp_pt'][iev][selectPVTPs]*np.sin(data['tp_phi'][iev][selectPVTPs]))
        tp_met_pt = math.sqrt(tp_met_px**2+tp_met_py**2)
        tp_met_phi = math.atan2(tp_met_py,tp_met_px)
        
        tfData['tp_met_pt'] = _float_feature(np.array([tp_met_pt],np.float32))
        tfData['tp_met_phi'] = _float_feature(np.array([tp_met_phi],np.float32))

        selectTracksInZ0Range = (abs(data['trk_z0'][iev]) <= 100)

        #calc PV position as pt-weighted z0 average of PV tracks
        selectPVTracks = (data['trk_fake'][iev]==1)
        selectPUTracks = (data['trk_fake'][iev]==2)

        # selectPVTracks = (data['trk_fake'][iev]==1)
        # selectPUTracks = (data['trk_fake'][iev]==0)

        if (np.sum(1.*selectPVTracks)<1):
            if skip_noPV_events:
                continue
            else:
                PVtrack_weight = 0 
                PUtrack_weight = 1

        else:
            PVtrack_weight = (1/len(data['trk_pt'][iev][selectTracksInZ0Range][selectPVTracks]))*((len(data['trk_pt'][iev][selectTracksInZ0Range]))/2)
            PUtrack_weight = (1/len(data['trk_pt'][iev][selectTracksInZ0Range][selectPUTracks]))*((len(data['trk_pt'][iev][selectTracksInZ0Range]))/2)
        
        tfData['trk_fromPV'] = _float_feature(padArray(1.*selectPVTracks*selectTracksInZ0Range,nMaxTracks))

       
        track_weight = np.where((data['trk_fake'][iev][selectTracksInZ0Range]==1), PVtrack_weight, PUtrack_weight)

        tfData['trk_class_weight'] = _float_feature(padArray(track_weight,nMaxTracks))
        pvz0 = data["pv_MC"][iev]

        res = res_bins[np.digitize(abs(data['trk_eta'][iev][selectTracksInZ0Range]),eta_bins)]

        tfData['trk_z0_res']= _float_feature(padArray(np.array(res,np.float32),nMaxTracks)) 


        tfData['pvz0'] = _float_feature(np.array(flip*pvz0,np.float32))
        trk_word_pT = data['trk_gtt_pt'][iev][selectTracksInZ0Range]
        trk_word_pT = np.clip(trk_word_pT,0, 128)
        trk_word_eta = abs(data['trk_gtt_eta'][iev][selectTracksInZ0Range])

        tfData['trk_word_pT'] = _float_feature(padArray(np.array(trk_word_pT,np.float32),nMaxTracks,num=0))
        tfData['trk_word_eta'] = _float_feature(padArray(np.array(trk_word_eta,np.float32),nMaxTracks,num=0))

        for trackFeature in trackFeatures:
            tfData[trackFeature] = _float_feature(padArray(data[trackFeature][iev][selectTracksInZ0Range],nMaxTracks))
        tfData['trk_z0'] =  _float_feature(padArray(np.array(flip*data['trk_z0'][iev][selectTracksInZ0Range],np.float32),nMaxTracks)) 
        tfData['trk_eta'] =  _float_feature(padArray(np.array(flip*data['trk_eta'][iev][selectTracksInZ0Range],np.float32),nMaxTracks)) 

        example = tf.train.Example(features = tf.train.Features(feature = tfData))
        tfwriter.write(example.SerializeToString())
        
    tfwriter.close()
    
