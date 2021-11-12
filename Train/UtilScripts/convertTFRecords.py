import uproot3 as uproot
import tensorflow as tf
import numpy as np
import math
from math import isnan
import sys

tf.compat.v1.disable_eager_execution()

#f = uproot.open("/vols/cms/cb719/VertexDatasets/OldKF_TTbar_170K_quality.root")

KFname =sys.argv[1]
filename = sys.argv[2]
f = uproot.open("/home/cebrown/Documents/Datasets/VertexDatasets/"+KFname+"_"+filename+".root")
#print (sorted(f['L1TrackNtuple']['eventTree'].keys()))

branches = [
    'tp_d0',
    'tp_d0_prod',
    'tp_dxy',
    'tp_eta', 
    'tp_eventid',
    'tp_nmatch', 
    'tp_nstub', 
    'tp_pdgid', 
    'tp_phi', 
    'tp_pt', 
    'tp_z0', 
    'tp_z0_prod', 
    'trk_MVA1', 
    'trk_bendchi2',
    'trk_chi2', 
    'trk_chi2rphi', 
    'trk_chi2rz', 
    'trk_combinatoric', 
    'trk_d0', 
    'trk_dhits', 
    'trk_eta', 
    'trk_fake', 
    'trk_genuine', 
    'trk_hitpattern', 
    'trk_lhits', 
    'trk_loose', 
    'trk_matchtp_dxy', 
    'trk_matchtp_eta', 
    'trk_matchtp_pdgid', 
    'trk_matchtp_phi',
    'trk_matchtp_pt',
    'trk_matchtp_z0', 
    'trk_nstub', 
    'trk_phi',
    'trk_phiSector',
    'trk_pt',
    'trk_seed',
    'trk_unknown',
    'trk_z0',
    'pv_MC',
    'genMETPx',
    'genMETPy',
    'genMET',
    'genMETPhi',
    "trk_MVA1"

]

trackFeatures = [
    'trk_z0',
    "trk_MVA1",
    'trk_pt',
    'trk_phi',
    'trk_eta', 
    'trk_chi2rphi', 
    'trk_chi2rz', 
    'trk_bendchi2',
    'corrected_trk_z0',
    'trk_fake',
]


def unpackbits(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits]) # decode as little endian
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])[::-1] #flip for big endian
    
def padArray(x,n):
    arr = np.zeros((n,),dtype=np.float32)
    for i in range(min(n,len(x))):
        arr[i] = x[i]
    return arr

def decodeHitPattern(hitpattern,eta):
    etaBins = np.array([0.0,0.2,0.41,0.62,0.9,1.26,1.68,2.08,2.5])
    table = np.array([
        [0,1,2,3,4,5,11],
        [0,1,2,3,4,5,11],
        [0,1,2,3,4,5,11],
        [0,1,2,3,4,5,11],
        [0,1,2,3,4,5,11],
        [0,1,2,6,7,8,9],
        [0,1,7,8,9,10,11],
        [0,6,7,8,9,10,11],
    ])
    patternBits = unpackbits(hitpattern.astype(np.uint8), 7)
    
    etaBits,_ = np.histogram(math.fabs(eta),bins=etaBins)
    bitShifts = np.dot(etaBits,table)
    targetBits = np.array([1<<shift for shift in bitShifts])
    
    #treat patternBits as bit mask
    return unpackbits(np.sum(targetBits*patternBits[::-1]),11)
    
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(
        value=np.nan_to_num(value.flatten(), nan=0.0, posinf=0.0, neginf=0.0)
    ))

def splitter(x,granularity,signed):
    import math

    mult = 1

    if signed: 
        mult = 2
      # Get the bin index
    t = (mult*x)/granularity

    return np.floor(t)

eta_bins = np.array([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2,2.2,2.4,np.inf])
res_bins = np.array([0.0,0.1,0.1,0.12,0.14,0.16,0.18,0.23,0.23,0.3,0.35,0.38,0.42,0.5,1])

chi2rz_bins   = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0, 20.0, 50.0,np.inf])
chi2rphi_bins = np.array([0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 10.0, 15.0, 20.0, 35.0, 60.0, 200.0, np.inf])
bendchi2_bins = np.array([0, 0.75, 1.0, 1.5, 2.25, 3.5, 5.0, 20.0,np.inf])

nMaxTracks = 250

chunkread = 5000

for ibatch,data in enumerate(f['L1TrackNtuple']['eventTree'].iterate(branches,entrysteps=chunkread)):
    data = {k.decode('utf-8'):v for k,v in data.items() }
    print ('processing batch:',ibatch+1,'/',math.ceil(1.*len(f['L1TrackNtuple']['eventTree'])/chunkread))
    
    tfwriter = tf.io.TFRecordWriter(
        '/home/cebrown/Documents/Datasets/VertexDatasets/%sData/%s%i.tfrecord'%(KFname,KFname,ibatch),
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
    data['corrected_trk_z0']= (data['trk_z0'] + (data['trk_z0']>0.)*0.03 - (data['trk_z0']<0.)*0.03) 

    #################################################
    
    tfData = {}
    

    for iev in range(len(data['trk_pt'])):
        selectPVTPs = (data['tp_eventid'][iev]==0)
        #tp met
        tp_met_px = np.sum(data['tp_pt'][iev][selectPVTPs]*np.cos(data['tp_phi'][iev][selectPVTPs]))
        tp_met_py = np.sum(data['tp_pt'][iev][selectPVTPs]*np.sin(data['tp_phi'][iev][selectPVTPs]))
        tp_met_pt = math.sqrt(tp_met_px**2+tp_met_py**2)
        tp_met_phi = math.atan2(tp_met_py,tp_met_px)
        
        tfData['tp_met_px'] = _float_feature(np.array([tp_met_px],np.float32))
        tfData['tp_met_py'] = _float_feature(np.array([tp_met_py],np.float32))
        tfData['tp_met_pt'] = _float_feature(np.array([tp_met_pt],np.float32))
        tfData['tp_met_phi'] = _float_feature(np.array([tp_met_phi],np.float32))

        #only consider well reconstructed tracks
        selectGoodTracks = (data['trk_fake'][iev]>=0.0)

        selectTracksInZ0Range = (abs(data['trk_z0'][iev]) <= 30.0)
        #print(selectTracksInZ0Range)
        
        #calc PV position as pt-weighted z0 average of PV tracks
        selectPVTracks = (data['trk_fake'][iev]==1)

        if (np.sum(1.*selectPVTracks)<1):
            continue

        

        #pv tk met
        pv_trk_met_px = np.sum(data['trk_pt'][iev][selectPVTracks]*np.cos(data['trk_phi'][iev][selectPVTracks]))
        pv_trk_met_py = np.sum(data['trk_pt'][iev][selectPVTracks]*np.sin(data['trk_phi'][iev][selectPVTracks]))
        pv_trk_met_pt = math.sqrt(pv_trk_met_px**2+pv_trk_met_py**2)
        pv_trk_met_phi = math.atan2(pv_trk_met_py,pv_trk_met_px)
        
        tfData['pv_trk_met_px'] = _float_feature(np.array([pv_trk_met_px],np.float32))
        tfData['pv_trk_met_py'] = _float_feature(np.array([pv_trk_met_py],np.float32))
        tfData['pv_trk_met_pt'] = _float_feature(np.array([pv_trk_met_pt],np.float32))
        tfData['pv_trk_met_phi'] = _float_feature(np.array([pv_trk_met_phi],np.float32))

        tfData['true_met_px'] = _float_feature(np.array(data['genMETPx'][iev],np.float32))
        tfData['true_met_py'] = _float_feature(np.array(data['genMETPy'][iev],np.float32))
        tfData['true_met_pt'] = _float_feature(np.array(data['genMET'][iev],np.float32))
        tfData['true_met_phi'] = _float_feature(np.array(data['genMETPhi'][iev],np.float32))
        
        tfData['trk_fromPV'] = _float_feature(padArray(1.*selectPVTracks*selectTracksInZ0Range,nMaxTracks))

        hist1,bin_edges = np.histogram(data['trk_z0'][iev][selectTracksInZ0Range],256,range=(-15,15),weights=selectPVTracks[selectTracksInZ0Range])
        hist2,bin_edges = np.histogram(data['trk_z0'][iev][selectTracksInZ0Range],256,range=(-15,15),weights=selectPVTracks[selectTracksInZ0Range]*data['trk_pt'][iev][selectTracksInZ0Range]*data['trk_pt'][iev][selectTracksInZ0Range])
        tfData['PV_hist'] = _float_feature(np.array(hist1,np.float32))
        tfData['PVpt_hist'] = _float_feature(np.array(hist2,np.float32))
        #sumZ0 = np.sum(data['trk_pt'][iev][selectPVTracks]*data['trk_z0'][iev][selectPVTracks])
        #sumWeights = np.sum(data['trk_pt'][iev][selectPVTracks])
        pvz0 = data["pv_MC"][iev]

        res = res_bins[np.digitize(abs(data['trk_eta'][iev][selectTracksInZ0Range]),eta_bins)]*2
        binned_trk_chi2rphi = np.digitize(data['trk_chi2rphi'][iev][selectTracksInZ0Range],chi2rphi_bins)/16
        binned_trk_chi2rz   = np.digitize(data['trk_chi2rz'][iev][selectTracksInZ0Range],chi2rz_bins)/16
        binned_trk_bendchi2 = np.digitize(data['trk_bendchi2'][iev][selectTracksInZ0Range],bendchi2_bins)/8

        tfData['binned_trk_chi2rphi'] =  _float_feature(padArray(np.array(binned_trk_chi2rphi,np.float32),nMaxTracks))
        tfData['binned_trk_chi2rz'] =  _float_feature(padArray(np.array(binned_trk_chi2rz,np.float32),nMaxTracks))
        tfData['binned_trk_bendchi2'] =  _float_feature(padArray(np.array(binned_trk_bendchi2,np.float32),nMaxTracks))

        tfData['trk_z0_res']= _float_feature(padArray(np.array(res,np.float32),nMaxTracks)) 

        sumZ0 = np.sum(data['trk_pt'][iev][selectPVTracks]*data['trk_z0'][iev][selectPVTracks])
        sumWeights = np.sum(data['trk_pt'][iev][selectPVTracks])

        tfData['pvz0'] = _float_feature(np.array(pvz0,np.float32))
        

        clipped_pt = np.clip(data['trk_pt'][iev][selectTracksInZ0Range],0, 512)
        normed_pt = clipped_pt / 512

        tfData['normed_trk_pt']               = _float_feature(padArray(np.array(normed_pt,np.float32),nMaxTracks))
        tfData['normed_trk_invR']             = _float_feature(padArray(np.array(1/normed_pt,np.float32),nMaxTracks))
        tfData['normed_trk_eta']              = _float_feature(padArray(np.array(abs(data['trk_eta'][iev][selectTracksInZ0Range]/2.4)),nMaxTracks))
        tfData['normed_trk_over_eta']         = _float_feature(padArray(np.array(2.4/abs(data['trk_eta'][iev][selectTracksInZ0Range])),nMaxTracks))
        tfData['normed_trk_over_eta_squared'] = _float_feature(padArray(np.array(5.76/abs(data['trk_eta'][iev][selectTracksInZ0Range])**2),nMaxTracks))

        tfData['trk_over_eta_squared'] = _float_feature(padArray(np.array(1/(0.1+0.2*(data['trk_eta'][iev][selectTracksInZ0Range])**2)),nMaxTracks))

        tanL = abs(np.sinh(data['trk_eta'][iev][selectTracksInZ0Range]))
        InvR = abs((3.8112*(3e8/1e11))/(2*data['trk_pt'][iev][selectTracksInZ0Range]))
        bit_InvR = splitter(InvR,5.20424e-07,True)
        bit_tanL = splitter(tanL,0.000244141,True)
        #print("#################################")
        #print(data['trk_eta'][iev][selectTracksInZ0Range])
        #print(bit_tanL)
        #print( np.log(np.sqrt(0.000244141**2*bit_tanL**2 + 1) + 0.000244141*bit_tanL))
        #print( np.log(np.sqrt(0.000244141**2*bit_tanL**2 + 1) + 0.000244141*bit_tanL)/0.000244141)
        #print("#################################")
        #print(data['trk_pt'][iev][selectTracksInZ0Range])
        #print(bit_InvR)
        #print( abs((3.8112*(3e8/1e11))/(bit_InvR)))
        #print( (abs((3.8112*(3e8/1e11))/(bit_InvR))/(5.20424e-07))/0.015625)

        bit_trk_invR = bit_InvR
        bit_trk_pt =  np.floor(abs((3.8112*(3e8/1e11))/(bit_InvR))/((5.20424e-07))/0.030517578125)
        bit_trk_tanL = bit_tanL
        bit_trk_eta = np.floor(np.log(np.sqrt(0.000244141**2*bit_tanL**2 + 1) + 0.000244141*bit_tanL)/0.000244141)
        bit_MVA1 = np.floor(data['trk_MVA1'][iev][selectTracksInZ0Range]*8)
        rescaled_bit_MVA1 = np.floor(data['trk_MVA1'][iev][selectTracksInZ0Range]*8)*1024
 
        bit_corrected_trk_z0 = 2048 + splitter((data['trk_z0'][iev][selectTracksInZ0Range] + (data['trk_z0'][iev][selectTracksInZ0Range]>0.)*0.03 - (data['trk_z0'][iev][selectTracksInZ0Range]<0.)*0.03),0.00999469,False) 
        bit_trk_z0 =  2048 + splitter((data['trk_z0'][iev][selectTracksInZ0Range]),0.00999469,False)
        #print(data['trk_z0'][iev][selectTracksInZ0Range])
        #print(bit_corrected_trk_z0)
        #print(bit_trk_z0)

        tfData['bit_trk_invR'] = _float_feature(padArray(np.array(bit_trk_invR,np.float32),nMaxTracks))
        tfData['bit_trk_pt'] =  _float_feature(padArray(np.array(bit_trk_pt,np.float32),nMaxTracks))
        tfData['bit_trk_tanL'] = _float_feature(padArray(np.array(bit_trk_tanL,np.float32),nMaxTracks))
        tfData['bit_trk_eta'] = _float_feature(padArray(np.array(bit_trk_eta,np.float32),nMaxTracks))
        tfData['bit_MVA1'] = _float_feature(padArray(np.array(bit_MVA1,np.float32),nMaxTracks))
        tfData['rescaled_bit_MVA1'] =_float_feature(padArray(np.array(rescaled_bit_MVA1,np.float32),nMaxTracks))

        tfData['bit_corrected_trk_z0']= _float_feature(padArray(np.array(bit_corrected_trk_z0,np.float32),nMaxTracks))
        tfData['bit_trk_z0']= _float_feature(padArray(np.array(bit_trk_z0,np.float32),nMaxTracks))


        #sum2Z0 = np.sum(np.square(data['trk_pt'][iev][selectPVTracks])*data['trk_z0'][iev][selectPVTracks])
        #sum2Weights = np.sum(np.square(data['trk_pt'][iev][selectPVTracks]))
        #pv2z0 = sum2Z0/sum2Weights
        #tfData['pv2z0'] = _float_feature(np.array([pv2z0],np.float32))
        
        
        
        for trackFeature in trackFeatures:
            tfData[trackFeature] = _float_feature(padArray(data[trackFeature][iev][selectTracksInZ0Range],nMaxTracks))
        
        hitPatternArr = np.zeros((nMaxTracks,11))
        #for itrack in range(min(nMaxTracks,len(data['trk_pt'][iev][selectGoodTracks]))): 
        #    hitPattern = decodeHitPattern(
        #        data['trk_hitpattern'][iev][selectGoodTracks][itrack],
        #        data['trk_eta'][iev][selectGoodTracks][itrack]
        #    )
        #    hitPatternArr[itrack] = hitPattern
            
        tfData['trk_hitpattern'] = _float_feature(hitPatternArr)
    
        example = tf.train.Example(features = tf.train.Features(feature = tfData))
        tfwriter.write(example.SerializeToString())
        
    tfwriter.close()
    #break
    
