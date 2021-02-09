import uproot
import tensorflow as tf
import numpy as np
import math

tf.compat.v1.disable_eager_execution()

f = uproot.open("/vols/cms/mkomm/VTX/samples/TTbar_170K_hybrid.root")
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
    'trk_z0'
]

trackFeatures = [
    'trk_z0',
    'trk_pt',
    'trk_eta', 
    'trk_chi2rphi', 
    'trk_chi2rz', 
    'trk_bendchi2',
    'trk_nstub', 
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
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))


nMaxTracks = 200

chunkread = 5000

for ibatch,data in enumerate(f['L1TrackNtuple']['eventTree'].iterate(branches,entrysteps=chunkread)):
    data = {k.decode('utf-8'):v for k,v in data.items()}
    print ('processing batch:',ibatch+1,'/',math.ceil(1.*len(f['L1TrackNtuple']['eventTree'])/chunkread))
    
    tfwriter = tf.io.TFRecordWriter(
        'TTbar_%i.tfrecord'%ibatch,
        options=tf.io.TFRecordOptions(
            compression_type='GZIP',
            compression_level = 4,
            input_buffer_size=100,
            output_buffer_size=100,
            mem_level = 8,
        )
    )
    
    tfData = {}
    for iev in range(len(data['trk_pt'])):

        #only consider well reconstructed tracks
        selectGoodTracks = (data['trk_pt'][iev]<500.)#*(data['trk_chi2'][iev]<500.)

        #calc PV position as pt-weighted z0 average of genuine tracks
        selectGenuineTracks = (data['trk_genuine'][iev]>0.5)*selectGoodTracks
        
        #TODO: check tp match instead of genuine flag
        
        sumZ0 = np.sum(data['trk_pt'][iev][selectGenuineTracks]*data['trk_z0'][iev][selectGenuineTracks])
        sumWeights = np.sum(data['trk_pt'][iev][selectGenuineTracks])
        pvz0 = sumZ0/sumWeights
        tfData['pvz0'] = _float_feature(np.array([pvz0],np.float32))
        
        sum2Z0 = np.sum(np.square(data['trk_pt'][iev][selectGenuineTracks])*data['trk_z0'][iev][selectGenuineTracks])
        sum2Weights = np.sum(np.square(data['trk_pt'][iev][selectGenuineTracks]))
        pv2z0 = sum2Z0/sum2Weights
        tfData['pv2z0'] = _float_feature(np.array([pv2z0],np.float32))
        
        
        for trackFeature in trackFeatures:
            tfData[trackFeature] = _float_feature(padArray(data[trackFeature][iev][selectGoodTracks],nMaxTracks))
        
        hitPatternArr = np.zeros((nMaxTracks,11))
        for itrack in range(min(nMaxTracks,len(data['trk_pt'][iev][selectGoodTracks]))): 
            hitPattern = decodeHitPattern(
                data['trk_hitpattern'][iev][selectGoodTracks][itrack],
                data['trk_eta'][iev][selectGoodTracks][itrack]
            )
            hitPatternArr[itrack] = hitPattern
            
        tfData['hitPattern'] = _float_feature(hitPatternArr)
    
        example = tf.train.Example(features = tf.train.Features(feature = tfData))
        tfwriter.write(example.SerializeToString())
        
    tfwriter.close()
    #break
    
