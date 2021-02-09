import uproot
import numpy as np
import math

def predictZ0(value,weight):
    z0List = []
    halfBinWidth = 0.5*30./256.
    for ibatch in range(value.shape[0]):
        hist,bin_edges = np.histogram(value[ibatch],256,range=(-15,15),weights=weight[ibatch])
        hist = np.convolve(hist,[1,1,1],mode='same')
        z0Index= np.argmax(hist)
        z0 = -15.+30.*z0Index/256.+halfBinWidth
        z0List.append([z0])
    return np.array(z0List,dtype=np.float32)

f = uproot.open("/vols/cms/mkomm/VTX/samples/TTbar_170K_hybrid.root")

branches = [
    'trk_genuine', 
    'trk_pt',
    'trk_z0'
]

chunkread = 5000

for ibatch,data in enumerate(f['L1TrackNtuple']['eventTree'].iterate(branches,entrysteps=chunkread)):
    data = {k.decode('utf-8'):v for k,v in data.items()}
    print ('processing batch:',ibatch+1,'/',math.ceil(1.*len(f['L1TrackNtuple']['eventTree'])/chunkread))
    
    predictedZ0 = predictZ0(data['trk_z0'],data['trk_pt'])
    
    pvz0 = []
    
    for iev in range(len(data['trk_pt'])):

        #calc PV position as pt-weighted z0 average of genuine tracks
        selectGenuineTracks = (data['trk_genuine'][iev]>0.5)
        
        sumZ0 = np.sum(data['trk_pt'][iev][selectGenuineTracks]*data['trk_z0'][iev][selectGenuineTracks])
        sumWeights = np.sum(data['trk_pt'][iev][selectGenuineTracks])
        pvz0.append(sumZ0/sumWeights)

    pvz0 = np.array(pvz0)
    
    print (np.percentile(predictedZ0-pvz0,[5,15,50,85,95]))
        
    break
