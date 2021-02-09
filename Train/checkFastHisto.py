import uproot
import numpy as np
import math

def predictZ0(value,weight,fake):
    z0List = []
    halfBinWidth = 0.5*30./256.
    for ibatch in range(value.shape[0]):
        selectNonFake = fake[ibatch]>0.5
        
        hist,bin_edges = np.histogram(value[ibatch][selectNonFake],256,range=(-15,15),weights=weight[ibatch][selectNonFake])
        hist = np.convolve(hist,[1,1,1],mode='same')
        z0Index= np.argmax(hist)
        z0 = -15.+30.*z0Index/256.+halfBinWidth
        z0List.append(z0)
    return np.array(z0List,dtype=np.float32)

f = uproot.open("/vols/cms/mkomm/VTX/samples/TTbar_170K_hybrid.root")
print (sorted(f['L1TrackNtuple']['eventTree'].keys()))
branches = [
    'trk_genuine', 
    'trk_pt',
    'trk_z0',
    'trk_fake',
]

chunkread = 5000

for ibatch,data in enumerate(f['L1TrackNtuple']['eventTree'].iterate(branches,entrysteps=chunkread)):
    data = {k.decode('utf-8'):v for k,v in data.items()}
    print ('processing batch:',ibatch+1,'/',math.ceil(1.*len(f['L1TrackNtuple']['eventTree'])/chunkread))
    
    predictedZ0 = predictZ0(data['trk_z0'],data['trk_pt'],data['trk_fake'])
    
    pvz0 = []
    
    for iev in range(len(data['trk_pt'])):
        #calc PV position as pt-weighted z0 average of genuine tracks
        selectGenuineTracks = (data['trk_fake'][iev]==1)#*(data['trk_pt'][iev]<500)
        
        #sumZ0 = np.sum(data['trk_pt'][iev][selectGenuineTracks]*data['trk_z0'][iev][selectGenuineTracks])
        #sumWeights = np.sum(data['trk_pt'][iev][selectGenuineTracks])

        #pvz0.append(sumZ0/sumWeights)
        
        pvz0.append(np.average(data["trk_z0"][iev][selectGenuineTracks],weights=data["trk_pt"][iev][selectGenuineTracks]))


    pvz0 = np.array(pvz0)
    print (predictedZ0.shape,pvz0.shape)
    print (np.std(predictedZ0-pvz0),np.percentile(predictedZ0-pvz0,[5,15,50,85,95]))
        
    break
