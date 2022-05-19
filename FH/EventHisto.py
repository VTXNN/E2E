import numpy as np
import uproot3 as uproot

import glob
import sklearn.metrics as metrics
import yaml
import sys

import math

import os
from textwrap import wrap


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
hep.cms.label()
hep.cms.text("Simulation")
plt.style.use(hep.style.CMS)

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

LEGEND_WIDTH = 20
LINEWIDTH = 3
MARKERSIZE = 20

colormap = "jet"

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('axes', linewidth=5)              # thickness of axes
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)            # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

matplotlib.rcParams['xtick.major.size'] = 20
matplotlib.rcParams['xtick.major.width'] = 5
matplotlib.rcParams['xtick.minor.size'] = 10
matplotlib.rcParams['xtick.minor.width'] = 4

matplotlib.rcParams['ytick.major.size'] = 20
matplotlib.rcParams['ytick.major.width'] = 5
matplotlib.rcParams['ytick.minor.size'] = 10
matplotlib.rcParams['ytick.minor.width'] = 4

colours=["red","green","blue","orange","purple","yellow"]

f = uproot.open("/home/cebrown/Documents/Datasets/VertexDatasets/GTT_TrackNtuple_FH_oldTQ.root")

branches = [
    'trk_MVA1', 
    'trk_bendchi2',
    'trk_chi2', 
    'trk_chi2rphi', 
    'trk_chi2rz', 
    'trk_eta', 
    'trk_hitpattern', 
    'trk_nstub', 
    'trk_phi',
    'trk_pt',
    'trk_z0',
    'trk_fake',
    'pv_MC',
]
def linear_res_function(x,return_bool = False):
        if return_bool:
            return np.full_like(x,True).astype(bool)

        else:
            return np.ones_like(x)

def predictFastHisto(value,weight, res_func, return_index = False, nbins=256, max_z0=15):
    z0List = []
    halfBinWidth = max_z0/nbins


    hist,bin_edges = np.histogram(value,nbins,range=(-1*max_z0,max_z0),weights=weight*res_func)
    hist = np.convolve(hist,[1,1,1],mode='same')
    z0Index= np.argmax(hist)
    if return_index:
        z0List.append([z0Index])
    else:
        z0 = -1*max_z0+2*max_z0*z0Index/nbins+halfBinWidth
        z0List.append([z0])
    return np.array(z0List,dtype=np.float32)

def CreateHisto(value,weight, res_func, return_index = False, nbins=256, max_z0=15,factor=1):

    hist,bin_edges = np.histogram(value,nbins,range=(-1*max_z0,max_z0),weights=weight*res_func,density=True)
    hist = np.clip(hist,0,1)
    
    return hist/factor,bin_edges

if __name__=="__main__":

    outputFolder = "EventHisto" + "/" 

    max_z0 = 20.46912512
    nbins = 256

    os.system("mkdir -p "+ outputFolder)
    histo_counter = 0

    for ibatch,data in enumerate(f['L1TrackNtuple']['eventTree'].iterate(branches,entrysteps=5000)):
        data = {k.decode('utf-8'):v for k,v in data.items() }

        

        if ((ibatch % 1 == 0) & (ibatch != 0)):
             print("Step: ", ibatch, " Out of ",int(len(f['L1TrackNtuple']['eventTree'])/5000))
                
        for iev in range(len(data['trk_pt'])):
            if (iev % 1000 == 0):
                print("Event: ", iev, " Out of ",len(data["trk_pt"]))

            trk_overEta = 1/(0.1+0.2*(data['trk_eta'][iev])**2)

            trk_fromPV = (data['trk_fake'][iev] == 1).astype(int)
            fake_trk = (data['trk_fake'][iev] == 0).astype(int)

            pt_histo = CreateHisto(data['trk_z0'][iev],data['trk_pt'][iev],res_func=linear_res_function(data['trk_pt'][iev]), nbins=nbins, max_z0=max_z0)
            eta_histo = CreateHisto(data['trk_z0'][iev],abs(data['trk_eta'][iev]),res_func=linear_res_function(data['trk_pt'][iev]), nbins=nbins, max_z0=max_z0)
            MVA_histo = CreateHisto(data['trk_z0'][iev],data['trk_MVA1'][iev],res_func=linear_res_function(data['trk_pt'][iev]), nbins=nbins, max_z0=max_z0)

            chi2rphi_histo = CreateHisto(data['trk_z0'][iev],data['trk_chi2rphi'][iev],res_func=linear_res_function(data['trk_pt'][iev]), nbins=nbins, max_z0=max_z0)
            chi2rz_histo = CreateHisto(data['trk_z0'][iev],data['trk_chi2rz'][iev],res_func=linear_res_function(data['trk_pt'][iev]), nbins=nbins, max_z0=max_z0)
            bendchi2_histo = CreateHisto(data['trk_z0'][iev],data['trk_bendchi2'][iev],res_func=linear_res_function(data['trk_pt'][iev]), nbins=nbins, max_z0=max_z0)
            phi_histo = CreateHisto(data['trk_z0'][iev],abs(data['trk_phi'][iev]),res_func=linear_res_function(data['trk_pt'][iev]), nbins=nbins, max_z0=max_z0)
            nstub_histo = CreateHisto(data['trk_z0'][iev],data['trk_nstub'][iev],res_func=linear_res_function(data['trk_pt'][iev]), nbins=nbins, max_z0=max_z0)
            overeta_histo = CreateHisto(data['trk_z0'][iev],trk_overEta,res_func=linear_res_function(data['trk_pt'][iev]), nbins=nbins, max_z0=max_z0)

            PVtrk_histo = CreateHisto(data['trk_z0'][iev],trk_fromPV,res_func=linear_res_function(data['trk_pt'][iev]), nbins=nbins, max_z0=max_z0,factor=10)
            faketrk_histo = CreateHisto(data['trk_z0'][iev],fake_trk,res_func=linear_res_function(data['trk_pt'][iev]), nbins=nbins, max_z0=max_z0,factor=10)



            reco_z0 = predictFastHisto(data['trk_z0'][iev],data['trk_pt'][iev],res_func=linear_res_function(data['trk_pt'][iev]), nbins=nbins, max_z0=max_z0)
            if ((abs(reco_z0 - data['pv_MC'][iev]) > 5) & (abs(reco_z0 - data['pv_MC'][iev]) < 10)):
                histo_counter += 1
                print("Large Error   ", reco_z0[0][0], "  ", data['pv_MC'][iev][0])

                nan_array = np.zeros_like(pt_histo[0])
                nan_array[:] = np.NaN

                histo_list = [pt_histo[0],eta_histo[0],MVA_histo[0],chi2rphi_histo[0],chi2rz_histo[0],bendchi2_histo[0],phi_histo[0],nstub_histo[0],overeta_histo[0],PVtrk_histo[0],faketrk_histo[0],nan_array]
                histo_names = ["$p_T$ ","$\eta$ ","MVA ","$\chi^2_{R\phi}$ ","$\chi^2_{rz}$ ","$\chi^2_{bend}$ ","$\phi$ ","# stub ","$\\frac{1}{\eta^2}$ ", "PV Tracks", "Fake Tracks","Vertex"][::-1]

                plt.clf()
                fig,ax = plt.subplots(1,1,figsize=(24,10))
                hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
                twod_hist = np.stack(histo_list, axis=0)

                hist2d = ax.imshow(twod_hist,cmap=colormap,aspect='auto',extent=[-1*max_z0,max_z0,0,len(histo_names)])

                ax.grid(True,axis='y',linewidth=2)
                ax.grid(True,axis='x',linewidth=1)
                ax.set_ylabel('Track Feature',ha="right",y=1)
                ax.set_xlabel('Track $z_{0}$ [cm]',ha="right",x=1)
        

                ax.set_yticklabels(histo_names)
                

                ax.set_yticks(np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5]))

                rect = plt.Rectangle((data['pv_MC'][iev]-((2.5*max_z0)/nbins), 1), 5*max_z0/nbins, len(histo_names),
                                    fill=False,linewidth=2,linestyle='--',edgecolor='r')
                ax.add_patch(rect)

                rect_reco = plt.Rectangle((reco_z0-((2.5*max_z0)/nbins), 1), 5*max_z0/nbins, len(histo_names),
                                    fill=False,linewidth=2,linestyle='--',edgecolor='g')
                ax.add_patch(rect_reco)

                ax.text(reco_z0-0.5, 0.5, "Reco Vertex", color='g')
                ax.text(data['pv_MC'][iev]-0.5, 0.5, "True Vertex", color='r')


                cbar = plt.colorbar(hist2d , ax=ax)
                cbar.set_label('# Tracks')

                cbar.set_label('# Tracks')
                plt.tight_layout()
                plt.savefig("%s/2dhist_%s.png" % (outputFolder, str(histo_counter)))
                plt.close()


            