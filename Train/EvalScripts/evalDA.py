import glob
import sys
from textwrap import wrap

import comet_ml
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import sklearn.metrics as metrics
import tensorflow as tf
import yaml

import vtx
from TrainingScripts.train import *
plt.style.use(hep.style.CMS)

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

LEGEND_WIDTH = 20
LINEWIDTH = 3
MARKERSIZE = 20

colormap = "seismic"

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

def predictFastHisto(value,weight):
    z0List = []
    halfBinWidth = 0.5*30./256.
    for ibatch in range(value.shape[0]):
        hist,bin_edges = np.histogram(value[ibatch],256,range=(-15,15),weights=weight[ibatch])
        hist = np.convolve(hist,[1,1,1],mode='same')
        z0Index= np.argmax(hist)
        z0 = -15.+30.*z0Index/256.+halfBinWidth
        z0List.append([z0])
    return np.array(z0List,dtype=np.float32)

def predictFastHistoZ0res(value,weight,eta):

    def res_function(eta):
        res = 0.1 + 0.2*eta**2
        return res
    
    z0List = []
    halfBinWidth = 0.5*30./256.
    for ibatch in range(value.shape[0]):
        #res = res_bins[np.digitize(abs(eta[ibatch]),eta_bins)]
        res = res_function(eta[ibatch])
        hist,bin_edges = np.histogram(value[ibatch],256,range=(-15,15),weights=weight[ibatch]/res)
        hist = np.convolve(hist,[1,1,1],mode='same')
        z0Index= np.argmax(hist)
        z0 = -15.+30.*z0Index/256.+halfBinWidth
        z0List.append([z0])
    return np.array(z0List,dtype=np.float32)

def predictFastHistoMVAcut(value,weight,MVA):

    def res_function(MVA):
        res = MVA > 0.3
        return res

    z0List = []
    halfBinWidth = 0.5*30./256.
    for ibatch in range(value.shape[0]):
        #res = res_bins[np.digitize(abs(eta[ibatch]),eta_bins)]
        
        res = res_function(MVA[ibatch])
        hist,bin_edges = np.histogram(value[ibatch][res],256,range=(-15,15),weights=(weight[ibatch][res]))
        #hist,bin_edges = np.histogram(value[ibatch][MVA[ibatch][MVA[ibatch] > 0.08]],256,range=(-15,15),weights=(weight[ibatch][MVA[ibatch][MVA[ibatch] > 0.08]]))
        hist = np.convolve(hist,[1,1,1],mode='same')
        z0Index= np.argmax(hist)
        z0 = -15.+30.*z0Index/256.+halfBinWidth
        z0List.append([z0])
    return np.array(z0List,dtype=np.float32)

def predictFastHistoZ0MVA(value,weight,eta,MVA):

    def res_function(eta):
        res = 0.1 + 0.2*eta**2
        return res
    
    z0List = []
    halfBinWidth = 0.5*30./256.
    for ibatch in range(value.shape[0]):
        #res = res_bins[np.digitize(abs(eta[ibatch]),eta_bins)]
        res = res_function(eta[ibatch])
        hist,bin_edges = np.histogram(value[ibatch],256,range=(-15,15),weights=(weight[ibatch]*MVA[ibatch])/res)
        hist = np.convolve(hist,[1,1,1],mode='same')
        z0Index= np.argmax(hist)
        z0 = -15.+30.*z0Index/256.+halfBinWidth
        z0List.append([z0])
    return np.array(z0List,dtype=np.float32)

def predictFastHistoNoFakes(value,weight,Fakes):

    def res_function(fakes):
        res = fakes != 0
        return res

    z0List = []
    halfBinWidth = 0.5*30./256.
    for ibatch in range(value.shape[0]):
        #res = res_bins[np.digitize(abs(eta[ibatch]),eta_bins)]
        
        res = res_function(Fakes[ibatch])
        hist,bin_edges = np.histogram(value[ibatch][res],256,range=(-15,15),weights=(weight[ibatch][res]))
        #hist,bin_edges = np.histogram(value[ibatch][MVA[ibatch][MVA[ibatch] > 0.08]],256,range=(-15,15),weights=(weight[ibatch][MVA[ibatch][MVA[ibatch] > 0.08]]))
        hist = np.convolve(hist,[1,1,1],mode='same')
        z0Index= np.argmax(hist)
        z0 = -15.+30.*z0Index/256.+halfBinWidth
        z0List.append([z0])
    return np.array(z0List,dtype=np.float32)

def FastHistoAssoc(PV,trk_z0,trk_eta,kf):
    if kf == "NewKF":
        deltaz_bins = np.array([0.0,0.41,0.55,0.66,0.825,1.1,1.76,0.0])
    elif kf == "OldKF":
        deltaz_bins = np.array([0.0,0.37,0.5,0.6,0.75,1.0,1.6,0.0])
    eta_bins = np.array([0.0,0.7,1.0,1.2,1.6,2.0,2.4])
    

    eta_bin = np.digitize(abs(trk_eta),eta_bins)
    deltaz = abs(trk_z0 - PV)

    assoc = (deltaz < deltaz_bins[eta_bin])

    return np.array(assoc,dtype=np.float32)

def FastHistoAssocMVAcut(PV,trk_z0,trk_eta,MVA,kf,threshold=0.3):
    if kf == "NewKF":
        deltaz_bins = np.array([0.0,0.41,0.55,0.66,0.825,1.1,1.76,0.0])
    elif kf == "OldKF":
        deltaz_bins = np.array([0.0,0.37,0.5,0.6,0.75,1.0,1.6,0.0])
    eta_bins = np.array([0.0,0.7,1.0,1.2,1.6,2.0,2.4])
    

    eta_bin = np.digitize(abs(trk_eta),eta_bins)
    deltaz = abs(trk_z0 - PV)

    assoc = (deltaz < deltaz_bins[eta_bin]) & (MVA > threshold)

    return np.array(assoc,dtype=np.float32)

def FastHistoAssocNoFakes(PV,trk_z0,trk_eta,Fakes,kf):
    if kf == "NewKF":
        deltaz_bins = np.array([0.0,0.41,0.55,0.66,0.825,1.1,1.76,0.0])
    elif kf == "OldKF":
        deltaz_bins = np.array([0.0,0.37,0.5,0.6,0.75,1.0,1.6,0.0])
    eta_bins = np.array([0.0,0.7,1.0,1.2,1.6,2.0,2.4])
    

    eta_bin = np.digitize(abs(trk_eta),eta_bins)
    deltaz = abs(trk_z0 - PV)

    assoc = (deltaz < deltaz_bins[eta_bin]) & (Fakes != 0)

    return np.array(assoc,dtype=np.float32)

def predictMET(pt,phi,predictedAssoc,threshold,quality=False,chi2rphi=None,chi2rz=None,bendchi2=None):
    met_pt_list = []
    met_phi_list = []

    def assoc_function(Assoc):
        res = Assoc > threshold
        return res

    def quality_function(chi2rphi,chi2rz,bendchi2):
        qrphi = chi2rphi < 12 
        qrz =  chi2rz < 9 
        qbend = bendchi2 < 4
        q = np.logical_and(qrphi,qrz)
        q = np.logical_and(q,qbend)
        return q


    for ibatch in range(pt.shape[0]):
        assoc = assoc_function(predictedAssoc[ibatch])
        if quality:
            quality_s = quality_function(chi2rphi[ibatch],chi2rz[ibatch],bendchi2[ibatch])
            selection = np.logical_and(assoc,quality_s)
        else:
            selection = assoc
        newpt = pt[ibatch][selection]
        newphi = phi[ibatch][selection]



        met_px = np.sum(newpt*np.cos(newphi))
        met_py = np.sum(newpt*np.sin(newphi))
        met_pt_list.append(math.sqrt(met_px**2+met_py**2))
        met_phi_list.append(math.atan2(met_py,met_px))
    return  [np.array(met_pt_list,dtype=np.float32),
             np.array(met_phi_list,dtype=np.float32)]

def plotz0_residual(NNdiff,FHdiff,NNnames,FHnames,colours=colours):
    plt.clf()
    fig,ax = plt.subplots(1,2,figsize=(20,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])
    
    
    items = 0
    for i,FH in enumerate(FHdiff):
        qz0_FH = np.percentile(FH,[32,50,68])
        ax[0].hist(FH,bins=50,range=(-15,15),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s \nRMS = %.4f" 
                 %(FHnames[i],np.sqrt(np.mean(FH**2))),LEGEND_WIDTH)))
        ax[1].hist(FH,bins=50,range=(-1,1),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s \nQuartile Width = %.4f" 
                 %(FHnames[i],qz0_FH[2]-qz0_FH[0]),LEGEND_WIDTH)))
        items+=1

    for i,NN in enumerate(NNdiff):
        qz0_NN = np.percentile(NN,[32,50,68])
        ax[0].hist(NN,bins=50,range=(-15,15),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s \nRMS = %.4f" 
                 %(NNnames[i],np.sqrt(np.mean(NN**2))),LEGEND_WIDTH)))
        ax[1].hist(NN,bins=50,range=(-1,1),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s \nQuartile Width = %.4f" 
                 %(NNnames[i],qz0_NN[2]-qz0_NN[0]),LEGEND_WIDTH)))
        items+=1
    
    ax[0].grid(True)
    ax[0].set_xlabel('$z^{PV}_0$ Residual [cm]',ha="right",x=1)
    ax[0].set_ylabel('Events',ha="right",y=1)
    ax[0].set_yscale("log")
    ax[0].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

    ax[1].grid(True)
    ax[1].set_xlabel('$z^{PV}_0$ Residual [cm]',ha="right",x=1)
    ax[1].set_ylabel('Events',ha="right",y=1)
    ax[1].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

    plt.tight_layout()
    return fig

def plotMET_residual(NNdiff,FHdiff,NNnames,FHnames,colours=colours,range=(-50,50),logrange=(-1,1),relative=False,actual=None,logbins=False):
    plt.clf()
    fig,ax = plt.subplots(1,2,figsize=(20,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])

    if logbins:
        bins = [0,5,10,20,30,45,60,80,100]
    else:
        bins = np.linspace(logrange[0],logrange[1],50)
    
    items = 0
    for i,FH in enumerate(FHdiff):
        if relative:
            FH = (FH - actual) / actual
            temp_actual = actual[~np.isnan(FH)]
            temp_actual = temp_actual[np.isfinite(FH)]
            FH = FH[~np.isnan(FH)]
            FH = FH[np.isfinite(FH)]

            if logbins:
                FH = FH + 1
            
        else:
            FH = (FH - actual)
            temp_actual = actual
        qz0_FH = np.percentile(FH,[32,50,68])

        ax[0].hist(FH,bins=bins,histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s RMS = %.4f" 
                 %(FHnames[i],metrics.mean_squared_error(temp_actual,FH,squared=False)),LEGEND_WIDTH)))
        ax[1].hist(FH,bins=50,range=range,histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s Quartile Width = %.4f   Centre = %.4f" 
                 %(FHnames[i],qz0_FH[2]-qz0_FH[0], qz0_FH[1]),25)))
        items+=1

    for i,NN in enumerate(NNdiff):
        if relative:
            NN = (NN - actual)/actual
            temp_actual = actual[~np.isnan(NN)]
            temp_actual = temp_actual[np.isfinite(NN)]
            NN = NN[~np.isnan(NN)]
            NN = NN[np.isfinite(NN)]

            if logbins:
                NN = NN + 1

        else:
            NN = (NN - actual)
            temp_actual = actual
        qz0_NN = np.percentile(NN,[32,50,68])
        ax[0].hist(NN,bins=bins,histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s RMS = %.4f" 
                 %(NNnames[i],metrics.mean_squared_error(temp_actual,NN,squared=False)),LEGEND_WIDTH)))
        ax[1].hist(NN,bins=50,range=range,histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s Quartile Width = %.4f   Centre = %.4f" 
                 %(NNnames[i],qz0_NN[2]-qz0_NN[0], qz0_NN[1]),25)))
        items+=1
    
    ax[0].grid(True)
    
    ax[0].set_ylabel('Events',ha="right",y=1)
    ax[0].set_yscale("log")
    ax[0].legend(loc='upper right', bbox_to_anchor=(0.95, 0.95)) 

    ax[1].grid(True)
    
    ax[1].set_ylabel('Events',ha="right",y=1)
    ax[1].legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))

    if relative:
        ax[0].set_xlabel('$E_{T}^{miss}$ Resolution \\Frac{Gen - True}{True}',ha="right",x=1)
        ax[1].set_xlabel('$E_{T}^{miss}$ Resolution \\Frac{Gen - True}{True}',ha="right",x=1)
    else:
        ax[0].set_xlabel('$E_{T}^{miss}$ Residual [GeV]',ha="right",x=1)
        ax[1].set_xlabel('$E_{T}^{miss}$ Residual [GeV]',ha="right",x=1)


    plt.tight_layout()
    return fig

def plotMETphi_residual(NNdiff,FHdiff,NNnames,FHnames,colours=colours,actual=None):
    plt.clf()
    fig,ax = plt.subplots(1,2,figsize=(20,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])
    

    items = 0
    for i,FH in enumerate(FHdiff):
        FH = FH - actual
        qz0_FH = np.percentile(FH,[32,50,68])
        ax[0].hist(FH,bins=50,range=(-np.pi,np.pi),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s RMS = %.4f,     Centre = %.4f" 
                 %(FHnames[i],np.sqrt(np.mean(FH**2)), qz0_FH[1]),LEGEND_WIDTH)))
        ax[1].hist(FH,bins=50,range=(-2*np.pi,2*np.pi),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s Quartile Width = %.4f, Centre = %.4f" 
                 %(FHnames[i],qz0_FH[2]-qz0_FH[0], qz0_FH[1]),LEGEND_WIDTH)))
        items+=1

    for i,NN in enumerate(NNdiff):
        NN = NN - actual
        qz0_NN = np.percentile(NN,[32,50,68])
        ax[0].hist(NN,bins=50,range=(-np.pi,np.pi),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s RMS = %.4f, Centre = %.4f" 
                 %(NNnames[i],np.sqrt(np.mean(NN**2)), qz0_NN[1]),LEGEND_WIDTH)))
        ax[1].hist(NN,bins=50,range=(-2*np.pi,2*np.pi),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s Quartile Width = %.4f, Centre = %.4f" 
                 %(NNnames[i],qz0_NN[2]-qz0_NN[0], qz0_NN[1]),LEGEND_WIDTH)))
        items+=1
    
    ax[0].grid(True)
    ax[0].set_xlabel('$E_{T,\\phi}^{miss}$ Residual [rad]',ha="right",x=1)
    ax[0].set_ylabel('Events',ha="right",y=1)
    ax[0].set_yscale("log")
    ax[0].legend(loc=2) 

    ax[1].grid(True)
    ax[1].set_xlabel('$E_{T,\\phi}^{miss}$  Residual [rad]',ha="right",x=1)
    ax[1].set_ylabel('Events',ha="right",y=1)
    ax[1].legend(loc=2) 

    plt.tight_layout()
    return fig

def plotPV_roc(actual,NNpred,FHpred,NNnames,FHnames,Nthresholds=50,colours=colours):
    plt.clf()
    fig,ax = plt.subplots(1,2,figsize=(20,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])
    

    items=0

    for i,FH in enumerate(FHpred):
        tnFH, fpFH, fnFH, tpFH = metrics.confusion_matrix(actual, FH).ravel()
        precisionFH = tpFH / (tpFH + fpFH) 
        recallFH = tpFH / (tpFH + fnFH) 
        TPRFH = recallFH
        FPRFH = fpFH / (fpFH + tnFH) 
        ax[0].plot(recallFH,precisionFH,label=str(FHnames[i]),linewidth=LINEWIDTH,color=colours[items],marker='o')
        ax[1].plot(TPRFH,FPRFH,label='\n'.join(wrap(f"%s AUC: %.4f" %(FHnames[i],metrics.roc_auc_score(actual,FH)),LEGEND_WIDTH)),color=colours[items],marker='o')
        items+=1

    for i,NN in enumerate(NNpred):
        precisionNN = []
        recallNN = []
        FPRNN = []

        NN = (NN - min(NN))/(max(NN) - min(NN))

        thresholds = np.linspace(0,1,Nthresholds)

        for j,threshold in enumerate(thresholds):
            print(str(NNnames[i]) + " Testing ROC threshold: "+str(j) + " out of "+str(len(thresholds)))
            tnNN, fpNN, fnNN, tpNN = metrics.confusion_matrix(actual, NN>threshold).ravel()
            precisionNN.append( tpNN / (tpNN + fpNN) )
            recallNN.append(tpNN / (tpNN + fnNN) )
            FPRNN.append(fpNN / (fpNN + tnNN) )

        
        ax[0].plot(recallNN,precisionNN,label=str(NNnames[i]),linewidth=LINEWIDTH,color=colours[items])
        ax[1].plot(recallNN,FPRNN,linewidth=LINEWIDTH,label='\n'.join(wrap(f"%s AUC: %.4f" %(NNnames[i],metrics.roc_auc_score(actual,NN)),LEGEND_WIDTH)),color=colours[items])
        items+=1

    
    
    ax[0].grid(True)
    ax[0].set_xlabel('Efficiency',ha="right",x=1)
    ax[0].set_ylabel('Purity',ha="right",y=1)
    ax[0].set_xlim([0.75,1])
    ax[0].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

    ax[1].grid(True)
    ax[1].set_yscale("log")
    ax[1].set_xlabel('Track to Vertex Association True Positive Rate',ha="right",x=1)
    ax[1].set_ylabel('Track to Vertex Association False Positive Rate',ha="right",y=1)
    ax[1].set_xlim([0.75,1])
    ax[1].set_ylim([1e-2,1])
    ax[1].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
    plt.tight_layout()
    return fig

def plotz0_percentile(NNdiff,FHdiff,NNnames,FHnames,colours=colours):
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    

    percentiles = np.linspace(0,100,100)

    items=0

    for i,FH in enumerate(FHdiff):
        FHpercentiles = np.percentile(FH,percentiles)
        ax.plot(percentiles,abs(FHpercentiles),linewidth=LINEWIDTH,color=colours[items],label='\n'.join(wrap(f"FH %s minimum: %.4f at : %.2f " %(FHnames[i],min(abs(FHpercentiles)),np.argmin(abs(FHpercentiles))),LEGEND_WIDTH)))
        items+=1

    for i,NN in enumerate(NNdiff):
        NNpercentiles = np.percentile(NN,percentiles)
        ax.plot(percentiles,abs(NNpercentiles),linewidth=LINEWIDTH,color=colours[items],label='\n'.join(wrap(f"NN %s minimum: %.4f at : %.2f " %(NNnames[i],min(abs(NNpercentiles)),np.argmin(abs(NNpercentiles))),LEGEND_WIDTH)))
        items+=1
    

    ax.grid(True)
    ax.set_xlabel('Percentile',ha="right",x=1)
    ax.set_ylabel('$|\\delta z_{0}| [cm]$',ha="right",y=1)
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    

    return fig

def plotKDEandTracks(tracks,assoc,genPV,predictedPV,weights,weight_label="KDE",threshold=-1):
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(20,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    fakes = assoc == 0
    PU = assoc == 2
    PV = assoc == 1

    PUhist,PUbin_edges = np.histogram(tracks[PU],256,range=(-15,15),weights=weights[PU])
    plt.bar(PUbin_edges[:-1],PUhist,width=30/256,color='b',alpha=0.5, label="PU Trk",bottom=2)

    Fakehist,Fakebin_edges = np.histogram(tracks[fakes],256,range=(-15,15),weights=weights[fakes])
    plt.bar(Fakebin_edges[:-1],Fakehist,width=30/256,color='r',alpha=0.5, label="Fake Trk",bottom=2+PUhist)

    PVhist,PVbin_edges = np.histogram(tracks[PV],256,range=(-15,15),weights=weights[PV])
    plt.bar(PVbin_edges[:-1],PVhist,width=30/256,color='g',alpha=0.5, label="PV Trk",bottom=2+PUhist+Fakehist)

    maxpt = np.max(2+PUhist+Fakehist+PVhist)*1.5



    #plt.plot(tracks[PV], [2] * len(tracks[PV]), '+g', label='PV Trk',markersize=MARKERSIZE)
    #plt.plot(tracks[PU], [2] * len(tracks[PU]), '+b', label='PU Trk',markersize=MARKERSIZE)
    #plt.plot(tracks[fakes], [2] * len(tracks[fakes]), '+r', label='Fake Trk',markersize=MARKERSIZE)

    ax.plot([predictedPV, predictedPV], [0,maxpt], '--k', label='Reco Vx',linewidth=LINEWIDTH)

    ax.plot([genPV, genPV], [0,maxpt], linestyle='--',color="m", label='True Vx',linewidth=LINEWIDTH)

    ax.set_xlabel('$z_0$ [cm]',ha="right",x=1)
    ax.set_ylabel('$\\sum p_T$ [GeV]',ha="right",y=1)
    ax.set_xlim(-15,15)
    ax.set_ylim(2,maxpt)
    ax.set_yscale("log")
    ax.grid()
    ax.legend(fontsize=24)
    plt.tight_layout()

    return fig

def plotMET_resolution(NNpred,FHpred,NNnames,FHnames,colours=colours,actual=None,Et_bins = [0,100,300]):
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    items = 0

    Et_bin_centres = []
    Et_bin_widths = []
    Et_bin_indices = []

    for i in range(len(Et_bins) - 1):
        Et_bin_centres.append(Et_bins[i] + (Et_bins[i+1] - Et_bins[i]) / 2)
        Et_bin_widths.append((Et_bins[i+1] - Et_bins[i]) / 2)

    for i,FH in enumerate(FHpred):
        FH = (FH - actual) / actual
        temp_actual = actual[~np.isnan(FH)]
        temp_actual = temp_actual[np.isfinite(FH)]
        FH = FH[~np.isnan(FH)]
        FH = FH[np.isfinite(FH)]
        FH_means = []
        FH_sdevs = []

        for j in range(len(Et_bins) - 1):
            Et_bin_indices = np.where(np.logical_and(temp_actual >= Et_bins[j], temp_actual < Et_bins[j+1]))
            FH_means.append(np.mean(FH[Et_bin_indices]))
            FH_sdevs.append(np.std(FH[Et_bin_indices]))

        ax.errorbar(x=Et_bin_centres,y=FH_means,xerr = Et_bin_widths, yerr = FH_sdevs,markersize=5,marker='s',color = colours[items],label=FHnames[i])
        items+=1

    for i,NN in enumerate(NNpred):
        NN_means = []
        NN_sdevs = []
        NN = (NN - actual)/actual
        temp_actual = actual[~np.isnan(NN)]
        temp_actual = temp_actual[np.isfinite(NN)]
        NN = NN[~np.isnan(NN)]
        NN = NN[np.isfinite(NN)]

        for j in range(len(Et_bins) - 1):
            Et_bin_indices = np.where(np.logical_and(temp_actual >= Et_bins[j], temp_actual < Et_bins[j+1]))
            NN_means.append(np.mean(NN[Et_bin_indices]))
            NN_sdevs.append(np.std(NN[Et_bin_indices]))

        ax.errorbar(x=Et_bin_centres,y=NN_means,xerr = Et_bin_widths, yerr = NN_sdevs,markersize=5,marker='s',color = colours[items],label=NNnames[i])
        items+=1
    
    ax.grid(True)
    
    ax.set_ylabel('$\\delta_{E_{T}^{miss}}/E_{T}^{miss}$',ha="right",y=1)
    ax.set_xlabel('True $E_{T}^{miss}$ ',ha="right",x=1)
    ax.set_ylim(-1,20)
    ax.legend(loc=1) 

    plt.tight_layout()
    return fig

if __name__=="__main__":
    kf = sys.argv[1]

    with open(sys.argv[2]+'.yaml', 'r') as f:
            config = yaml.load(f,Loader=yaml.FullLoader)

    if kf == "NewKF":
        test_files = glob.glob(config["data_folder"]+"NewKFData/MET/*.tfrecord")
        z0 = 'trk_z0'
    elif kf == "OldKF":
        test_files = glob.glob(config["data_folder"]+"OldKFData/MET/*.tfrecord")
        z0 = 'corrected_trk_z0'

    nMaxTracks = 250

    with open(kf+'experimentkey.txt') as f:
        first_line = f.readline()

    EXPERIMENT_KEY = first_line

    if (EXPERIMENT_KEY is not None):
        # There is one, but the experiment might not exist yet:
        api = comet_ml.API() # Assumes API key is set in config/env
        try:
            api_experiment = api.get_experiment_by_key(EXPERIMENT_KEY)
        except Exception:
            api_experiment = None

    experiment = comet_ml.ExistingExperiment(
            previous_experiment=EXPERIMENT_KEY,
            log_env_details=True, # to continue env logging
            log_env_gpu=True,     # to continue GPU logging
            log_env_cpu=True,     # to continue CPU logging
        )

    with open(sys.argv[2]+'.yaml', 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    outputFolder = kf+config['eval_folder']
    trainable = config["trainable"]
    trackfeat = config["track_features"] 
    weightfeat = config["weight_features"] 


    features = {
            "pvz0": tf.io.FixedLenFeature([1], tf.float32),
            "trk_fromPV":tf.io.FixedLenFeature([nMaxTracks], tf.float32),
            "trk_hitpattern": tf.io.FixedLenFeature([nMaxTracks*11], tf.float32), 
            "PV_hist"  :tf.io.FixedLenFeature([256,1], tf.float32),
            "tp_met_pt" : tf.io.FixedLenFeature([1], tf.float32),
            "tp_met_phi" : tf.io.FixedLenFeature([1], tf.float32)
    }

    def decode_data(raw_data):
        decoded_data = tf.io.parse_example(raw_data,features)
        decoded_data['trk_hitpattern'] = tf.reshape(decoded_data['trk_hitpattern'],[-1,nMaxTracks,11])
        return decoded_data

    def setup_pipeline(fileList):
        ds = tf.data.Dataset.from_tensor_slices(fileList)
        ds.shuffle(len(fileList),reshuffle_each_iteration=True)
        ds = ds.interleave(
            lambda x: tf.data.TFRecordDataset(
                x, compression_type='GZIP', buffer_size=100000000
            ),
            cycle_length=6, 
            block_length=200, 
            num_parallel_calls=6
        )
        ds = ds.batch(200) #decode in batches (match block_length?)
        ds = ds.map(decode_data, num_parallel_calls=6)
        ds = ds.unbatch()
        ds = ds.shuffle(5000,reshuffle_each_iteration=True)
        ds = ds.batch(2000)
        ds = ds.prefetch(5)
        
        return ds

    trackFeatures = [
            'trk_z0',
            'normed_trk_pt',
            'normed_trk_eta', 
            'normed_trk_invR',
            'binned_trk_chi2rphi', 
            'binned_trk_chi2rz', 
            'binned_trk_bendchi2',
            'trk_z0_res',
            'trk_pt',
            'trk_eta',
            'trk_phi',
            'trk_MVA1',
            'trk_fake',
            'corrected_trk_z0',
            'normed_trk_over_eta',
            'normed_trk_over_eta_squared',
            'trk_over_eta_squared'

        ]

    for trackFeature in trackFeatures:
        features[trackFeature] = tf.io.FixedLenFeature([nMaxTracks], tf.float32)

            
    if trainable == "DiffArgMax":
        
        nlatent = 2

        network = vtx.nn.E2EDiffArgMax(
            nbins=256,
            ntracks=max_ntracks, 
            nweightfeatures=len(weightfeat), 
            nfeatures=len(trackfeat), 
            nweights=1, 
            nlatent = nlatent,
            activation='relu',
            regloss=1e-10
        )


    elif trainable == "FH":
        nlatent = 0

        network = vtx.nn.E2EFH(
            nbins=256,
            ntracks=max_ntracks, 
            nweightfeatures=len(weightfeat), 
            nfeatures=len(trackfeat), 
            nweights=1, 
            activation='relu',
            regloss=1e-10
        )

    elif trainable == "FullNetwork":
        nlatent = 2

        network = vtx.nn.E2Ecomparekernel(
            nbins=256,
            ntracks=max_ntracks, 
            nweightfeatures=len(weightfeat), 
            nfeatures=len(trackfeat), 
            nweights=1, 
            nlatent = nlatent,
            activation='relu',
            regloss=1e-10
        )

    elif trainable == "QDiffArgMax":
        nlatent = 2

        network = vtx.nn.E2EQKerasDiffArgMax(
            nbins=256,
            ntracks=max_ntracks, 
            nweightfeatures=len(weightfeat), 
            nfeatures=len(trackfeat), 
            nweights=1, 
            nlatent = nlatent,
            activation='relu',
            l1regloss = config['l1regloss'],
            l2regloss = config['l2regloss'],
            nweightnodes = config['nweightnodes'],
            nweightlayers = config['nweightlayers'],
            nassocnodes = config['nassocnodes'],
            nassoclayers = config['nassoclayers'],
            bits = config['bits'],
            integer = config['integer'],
            alpha = config['alpha'],
        )




    model = network.createE2EModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(
        optimizer,
        loss=[
            tf.keras.losses.MeanAbsoluteError(),
            tf.keras.losses.BinaryCrossentropy(from_logits=True),
            lambda y,x: 0.
        ],
        metrics=[
            tf.keras.metrics.BinaryAccuracy(threshold=0.,name='assoc_acc') #use thres=0 here since logits are used
        ],
        loss_weights=[config['z0_loss_weight'],
                      config['crossentropy_loss_weight'],
                      0]
    )
    model.summary()
    model.load_weights(kf+"best_weights.tf")

    predictedZ0_FH = []
    predictedZ0_FHz0res = []
    predictedZ0_FHz0MVA = []
    predictedZ0_FHnoFake = []
    predictedZ0_NN = []

    predictedWeights = []

    predictedAssoc_NN = []
    predictedAssoc_FH = []
    predictedAssoc_FHres = []
    predictedAssoc_FHMVA = []
    predictedAssoc_FHnoFake = []

    num_threshold = 10
    thresholds = [str(i/num_threshold) for i in range(0,num_threshold)]
    predictedMET_NN = {key: value for key, value in zip(thresholds, [[] for i in range(0,num_threshold)])}
    predictedMETphi_NN = {key: value for key, value in zip(thresholds, [[] for i in range(0,num_threshold)])}

    predictedMET_FH = []
    predictedMET_FHres = []
    predictedMET_FHMVA = []
    predictedMET_FHnoFake = []

    predictedMETphi_FH = []
    predictedMETphi_FHres = []
    predictedMETphi_FHMVA = []
    predictedMETphi_FHnoFake = []

    actual_Assoc = []
    actual_PV = []
    actual_MET = []
    actual_METphi = []
    actual_trkMET = []
    actual_trkMETphi = []

    trk_z0 = []
    trk_MVA = []
    trk_phi = []
    trk_pt = []
    trk_eta = []

    trk_chi2rphi = []
    trk_chi2rz = []
    trk_bendchi2 = []

    threshold = -1

    for step,batch in enumerate(setup_pipeline(test_files)):

        trackFeatures = np.stack([batch[feature] for feature in trackfeat],axis=2)
        WeightFeatures = np.stack([batch[feature] for feature in weightfeat],axis=2)
        nBatch = batch['pvz0'].shape[0]
        FH = predictFastHisto(batch[z0],batch['trk_pt'])
        predictedZ0_FH.append(FH)
        predictedZ0_FHz0res.append(predictFastHistoZ0res(batch[z0],batch['trk_pt'],batch['trk_eta']))
        predictedZ0_FHz0MVA.append(predictFastHistoMVAcut(batch[z0],batch['trk_pt'],batch['trk_MVA1']))
        predictedZ0_FHnoFake.append(predictFastHistoNoFakes(batch[z0],batch['trk_pt'],batch['trk_fake']))

        trk_z0.append(batch[z0])
        trk_MVA.append(batch["trk_MVA1"])
        trk_pt.append(batch['normed_trk_pt'])
        trk_eta.append(batch['normed_trk_eta'])
        trk_phi.append(batch['trk_phi'])

        trk_chi2rphi.append(batch['binned_trk_chi2rphi'])
        trk_chi2rz.append(batch['binned_trk_chi2rz'])
        trk_bendchi2.append(batch['binned_trk_bendchi2'])

        actual_Assoc.append(batch["trk_fromPV"])
        actual_PV.append(batch['pvz0'])
        FHassoc = FastHistoAssoc(predictFastHisto(batch[z0],batch['trk_pt']),batch[z0],batch['trk_eta'],kf)
        predictedAssoc_FH.append(FHassoc)
        FHassocres = FastHistoAssoc(predictFastHistoZ0res(batch[z0],batch['trk_pt'],batch['trk_eta']),batch[z0],batch['trk_eta'],kf)
        predictedAssoc_FHres.append(FHassocres)

        FHassocMVA = FastHistoAssocMVAcut(predictFastHistoMVAcut(batch[z0],batch['trk_pt'],batch['trk_MVA1']),batch[z0],batch['trk_eta'],batch['trk_MVA1'],kf)
        predictedAssoc_FHMVA.append(FHassocMVA)

        FHassocnoFake = FastHistoAssocNoFakes(predictFastHistoNoFakes(batch[z0],batch['trk_pt'],batch['trk_fake']),batch[z0],batch['trk_eta'],batch['trk_fake'],kf)
        predictedAssoc_FHnoFake.append(FHassocnoFake)

        predictedZ0_NN_temp, predictedAssoc_NN_temp, predictedWeights_NN = model.predict_on_batch(
                        [batch[z0],WeightFeatures,trackFeatures]
                    )

        predictedAssoc_NN_temp = tf.math.divide( tf.math.subtract( predictedAssoc_NN_temp,tf.reduce_min(predictedAssoc_NN_temp)), 
                            tf.math.subtract( tf.reduce_max(predictedAssoc_NN_temp), tf.reduce_min(predictedAssoc_NN_temp) ))


        predictedZ0_NN.append(predictedZ0_NN_temp)
        predictedAssoc_NN.append(predictedAssoc_NN_temp)
        predictedWeights.append(predictedWeights_NN)

        actual_MET.append(batch['tp_met_pt'])
        actual_METphi.append(batch['tp_met_phi'])

        temp_met,temp_metphi = predictMET(batch['trk_pt'],batch['trk_phi'],batch['trk_fromPV'],threshold=0.5)
        actual_trkMET.append(temp_met)
        actual_trkMETphi.append(temp_metphi)

        temp_met,temp_metphi = predictMET(batch['trk_pt'],batch['trk_phi'],FHassocnoFake,threshold=0.5)
        predictedMET_FHnoFake.append(temp_met)
        predictedMETphi_FHnoFake.append(temp_metphi)

        temp_met,temp_metphi = predictMET(batch['trk_pt'],batch['trk_phi'],FHassocMVA,threshold=0.5)
        predictedMET_FHMVA.append(temp_met)
        predictedMETphi_FHMVA.append(temp_metphi)

        temp_met,temp_metphi = predictMET(batch['trk_pt'],batch['trk_phi'],FHassocres,threshold=0.5)
        predictedMET_FHres.append(temp_met)
        predictedMETphi_FHres.append(temp_metphi)

        temp_met,temp_metphi = predictMET(batch['trk_pt'],batch['trk_phi'],FHassoc,threshold=0.5)
        predictedMET_FH.append(temp_met)
        predictedMETphi_FH.append(temp_metphi)

        for i in range(0,num_threshold):
            temp_met,temp_metphi = predictMET(batch['trk_pt'],batch['trk_phi'],predictedAssoc_NN_temp.numpy().squeeze(),threshold=i/num_threshold)
            predictedMET_NN[str(i/num_threshold)].append(temp_met)
            predictedMETphi_NN[str(i/num_threshold)].append(temp_metphi)

    z0_NN_array = np.concatenate(predictedZ0_NN).ravel()
    z0_FH_array = np.concatenate(predictedZ0_FH).ravel()
    z0_FHres_array = np.concatenate(predictedZ0_FHz0res).ravel()
    z0_FHMVA_array = np.concatenate(predictedZ0_FHz0MVA).ravel()
    z0_FHnoFake_array = np.concatenate(predictedZ0_FHnoFake).ravel()
    z0_PV_array = np.concatenate(actual_PV).ravel()

    predictedWeightsarray = np.concatenate(predictedWeights).ravel()

    trk_z0_array = np.concatenate(trk_z0).ravel()
    trk_mva_array = np.concatenate(trk_MVA).ravel()
    trk_pt_array = np.concatenate(trk_pt).ravel()
    trk_eta_array = np.concatenate(trk_eta).ravel()
    trk_phi_array = np.concatenate(trk_phi).ravel()

    trk_chi2rphi_array = np.concatenate(trk_chi2rphi).ravel()
    trk_chi2rz_array = np.concatenate(trk_chi2rz).ravel()
    trk_bendchi2_array = np.concatenate(trk_bendchi2).ravel()


    assoc_NN_array = np.concatenate(predictedAssoc_NN).ravel()
    assoc_FH_array = np.concatenate(predictedAssoc_FH).ravel()
    assoc_FHres_array = np.concatenate(predictedAssoc_FHres).ravel()
    assoc_FHMVA_array = np.concatenate(predictedAssoc_FHMVA).ravel()
    assoc_FHnoFake_array = np.concatenate(predictedAssoc_FHnoFake).ravel()
    assoc_PV_array = np.concatenate(actual_Assoc).ravel()

    actual_MET_array = np.concatenate(actual_MET).ravel()
    actual_METphi_array = np.concatenate(actual_METphi).ravel()
    actual_trkMET_array = np.concatenate(actual_trkMET).ravel()
    actual_trkMETphi_array = np.concatenate(actual_trkMETphi).ravel()
    MET_FHnoFake_array = np.concatenate(predictedMET_FHnoFake).ravel()
    METphi_FHnoFake_array = np.concatenate(predictedMETphi_FHnoFake).ravel()
    MET_FHMVA_array = np.concatenate(predictedMET_FHMVA).ravel()
    METphi_FHMVA_array = np.concatenate(predictedMETphi_FHMVA).ravel()
    MET_FHres_array = np.concatenate(predictedMET_FHres).ravel()
    METphi_FHres_array = np.concatenate(predictedMETphi_FHres).ravel()
    MET_FH_array = np.concatenate(predictedMET_FH).ravel()
    METphi_FH_array = np.concatenate(predictedMETphi_FH).ravel()

    MET_NN_RMS_array = np.zeros([num_threshold])
    MET_NN_Quartile_array = np.zeros([num_threshold])
    MET_NN_Centre_array = np.zeros([num_threshold])
    METphi_NN_RMS_array = np.zeros([num_threshold])
    METphi_NN_Quartile_array = np.zeros([num_threshold])
    METphi_NN_Centre_array = np.zeros([num_threshold])
    

    for i in range(0,num_threshold):
        MET_NN_array = np.concatenate(predictedMET_NN[str(i/num_threshold)]).ravel()
        METphi_NN_array = np.concatenate(predictedMETphi_NN[str(i/num_threshold)]).ravel()

        Diff = MET_NN_array - actual_MET_array
        PhiDiff = METphi_NN_array - actual_METphi_array

        MET_NN_RMS_array[i] = np.sqrt(np.mean(Diff**2))
        METphi_NN_RMS_array[i] = np.sqrt(np.mean(PhiDiff**2))

        qMET = np.percentile(Diff,[32,50,68])
        qMETphi = np.percentile(PhiDiff,[32,50,68])

        MET_NN_Quartile_array[i] = qMET[2] - qMET[0]
        METphi_NN_Quartile_array[i] = qMETphi[2] - qMETphi[0]

        MET_NN_Centre_array[i] = qMET[1]
        METphi_NN_Centre_array[i] = qMETphi[1]


    MET_NN_bestQ_array = np.concatenate(predictedMET_NN[str(np.argmin(MET_NN_Quartile_array)/num_threshold)]).ravel()
    METphi_NN_bestQ_array = np.concatenate(predictedMETphi_NN[str(np.argmin(MET_NN_Quartile_array)/num_threshold)]).ravel()

    MET_NN_bestRMS_array = np.concatenate(predictedMET_NN[str(np.argmin(MET_NN_RMS_array)/num_threshold)]).ravel()
    METphi_NN_bestRMS_array = np.concatenate(predictedMETphi_NN[str(np.argmin(MET_NN_RMS_array)/num_threshold)]).ravel()

    pv_track_sel = assoc_PV_array == 1
    pu_track_sel = assoc_PV_array == 0

    weightmax = np.max(predictedWeightsarray)
    
    fig,ax = plt.subplots(1,1,figsize=(10,10))

    '''
    plt.clf()

    plt.clf()
    plt.hist(trk_bendchi2_array[pv_track_sel],range=(0,1), bins=50, label="PV tracks", alpha=0.5, density=True)
    plt.hist(trk_bendchi2_array[pu_track_sel],range=(0,1), bins=50, label="PU tracks", alpha=0.5, density=True)
    plt.xlabel("Track $\\chi^2_{bend}$", horizontalalignment='right', x=1.0)
    # plt.ylabel("tracks/counts", horizontalalignment='right', y=1.0)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/bendchi2hist.png" % outputFolder)
    plt.clf()


    plt.clf()
    plt.hist(predictedWeightsarray[pv_track_sel],range=(0,weightmax), bins=50, label="PV tracks", alpha=0.5, density=True)
    plt.hist(predictedWeightsarray[pu_track_sel],range=(0,weightmax), bins=50, label="PU tracks", alpha=0.5, density=True)
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    # plt.ylabel("tracks/counts", horizontalalignment='right', y=1.0)
    plt.yscale("log")
    plt.title("Histogram weights for PU and PV tracks")
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/corr-assoc-1d.png" % outputFolder)
    plt.clf()

    pv_track_no = np.sum(pv_track_sel)
    pu_track_no = np.sum(pu_track_sel)


    assoc_scale = (pv_track_no / pu_track_no)
    # plt.bar(b[:-1], h, width=bw, label="PV tracks", alpha=0.5)

    

    plt.hist(predictedWeightsarray[pv_track_sel],range=(0,weightmax), bins=50, label="PV tracks", alpha=0.5, weights=np.ones_like(predictedWeightsarray[pv_track_sel]) / assoc_scale)
    plt.hist(predictedWeightsarray[pu_track_sel],range=(0,weightmax), bins=50, label="PU tracks", alpha=0.5)
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("a.u.", horizontalalignment='right', y=1.0)
    plt.title("Histogram weights for PU and PV tracks (normalised)")
    plt.yscale("log")

    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/corr-assoc-1d-norm.png" % outputFolder)

    plt.clf()
    plt.hist2d(predictedWeightsarray, assoc_NN_array, range=((0,weightmax),(0,1)),bins=50, norm=matplotlib.colors.LogNorm());
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("track-to-vertex association flag", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/corr-assoc.png" %  outputFolder)

    plt.clf()
    plt.hist2d(predictedWeightsarray, trk_z0_array, bins=50, norm=matplotlib.colors.LogNorm());
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("track $z_0$ [cm]", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/corr-z0.png" %  outputFolder)

    plt.clf()
    plt.hist2d(predictedWeightsarray, trk_mva_array, bins=50, norm=matplotlib.colors.LogNorm());
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("track MVA", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/corr-mva.png" %  outputFolder)

    plt.clf()
    plt.hist2d(predictedWeightsarray, trk_pt_array, bins=50, norm=matplotlib.colors.LogNorm());
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("track $p_T$ [GeV]", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/corr-pt.png" %  outputFolder)

    plt.clf()
    plt.hist2d(predictedWeightsarray, np.abs(trk_eta_array), bins=50, norm=matplotlib.colors.LogNorm());
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("track $|\\eta|$", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/corr-abs-eta.png" %  outputFolder)

    plt.clf()
    plt.hist2d(predictedWeightsarray, trk_eta_array, bins=50, norm=matplotlib.colors.LogNorm());
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("track $\\eta$", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/corr-eta.png" %  outputFolder)

    plt.clf()
    plt.hist2d(predictedWeightsarray, trk_chi2rphi_array, bins=50, norm=matplotlib.colors.LogNorm());
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("track $\\chi^2_{r\\phi}$", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/corr-chi2rphi.png" % outputFolder)

    plt.clf()
    plt.hist2d(predictedWeightsarray, trk_chi2rz_array, bins=50, norm=matplotlib.colors.LogNorm());
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("track $\\chi^2_{rz}$", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/corr-chi2rz.png" % outputFolder)

    plt.clf()
    plt.hist2d(predictedWeightsarray, trk_bendchi2_array, bins=50, norm=matplotlib.colors.LogNorm());
    plt.xlabel("weights", horizontalalignment='right', x=1.0)
    plt.ylabel("track $\\chi^2_{bend}$", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/corr-chi2bend.png" % outputFolder)
    '''
    do_scatter = False
    if (do_scatter):
        plt.clf()
        plt.scatter(predictedWeightsarray, trk_z0_array, label="z0")
        plt.xlabel("weight")
        plt.ylabel("variable")
        plt.title("Correlation between predicted weight and track variables")
        plt.legend()
        plt.savefig("%s/scatter-z0.png" %  outputFolder)
        # plt.show()

        plt.clf()
        plt.scatter(predictedWeightsarray, trk_pt_array, label="pt")
        plt.xlabel("weight")
        plt.ylabel("variable")
        plt.title("Correlation between predicted weight and track variables")
        plt.legend()
        plt.savefig("%s/scatter-pt.png" % outputFolder)

        plt.clf()
        plt.scatter(predictedWeightsarray, trk_eta_array, label="eta")
        plt.xlabel("weight")
        plt.ylabel("variable")
        plt.title("Correlation between predicted weight and track variables")
        plt.legend()
        plt.savefig("%s/scatter-eta.png" %  outputFolder)

    plt.clf()
    figure=plotz0_residual([(z0_PV_array-z0_NN_array)],
                          [(z0_PV_array-z0_FH_array),(z0_PV_array-z0_FHMVA_array),(z0_PV_array-z0_FHnoFake_array)],
                          ["ArgMax"],
                          ["Base","MVA Cut","No Fakes"])
    plt.savefig("%s/Z0Residual.png" % outputFolder)

    #plt.clf()
    #figure=plotPV_roc(assoc_PV_array,[assoc_NN_array],
    #                 [assoc_FH_array,assoc_FHMVA_array,assoc_FHnoFake_array],
    #                 ["ArgMax"],
    #                 ["Base","MVA Cut","No Fakes"])
    #plt.savefig("%s/PVROC.png" % outputFolder)

    #plt.clf()
    #figure=plotz0_percentile([(z0_PV_array-z0_NN_array)],
    #                         [(z0_PV_array-z0_FH_array),(z0_PV_array-z0_FHMVA_array),(z0_PV_array-z0_FHnoFake_array)],
    #                         ["ArgMax"],
    #                         ["Base","MVA Cut","No Fakes"])
    #plt.savefig("%s/Z0percentile.png" % outputFolder)

    plt.clf()
    figure=plotMET_residual([(actual_MET_array-MET_NN_bestQ_array)],
                             [(actual_MET_array-MET_FH_array),(actual_MET_array-actual_trkMET_array)],
                             ["ArgMax thresh=" + str(np.argmin(MET_NN_Quartile_array)/num_threshold)],
                             ["Base","PV Tracks"],range=(-100,100),logrange=(-300,300))
    plt.savefig("%s/METbestQresidual.png" % outputFolder)

    plt.clf()
    figure=plotMETphi_residual([(actual_METphi_array-METphi_NN_bestQ_array)],
                             [(actual_METphi_array-METphi_FH_array),(actual_METphi_array-actual_trkMETphi_array)],
                             ["ArgMax thresh=" + str(np.argmin(MET_NN_Quartile_array)/num_threshold)],
                             ["Base","PV Tracks"])
    plt.savefig("%s/METphibestQresidual.png" % outputFolder)

    plt.clf()
    figure=plotMET_residual([(actual_MET_array-MET_NN_bestRMS_array)],
                             [(actual_MET_array-MET_FH_array),(actual_MET_array-actual_trkMET_array)],
                             ["ArgMax thresh=" + str(np.argmin(MET_NN_RMS_array)/num_threshold)],
                             ["Base","PV Tracks"],range=(-100,100),logrange=(-300,300))
    plt.savefig("%s/METbestRMSresidual.png" % outputFolder)

    plt.clf()
    figure=plotMETphi_residual([(actual_METphi_array-METphi_NN_bestRMS_array)],
                             [(actual_METphi_array-METphi_FH_array),(actual_METphi_array-actual_trkMETphi_array)],
                             ["ArgMax thresh=" + str(np.argmin(MET_NN_RMS_array)/num_threshold)],
                             ["Base","PV Tracks"])
    plt.savefig("%s/METphibestRMSresidual.png" % outputFolder)

    fig,ax = plt.subplots(1,1,figsize=(10,10))

    plt.clf()
    plt.hist(z0_FH_array,range=(-15,15),bins=120,density=True,color='r',histtype="step",label="FastHisto Base")
    plt.hist(z0_FHres_array,range=(-15,15),bins=120,density=True,color='g',histtype="step",label="FastHisto with z0 res")
    plt.hist(z0_NN_array,range=(-15,15),bins=120,density=True,color='b',histtype="step",label="CNN")
    plt.hist(z0_PV_array,range=(-15,15),bins=120,density=True,color='y',histtype="step",label="Truth")
    plt.grid(True)
    plt.xlabel('$z_0$ [cm]',ha="right",x=1)
    plt.ylabel('Events',ha="right",y=1)
    plt.legend() 
    plt.tight_layout()
    plt.savefig("%s/z0hist.png" % outputFolder)

    plt.clf()
    plt.hist2d(z0_PV_array, (z0_PV_array-z0_FH_array), bins=60,range=((-15,15),(-30,30)) ,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("PV", horizontalalignment='right', x=1.0)
    plt.ylabel("PV - $z_0^{FH}$", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/FHerr_vs_z0.png" %  outputFolder)

    plt.clf()
    plt.hist2d(z0_PV_array, (z0_PV_array-z0_FHnoFake_array), bins=60,range=((-15,15),(-30,30)) ,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("PV", horizontalalignment='right', x=1.0)
    plt.ylabel("PV - $z_0^{FH}$", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/FHnoFakeerr_vs_z0.png" %  outputFolder)

    plt.clf()
    plt.hist2d(z0_PV_array, (z0_PV_array-z0_FHMVA_array), bins=60,range=((-15,15),(-30,30)) ,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("PV", horizontalalignment='right', x=1.0)
    plt.ylabel("PV - $z_0^{FH}$", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/FHMVAerr_vs_z0.png" %  outputFolder)

    plt.clf()
    plt.hist2d(z0_PV_array, (z0_PV_array-z0_NN_array), bins=60,range=((-15,15),(-30,30)), norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("PV", horizontalalignment='right', x=1.0)
    plt.ylabel("PV - $z_0^{NN}$", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/NNerr_vs_z0.png" %  outputFolder)

    plt.clf()
    plt.hist2d(z0_PV_array, z0_FH_array, bins=60,range=((-15,15),(-15,15)) ,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("PV", horizontalalignment='right', x=1.0)
    plt.ylabel("$z_0^{FH}$", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/FH_vs_z0.png" %  outputFolder)

    plt.clf()
    plt.hist2d(z0_PV_array, z0_FHMVA_array, bins=60,range=((-15,15),(-15,15)) ,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("PV", horizontalalignment='right', x=1.0)
    plt.ylabel("$z_0^{FH}$", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/FHMVA_vs_z0.png" %  outputFolder)

    plt.clf()
    plt.hist2d(z0_PV_array, z0_FHnoFake_array, bins=60,range=((-15,15),(-15,15)) ,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("PV", horizontalalignment='right', x=1.0)
    plt.ylabel("$z_0^{FH}$", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/FHnoFake_vs_z0.png" %  outputFolder)

    plt.clf()
    plt.hist2d(z0_PV_array, z0_NN_array, bins=60,range=((-15,15),(-15,15)), norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("PV", horizontalalignment='right', x=1.0)
    plt.ylabel("$z_0^{NN}$", horizontalalignment='right', y=1.0)
    #plt.colorbar(vmin=0,vmax=1000)
    plt.tight_layout()
    plt.savefig("%s/NN_vs_z0.png" %  outputFolder)

    fig,ax = plt.subplots(1,1,figsize=(10,10))

    plt.clf()
    plt.hist2d(actual_MET_array, MET_NN_bestQ_array, bins=60,range=((0,300),(0,300)), norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("$E_{T}^{miss}$", horizontalalignment='right', x=1.0)
    plt.ylabel("$E_{T}^{miss,NN}$", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/NNbestQ_vs_MET.png" %  outputFolder)

    plt.clf()
    plt.hist2d(actual_MET_array, MET_NN_bestRMS_array, bins=60,range=((0,300),(0,300)), norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("$E_{T}^{miss}$", horizontalalignment='right', x=1.0)
    plt.ylabel("$E_{T}^{miss,NN}$", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/NNbestRMS_vs_MET.png" %  outputFolder)

    plt.clf()
    plt.hist2d(actual_MET_array, MET_FH_array, bins=60,range=((0,300),(0,300)), norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("$E_{T}^{miss}$", horizontalalignment='right', x=1.0)
    plt.ylabel("$E_{T}^{miss,FH}$", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/FH_vs_MET.png" %  outputFolder)

    plt.clf()
    plt.hist2d(actual_METphi_array, METphi_NN_bestQ_array, bins=60,range=((-np.pi,np.pi),(-np.pi,np.pi)), norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("$E_{T,\\phi}^{miss}$", horizontalalignment='right', x=1.0)
    plt.ylabel("$E_{T,\\phi}^{miss,NN}$", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/NNbestQ_vs_METphi.png" %  outputFolder)

    plt.clf()
    plt.hist2d(actual_METphi_array, METphi_NN_bestRMS_array, bins=60,range=((-np.pi,np.pi),(-np.pi,np.pi)), norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("$E_{T,\\phi}^{miss}$", horizontalalignment='right', x=1.0)
    plt.ylabel("$E_{T,\\phi}^{miss,NN}$", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/NNbestRMS_vs_METphi.png" %  outputFolder)

    plt.clf()
    plt.hist2d(actual_METphi_array, METphi_FH_array, bins=60,range=((-np.pi,np.pi),(-np.pi,np.pi)), norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
    plt.xlabel("$E_{T,\\phi}^{miss}$", horizontalalignment='right', x=1.0)
    plt.ylabel("$E_{T,\\phi}^{miss,FH}$", horizontalalignment='right', y=1.0)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("%s/FH_vs_METphi.png" %  outputFolder)

    def calc_widths(actual,predicted):
        diff = (actual-predicted)
        RMS = np.sqrt(np.mean(diff**2))
        qs = np.percentile(diff,[32,50,68])
        qwidth = qs[2] - qs[0]
        qcentre = qs[1]

        return [RMS,qwidth,qcentre]

    FHwidths = calc_widths(actual_MET_array,MET_FH_array)
    FHMVAwidths = calc_widths(actual_MET_array,MET_FHMVA_array)
    FHNoFakeWidths = calc_widths(actual_MET_array,MET_FHnoFake_array)
    TrkMETWidths = calc_widths(actual_MET_array,actual_trkMET_array)

    FHphiwidths = calc_widths(actual_METphi_array,METphi_FH_array)
    FHMVAphiwidths = calc_widths(actual_METphi_array,METphi_FHMVA_array)
    FHNoFakephiWidths = calc_widths(actual_METphi_array,METphi_FHnoFake_array)
    TrkMETphiWidths = calc_widths(actual_METphi_array,actual_trkMETphi_array)


    plt.clf()
    plt.plot(thresholds,MET_NN_RMS_array,label="Argmax NN",markersize=10,linewidth=LINEWIDTH,marker='o')
    plt.plot(thresholds,np.full(len(thresholds),FHwidths[0]),label="Base FH",linestyle='--',linewidth=LINEWIDTH)
    plt.plot(thresholds,np.full(len(thresholds),FHMVAwidths[0]),label="BDT Cut FH",linestyle='--',linewidth=LINEWIDTH)
    plt.plot(thresholds,np.full(len(thresholds),FHNoFakeWidths[0]),label="No Fakes FH",linestyle='--',linewidth=LINEWIDTH)
    plt.plot(thresholds,np.full(len(thresholds),TrkMETWidths[0]),label="PV Tracks FH",linestyle='--',linewidth=LINEWIDTH)
    plt.ylabel("$E_{T}^{miss}$ Residual RMS", horizontalalignment='right', y=1.0)
    plt.xlabel("Track-to-vertex association threshold", horizontalalignment='right', x=1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/METRMSvsThreshold.png" %  outputFolder)

    plt.clf()
    plt.plot(thresholds,MET_NN_Quartile_array,label="Argmax NN",markersize=10,linewidth=LINEWIDTH,marker='o')
    plt.plot(thresholds,np.full(len(thresholds),FHwidths[1]),label="Base FH",linestyle='--',linewidth=LINEWIDTH)
    plt.plot(thresholds,np.full(len(thresholds),FHMVAwidths[1]),label="BDT Cut FH",linestyle='--',linewidth=LINEWIDTH)
    plt.plot(thresholds,np.full(len(thresholds),FHNoFakeWidths[1]),label="No Fakes FH",linestyle='--',linewidth=LINEWIDTH)
    plt.plot(thresholds,np.full(len(thresholds),TrkMETWidths[1]),label="PV Tracks FH",linestyle='--',linewidth=LINEWIDTH)
    plt.ylabel("$E_{T}^{miss}$ Residual Quartile Width", horizontalalignment='right', y=1.0)
    plt.xlabel("Track-to-vertex association threshold", horizontalalignment='right', x=1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/METQsvsThreshold.png" %  outputFolder)

    plt.clf()
    plt.plot(thresholds,MET_NN_Centre_array,label="Argmax NN",markersize=10,linewidth=LINEWIDTH,marker='o')
    plt.plot(thresholds,np.full(len(thresholds),FHwidths[2]),label="Base FH",linestyle='--',linewidth=LINEWIDTH)
    plt.plot(thresholds,np.full(len(thresholds),FHMVAwidths[2]),label="BDT Cut FH",linestyle='--',linewidth=LINEWIDTH)
    plt.plot(thresholds,np.full(len(thresholds),FHNoFakeWidths[2]),label="No Fakes FH",linestyle='--',linewidth=LINEWIDTH)
    plt.plot(thresholds,np.full(len(thresholds),TrkMETWidths[2]),label="PV Tracks FH",linestyle='--',linewidth=LINEWIDTH)
    plt.ylabel("$E_{T}^{miss}$ Residual Centre", horizontalalignment='right', y=1.0)
    plt.xlabel("Track-to-vertex association threshold", horizontalalignment='right', x=1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/METCentrevsThreshold.png" %  outputFolder)

    plt.clf()
    plt.plot(thresholds,METphi_NN_RMS_array,label="Argmax NN",markersize=10,linewidth=LINEWIDTH,marker='o')
    plt.plot(thresholds,np.full(len(thresholds),FHphiwidths[0]),label="Base FH",linestyle='--',linewidth=LINEWIDTH)
    plt.plot(thresholds,np.full(len(thresholds),FHMVAphiwidths[0]),label="BDT Cut FH",linestyle='--',linewidth=LINEWIDTH)
    plt.plot(thresholds,np.full(len(thresholds),FHNoFakephiWidths[0]),label="No Fakes FH",linestyle='--',linewidth=LINEWIDTH)
    plt.plot(thresholds,np.full(len(thresholds),TrkMETphiWidths[0]),label="PV Tracks FH",linestyle='--',linewidth=LINEWIDTH)
    plt.ylabel("$E_{T,\\phi}^{miss}$ Residual RMS", horizontalalignment='right', y=1.0)
    plt.xlabel("Track-to-vertex association threshold", horizontalalignment='right', x=1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/METphiRMSvsThreshold.png" %  outputFolder)

    plt.clf()
    plt.plot(thresholds,METphi_NN_Quartile_array,label="Argmax NN",markersize=10,linewidth=LINEWIDTH,marker='o')
    plt.plot(thresholds,np.full(len(thresholds),FHphiwidths[1]),label="Base FH",linestyle='--',linewidth=LINEWIDTH)
    plt.plot(thresholds,np.full(len(thresholds),FHMVAphiwidths[1]),label="BDT Cut FH",linestyle='--',linewidth=LINEWIDTH)
    plt.plot(thresholds,np.full(len(thresholds),FHNoFakephiWidths[1]),label="No Fakes FH",linestyle='--',linewidth=LINEWIDTH)
    plt.plot(thresholds,np.full(len(thresholds),TrkMETphiWidths[1]),label="PV Tracks FH",linestyle='--',linewidth=LINEWIDTH)
    plt.ylabel("$E_{T,\\phi}^{miss}$ Residual Quartile Width", horizontalalignment='right', y=1.0)
    plt.xlabel("Track-to-vertex association threshold", horizontalalignment='right', x=1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/METphiQsvsThreshold.png" %  outputFolder)

    plt.clf()
    plt.plot(thresholds,METphi_NN_Centre_array,label="Argmax NN",markersize=10,linewidth=LINEWIDTH,marker='o')
    plt.plot(thresholds,np.full(len(thresholds),FHphiwidths[2]),label="Base FH",linestyle='--',linewidth=LINEWIDTH)
    plt.plot(thresholds,np.full(len(thresholds),FHMVAphiwidths[2]),label="BDT Cut FH",linestyle='--',linewidth=LINEWIDTH)
    plt.plot(thresholds,np.full(len(thresholds),FHNoFakephiWidths[2]),label="No Fakes FH",linestyle='--',linewidth=LINEWIDTH)
    plt.plot(thresholds,np.full(len(thresholds),TrkMETphiWidths[2]),label="PV Tracks FH",linestyle='--',linewidth=LINEWIDTH)
    plt.ylabel("$E_{T,\\phi}^{miss}$ Residual Centre", horizontalalignment='right', y=1.0)
    plt.xlabel("Track-to-vertex association threshold", horizontalalignment='right', x=1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/METphiCentrevsThreshold.png" %  outputFolder)

    experiment.log_asset_folder(outputFolder, step=None, log_file_name=True)
    experiment.log_asset(sys.argv[2]+'.yaml')
