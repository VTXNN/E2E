from re import L, X
import tensorflow as tf
import numpy as np

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
#hep.set_style("CMSTex")

import glob
import sklearn.metrics as metrics
import vtx
import math
from textwrap import wrap

hep.cms.label()
hep.cms.text("Simulation")
plt.style.use(hep.style.CMS)

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

LEGEND_WIDTH = 20
LINEWIDTH = 3
MARKERSIZE = 20

nbins = 256
max_z0 = 20.46912512

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

from TrainingScripts.train import *

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

def comb_res_function(mva,eta):
        res = 0.1 + 0.2*eta**2
        return ((mva)/8)/res

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

def predictFastHisto(value,weight, res_func, return_index = False):
    z0List = []
    halfBinWidth = 0.5*(2*max_z0)/nbins

    for ibatch in range(value.shape[0]):
        hist,bin_edges = np.histogram(value[ibatch],nbins,range=(-1*max_z0,max_z0),weights=weight[ibatch]*res_func[ibatch])
        hist = np.convolve(hist,[1,1,1],mode='same')
        z0Index= np.argmax(hist)
        if return_index:
            z0List.append([z0Index])
        else:
            z0 = -1*max_z0 +(2*max_z0)*z0Index/nbins+halfBinWidth
            z0List.append([z0])
    return np.array(z0List,dtype=np.float32)

def FastHistoAssoc(PV,trk_z0,trk_eta, res_func):

    deltaz_bins = np.array([0.0,0.37,0.5,0.6,0.75,1.0,1.6,0.0])
    eta_bins = np.array([0.0,0.7,1.0,1.2,1.6,2.0,2.4])
    

    eta_bin = np.digitize(abs(trk_eta),eta_bins)
    deltaz = abs(trk_z0 - PV)

    assoc = (deltaz < deltaz_bins[eta_bin]) & res_func

    return np.array(assoc,dtype=np.float32)

def predictMET(pt,phi,predictedAssoc,threshold,quality_func):
    met_pt_list = []
    met_phi_list = []

    def assoc_function(Assoc):
        res = Assoc > threshold
        return res

    for ibatch in range(pt.shape[0]):
        assoc = assoc_function(predictedAssoc[ibatch])
        selection = np.logical_and(assoc,quality_func[ibatch])

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
        ax[0].hist(FH,bins=50,range=(-1*max_z0,max_z0),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s \nRMS = %.4f" 
                 %(FHnames[i],np.sqrt(np.mean(FH**2))),LEGEND_WIDTH)),density=True)
        ax[1].hist(FH,bins=50,range=(-1,1),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s \nQuartile Width = %.4f" 
                 %(FHnames[i],qz0_FH[2]-qz0_FH[0]),LEGEND_WIDTH)),density=True)
        items+=1

    for i,NN in enumerate(NNdiff):
        qz0_NN = np.percentile(NN,[32,50,68])
        ax[0].hist(NN,bins=50,range=(-1*max_z0,max_z0),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s \nRMS = %.4f" 
                 %(NNnames[i],np.sqrt(np.mean(NN**2))),LEGEND_WIDTH)),density=True)
        ax[1].hist(NN,bins=50,range=(-1,1),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s \nQuartile Width = %.4f" 
                 %(NNnames[i],qz0_NN[2]-qz0_NN[0]),LEGEND_WIDTH)),density=True)
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
            FH = (FH - actual[i]) / actual[i]
            temp_actual = actual[i][~np.isnan(FH)]
            FH = FH[~np.isnan(FH)]
            temp_actual = temp_actual[np.isfinite(FH)]
            FH = FH[np.isfinite(FH)]

            if logbins:
                FH = FH + 1
            
        else:
            FH = (FH - actual[i])
            temp_actual = actual[i]
        qz0_FH = np.percentile(FH,[32,50,68])

        ax[0].hist(FH,bins=bins,histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s RMS = %.4f" 
                 %(FHnames[i],metrics.mean_squared_error(temp_actual,FH,squared=False)),LEGEND_WIDTH)),density=True)
        ax[1].hist(FH,bins=50,range=range,histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s Quartile Width = %.4f   Centre = %.4f" 
                 %(FHnames[i],qz0_FH[2]-qz0_FH[0], qz0_FH[1]),25)),density=True)
        items+=1

    for i,NN in enumerate(NNdiff):
        if relative:
            NN = (NN - actual[i])/actual[i]
            temp_actual = actual[i][~np.isnan(NN)]
            NN = NN[~np.isnan(NN)]
            temp_actual = temp_actual[np.isfinite(NN)]
            NN = NN[np.isfinite(NN)]

            if logbins:
                NN = NN + 1

        else:
            NN = (NN - actual[i])
            temp_actual = actual[i]
        qz0_NN = np.percentile(NN,[32,50,68])
        ax[0].hist(NN,bins=bins,histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s RMS = %.4f" 
                 %(NNnames[i],metrics.mean_squared_error(temp_actual,NN,squared=False)),LEGEND_WIDTH)),density=True)
        ax[1].hist(NN,bins=50,range=range,histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s Quartile Width = %.4f   Centre = %.4f" 
                 %(NNnames[i],qz0_NN[2]-qz0_NN[0], qz0_NN[1]),25)),density=True)
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
        tnFH, fpFH, fnFH, tpFH = metrics.confusion_matrix(actual[i], FH).ravel()
        precisionFH = tpFH / (tpFH + fpFH) 
        recallFH = tpFH / (tpFH + fnFH) 
        TPRFH = recallFH
        FPRFH = fpFH / (fpFH + tnFH) 
        ax[0].plot(recallFH,precisionFH,label=str(FHnames[i]),linewidth=LINEWIDTH,color=colours[items],marker='o')
        ax[1].plot(TPRFH,FPRFH,label='\n'.join(wrap(f"%s AUC: %.4f" %(FHnames[i],metrics.roc_auc_score(actual[i],FH)),LEGEND_WIDTH)),color=colours[items],marker='o')
        items+=1

    for i,NN in enumerate(NNpred):
        precisionNN = []
        recallNN = []
        FPRNN = []

        NN = (NN - min(NN))/(max(NN) - min(NN))

        thresholds = np.linspace(0,1,Nthresholds)

        for j,threshold in enumerate(thresholds):
            print(str(NNnames[i]) + " Testing ROC threshold: "+str(j) + " out of "+str(len(thresholds)))
            tnNN, fpNN, fnNN, tpNN = metrics.confusion_matrix(actual[i], NN>threshold).ravel()
            precisionNN.append( tpNN / (tpNN + fpNN) )
            recallNN.append(tpNN / (tpNN + fnNN) )
            FPRNN.append(fpNN / (fpNN + tnNN) )

        
        ax[0].plot(recallNN,precisionNN,label=str(NNnames[i]),linewidth=LINEWIDTH,color=colours[items])
        ax[1].plot(recallNN,FPRNN,linewidth=LINEWIDTH,label='\n'.join(wrap(f"%s AUC: %.4f" %(NNnames[i],metrics.roc_auc_score(actual[i],NN)),LEGEND_WIDTH)),color=colours[items])
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

    PUhist,PUbin_edges = np.histogram(tracks[PU],nbins,range=(-1*max_z0,max_z0),weights=weights[PU])
    plt.bar(PUbin_edges[:-1],PUhist,width=(2*max_z0)/nbins,color='b',alpha=0.5, label="PU Trk",bottom=2)

    Fakehist,Fakebin_edges = np.histogram(tracks[fakes],nbins,range=(-1*max_z0,max_z0),weights=weights[fakes])
    plt.bar(Fakebin_edges[:-1],Fakehist,width=(2*max_z0)/nbins,color='r',alpha=0.5, label="Fake Trk",bottom=2+PUhist)

    PVhist,PVbin_edges = np.histogram(tracks[PV],nbins,range=(-1*max_z0,max_z0),weights=weights[PV])
    plt.bar(PVbin_edges[:-1],PVhist,width=(2*max_z0)/nbins,color='g',alpha=0.5, label="PV Trk",bottom=2+PUhist+Fakehist)

    maxpt = np.max(2+PUhist+Fakehist+PVhist)*1.5



    #plt.plot(tracks[PV], [2] * len(tracks[PV]), '+g', label='PV Trk',markersize=MARKERSIZE)
    #plt.plot(tracks[PU], [2] * len(tracks[PU]), '+b', label='PU Trk',markersize=MARKERSIZE)
    #plt.plot(tracks[fakes], [2] * len(tracks[fakes]), '+r', label='Fake Trk',markersize=MARKERSIZE)

    ax.plot([predictedPV, predictedPV], [0,maxpt], '--k', label='Reco Vx',linewidth=LINEWIDTH)

    ax.plot([genPV, genPV], [0,maxpt], linestyle='--',color="m", label='True Vx',linewidth=LINEWIDTH)

    ax.set_xlabel('$z_0$ [cm]',ha="right",x=1)
    ax.set_ylabel('$\\sum p_T$ [GeV]',ha="right",y=1)
    ax.set_xlim(-1*max_z0,max_z0)
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
        FH = (FH - actual[i]) / actual[i]
        temp_actual = actual[i][~np.isnan(FH)]
        FH = FH[~np.isnan(FH)]
        temp_actual = temp_actual[np.isfinite(FH)]
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
        NN = (NN - actual[i])/actual[i]
        temp_actual = actual[i][~np.isnan(NN)]
        NN = NN[~np.isnan(NN)]
        temp_actual = temp_actual[np.isfinite(NN)]
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
