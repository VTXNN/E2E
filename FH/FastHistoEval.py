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

colours=["red","black","blue","orange","purple","green"]
linestyles = ["-","--","dotted",(0, (3, 5, 1, 5)),(0, (3, 5, 1, 5, 1, 5)),(0, (3, 10, 1, 10)),(0, (3, 10, 1, 10, 1, 10))]

kf = sys.argv[1]

if kf == "NewKF":
    f = uproot.open("/home/cb719/Documents/DataSets/NewKF_TTbar_170K_quality.root")
    z0 = 'trk_z0'
elif kf == "OldKF":
    f = uproot.open("/home/cebrown/Documents/Datasets/VertexDatasets/GTT_TrackNtuple_FH_oldTQ.root")
    z0 = 'corrected_trk_z0'

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
    "trk_MVA1"
]

def linear_res_function(x,return_bool = False):
        if return_bool:
            return np.full_like(x,True).astype(bool)

        else:
            return np.ones_like(x)

def eta_res_function(eta):
        res = 0.1 + 0.2*eta**2
        return 1/res
    
def MVA_res_function(MVA,threshold=0.3,return_bool = False):
        res = MVA > threshold
        if return_bool:
            return res.astype(bool)
        else:
            return res.astype(np.float32)

def chi_res_function(chi2rphi,chi2rz,bendchi2,return_bool = False):
        qrphi = chi2rphi < 20 
        qrz =  chi2rz < 5 
        qbend = bendchi2 < 2.25
        q = np.logical_and(qrphi,qrz)
        q = np.logical_and(q,qbend)
        if return_bool:
            return q.astype(bool)
        else:
            return q.astype(np.float32)

def fake_res_function(fakes,return_bool = False):
        res = fakes != 0
        if return_bool:
            return res.astype(bool)
        else:
            return res.astype(np.float32)

def pv_res_function(pv,return_bool = False):
        res = pv == 1
        if return_bool:
            return res.astype(bool)
        else:
            return res.astype(np.float32)

def predictFastHisto(value,weight, res_func, return_index = False, nbins=256):
    z0List = []
    halfBinWidth = 0.5*30./nbins


    hist,bin_edges = np.histogram(value,nbins,range=(-15,15),weights=weight*res_func)
    hist = np.convolve(hist,[1,1,1],mode='same')
    z0Index= np.argmax(hist)
    if return_index:
        z0List.append([z0Index])
    else:
        z0 = -15.+30.*z0Index/nbins+halfBinWidth
        z0List.append([z0])
    return np.array(z0List,dtype=np.float32)

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

def FastHistoAssoc(PV,trk_z0,trk_eta, res_func, kf):
    if (kf == "NewKF") | (kf == "NewKF_intZ"):
        deltaz_bins = np.array([0.0,0.41,0.55,0.66,0.825,1.1,1.76,0.0])
    elif (kf == "OldKF") | (kf == "OldKF_intZ"):
        deltaz_bins = np.array([0.0,0.37,0.5,0.6,0.75,1.0,1.6,0.0])
    eta_bins = np.array([0.0,0.7,1.0,1.2,1.6,2.0,2.4])
    

    eta_bin = np.digitize(abs(trk_eta),eta_bins)
    deltaz = abs(trk_z0 - PV)

    assoc = (deltaz < deltaz_bins[eta_bin])[0] & res_func

    return np.array(assoc,dtype=bool)

def plotz0_residual(FHdiff,FHnames,colours=colours,linestyle=linestyles,splitplots=False):
    fig,ax = plt.subplots(1,2,figsize=(20,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])
    items = 0

    for i,FH in enumerate(FHdiff):
        qz0_FH = np.percentile(FH,[32,50,68])
        ax[0].hist(FH,bins=50,range=(-15,15),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items], linestyle=linestyles[items],
                 )
        ax[0].plot(0,1,label='\n'.join(wrap(f"%s \n RMS = %.4f" 
                 %(FHnames[i],np.sqrt(np.mean(FH**2))),LEGEND_WIDTH)),color = colours[items],markersize=0,linewidth=LINEWIDTH)
        ax[1].hist(FH,bins=50,range=(-1,1),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items], linestyle=linestyles[items],
                 )
        ax[1].plot(0,1,label='\n'.join(wrap(f"%s \nQuartile Width = %.4f" 
                 %(FHnames[i],qz0_FH[2]-qz0_FH[0]),LEGEND_WIDTH)),color = colours[items],markersize=0,linewidth=LINEWIDTH)
        items+=1

    ax[0].grid(True)
    ax[0].set_xlabel('$z^{PV}_0$ Residual [cm]',ha="right",x=1)
    ax[0].set_ylabel('Events',ha="right",y=1)
    ax[0].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95),frameon=True,facecolor='w',edgecolor='w')
    ax[0].set_yscale("log")
    ax[0].set_ylim([5,200000])

    ax[1].grid(True)
    ax[1].set_xlabel('$z^{PV}_0$ Residual [cm]',ha="right",x=1)
    ax[1].set_ylabel('Events',ha="right",y=1)
    ax[1].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95),frameon=True,facecolor='w',edgecolor='w')
    ax[1].set_ylim([0,65000])

    plt.tight_layout()

    return fig

def plot_split_z0_residual(FHdiff,FHnames,colours=colours,linestyle=linestyles):
    fig_1,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    items = 0

    for i,FH in enumerate(FHdiff):
        qz0_FH = np.percentile(FH,[32,50,68])
        ax.hist(FH,bins=50,range=(-15,15),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items], linestyle=linestyles[items])
        ax.plot(0,1,label='\n'.join(wrap(f"%s \n RMS = %.4f" 
                 %(FHnames[i],np.sqrt(np.mean(FH**2))),LEGEND_WIDTH)),color = colours[items],markersize=0,linewidth=LINEWIDTH)
        items+=1

    ax.grid(True)
    ax.set_xlabel('$z^{PV}_0$ Residual [cm]',ha="right",x=1)
    ax.set_ylabel('Events',ha="right",y=1)
    ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95),frameon=True,facecolor='w',edgecolor='w')
    ax.set_yscale("log")
    ax.set_ylim([5,200000])
    fig_1.tight_layout()

    #============================================================================================#
    plt.close()
    plt.clf()
    fig_2,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    items = 0

    for i,FH in enumerate(FHdiff):
        qz0_FH = np.percentile(FH,[32,50,68])
        ax.hist(FH,bins=50,range=(-1,1),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items], linestyle=linestyles[items])
        ax.plot(0,1,label='\n'.join(wrap(f"%s \nQuartile Width = %.4f" 
                 %(FHnames[i],qz0_FH[2]-qz0_FH[0]),LEGEND_WIDTH)),color = colours[items],markersize=0,linewidth=LINEWIDTH)
        items+=1

    ax.grid(True)
    ax.set_xlabel('$z^{PV}_0$ Residual [cm]',ha="right",x=1)
    ax.set_ylabel('Events',ha="right",y=1)
    ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95),frameon=True,facecolor='w',edgecolor='w')
    ax.set_ylim([0,65000])
    fig_2.tight_layout()

    return fig_1,fig_2

def plotPV_roc(actual,FHpred,FHnames,colours=colours):
    plt.close()
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
    
    ax[0].grid(True)
    ax[0].set_xlabel('Efficiency',ha="right",x=1)
    ax[0].set_ylabel('Purity',ha="right",y=1)
    ax[0].set_xlim([0.75,1])
    ax[0].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95),frameon=True,facecolor='w',edgecolor='k')

    ax[1].grid(True)
    ax[1].set_yscale("log")
    ax[1].set_xlabel('Track to Vertex Association True Positive Rate',ha="right",x=1)
    ax[1].set_ylabel('Track to Vertex Association False Positive Rate',ha="right",y=1)
    ax[1].set_xlim([0.75,1])
    ax[1].set_ylim([1e-2,1])
    ax[1].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95),frameon=True,facecolor='w',edgecolor='k')
    plt.tight_layout()
    return fig

def plotPV_classdistribution(actual,NNpred,FHpred):
    #NNpred = (NNpred - min(NNpred))/(max(NNpred) - min(NNpred))
    genuine = []
    fake = []

    FHgenuine = []
    FHfake = []

    for i in range(len(actual)):
        if actual[i] == 1:
            genuine.append(NNpred[i])
            FHgenuine.append(FHpred[i])
        else:
            fake.append(NNpred[i])
            FHfake.append(FHpred[i])


    plt.close()
    plt.clf()  
    fig, ax = plt.subplots(1,1, figsize=(20,10)) 

    ax[0].set_title("FH Normalised Class Distribution" ,loc='left')
    ax[0].hist(FHgenuine,color='g',bins=20,range=(0,1),histtype="step",label="Genuine",density=True,linewidth=2)
    ax[0].hist(FHfake,color='r',bins=20,range=(0,1),histtype="step",label="Fake",density=True,linewidth=2)
    ax[0].grid()
    #ax01].set_yscale("log")
    ax[0].set_xlabel("FH Output",ha="right",x=1)
    ax[0].set_ylabel("a.u.",ha="right",y=1)
    ax[0].legend(loc="upper center",frameon=True,facecolor='w',edgecolor='k')
    plt.tight_layout()
    return fig

if __name__=="__main__":

    outputFolder = kf+"FHOutput" + "/" 

    os.system("mkdir -p "+ outputFolder)

    num_threshold = 10
    thresholds = [str(i/num_threshold) for i in range(0,num_threshold)]

    predictedZ0_MVA = {key: value for key, value in zip(thresholds, [[] for i in range(0,num_threshold)])}

    predictedZ0_FH = []
    predictedZ0_FHz0res = []
    predictedZ0_nofakes = []
    predictedZ0_onlyPV = []
    predictedZ0_chi2 = []


    predictedAssoc_MVA = {key: value for key, value in zip(thresholds, [[] for i in range(0,num_threshold)])}
    predictedAssoc_FH = []
    predictedAssoc_FHz0res = []
    predictedAssoc_nofakes = []
    predictedAssoc_onlyPV = []
    predictedAssoc_chi2 = []

    actual_Assoc = []
    actual_PV = []

    for ibatch,data in enumerate(f['L1TrackNtuple']['eventTree'].iterate(branches,entrysteps=5000)):
        data = {k.decode('utf-8'):v for k,v in data.items() }
        data['corrected_trk_z0']= (data['trk_z0'] + (data['trk_z0']>0.)*0.03 - (data['trk_z0']<0.)*0.03) 

        #if ibatch > 0:
        #   break

        if ((ibatch % 1 == 0) & (ibatch != 0)):
             print("Step: ", ibatch, " Out of ",int(len(f['L1TrackNtuple']['eventTree'])/5000))
                
        for iev in range(len(data['trk_pt'])):
            if (iev % 1000 == 0):
                print("Event: ", iev, " Out of ",len(data["trk_pt"]))

            selectPVTracks = (data['trk_fake'][iev]==1)
            numPVs = len(data["trk_eta"][iev][selectPVTracks])

            FH = predictFastHisto(data[z0][iev],data['trk_pt'][iev],res_func=linear_res_function(data['trk_pt'][iev]))
            FHres = predictFastHisto(data[z0][iev],data['trk_pt'][iev],res_func=eta_res_function(data['trk_eta'][iev]))
            FHnofake = predictFastHisto(data[z0][iev],data['trk_pt'][iev],res_func=fake_res_function(data['trk_fake'][iev]))
            FHonlyPV = predictFastHisto(data[z0][iev],data['trk_pt'][iev],res_func=pv_res_function(data['trk_fake'][iev]))
            FHchi2 = predictFastHisto(data[z0][iev],data['trk_pt'][iev],res_func=chi_res_function(data['trk_chi2rphi'][iev],data['trk_chi2rz'][iev],data['trk_bendchi2'][iev]))

            predictedZ0_FH.append(FH)
            predictedZ0_FHz0res.append(FHres)
            predictedZ0_nofakes.append(FHnofake)
            predictedZ0_onlyPV.append(FHonlyPV)
            predictedZ0_chi2.append(FHchi2)

            for i in range(0,num_threshold):
                FH_MVA = predictFastHisto(data[z0][iev],data['trk_pt'][iev],res_func=MVA_res_function(data['trk_MVA1'][iev],threshold=i/num_threshold))
                predictedZ0_MVA[str(i/num_threshold)].append(FH_MVA)
                predictedAssoc_MVA[str(i/num_threshold)].append(FastHistoAssoc(FH_MVA,data[z0][iev],data['trk_eta'][iev],res_func=linear_res_function(data['trk_pt'][iev],return_bool=True),kf=kf))

            actual_Assoc.append((data['trk_fake'][iev]==1))
            actual_PV.append(data["pv_MC"][iev])

            predictedAssoc_FH.append(FastHistoAssoc(FH,data[z0][iev],data['trk_eta'][iev],res_func=linear_res_function(data['trk_pt'][iev],return_bool=True),kf=kf))
            predictedAssoc_FHz0res.append(FastHistoAssoc(FHres,data[z0][iev],data['trk_eta'][iev],res_func=linear_res_function(data['trk_pt'][iev],return_bool=True),kf=kf))
            predictedAssoc_nofakes.append(FastHistoAssoc(FHnofake,data[z0][iev],data['trk_eta'][iev],res_func=linear_res_function(data['trk_pt'][iev],return_bool=True),kf=kf))
            predictedAssoc_onlyPV.append(FastHistoAssoc(FHonlyPV,data[z0][iev],data['trk_eta'][iev],res_func=linear_res_function(data['trk_pt'][iev],return_bool=True),kf=kf))
            predictedAssoc_chi2.append(FastHistoAssoc(FHchi2,data[z0][iev],data['trk_eta'][iev],res_func=linear_res_function(data['trk_pt'][iev],return_bool=True),kf=kf))

    z0_FH_array = np.concatenate(predictedZ0_FH).ravel()
    z0_FHzres_array = np.concatenate(predictedZ0_FHz0res).ravel()
    z0_FH_nofakes_array = np.concatenate(predictedZ0_nofakes).ravel()
    z0_FH_onlyPV_array = np.concatenate(predictedZ0_onlyPV).ravel()
    z0_FH_chi2_array = np.concatenate(predictedZ0_chi2).ravel()

    z0_PV_array = np.concatenate(actual_PV).ravel()

    assoc_FH_array = np.concatenate(predictedAssoc_FH).ravel()
    assoc_PV_array = np.concatenate(actual_Assoc).ravel()
    assoc_FHzres_array = np.concatenate(predictedAssoc_FHz0res).ravel()
    assoc_FH_nofakes_array = np.concatenate(predictedAssoc_nofakes).ravel()
    assoc_FH_onlyPV_array = np.concatenate(predictedAssoc_onlyPV).ravel()
    assoc_FH_chi2_array = np.concatenate(predictedAssoc_chi2).ravel()

    predictedZ0_MVA_array = {key: value for key, value in zip(thresholds, [np.zeros(1) for i in range(0,num_threshold)])}
    predictedAssoc_MVA_array = np.zeros([num_threshold,len(assoc_FH_chi2_array)])

    predictedZ0_MVA_RMS_array = np.zeros([num_threshold])
    predictedZ0_MVA_Quartile_array = np.zeros([num_threshold])
    predictedZ0_MVA_Centre_array = np.zeros([num_threshold])

    MVA_histos = []
    MVA_log_histos = []

    for i in range(0,num_threshold):
        z0_MVA_array  = np.concatenate(predictedZ0_MVA[str(i/num_threshold)]).ravel()
        Assoc_MVA_array  = np.concatenate(predictedAssoc_MVA[str(i/num_threshold)]).ravel()
        predictedZ0_MVA_array[str(i/num_threshold)] = z0_MVA_array
        predictedAssoc_MVA_array[i] = Assoc_MVA_array
        Diff = z0_PV_array - z0_MVA_array

        predictedZ0_MVA_RMS_array[i] = np.sqrt(np.mean(Diff**2))
        qMVA = np.percentile(Diff,[32,50,68])

        predictedZ0_MVA_Quartile_array[i] = qMVA[2] - qMVA[0]
        predictedZ0_MVA_Centre_array[i] = qMVA[1]

        hist,bin_edges = np.histogram((z0_PV_array - z0_MVA_array),bins=50,range=(-1,1))
        hist_log,bin_edges = np.histogram((z0_PV_array - z0_MVA_array),bins=50,range=(-15,15))
        MVA_histos.append(hist)
        MVA_log_histos.append(hist_log)

    plt.clf()
    fig,ax = plt.subplots(1,2,figsize=(24,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])
    twod_hist = np.stack(MVA_histos, axis=1)
    twod_log_hist = np.stack(MVA_log_histos, axis=1)

    hist2d = ax[0].imshow(twod_hist,cmap=colormap,aspect='auto',extent=[0,1,-1,1])
    hist2d_log = ax[1].imshow(twod_log_hist,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),aspect='auto',cmap=colormap,extent=[0,1,-15,15])

    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].set_xlabel('MVA Threshold',ha="right",x=1)
    ax[0].set_ylabel('Reconstructed $z_{0}^{PV}$ [cm]',ha="right",y=1)
    ax[1].set_xlabel('MVA Threshold',ha="right",x=1)
    ax[1].set_ylabel('Reconstructed $z_{0}^{PV}$ [cm]',ha="right",y=1)

    cbar = plt.colorbar(hist2d , ax=ax[0])
    cbar.set_label('# Events')

    cbar = plt.colorbar(hist2d_log , ax=ax[1])
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/2dhist.pdf" % outputFolder)

    Quartilethreshold_choice = str(np.argmin(predictedZ0_MVA_Quartile_array)/num_threshold)
    RMSthreshold_choice= str(np.argmin(predictedZ0_MVA_RMS_array)/num_threshold)

    MVA_bestQ_array = predictedZ0_MVA_array[Quartilethreshold_choice]
    MVA_bestRMS_array = predictedZ0_MVA_array[RMSthreshold_choice]

    plt.close()
    plt.clf()
    figure=plotz0_residual([(z0_PV_array-MVA_bestRMS_array),
                                              (z0_PV_array-z0_FH_array), 
                                              (z0_PV_array-z0_FHzres_array),
                                              (z0_PV_array-z0_FH_nofakes_array),
                                              (z0_PV_array-z0_FH_onlyPV_array),
                                              (z0_PV_array-z0_FH_chi2_array)],
                                              ['BDT              ','Baseline        ', '$\eta$ Corrected  ', 'No Fakes       ', 'Only PV        ', '$\chi^{2}$ Corrected '])
    plt.savefig("%s/Z0Residual.pdf" % outputFolder)
    plt.savefig("%s/Z0Residual.png" % outputFolder)

    plt.close()
    plt.clf()
    figure=plotz0_residual([(z0_PV_array-z0_FH_array), 
                                              (z0_PV_array-z0_FHzres_array)],
                                             ['Baseline FH     ', '$\eta$ Corrected FH'])
    plt.savefig("%s/Z0EtaResidual.pdf" % outputFolder)
    plt.savefig("%s/Z0EtaResidual.png" % outputFolder)

    plt.close()
    plt.clf()
    figure=plotz0_residual([(z0_PV_array-z0_FH_array), 
                                              (z0_PV_array-MVA_bestRMS_array)],
                                             ['Baseline FH     ','BDT Corrected FH,    Thresh: ' + RMSthreshold_choice          ])
    plt.savefig("%s/Z0BDTResidual.pdf" % outputFolder)
    plt.savefig("%s/Z0BDTResidual.png" % outputFolder)

    plt.close()
    plt.clf()
    figure=plotz0_residual([(z0_PV_array-z0_FH_array), 
                                              (z0_PV_array-z0_FH_chi2_array)],
                                             ['Baseline FH      ', '$\chi^{2}$ Corrected FH'])
    plt.savefig("%s/Z0Chi2Residual.pdf" % outputFolder)
    plt.savefig("%s/Z0Chi2Residual.png" % outputFolder)

    plt.close()
    plt.clf()
    figure=plotz0_residual([(z0_PV_array-z0_FH_array), 
                                              (z0_PV_array-MVA_bestRMS_array),
                                              (z0_PV_array-z0_FH_chi2_array)],
                                             ['Baseline FH         ','BDT Corrected FH,    Thresh: ' + RMSthreshold_choice          ,'$\chi^{2}$ Corrected FH'])
    plt.savefig("%s/Z0BDTChi2Residual.pdf" % outputFolder)
    plt.savefig("%s/Z0BDTChi2Residual.png" % outputFolder)

    plt.close()
    plt.clf()
    figure=plotz0_residual([(z0_PV_array-z0_FH_array)],
                                             ['Baseline FH      '])
    plt.savefig("%s/Z0BaselineResidual.pdf" % outputFolder)
    plt.savefig("%s/Z0BaselineResidual.png" % outputFolder)

    plt.close()
    plt.clf()
    figure = plotPV_roc(assoc_PV_array,[predictedAssoc_MVA_array[int(np.argmin(predictedZ0_MVA_RMS_array))],
                                        assoc_FH_array,
                                        assoc_FHzres_array,
                                        assoc_FH_nofakes_array,
                                        assoc_FH_onlyPV_array,
                                        assoc_FH_chi2_array],['BDT FH','Baseline FH', '$\eta$ Corrected FH', 'No Fakes FH', 'Only PV FH', '$\chi^{2}$ Corrected FH'])
    plt.savefig("%s/PVROC.pdf" % outputFolder)


    ######################################################

    plt.close()
    plt.clf()
    fig_1,fig_2=plot_split_z0_residual([(z0_PV_array-MVA_bestRMS_array),
                                              (z0_PV_array-z0_FH_array), 
                                              (z0_PV_array-z0_FHzres_array),
                                              (z0_PV_array-z0_FH_nofakes_array),
                                              (z0_PV_array-z0_FH_onlyPV_array),
                                              (z0_PV_array-z0_FH_chi2_array)],
                                              ['BDT FH','Baseline FH', '$\eta$ Corrected FH', 'No Fakes FH', 'Only PV FH', '$\chi^{2}$ Corrected FH'])
    fig_1.savefig("%s/Z0Residual_Log.pdf" % outputFolder)
    fig_1.savefig("%s/Z0Residual_Log.png" % outputFolder)
    fig_2.savefig("%s/Z0Residual_NonLog.pdf" % outputFolder)
    fig_2.savefig("%s/Z0Residual_NonLog.png" % outputFolder)

    plt.close()
    plt.clf()
    fig_1,fig_2=plot_split_z0_residual([(z0_PV_array-z0_FH_array), 
                                              (z0_PV_array-z0_FHzres_array)],
                                             ['Baseline FH', '$\eta$ Corrected FH'])
    fig_1.savefig("%s/Z0EtaResidual_Log.pdf" % outputFolder)
    fig_1.savefig("%s/Z0EtaResidual_Log.png" % outputFolder)
    fig_2.savefig("%s/Z0EtaResidual_NonLog.pdf" % outputFolder)
    fig_2.savefig("%s/Z0EtaResidual_NonLog.png" % outputFolder)

    plt.close()
    plt.clf()
    fig_1,fig_2=plot_split_z0_residual([(z0_PV_array-z0_FH_array), 
                                              (z0_PV_array-MVA_bestRMS_array)],
                                             ['Baseline FH','BDT Corrected FH, thresh: ' + RMSthreshold_choice ,'$\chi^{2}$ Corrected FH'])
    fig_1.savefig("%s/Z0BDTResidual_Log.pdf" % outputFolder)
    fig_1.savefig("%s/Z0BDTResidual_Log.png" % outputFolder)
    fig_2.savefig("%s/Z0BDTResidual_NonLog.pdf" % outputFolder)
    fig_2.savefig("%s/Z0BDTResidual_NonLog.png" % outputFolder)

    plt.close()
    plt.clf()
    fig_1,fig_2=plot_split_z0_residual([(z0_PV_array-z0_FH_array), 
                                              (z0_PV_array-z0_FH_chi2_array)],
                                             ['Baseline FH', '$\chi^{2}$ Corrected FH'])
    fig_1.savefig("%s/Z0Chi2Residual_Log.pdf" % outputFolder)
    fig_1.savefig("%s/Z0Chi2Residual_Log.png" % outputFolder)
    fig_2.savefig("%s/Z0Chi2Residual_NonLog.pdf" % outputFolder)
    fig_2.savefig("%s/Z0Chi2Residual_NonLog.png" % outputFolder)

    plt.close()
    plt.clf()
    fig_1,fig_2=plot_split_z0_residual([(z0_PV_array-z0_FH_array), 
                                              (z0_PV_array-MVA_bestRMS_array),
                                              (z0_PV_array-z0_FH_chi2_array)],
                                             ['Baseline FH','BDT Corrected FH, thresh: ' + RMSthreshold_choice ,'$\chi^{2}$ Corrected FH'])
    fig_1.savefig("%s/Z0BDTChi2Residual_Log.pdf" % outputFolder)
    fig_1.savefig("%s/Z0BDTChi2Residual_Log.png" % outputFolder)
    fig_2.savefig("%s/Z0BDTChi2Residual_NonLog.pdf" % outputFolder)
    fig_2.savefig("%s/Z0BDTChi2Residual_NonLog.png" % outputFolder)
    

    plt.close()
    plt.close()
    plt.clf()
    fig_1,fig_2=plot_split_z0_residual([(z0_PV_array-z0_FH_array)],
                                             ['Baseline FH'])
    fig_1.savefig("%s/Z0BaselineResidual_Log.pdf" % outputFolder)
    fig_1.savefig("%s/Z0BaselineResidual_Log.png" % outputFolder)
    fig_2.savefig("%s/Z0BaselineResidual_NonLog.pdf" % outputFolder)
    fig_2.savefig("%s/Z0BaselineResidual_NonLog.png" % outputFolder)


    plt.close()
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    ax.hist(z0_FH_array,range=(-15,15),bins=64,density=True,color='r',histtype="step",label="FastHisto Base")
    ax.hist(z0_FHzres_array,range=(-15,15),bins=64,density=True,color='g',histtype="step",label="FastHisto with z0 res")
    ax.hist(z0_PV_array,range=(-15,15),bins=64,density=True,color='y',histtype="step",label="Truth")
    ax.grid(True)
    ax.set_xlabel('Reconstructed $z_{0}^{PV}$ [cm]',ha="right",x=1)
    ax.set_ylabel('Events',ha="right",y=1)
    ax.legend(frameon=True,facecolor='w',edgecolor='k') 
    plt.tight_layout()
    plt.savefig("%s/z0hist.pdf" % outputFolder)
    plt.savefig("%s/z0hist.png" % outputFolder)

    plt.close()
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_PV_array, (z0_PV_array-z0_FH_array), bins=60,range=((-15,15),(-30,30)) ,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("PV", horizontalalignment='right', x=1.0)
    ax.set_ylabel("PV - $z_0^{FH}$", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/FHerr_vs_z0.pdf" %  outputFolder)
    plt.savefig("%s/FHerr_vs_z0.png" %  outputFolder)
    plt.close()

    plt.close()
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    hist2d = ax.hist2d(z0_PV_array, z0_FH_array, bins=60,range=((-15,15),(-15,15)) ,norm=matplotlib.colors.LogNorm(vmin=1,vmax=1000),cmap=colormap)
    ax.set_xlabel("PV", horizontalalignment='right', x=1.0)
    ax.set_ylabel("$z_0^{FH}$", horizontalalignment='right', y=1.0)
    cbar = plt.colorbar(hist2d[3] , ax=ax)
    cbar.set_label('# Events')
    plt.tight_layout()
    plt.savefig("%s/FH_vs_z0.pdf" %  outputFolder)
    plt.savefig("%s/FH_vs_z0.png" %  outputFolder)
    plt.close()

    def calc_widths(actual,predicted):
        diff = (actual-predicted)
        RMS = np.sqrt(np.mean(diff**2))
        qs = np.percentile(diff,[32,50,68])
        qwidth = qs[2] - qs[0]
        qcentre = qs[1]

        return [RMS,qwidth,qcentre]

    FHwidths = calc_widths(z0_PV_array,z0_FH_array)
    FHNoFakeWidths = calc_widths(z0_PV_array,z0_FH_nofakes_array)
    FHPVWidths = calc_widths(z0_PV_array,z0_FH_onlyPV_array)
    FHchi2Widths = calc_widths(z0_PV_array,z0_FH_chi2_array)
    FHresWidths = calc_widths(z0_PV_array,z0_FHzres_array)

    FHwidths = [0,0,0,0]
    FHresWidths = [0,0,0,0]
    FHNoFakeWidths = [0,0,0,0]
    FHPVWidths = [0,0,0,0]
    FHchi2Widths = [0,0,0,0]

    predictedZ0_MVA_RMS_array  = [1.3301,0.9896,0.9358,0.9120,0.8981,0.8933,0.8892,0.8892,0.8973,0.9147]
    predictedZ0_MVA_Quartile_array = [ 0.1920, 0.1876,0.1865, 0.1862, 0.1861, 0.1859, 0.1859, 0.1859, 0.1859, 0.1865]
    predictedZ0_MVA_Centre_array = [ -0.0255,-0.0270,-0.0273, -0.0277, -0.0281, -0.0284, -0.0288, -0.0282,-0.0268, -0.0273]
    predictedZ0_MVA_Efficiency_array = [0.948375,0.96456,0.967723,0.969371,0.970305,0.970829,0.971227,0.97127,0.971163,0.970467]

    FHwidths[0] = 1.3301
    FHresWidths[0] = 1.5262
    FHNoFakeWidths[0] = 0.8620
    FHPVWidths[0] = 0.2015
    FHchi2Widths[0] = 1.1373

    FHwidths[1] = 0.1920
    FHresWidths[1] = 0.1800
    FHNoFakeWidths[1] = 0.1852
    FHPVWidths[1] = 0.1788
    FHchi2Widths[1] = 0.1984

    FHwidths[2] = -0.0255
    FHresWidths[2] = -0.0279
    FHNoFakeWidths[2] = -0.0272
    FHPVWidths[2] = -0.0290
    FHchi2Widths[2] = -0.0264

    FHwidths[3] = 0.948375
    FHresWidths[3] = 0.928658
    FHNoFakeWidths[3] = 0.972631
    FHPVWidths[3] = 0.993444
    FHchi2Widths[3] = 0.95473

    plt.close()
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        
    ax.plot(thresholds,predictedZ0_MVA_RMS_array,label="BDT FH",markersize=10,linestyle=linestyles[0],linewidth=LINEWIDTH,marker='o',color=colours[0])
    ax.plot(thresholds,np.full(len(thresholds),FHwidths[0]),label="Base FH",linestyle=linestyles[1],linewidth=LINEWIDTH,color=colours[1])
    ax.plot(thresholds,np.full(len(thresholds),FHresWidths[0]),label="$\eta$ Corrected FH",linestyle=linestyles[2],linewidth=LINEWIDTH,color=colours[2])
    ax.plot(thresholds,np.full(len(thresholds),FHNoFakeWidths[0]),label="No Fakes FH",linestyle=linestyles[3],linewidth=LINEWIDTH,color=colours[3])
    ax.plot(thresholds,np.full(len(thresholds),FHPVWidths[0]),label="Only PV FH",linestyle=linestyles[4],linewidth=LINEWIDTH,color=colours[4])
    ax.plot(thresholds,np.full(len(thresholds),FHchi2Widths[0]),label="$\chi^{2}$ Corrected FH",linestyle=linestyles[5],linewidth=LINEWIDTH,color=colours[5])
    
    ax.set_ylabel("$z_{0}^{PV}$ Residual RMS [cm]", horizontalalignment='right', y=1.0)
    ax.set_xlabel("BDT classification threshold", horizontalalignment='right', x=1.0)
    ax.legend(frameon=True,facecolor='w',edgecolor='w',loc=3)
    plt.tight_layout()
    plt.savefig("%s/BDTRMSvsThreshold.pdf" %  outputFolder)
    plt.savefig("%s/BDTRMSvsThreshold.png" %  outputFolder)
    plt.close()

    plt.close()
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        
    ax.plot(thresholds,predictedZ0_MVA_Quartile_array,label="BDT FH",markersize=10,linestyle=linestyles[0],linewidth=LINEWIDTH,marker='o',color=colours[0])
    ax.plot(thresholds,np.full(len(thresholds),FHwidths[1]),label="Baseline FH",linestyle=linestyles[1],linewidth=LINEWIDTH,color=colours[1])
    ax.plot(thresholds,np.full(len(thresholds),FHresWidths[1]),label="$\eta$ Corrected FH",linestyle=linestyles[2],linewidth=LINEWIDTH,color=colours[2])
    ax.plot(thresholds,np.full(len(thresholds),FHNoFakeWidths[1]),label="No Fakes FH",linestyle=linestyles[3],linewidth=LINEWIDTH,color=colours[3])
    ax.plot(thresholds,np.full(len(thresholds),FHPVWidths[1]),label="Only PV FH",linestyle=linestyles[4],linewidth=LINEWIDTH,color=colours[4])
    ax.plot(thresholds,np.full(len(thresholds),FHchi2Widths[1]),label="$\chi^{2}$ Corrected FH",linestyle=linestyles[5],linewidth=LINEWIDTH,color=colours[5])
    
    ax.set_ylabel("$z_{0}^{PV}$ Residual Quartile Width [cm]", horizontalalignment='right', y=1.0)
    ax.set_xlabel("BDT classification threshold", horizontalalignment='right', x=1.0)
    ax.legend(frameon=True,facecolor='w',edgecolor='w',loc=3)
    plt.tight_layout()
    plt.savefig("%s/BDTQuartilevsThreshold.pdf" %  outputFolder)
    plt.savefig("%s/BDTQuartilevsThreshold.png" %  outputFolder)
    plt.close()

    plt.close()
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        
    ax.plot(thresholds,predictedZ0_MVA_Centre_array,label="BDT FH",markersize=10,linestyle=linestyles[0],linewidth=LINEWIDTH,marker='o',color=colours[0])
    ax.plot(thresholds,np.full(len(thresholds),FHwidths[2]),label="Base FH",linestyle=linestyles[1],linewidth=LINEWIDTH,color=colours[1])
    ax.plot(thresholds,np.full(len(thresholds),FHresWidths[2]),label="$\eta$ Corrected FH",linestyle=linestyles[2],linewidth=LINEWIDTH,color=colours[2])
    ax.plot(thresholds,np.full(len(thresholds),FHNoFakeWidths[2]),label="No Fakes FH",linestyle=linestyles[3],linewidth=LINEWIDTH,color=colours[3])
    ax.plot(thresholds,np.full(len(thresholds),FHPVWidths[2]),label="Only PV FH",linestyle=linestyles[4],linewidth=LINEWIDTH,color=colours[4])
    ax.plot(thresholds,np.full(len(thresholds),FHchi2Widths[2]),label="$\chi^{2}$ Corrected FH",linestyle=linestyles[5],linewidth=LINEWIDTH,color=colours[5])
    
    ax.set_ylabel("$z_{0}^{PV}$ Residual Centre [cm]", horizontalalignment='right', y=1.0)
    ax.set_xlabel("BDT classification threshold", horizontalalignment='right', x=1.0)
    ax.legend(frameon=True,facecolor='w',edgecolor='w',loc=3)
    plt.tight_layout()
    plt.savefig("%s/BDTCentrevsThreshold.pdf" %  outputFolder)
    plt.savefig("%s/BDTCentrevsThreshold.png" %  outputFolder)
    plt.close()

    plt.close()
    plt.clf()
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
        
    ax.plot(thresholds,predictedZ0_MVA_Efficiency_array,label="BDT FH",markersize=10,linestyle=linestyles[0],linewidth=LINEWIDTH,marker='o',color=colours[0])
    ax.plot(thresholds,np.full(len(thresholds),FHwidths[3]),label="Base FH",linestyle=linestyles[1],linewidth=LINEWIDTH,color=colours[1])
    ax.plot(thresholds,np.full(len(thresholds),FHresWidths[3]),label="$\eta$ Corrected FH",linestyle=linestyles[2],linewidth=LINEWIDTH,color=colours[2])
    ax.plot(thresholds,np.full(len(thresholds),FHNoFakeWidths[3]),label="No Fakes FH",linestyle=linestyles[3],linewidth=LINEWIDTH,color=colours[3])
    ax.plot(thresholds,np.full(len(thresholds),FHPVWidths[3]),label="Only PV FH",linestyle=linestyles[4],linewidth=LINEWIDTH,color=colours[4])
    ax.plot(thresholds,np.full(len(thresholds),FHchi2Widths[3]),label="$\chi^{2}$ Corrected FH",linestyle=linestyles[5],linewidth=LINEWIDTH,color=colours[5])
    
    ax.set_ylabel("Vertex Finding Efficiency (threshold 0.5 cm)", horizontalalignment='right', y=1.0)
    ax.set_xlabel("BDT classification threshold", horizontalalignment='right', x=1.0)
    ax.legend(frameon=True,facecolor='w',edgecolor='w',loc=3)
    plt.tight_layout()
    plt.savefig("%s/BDTEfficiencyvsThreshold.pdf" %  outputFolder)
    plt.savefig("%s/BDTEfficiencyvsThreshold.png" %  outputFolder)
    plt.close()