import numpy as np
import uproot

import glob
import sklearn.metrics as metrics
import yaml
import sys

import math

import os

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
#hep.set_style("CMSTex")
#hep.cms.label()
#hep.cms.text("Simulation")
#plt.style.use(hep.style.CMS)

SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('axes', linewidth=5)              # thickness of axes
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=25)            # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

matplotlib.rcParams['xtick.major.size'] = 20
matplotlib.rcParams['xtick.major.width'] = 5
matplotlib.rcParams['xtick.minor.size'] = 10
matplotlib.rcParams['xtick.minor.width'] = 4

matplotlib.rcParams['ytick.major.size'] = 20
matplotlib.rcParams['ytick.major.width'] = 5
matplotlib.rcParams['ytick.minor.size'] = 10
matplotlib.rcParams['ytick.minor.width'] = 4


kf = sys.argv[1]

if kf == "NewKF":
    f = uproot.open("/home/cebrown/Documents/Datasets/VertexDatasets/NewKF_TTbar_170K_quality.root")
    z0 = 'trk_z0'
elif kf == "OldKF":
    f = uproot.open("/home/cebrown/Documents/Datasets/VertexDatasets/OldKF_TTbar_170K_quality.root")
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
    'genMETPx',
    'genMETPy',
    'genMET',
    'genMETPhi',
    "trk_MVA1"
]

def predictFastHisto(value,weight):
    halfBinWidth = 0.5*30./256.
    hist,bin_edges = np.histogram(value,256,range=(-15,15),weights=weight)
    hist = np.convolve(hist,[1,1,1],mode='same')
    z0Index= np.argmax(hist)
    z0 = -15.+30.*z0Index/256.+halfBinWidth

    return z0

def predictFastHistoZ0res(value,weight,eta):

    def res_function(eta):
        res = 0.1 + 0.2*eta**2
        return res
    
    halfBinWidth = 0.5*30./256.
    res = res_function(eta)
    hist,bin_edges = np.histogram(value,256,range=(-15,15),weights=weight/res)
    hist = np.convolve(hist,[1,1,1],mode='same')
    z0Index= np.argmax(hist)
    z0 = -15.+30.*z0Index/256.+halfBinWidth

    return z0

def predictFastHistoMVA(value,weight,eta,MVA):
    def res_function(eta):
        res = 0.1 + 0.2*eta**2
        return res
    
    halfBinWidth = 0.5*30./256.
    res = res_function(eta)
    hist,bin_edges = np.histogram(value,256,range=(-15,15),weights=(weight*MVA)/res)
    hist = np.convolve(hist,[1,1,1],mode='same')
    z0Index= np.argmax(hist)
    z0 = -15.+30.*z0Index/256.+halfBinWidth
    return z0

def predictFastHistoChi2(value,weight,eta,bendchi,chirphi,chirz):
    def res_function(eta):
        res = 0.1 + 0.2*eta**2
        return res

    def chi_cut(bendchi,chirphi,chirz):
        if bendchi > 2.4:
            return 0
        elif chirphi > 10:
            return 0
        elif chirz > 10:
            return 0
        else:
            return 1
    
    
    halfBinWidth = 0.5*30./256.
    res = res_function(eta)
    hist,bin_edges = np.histogram(value,256,range=(-15,15),weights=(weight*chi_cut(bendchi,chirphi,chirz))/res)
    hist = np.convolve(hist,[1,1,1],mode='same')
    z0Index= np.argmax(hist)
    z0 = -15.+30.*z0Index/256.+halfBinWidth
    return z0

def predictMET(pt,phi,predictedAssoc,threshold):

    met_px_list = []
    met_py_list = []
    met_pt_list = []
    met_phi_list = []


    

    for ibatch in range(pt.shape[0]):
        if type(predictedAssoc) == np.ndarray:
            assoc = np.expand_dims(predictedAssoc[ibatch],1)
        else:
            assoc = predictedAssoc[ibatch].numpy()
        newpt = pt[ibatch].numpy()
        newphi = phi[ibatch].numpy()

        assoc[assoc > threshold] = 1
        NN_track_sel = assoc == 1


        met_px = np.sum(newpt[NN_track_sel[:,0]]*np.cos(newphi[NN_track_sel[:,0]]))
        met_py = np.sum(newpt[NN_track_sel[:,0]]*np.sin(newphi[NN_track_sel[:,0]]))
        met_px_list.append([met_px])
        met_py_list.append([met_py])
        met_pt_list.append([math.sqrt(met_px**2+met_py**2)])
        met_phi_list.append([math.atan2(met_py,met_px)])
    return  [np.array(met_px_list,dtype=np.float32),
            np.array(met_py_list,dtype=np.float32),
            np.array(met_pt_list,dtype=np.float32),
            np.array(met_phi_list,dtype=np.float32)]

def FastHistoAssoc(PV,trk_z0,trk_eta,threshold=1):
    eta_bins = np.array([0.0,0.7,1.0,1.2,1.6,2.0,2.4])
    deltaz_bins = threshold*np.array([0.0,0.4,0.6,0.76,1.0,1.7,2.2,0.0])
    

    eta_bin = np.digitize(abs(trk_eta),eta_bins)
    deltaz = abs(trk_z0 - PV)

    assoc = (deltaz < deltaz_bins[eta_bin])

    return assoc

def plotz0_residual(FHdiff):
    plt.clf()
    figure = plt.figure(figsize=(10,10))
    qz0_FH = np.percentile(FHdiff,[32,50,68])
    plt.hist(FHdiff,bins=50,range=(-1,1),histtype="step",linewidth=3,label=f"FH Width = %.4f, Centre = %.4f" %(qz0_FH[2]-qz0_FH[0], qz0_FH[1]))
    plt.grid(True)
    plt.xlabel('$z_0$ Residual [cm]',ha="right",x=1)
    plt.ylabel('Events',ha="right",y=1)
    plt.legend() 
    plt.tight_layout()
    return figure,(qz0_FH[2]-qz0_FH[0])

def plotPV_roc(actual,FHpred):

    tnFH, fpFH, fnFH, tpFH = metrics.confusion_matrix(actual, FHpred).ravel()

    precisionFH = tpFH / (tpFH + fpFH) 
    recallFH = tpFH / (tpFH + fnFH) 

    TPRFH = recallFH
    FPRFH = fpFH / (fpFH + tnFH) 

    plt.clf()
    fig,ax = plt.subplots(1,2,figsize=(20,10))

    ax[0].set_title("Purity Efficiency Plot" ,loc='left')
    ax[0].scatter(recallFH,precisionFH,color='orange',label="FH",s=50)
    ax[0].grid(True)
    ax[0].set_xlabel('Efficiency',ha="right",x=1)
    ax[0].set_ylabel('Purity',ha="right",y=1)
    ax[0].set_xlim([0.01,1])
    ax[0].legend()

    ax[1].set_title("Reciever Operator Characteristic Plot" ,loc='left')
    ax[1].scatter(TPRFH,FPRFH,color='orange',label=f"FH AUC: %.4f" %(metrics.roc_auc_score(actual,FHpred)),s=50)
    ax[1].grid(True)
    ax[1].set_yscale("log")
    ax[1].set_xlabel('True Positive Rate',ha="right",x=1)
    ax[1].set_ylabel('False Positive Rate',ha="right",y=1)
    ax[0].set_ylim([1e-4,1])
    ax[1].legend()
    plt.tight_layout()
    return fig,metrics.roc_auc_score(actual,FHpred)

def plotz0_percentile(FHdiff):
    plt.clf()
    figure = plt.figure(figsize=(10,10))

    percentiles = np.linspace(0,100,100)
    FHpercentiles = np.percentile(FHdiff,percentiles)
    
    plt.plot(percentiles,abs(FHpercentiles),label=f"FH minimum: %.4f at : %.2f " %(min(abs(FHpercentiles)),np.argmin(abs(FHpercentiles))),linewidth=3)
    plt.grid(True)
    plt.xlabel('Percentile',ha="right",x=1)
    plt.ylabel('$|\\delta z_{0}| [cm]$',ha="right",y=1)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    

    return figure

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


    plt.clf()  
    fig, ax = plt.subplots(1,1, figsize=(20,10)) 

    ax[0].set_title("FH Normalised Class Distribution" ,loc='left')
    ax[0].hist(FHgenuine,color='g',bins=20,range=(0,1),histtype="step",label="Genuine",density=True,linewidth=2)
    ax[0].hist(FHfake,color='r',bins=20,range=(0,1),histtype="step",label="Fake",density=True,linewidth=2)
    ax[0].grid()
    #ax01].set_yscale("log")
    ax[0].set_xlabel("FH Output",ha="right",x=1)
    ax[0].set_ylabel("a.u.",ha="right",y=1)
    ax[0].legend(loc="upper center")
    plt.tight_layout()
    return fig

def plotKDEandTracks(tracks,MVA,assoc,genPV,predictedPV,weights,weight_label="KDE",threshold=-1):
  plt.clf()
  hist,bin_edges = np.histogram(tracks,256,range=(-15,15),weights=weights)
  maxy = max(hist)
  plt.bar(bin_edges[:-1],hist,width=30/256,color='grey',alpha=0.5, label=weight_label)

  assoc[assoc > threshold] = 1
  assoc[assoc < threshold] = 0
  pv_track_sel = assoc == 1

  print(MVA)

  MVA[MVA > 0.3] = 1
  MVA[MVA < 0.3] = 0
  fake_track_sel = MVA == 0


  pu_track_sel = assoc == 0
  '''
  #plt.plot(tracks[pv_track_sel], [0] * len(tracks[pv_track_sel]), '+g', label='PV Trk',markersize=20)
  #plt.plot(tracks[pu_track_sel], [0] * len(tracks[pu_track_sel]), '+b', label='PU Trk',markersize=20)
  #plt.plot(tracks[fake_track_sel], [0] * len(tracks[fake_track_sel]), '+r', label='Fake Trk',markersize=20)


  plt.plot(tracks, [0] * len(tracks), '+g', label='Tracks',markersize=20)


  #plt.plot([predictedPV, predictedPV], [0, max(hist)], '--k', label='Reco Vx',linewidth=5)

  #plt.plot([genPV, genPV], [0,max(hist)], '--g', label='True Vx',linewidth=5)

  plt.xlabel('z / cm')
  plt.ylabel('density')
  plt.xlim()
  plt.ylim(-0.05*maxy,maxy)
  plt.legend()
  plt.show()

  plt.bar(bin_edges[:-1],hist,width=30/256,color='grey',alpha=0.5, label=weight_label)

  plt.plot(tracks, [0] * len(tracks), '+g', label='Tracks',markersize=20)
    #plt.plot(tracks[pv_track_sel], [0] * len(tracks[pv_track_sel]), '+g', label='PV Trk',markersize=20)
  #plt.plot(tracks[pu_track_sel], [0] * len(tracks[pu_track_sel]), '+b', label='PU Trk',markersize=20)
  plt.plot(tracks[fake_track_sel], [0] * len(tracks[fake_track_sel]), '+r', label='Fake Trk',markersize=20)
  #plt.plot([predictedPV, predictedPV], [0, max(hist)], '--k', label='Reco Vx',linewidth=5)

  #plt.plot([genPV, genPV], [0,max(hist)], '--g', label='True Vx',linewidth=5)

  plt.xlabel('z / cm')
  plt.ylabel('density')
  plt.xlim()
  plt.ylim(-0.05*maxy,maxy)
  plt.legend()
  plt.show()

  plt.bar(bin_edges[:-1],hist,width=30/256,color='grey',alpha=0.5, label=weight_label)

  plt.plot(tracks, [0] * len(tracks), '+g', label='Tracks',markersize=20)
    #plt.plot(tracks[pv_track_sel], [0] * len(tracks[pv_track_sel]), '+g', label='PV Trk',markersize=20)
  #plt.plot(tracks[pu_track_sel], [0] * len(tracks[pu_track_sel]), '+b', label='PU Trk',markersize=20)
  plt.plot(tracks[fake_track_sel], [0] * len(tracks[fake_track_sel]), '+r', label='Fake Trk',markersize=20)
  plt.plot([predictedPV, predictedPV], [0, max(hist)], '--k', label='Reco Vx',linewidth=5)
  plt.plot([genPV, genPV], [0,max(hist)], '--g', label='True Vx',linewidth=5)

  plt.xlabel('z / cm')
  plt.ylabel('density')
  plt.xlim()
  plt.ylim(-0.05*maxy,maxy)
  plt.legend()
  plt.show()
  '''

  plt.plot(tracks[pu_track_sel], [0] * len(tracks[pu_track_sel]), '+b', label='PU Trk',markersize=20)
  plt.plot(tracks[fake_track_sel], [0] * len(tracks[fake_track_sel]), '+r', label='Fake Trk',markersize=20)
  plt.plot(tracks[pv_track_sel], [0] * len(tracks[pv_track_sel]), '+g', label='PV Trk',markersize=20)
  plt.plot([predictedPV, predictedPV], [0, max(hist)], '--k', label='Reco Vx',linewidth=5)
  plt.plot([genPV, genPV], [0,max(hist)], '--g', label='True Vx',linewidth=5)

  plt.xlabel('z / cm')
  plt.ylabel('density')
  plt.xlim()
  plt.ylim(-0.05*maxy,maxy)
  plt.legend()
  plt.show()


if __name__=="__main__":

    eta_boundaries = [2.4]#[1.0,1.2,1.6,2.0,2.4]
    eta_thresholds = [1.0]#[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    widths = np.ndarray([len(eta_boundaries),len(eta_thresholds)])
    widths_res = np.ndarray([len(eta_boundaries),len(eta_thresholds)])
    aucs = np.ndarray([len(eta_boundaries),len(eta_thresholds)])

    for b,eta_boundary in enumerate(eta_boundaries):
        for t,eta_threshold in enumerate(eta_thresholds):
            boundary_folder_name = "E_"+str(eta_boundary).replace(".", "_")        
            folder_name = "T_"+str(eta_threshold).replace(".", "_")
            outputFolder = kf+"TestOutput" + "/" + str(boundary_folder_name) + "/" + str(folder_name)

            os.system("mkdir -p "+ outputFolder)

            predictedZ0_FH = []
            predictedZ0_FHz0res = []
            predictedAssoc_FH = []
            predictedZ0_FHMVA = []
            predictedZ0_FH = []

            actual_Assoc = []
            actual_PV = []

            for ibatch,data in enumerate(f['L1TrackNtuple']['eventTree'].iterate(branches,entrysteps=5000)):
                data = {k.decode('utf-8'):v for k,v in data.items() }
                data['corrected_trk_z0']= (data['trk_z0'] + (data['trk_z0']>0.)*0.03 - (data['trk_z0']<0.)*0.03) 
            

                if ((ibatch % 1 == 0) & (ibatch != 0)):
                    print("Step: ", ibatch, " Out of ",len(f['L1TrackNtuple']['eventTree']))
                
                for iev in range(len(data['trk_pt'])):
                    if (iev % 1000 == 0):
                        print("Event: ", iev, " Out of ",len(data["trk_pt"]))
                    selectPVTracks = (data['trk_fake'][iev]==1)
                    numPVs = len(data["trk_eta"][iev][selectPVTracks])

                    eta_indices = (data["trk_eta"][iev] <= eta_boundary)

                    num_indices = len(eta_indices)
                    num_falses = num_indices - np.count_nonzero(eta_indices)
                    num_cut_falses = (int)(num_falses * eta_threshold)

                    false = 0

                    for i,item in enumerate(eta_indices):
                        if (item == False):
                            false+=1
                        if (false >= num_cut_falses):
                            eta_indices[i] = True

                    FH = predictFastHisto(data[z0][iev][eta_indices],data['trk_pt'][iev][eta_indices])
                    predictedZ0_FH.append(FH)
                    predictedZ0_FHz0res.append(predictFastHistoZ0res(data[z0][iev][eta_indices],data['trk_pt'][iev][eta_indices],data['trk_eta'][iev][eta_indices]))
                    predictedZ0_FHMVA.append(predictFastHistoMVA(data[z0][iev],data['trk_pt'][iev],data['trk_eta'][iev],data['trk_MVA1'][iev]))

                    actual_Assoc.append((data['trk_fake'][iev]==1))
                    actual_PV.append(data["pv_MC"][iev][0])
                    FHassoc = FastHistoAssoc(predictFastHisto(data[z0][iev],data['trk_pt'][iev]),data[z0][iev],data['trk_eta'][iev])
                    predictedAssoc_FH.append(FHassoc)

            z0_FH_array = np.array(predictedZ0_FH)
            z0_FHzres_array = np.array(predictedZ0_FHz0res)
            z0_FHMVA_array = np.array(predictedZ0_FHMVA)
            z0_PV_array = np.array(actual_PV)

            assoc_FH_array = np.array(predictedAssoc_FH)
            assoc_PV_array = np.array(actual_Assoc)

            plt.clf()
            figure,width=plotz0_residual((z0_PV_array-z0_FH_array))
            plt.savefig("%s/Z0Residual.png" % outputFolder)

            plt.clf()
            figure,res_width=plotz0_residual((z0_PV_array-z0_FHzres_array))
            plt.savefig("%s/Z0Residualzres.png" % outputFolder)

            plt.clf()
            figure,res_width=plotz0_residual((z0_PV_array-predictedZ0_FHMVA))
            plt.savefig("%s/Z0ResidualMVA.png" % outputFolder)

            widths[b][t] = width
            widths_res[b][t] = res_width

            plt.clf()
            figure,ROCauc=plotPV_roc(np.concatenate(assoc_PV_array),np.concatenate(assoc_FH_array))
            plt.savefig("%s/PVROC.png" % outputFolder)

            aucs[b][t] = ROCauc

            plt.clf()
            figure=plotz0_percentile((z0_PV_array-z0_FH_array))
            plt.savefig("%s/Z0percentile.png" % outputFolder)

            plt.clf()
            plt.hist(z0_FH_array,range=(-15,15),bins=120,density=True,color='r',histtype="step",label="FastHisto Base")
            plt.hist(z0_FHzres_array,range=(-15,15),bins=120,density=True,color='g',histtype="step",label="FastHisto with z0 res")
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
            plt.hist2d(z0_PV_array, (z0_PV_array-predictedZ0_FHMVA), bins=60,range=((-15,15),(-30,30)) ,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
            plt.xlabel("PV", horizontalalignment='right', x=1.0)
            plt.ylabel("PV - $z_0^{FH}$", horizontalalignment='right', y=1.0)
            #plt.colorbar(vmin=0,vmax=1000)
            plt.tight_layout()
            plt.savefig("%s/MVAFHerr_vs_z0.png" %  outputFolder)

            plt.clf()
            plt.hist2d(z0_PV_array, z0_FH_array, bins=60,range=((-15,15),(-15,15)) ,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=1000)
            plt.xlabel("PV", horizontalalignment='right', x=1.0)
            plt.ylabel("$z_0^{FH}$", horizontalalignment='right', y=1.0)
            #plt.colorbar(vmin=0,vmax=1000)
            plt.tight_layout()
            plt.savefig("%s/FH_vs_z0.png" %  outputFolder)
            
            plt.clf()
            figure=plotz0_percentile((z0_PV_array-z0_FHzres_array))
            plt.savefig("%s/Z0withrespercentile.png" % outputFolder)


    plt.clf()
    plt.imshow(widths_res)
    plt.ylabel("$\\eta$ Boundaries", horizontalalignment='right', y=1.0)
    plt.xlabel("\% PV Tracks within $\\eta$ Boundary ", horizontalalignment='right', x=1.0)
    plt.colorbar(label="FH Res Resolution")
    plt.yticks(np.arange(len(eta_boundaries)),eta_boundaries)
    plt.xticks(np.arange(len(eta_thresholds)),eta_thresholds)
    plt.tight_layout()
    plt.savefig(kf+"TestOutput/ResWidth.png")

    plt.clf()
    plt.imshow(widths)
    plt.ylabel("$\\eta$ Boundaries", horizontalalignment='right', y=1.0)
    plt.xlabel("\% PV Tracks within $\\eta$ Boundary ", horizontalalignment='right', x=1.0)
    plt.colorbar(label="FH Resolution")
    plt.yticks(np.arange(len(eta_boundaries)),eta_boundaries)
    plt.xticks(np.arange(len(eta_thresholds)),eta_thresholds)
    plt.tight_layout()
    plt.savefig(kf+"TestOutput/Width.png")

    plt.clf()
    plt.imshow(aucs)
    plt.ylabel("$\\eta$ Boundaries", horizontalalignment='right', y=1.0)
    plt.xlabel("\% PV Tracks within $\\eta$ Boundary ", horizontalalignment='right', x=1.0)
    plt.colorbar(label="AUC PV track association")
    plt.yticks(np.arange(len(eta_boundaries)),eta_boundaries)
    plt.xticks(np.arange(len(eta_thresholds)),eta_thresholds)
    plt.tight_layout()
    plt.savefig(kf+"TestOutput/AUCS.png")







    

    

