import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
#hep.set_style("CMSTex")

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
import pickle


predicted_z0 = []
true_z0 = []
predicted_association = []
trk_fromPV = []

predicted_weight = []
trk_z0 = []

weight_1 = { 'weight_1_'+str(i) : [] for i in range(0,10)}
weight_2 = { 'weight_2_'+str(i) : [] for i in range(0,10)}
weight_3_0 = []

association_1 = { 'association_1_'+str(i) : [] for i in range(0,20)}
association_2 = { 'association_2_'+str(i) : [] for i in range(0,20)}

predicted_z0_residual = []
predicted_association_residual = []

savingfolder = 'SavedDFs/Train'

for j in range(0,20):
        with open('SavedDFs/events_batch'+str(j)+'.pkl', 'rb') as outp:
                Events = pickle.load(outp)
        for i,event in enumerate(Events):
                if (i % 1000 == 0):
                        print('File: ',j,' Event: ',i, " out of ",len(Events))
                predicted_z0.append(event['predicted_z0'])
                true_z0.append(event['true_z0'])
                predicted_association.append(event['predicted_association'])
                trk_fromPV.append(event['trk_fromPV'])

                predicted_weight.append(event['predicted_weight'])
                trk_z0.append(event['trk_z0'])

                [weight_1['weight_1_'+str(k)].append(event['weight_1_'+str(k)]) for k in range(0,10)]
                [weight_2['weight_2_'+str(k)].append(event['weight_2_'+str(k)]) for k in range(0,10)]
                weight_3_0.append(event['weight_3_0'])
                [association_1['association_1_'+str(k)].append(event['association_1_'+str(k)]) for k in range(0,20)]
                [association_2['association_2_'+str(k)].append(event['association_2_'+str(k)]) for k in range(0,20)]


weight_1_array = np.array([np.concatenate(weight_1['weight_1_'+str(k)]).ravel() for k in range(0,10)])

weight_2_array = np.array([np.concatenate(weight_2['weight_2_'+str(k)]).ravel() for k in range(0,10)])
weight_3_0_array = np.concatenate(weight_3_0).ravel()
association_1_array = np.array([np.concatenate(association_1['association_1_'+str(k)]).ravel() for k in range(0,20)])
association_2_array =  np.array([np.concatenate(association_2['association_2_'+str(k)]).ravel() for k in range(0,20)])

predicted_weight_array =  np.concatenate(predicted_weight).ravel()

predicted_association_array =  np.concatenate(predicted_association).ravel()
trk_fromPV_array =  np.concatenate(trk_fromPV).ravel()
trk_z0_array = np.concatenate(trk_z0).ravel()
predicted_z0_array =  np.concatenate(predicted_z0).ravel()
true_z0_array =  np.concatenate(true_z0).ravel()

z0_residual = abs(true_z0_array - predicted_z0_array)
association_residual = abs(predicted_association_array - trk_fromPV_array)


### V. Correct Postive -> 0.999 residual = small negative
### V. Wrong Positive -> 0.9999 residual = big positive
### V. Correct Negative -> 0.001 residual = small positive
### V. Wrong Negative -> 0.0001 residual = big negative
'''
plt.clf()
fig,ax = plt.subplots(2,5,figsize=(50,20),sharey='row')

for i in range(0,5):
        ax[0,i].hist2d(weight_1_array[i], z0_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[0,i].grid(True)
        ax[0,i].set_xlabel('weight_1_'+str(i),ha="right",x=1)


        ax[1,i].hist2d(weight_1_array[i+5], z0_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[1,i].grid(True)
        ax[1,i].set_xlabel('weight_1_'+str(i+5),ha="right",x=1)
        
ax[1,0].set_ylabel('$z^{PV}_0$ Residual [cm]',ha="right",y=1)
ax[0,0].set_ylabel('$z^{PV}_0$ Residual [cm]',ha="right",y=1)
plt.tight_layout()
plt.savefig('weight_1_z0_residual.png')

plt.clf()
fig,ax = plt.subplots(2,5,figsize=(50,20),sharey='row')

for i in range(0,5):
        ax[0,i].hist2d(weight_2_array[i], z0_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[0,i].grid(True)
        ax[0,i].set_xlabel('weight_2_'+str(i),ha="right",x=1)
        

        ax[1,i].hist2d(weight_2_array[i+5], z0_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[1,i].grid(True)
        ax[1,i].set_xlabel('weight_2_'+str(i+5),ha="right",x=1)

ax[0,0].set_ylabel('$z^{PV}_0$ Residual [cm]',ha="right",y=1)
ax[1,0].set_ylabel('$z^{PV}_0$ Residual [cm]',ha="right",y=1)
plt.tight_layout()
plt.savefig('weight_2_z0_residual.png')

plt.clf()
fig,ax = plt.subplots(4,5,figsize=(50,40),sharey='row')

for i in range(0,5):
        ax[0,i].hist2d(association_1_array[i], z0_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[0,i].grid(True)
        ax[0,i].set_xlabel('association_1_'+str(i),ha="right",x=1)
        
        ax[1,i].hist2d(association_1_array[i+5], z0_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[1,i].grid(True)
        ax[1,i].set_xlabel('association_1_'+str(i+5),ha="right",x=1)
        
        ax[2,i].hist2d(association_1_array[i+10], z0_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[2,i].grid(True)
        ax[2,i].set_xlabel('association_1_'+str(i+10),ha="right",x=1)

        ax[3,i].hist2d(association_1_array[i+15], z0_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[3,i].grid(True)
        ax[3,i].set_xlabel('association_1_'+str(i+15),ha="right",x=1)

ax[0,0].set_ylabel('$z^{PV}_0$ Residual [cm]',ha="right",y=1)
ax[0,1].set_ylabel('$z^{PV}_0$ Residual [cm]',ha="right",y=1)
ax[0,2].set_ylabel('$z^{PV}_0$ Residual [cm]',ha="right",y=1)
ax[0,3].set_ylabel('$z^{PV}_0$ Residual [cm]',ha="right",y=1)
plt.tight_layout()
plt.savefig('association_1_z0_residual.png')

plt.clf()
fig,ax = plt.subplots(4,5,figsize=(50,40),sharey='row')

for i in range(0,5):
        ax[0,i].hist2d(association_2_array[i], z0_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[0,i].grid(True)
        ax[0,i].set_xlabel('association_2_'+str(i),ha="right",x=1)


        ax[1,i].hist2d(association_2_array[i+5], z0_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[1,i].grid(True)
        ax[1,i].set_xlabel('association_2_'+str(i+5),ha="right",x=1)

        ax[2,i].hist2d(association_2_array[i+10], z0_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[2,i].grid(True)
        ax[2,i].set_xlabel('association_2_'+str(i+10),ha="right",x=1)

        ax[3,i].hist2d(association_2_array[i+15], z0_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[3,i].grid(True)
        ax[3,i].set_xlabel('association_2_'+str(i+15),ha="right",x=1)

ax[0,0].set_ylabel('$z^{PV}_0$ Residual [cm]',ha="right",y=1)
ax[0,1].set_ylabel('$z^{PV}_0$ Residual [cm]',ha="right",y=1)
ax[0,2].set_ylabel('$z^{PV}_0$ Residual [cm]',ha="right",y=1)
ax[0,3].set_ylabel('$z^{PV}_0$ Residual [cm]',ha="right",y=1)
plt.tight_layout()
plt.savefig('association_2_z0_residual.png')

plt.clf()
fig,ax = plt.subplots(1,3,figsize=(40,20),sharey='row')

ax[0].hist2d(weight_3_0_array, z0_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
ax[0].grid(True)
ax[0].set_xlabel('weight_3_0',ha="right",x=1)
ax[0].set_ylabel('$z^{PV}_0$ Residual [cm]',ha="right",y=1)

ax[1].hist2d(predicted_weight_array, z0_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
ax[1].grid(True)
ax[1].set_xlabel('predicted_weight',ha="right",x=1)

ax[2].hist2d(trk_z0_array, z0_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
ax[2].grid(True)
ax[2].set_xlabel('trk_z0',ha="right",x=1)


plt.tight_layout()
plt.savefig('other_z0_residual.png')


##################################################
#                                                #
##################################################

plt.clf()
fig,ax = plt.subplots(2,5,figsize=(50,20),sharey='row')

for i in range(0,5):
        ax[0,i].hist2d(weight_1_array[i], association_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[0,i].grid(True)
        ax[0,i].set_xlabel('weight_1_'+str(i),ha="right",x=1)

        ax[1,i].hist2d(weight_1_array[i+5], association_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[1,i].grid(True)
        ax[1,i].set_xlabel('weight_1_'+str(i+5),ha="right",x=1)
        
ax[0,0].set_ylabel('Association Residual',ha="right",y=1)
ax[1,0].set_ylabel('Association Residual',ha="right",y=1)
plt.tight_layout()
plt.savefig('weight_1_association_residual.png')

plt.clf()
fig,ax = plt.subplots(2,5,figsize=(50,20),sharey='row')

for i in range(0,5):
        ax[0,i].hist2d(weight_2_array[i], association_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[0,i].grid(True)
        ax[0,i].set_xlabel('weight_2_'+str(i),ha="right",x=1)

        ax[1,i].hist2d(weight_2_array[i+5], association_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[1,i].grid(True)
        ax[1,i].set_xlabel('weight_2_'+str(i+5),ha="right",x=1)

ax[0,0].set_ylabel('Association Residual',ha="right",y=1)
ax[1,0].set_ylabel('Association Residual',ha="right",y=1)
plt.tight_layout()
plt.savefig('weight_2_association_residual.png')

plt.clf()
fig,ax = plt.subplots(4,5,figsize=(50,40),sharey='row')

for i in range(0,5):
        ax[0,i].hist2d(association_1_array[i], association_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[0,i].grid(True)
        ax[0,i].set_xlabel('association_1_'+str(i),ha="right",x=1)

        ax[1,i].hist2d(association_1_array[i+5], association_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[1,i].grid(True)
        ax[1,i].set_xlabel('association_1_'+str(i+5),ha="right",x=1)

        ax[2,i].hist2d(association_1_array[i+10], association_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[2,i].grid(True)
        ax[2,i].set_xlabel('association_1_'+str(i+10),ha="right",x=1)

        ax[3,i].hist2d(association_1_array[i+15], association_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[3,i].grid(True)
        ax[3,i].set_xlabel('association_1_'+str(i+15),ha="right",x=1)


ax[0,0].set_ylabel('Association Residual',ha="right",y=1)
ax[1,0].set_ylabel('Association Residual',ha="right",y=1)
ax[2,0].set_ylabel('Association Residual',ha="right",y=1)
ax[3,0].set_ylabel('Association Residual',ha="right",y=1)
plt.tight_layout()
plt.savefig('association_1_association_residual.png')

plt.clf()
fig,ax = plt.subplots(4,5,figsize=(50,40),sharey='row')

for i in range(0,5):
        ax[0,i].hist2d(association_2_array[i], association_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[0,i].grid(True)
        ax[0,i].set_xlabel('association_2_'+str(i),ha="right",x=1)

        ax[1,i].hist2d(association_2_array[i+5], association_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[1,i].grid(True)
        ax[1,i].set_xlabel('association_2_'+str(i+5),ha="right",x=1)

        ax[2,i].hist2d(association_2_array[i+10], association_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[2,i].grid(True)
        ax[2,i].set_xlabel('association_2_'+str(i+10),ha="right",x=1)

        ax[3,i].hist2d(association_2_array[i+15], association_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
        ax[3,i].grid(True)
        ax[3,i].set_xlabel('association_2_'+str(i+15),ha="right",x=1)

ax[0,0].set_ylabel('Association Residual',ha="right",y=1)
ax[1,0].set_ylabel('Association Residual',ha="right",y=1)
ax[2,0].set_ylabel('Association Residual',ha="right",y=1)
ax[3,0].set_ylabel('Association Residual',ha="right",y=1)
plt.tight_layout()
plt.savefig('association_2_association_residual.png')

plt.clf()
fig,ax = plt.subplots(1,3,figsize=(40,20),sharey='row')

ax[0].hist2d(weight_3_0_array, association_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
ax[0].grid(True)
ax[0].set_xlabel('weight_3_0',ha="right",x=1)
ax[0].set_ylabel('Association Residual',ha="right",y=1)

ax[1].hist2d(predicted_weight_array, association_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
ax[1].grid(True)
ax[1].set_xlabel('predicted_weight',ha="right",x=1)

ax[2].hist2d(trk_z0_array, association_residual,bins=50,cmap=colormap,norm=matplotlib.colors.LogNorm())
ax[2].grid(True)
ax[2].set_xlabel('trk_z0',ha="right",x=1)

plt.tight_layout()
plt.savefig('other_association_residual.png')
'''