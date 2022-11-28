#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node

conda activate qtf

dir=$PWD

mkdir /home/cebrown/Documents/Datasets/VertexDatasets/OldKFGTTData_EmuTQ/NewData

python convertTFRecords.py /home/cebrown/Documents/Datasets/VertexDatasets/OldKFGTTData_EmuTQ/GTT_TrackNtuple_TT.root /home/cebrown/Documents/Datasets/VertexDatasets/OldKFGTTData_EmuTQ/NewData

cd /home/cebrown/Documents/Datasets/VertexDatasets/OldKFGTTData_EmuTQ/NewData

mkdir Val
mkdir Test
mkdir Train

mv data{0..2}.* Val/
mv data{3..6}.* Test/
mv data* Train/

cd $dir

python convertTFRecords.py /home/cebrown/Documents/Datasets/VertexDatasets/OldKFGTTData_EmuTQ/GTT_TrackNtuple_TTsl.root /home/cebrown/Documents/Datasets/VertexDatasets/OldKFGTTData_EmuTQ/NewData

cd /home/cebrown/Documents/Datasets/VertexDatasets/OldKFGTTData_EmuTQ/NewData

mkdir MET

mv data* MET/

