#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node

conda activate qtf

dir=$PWD

python convertTFRecords.py /home/cebrown/Documents/Datasets/VertexDatasets/OldKFGTTData_New12/GTT_TrackNtuple_TT_new12.root /home/cebrown/Documents/Datasets/VertexDatasets/OldKFGTTData_New12

cd /home/cebrown/Documents/Datasets/VertexDatasets/OldKFGTTData_New12

mkdir Val
mkdir Test
mkdir Train

mv data{0..2}.* Val/
mv data{3..6}.* Test/
mv data* Train/

cd $dir

python convertTFRecords.py /home/cebrown/Documents/Datasets/VertexDatasets/OldKFGTTData_New12/GTT_TrackNtuple_TTsl_new12.root /home/cebrown/Documents/Datasets/VertexDatasets/OldKFGTTData_New12

cd /home/cebrown/Documents/Datasets/VertexDatasets/OldKFGTTData_New12

mkdir MET

mv data* MET/


