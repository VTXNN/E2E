#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node

conda activate qtf
export PYTHONPATH=Ops/release:$PYTHONPATH  

export COMET_API_KEY=
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=8

cp setupQDANewKF.yaml Train
cd Train
cp TrainingScripts/trainQprune.py .
cp TrainingScripts/trainDA.py .
cp EvalScripts/evalQprune.py .

python trainDA.py NewKF setupQDANewKF

mv NewKFbest_weights.tf.data-00000-of-00001 NewKFbest_weights_unquantised.tf.data-00000-of-00001 
mv NewKFbest_weights.tf.index NewKFbest_weights_unquantised.tf.index 

python trainQprune.py NewKF setupQDANewKF

mkdir NewKFQplots

python evalQprune.py NewKF setupQDANewKF

rm NewKFexperimentkey.txt
rm checkpoint
rm setupQDANewKF.yaml
rm trainQprune.py
rm trainDA.py
rm evalQprune.py
