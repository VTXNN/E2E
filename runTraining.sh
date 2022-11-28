#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node

conda activate qtf

###### CHANGE ##########
export PYTHONPATH=/home/cebrown/Documents/Trigger/E2E/Ops/release:$PYTHONPATH  
export PYTHONPATH=/home/cebrown/Documents/Trigger/E2E/Train:$PYTHONPATH
export COMET_API_KEY=expKifKow3Mn4dnjc1UGGOqrg


export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=8

cd Train

time_stamp=$(date +%Y-%m-%d-%T)
mkdir -p Assets_${time_stamp}

cd Assets_${time_stamp}

cp ../../setup.yaml .
cp ../EvalScripts/eval.py .
cp ../TrainingScripts/train.py .
cp ../TrainingScripts/prune.py .
cp ../UtilScripts/convertModels.py .
cp ../UtilScripts/upload_comet.py .

python train.py setup DA
python train.py setup QDA

for iter in {1..8}
do
    python prune.py setup $iter
    python train.py setup QDA_prune $iter
done

mkdir PruneIterations
mv *.pdf PruneIterations

python convertModels.py OldKF $1OldKF False

mkdir ModelFiles
mv *.pb ModelFiles
mv *.json ModelFiles
mv *.hdf5 ModelFiles
mv *_associationModel ModelFiles
mv *_patternModel ModelFiles
mv *_weightModel ModelFiles

mkdir hls4ml_profile
mv *.png hls4ml_profile
mkdir hls4ml_models
mv *hls* hls4ml_models

mkdir plots
mkdir SavedArrays

python eval.py setup

mkdir tf_model_files
mv *.tf.* tf_model_files
mv *.h5 tf_model_files

mv ModelFiles Assets_${time_stamp}
mv tf_model_files Assets_${time_stamp}
mv hls4ml_profile Assets_${time_stamp}
mv hls4ml_models Assets_${time_stamp}
mv PruneIterations Assets_${time_stamp}

mv plots Assets_${time_stamp}
mv SavedArrays Assets_${time_stamp}
mv *.py Assets_${time_stamp}
mv Assets_${time_stamp}/__init__.py .
mv Assets_${time_stamp}/upload_comet.py .
mv *.yaml Assets_${time_stamp}
mv experimentkey.txt Assets_${time_stamp}
mv *checkpoint Assets_${time_stamp}

python upload_comet.py OldKF Assets_${time_stamp}

rm upload_comet.py