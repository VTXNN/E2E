#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node

conda activate qtf

###### CHANGE ##########
export PYTHONPATH=/home/cebrown/Documents/Trigger/E2E/Ops/release:$PYTHONPATH  
export PYTHONPATH=/home/cebrown/TempE2E/E2E/Train:$PYTHONPATH
export COMET_API_KEY=expKifKow3Mn4dnjc1UGGOqrg

export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=8

cd Train

time_stamp=$(date +%Y-%m-%d-%T)
mkdir -p Assets_${time_stamp}

cd Assets_${time_stamp}

cp ../../Unquantised* .
cp ../../setup.yaml .
cp ../EvalScripts/eval.py .
cp ../TrainingScripts/train.py .
cp ../TrainingScripts/prune.py .
cp ../UtilScripts/convertModels.py .
cp ../UtilScripts/upload_comet.py .
cp -r ../ConvertFolder .

#python train.py setup DA
python train.py setup QDA

for iter in {1..8}
do
    python prune.py setup $iter
    python train.py setup QDA_prune $iter
done

mkdir PruneIterations
mv *.pdf PruneIterations

mkdir plots
mkdir SavedArrays

python eval.py setup

mkdir ConvertFolder/NetworkFiles
mv *.tf.* ConvertFolder/NetworkFiles
mv *.h5 ConvertFolder/NetworkFiles
cp setup.yaml ConvertFolder
cd ConvertFolder

### ONLY IF YOU HAVE VIVADO INSTALLED
./conv.sh 

cd ..

cp upload_comet.py ..
cp experimentkey.txt ..

cd ..

tar --force-local -zcvf Assets_${time_stamp}.tgz Assets_${time_stamp}

python upload_comet.py Assets_${time_stamp}.tgz

mv Assets_${time_stamp}.tgz Assets

rm -rf Assets_${time_stamp}

rm upload_comet.py
rm experimentkey.txt
