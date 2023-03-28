#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node

conda activate qtf

###### CHANGE ##########
export PYTHONPATH=/home/cebrown/Documents/Trigger/E2E/Ops/release:$PYTHONPATH  
export PYTHONPATH=/home/cebrown/TempE2E/E2E/Train:$PYTHONPATH
export COMET_API_KEY=expKifKow3Mn4dnjc1UGGOqrg

source /opt/Xilinx/Vivado/2021.2/settings64.sh
export PATH="/opt/modelsim/2019.2/modeltech/bin/:$PATH"
export LD_LIBRARY_PATH=/opt/cactus/lib:$LD_LIBRARY_PATH

export TF_CPP_MIN_LOG_LEVEL=10
export OMP_NUM_THREADS=8

# pip uninstall -y hls4ml
# cd ~/hls4ml_vitis/hls4ml
# pip install .[profiling]

# cd ~/TempE2E/E2E

cd Train

time_stamp=$(date +%Y-%m-%d-%T)
mkdir -p Assets_${time_stamp}

cd Assets_${time_stamp}

#cp ../../Unquantised* .
cp ../../setup.yaml .
cp ../EvalScripts/eval.py .
cp ../TrainingScripts/train.py .
cp ../TrainingScripts/prune.py .
cp ../UtilScripts/convertModels.py .
cp ../UtilScripts/upload_comet.py .
cp -r ../ConvertFolder .
cp -r ../Quantising .

python train.py setup DA

cp Unquantised_model*.tf.* Quantising/
cp setup.yaml Quantising/
cd Quantising

python quantiseFPWeightModel.py setup DA 0 True &> unquantisedmodel_weight_build.txt
python quantiseFPPatternModel.py setup DA 0 True &> unquantisedmodel_pattern_build.txt
python quantiseFPAssociationModel.py setup DA 0 True &> unquantisedmodel_association_build.txt

mkdir UnquantisedModel

mv *.png UnquantisedModel
mv *.txt UnquantisedModel
mv Unquantised_model_hls_* ../ConvertFolder
cp *QConfig.yaml UnquantisedModel
mv *.pb ../ConvertFolder/ExportedNetworkFiles
rm *.tf.*
cp Unquantised*QConfig.yaml ..

cd ..

python train.py setup QDA

cp Quantised_model*.tf.* Quantising/
cd Quantising

python quantiseFPWeightModel.py setup QDA 0 True &> quantisedmodel_weight_build.txt
python quantiseFPPatternModel.py setup QDA 0 True &> quantisedmodel_pattern_build.txt
python quantiseFPAssociationModel.py setup QDA 0 True &> quantisedmodel_association_build.txt

mkdir QuantisedModel

mv *.png QuantisedModel
mv *.txt QuantisedModel
mv Quantised_model_prune_iteration_0_hls_* ../ConvertFolder
cp Quantised_model_prune_iteration_0*QConfig.yaml QuantisedModel
mv *.pb ../ConvertFolder/ExportedNetworkFiles
rm *.tf.*
cp *QConfig.yaml ..

cd ..

for iter in {1..8}
do
    python prune.py setup $iter
    cp Quantised_model*.tf.* Quantising/
    cp *.h5 Quantising/

    cd Quantising

    python quantiseFPWeightModel.py setup QDAPrune $iter False &> quantisedmodel_${iter}_weight_build.txt
    python quantiseFPPatternModel.py setup QDAPrune $iter False &> quantisedmodel_${iter}_pattern_build.txt
    python quantiseFPAssociationModel.py setup QDAPrune $iter False &> quantisedmodel_${iter}_association_build.txt

    mkdir QuantisedModel$iter

    mv *.png QuantisedModel$iter
    mv *.txt QuantisedModel$iter
    rm Quantised_model_prune_iteration_${iter}_hls_* 
    rm *.pb
    cp Quantised_model_prune_iteration_${iter}*.yaml QuantisedModel$iter
    rm *.tf.*
    rm *.h5
    cp *QConfig.yaml ..


    cd ..

    python train.py setup QDA_prune $iter
done

cp Quantised_model*.tf.* Quantising/
cp *.h5 Quantising/
cd Quantising
cp Quantised_model_drop_weights_iteration_8.h5 Quantised_model_drop_weights_iteration_9.h5

python quantiseFPWeightModel.py setup QDAPrune 9 True &> quantisedmodel_9_weight_build.txt
python quantiseFPPatternModel.py setup QDAPrune 9 True &> quantisedmodel_9_pattern_build.txt
python quantiseFPAssociationModel.py setup QDAPrune 9 True &> quantisedmodel_9_association_build.txt

mkdir QuantisedModel9

mv *.png QuantisedModel9
mv *.txt QuantisedModel9
mv Quantised_model_prune_iteration_9_hls_* ../ConvertFolder
cp Quantised_model_prune_iteration_9*.yaml QuantisedModel9
mv *.pb ../ConvertFolder/ExportedNetworkFiles
rm *.tf.*
rm *.h5
mv *QConfig.yaml  ..
rm setup.yaml


cd ..

mkdir PruneIterations
mv *.png PruneIterations

mkdir plots
mkdir SavedArrays

python eval.py setup

mv *.tf.* ConvertFolder/NetworkFiles
mv *.h5 ConvertFolder/NetworkFiles
cp setup.yaml ConvertFolder
rm *.yaml
cd ConvertFolder

### ONLY IF YOU HAVE VIVADO INSTALLED
./conv.sh 

cd ..

cp upload_comet.py ..
cp experimentkey.txt ..

cd ..

#tar --force-local -zcvf Assets_${time_stamp}.tgz Assets_${time_stamp}

#python upload_comet.py Assets_${time_stamp}.tgz

#mv Assets_${time_stamp}.tgz Assets

#rm -rf Assets_${time_stamp}

rm upload_comet.py
rm experimentkey.txt
