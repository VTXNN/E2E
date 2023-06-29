#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node

conda activate qtf

###### CHANGE ##########
export PYTHONPATH=/home/cb719/TempE2E/E2E/Ops/release:$PYTHONPATH  
export PYTHONPATH=/home/cb719/TempE2E/E2E/Train:$PYTHONPATH
export COMET_API_KEY=expKifKow3Mn4dnjc1UGGOqrg

export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=8

source /opt/Xilinx/Vivado/2021.2/settings64.sh
export PATH="/opt/modelsim/2019.2/modeltech/bin/:$PATH"
export LD_LIBRARY_PATH=/opt/cactus/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/cb719/anaconda3/envs/old_qtf/x86_64-conda-linux-gnu/sysroot/usr/lib64/:$LD_LIBRARY_PATH