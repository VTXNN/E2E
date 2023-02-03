# #!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate qtf
source /opt/Xilinx/Vivado/2021.2/settings64.sh
export PATH="/opt/modelsim/2019.2/modeltech/bin/:$PATH"
export LD_LIBRARY_PATH=/opt/cactus/lib:$LD_LIBRARY_PATH

export COMET_API_KEY=expKifKow3Mn4dnjc1UGGOqrg
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=8


cp -r Quantised_model_prune_iteration_9_hls_association Quantised_model_prune_iteration_9_hls_association_vitis

#source ~/firmware.sh 2021

cp nnet_utils_0_6_0/* Quantised_model_prune_iteration_9_hls_association_vitis/firmware/nnet_utils

sed -i '/config_array_partition maximum_size="4096"/d' Quantised_model_prune_iteration_9_hls_association_vitis/myproject_prj/solution1/solution1.aps
sed -i '/config_compile name_max_length="80"/d' Quantised_model_prune_iteration_9_hls_association_vitis/myproject_prj/solution1/solution1.aps

cp build_vitis_prj.tcl Quantised_model_prune_iteration_9_hls_association_vitis
cd Quantised_model_prune_iteration_9_hls_association_vitis

vitis_hls -f build_vitis_prj.tcl
