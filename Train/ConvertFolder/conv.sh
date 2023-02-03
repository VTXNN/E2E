# #!/bin/bash
source /opt/Xilinx/Vivado/2019.2/settings64.sh
export PATH="/opt/modelsim/2019.2/modeltech/bin/:$PATH"
export LD_LIBRARY_PATH=/opt/cactus/lib:$LD_LIBRARY_PATH

python convertModels.py setup

mv *.png profiles

./vitis_conv_weight.sh > weightsout.txt
./vitis_conv_assoc.sh   > assocout.txt
./vitis_conv_pattern.sh > patternout.txt

python readreports.py setup > ResourceUsage.txt

mv *_hls_* Projects
