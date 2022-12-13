# #!/bin/bash
source /opt/Xilinx/Vivado/2019.2/settings64.sh
export PATH="/opt/modelsim/2019.2/modeltech/bin/:$PATH"

python convertModels.py setup

mv *.png profiles

./vitis_conv_assoc.sh
./vitis_conv_pattern.sh
./vitis_conv_weights.sh

python readreports.py setup > ResourceUsage.txt

mv *_hls_* Projects
