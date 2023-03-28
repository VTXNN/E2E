# #!/bin/bash

#python convertModels.py setup

#mv *.png profiles

#./vitis_conv_weight.sh > weightsout.txt
#./vitis_conv_assoc.sh   > assocout.txt
#./vitis_conv_pattern.sh > patternout.txt

python readreports.py setup > ResourceUsage.txt

mv *_hls_* Projects
