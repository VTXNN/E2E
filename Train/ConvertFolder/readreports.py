import numpy as np
import yaml
import sys
from pathlib import Path
import hls4ml

def getReports(indir,vitis=False):
    data_ = {}
    
    report_vsynth = Path('{}/vivado_synth.rpt'.format(indir))
    report_csynth = Path('{}/myproject_prj/solution1/syn/report/myproject_csynth.rpt'.format(indir))
    
    if report_vsynth.is_file() and report_csynth.is_file():
        print('Found valid vsynth and synth in {}! Fetching numbers'.format(indir))
        
        # Get the resources from the logic synthesis report 
        if vitis:
            with report_vsynth.open() as report:
                lines = np.array(report.readlines())
                data_['lut']     = int(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[2])
                data_['ff']      = int(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[2])
                data_['bram']    = float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[2])
                data_['dsp']     = int(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[2])
                data_['lut_rel'] = float(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[6])
                data_['ff_rel']  = float(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[6])
                data_['bram_rel']= float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[6])
                data_['dsp_rel'] = float(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[6])
        else:
            with report_vsynth.open() as report:
                lines = np.array(report.readlines())
                data_['lut']     = int(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[2])
                data_['ff']      = int(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[2])
                data_['bram']    = float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[2])
                data_['dsp']     = int(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[2])
                data_['lut_rel'] = float(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[5])
                data_['ff_rel']  = float(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[5])
                data_['bram_rel']= float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[5])
                data_['dsp_rel'] = float(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[5])

        with report_csynth.open() as report:
            lines = np.array(report.readlines())
            lat_line = lines[np.argwhere(np.array(['Latency (cycles)' in line for line in lines])).flatten()[0] + 3]
            data_['latency_clks'] = int(lat_line.split('|')[2])
            data_['latency_mus']  = float(lat_line.split('|')[2])*5.0/1000.
            data_['latency_ii']   = int(lat_line.split('|')[6])
    
    return data_

with open(sys.argv[1]+'.yaml', 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

QuantisedPrunedModelName = config["QuantisedPrunedModelName"] 
QuantisedModelName = config["QuantisedModelName"] + "_prune_iteration_0"
UnQuantisedModelName = config["UnquantisedModelName"] 

weight_report = getReports(UnQuantisedModelName+"_hls_weight",True)
print("\n Resource usage and latency: Weight Model")
print(weight_report)

q_weight_report = getReports(QuantisedModelName+"_hls_weight",True)
print("\n Quantised Resource usage and latency: Weight Model")
print(q_weight_report)

qp_weight_report = getReports(QuantisedPrunedModelName+"_hls_weight",True)
print("\n Quantised Pruned Resource usage and latency: Weight Model")
print(qp_weight_report)

# qpv_weight_report = getReports(QuantisedPrunedModelName+"_hls_weight_vitis",True)
# print("\n Quantised Pruned Vitis Resource usage and latency: Weight Model")
# print(qpv_weight_report)

############################################################################

association_report = getReports(UnQuantisedModelName+"_hls_assoc",True)
print("\n Resource usage and latency: Association Model")
print(association_report)

q_association_report = getReports(QuantisedModelName+"_hls_assoc",True)
print("\n Quantised Resource usage and latency: Association Model")
print(q_association_report)

qp_association_report = getReports(QuantisedPrunedModelName+"_hls_assoc",True)
print("\n Quantised Pruned Resource usage and latency: Association Model")
print(qp_association_report)

# qpv_association_report = getReports(QuantisedPrunedModelName+"_hls_assoc",True)
# print("\n Quantised Pruned Vitis Resource usage and latency: Association Model")
# print(qpv_association_report)

############################################################################

pattern_report = getReports(UnQuantisedModelName+"_hls_pattern",True)
print("\n Resource usage and latency: Pattern Model")
print(pattern_report)

q_pattern_report = getReports(QuantisedModelName+"_hls_pattern",True)
print("\n Quantised Resource usage and latency: Pattern Model")
print(q_pattern_report)

qp_pattern_report = getReports(QuantisedPrunedModelName+"_hls_pattern",True)
print("\n Quantised Pruned Resource usage and latency: Pattern Model")
print(qp_pattern_report)

# qpv_pattern_report = getReports(QuantisedPrunedModelName+"_hls_pattern_vitis",True)
# print("\n Quantised Pruned Vitis Resource usage and latency: Pattern Model")
# print(qpv_pattern_report)
