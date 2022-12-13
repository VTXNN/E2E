open_project -reset myproject_prj

set_top myproject
add_files firmware/myproject.cpp -cflags "-std=c++0x"
add_files -tb myproject_test.cpp -cflags "-std=c++0x"
add_files -tb firmware/weights

open_solution -reset "solution1"

set_part {xcvu13p-flga2577-2-e}
#set_part {xcvu9p-flga2104-2L-e}
#set_part {xcku15p-ffval1760-2-3}
create_clock -period 2.8 -name default
config_compile -name_max_length 80
config_schedule -enable_dsp_full_reg=false
config_compile -pipeline_style flp
set_clock_uncertainty 0.35 default

puts "***** C/RTL SYNTHESIS *****"
csynth_design

puts "***** C SIMULATION *****"
csim_design

puts "***** C/RTL SIMULATION *****"
add_files -tb myproject_test.cpp -cflags "-std=c++0x -DRTL_SIM"
set time_start [clock clicks -milliseconds]

cosim_design -trace_level all -setup


set old_pwd [pwd]
cd myproject_prj/solution1/sim/verilog/
source run_sim.tcl
cd $old_pwd

set time_end [clock clicks -milliseconds]
puts "INFO:"
puts [read [open myproject_prj/solution1/sim/report/myproject_cosim.rpt r]]

puts "***** VIVADO SYNTHESIS *****"
exec vivado -mode batch -source vivado_synth.tcl >@ stdout

exit
