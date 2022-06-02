# CMake generated Testfile for 
# Source directory: /home/cb719/Documents/L1Trigger/Trigger/GTT/Vertexing/E2E/Ops
# Build directory: /home/cb719/Documents/L1Trigger/Trigger/GTT/Vertexing/E2E/Ops
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_import "/home/cb719/anaconda3/envs/qtf/bin/python3" "-c" "import vtxops")
set_tests_properties(test_import PROPERTIES  WORKING_DIRECTORY "/home/cb719/Documents/L1Trigger/Trigger/GTT/Vertexing/E2E/Ops/release" _BACKTRACE_TRIPLES "/home/cb719/Documents/L1Trigger/Trigger/GTT/Vertexing/E2E/Ops/CMakeLists.txt;72;add_test;/home/cb719/Documents/L1Trigger/Trigger/GTT/Vertexing/E2E/Ops/CMakeLists.txt;0;")
add_test(test_ops "/home/cb719/anaconda3/envs/qtf/bin/python3" "/home/cb719/Documents/L1Trigger/Trigger/GTT/Vertexing/E2E/Ops/test_ops.py")
set_tests_properties(test_ops PROPERTIES  WORKING_DIRECTORY "/home/cb719/Documents/L1Trigger/Trigger/GTT/Vertexing/E2E/Ops" _BACKTRACE_TRIPLES "/home/cb719/Documents/L1Trigger/Trigger/GTT/Vertexing/E2E/Ops/CMakeLists.txt;78;add_test;/home/cb719/Documents/L1Trigger/Trigger/GTT/Vertexing/E2E/Ops/CMakeLists.txt;0;")
