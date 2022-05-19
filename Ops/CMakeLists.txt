project(VTXNN)
cmake_minimum_required(VERSION 3.14)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

macro(copy_file_if_changed in_file out_file target)  
    add_custom_command (
	    TARGET     ${target}
	    POST_BUILD
	    COMMAND    ${CMAKE_COMMAND}
	    ARGS       -E copy_if_different ${in_file} ${out_file}
	    COMMENT "Copying file: ${in_file} to: ${out_file}"
    )
endmacro(copy_file_if_changed)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specific - will use 'Release' as default")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release
        RelWithDebInfo MinSizeRel."
        FORCE
    )
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic ${CMAKE_CXX_FLAGS}")

find_package(PythonInterp 3 REQUIRED)
find_package(TensorFlow REQUIRED)

set(MODULENAME "vtxops" CACHE STRING "Module name")
message(STATUS "Module name: ${MODULENAME}")

message(STATUS "Install dir: ${CMAKE_INSTALL_PREFIX}")

set(CMAKE_CXX_STANDARD 14) #binary incompatible with c++17
add_definitions(${TF_DEF})

include_directories(${CURRENT_CMAKE_SOURCE_DIR})
include_directories(SYSTEM ${TensorFlow_INCLUDE_DIRS})
add_library(KDEHistogram
    MODULE
    KDEHistogram.cc
    KDEHistogramGrad.cc
)
target_link_libraries(KDEHistogram ${TensorFlow_LIBRARY})
install(TARGETS KDEHistogram LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${MODULENAME})

add_library(HistogramMaxSample
    MODULE
    HistogramMaxSample.cc
)
target_link_libraries(HistogramMaxSample ${TensorFlow_LIBRARY})
install(TARGETS HistogramMaxSample LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${MODULENAME})

set(pythonfiles
    __init__.py
    test_ops.py
    kde_histogram.py
    histogram_max.py
)

foreach(pyfile ${pythonfiles})
    copy_file_if_changed(
        ${CMAKE_SOURCE_DIR}/${pyfile}
        ${CMAKE_CURRENT_BINARY_DIR}/${pyfile}
        KDEHistogram
    )
    install(FILES ${CMAKE_SOURCE_DIR}/${pyfile} DESTINATION "${CMAKE_INSTALL_PREFIX}/${MODULENAME}")
endforeach(pyfile)

enable_testing()

add_test(
    NAME test_import
    COMMAND ${PYTHON_EXECUTABLE} -c "import ${MODULENAME}"
    WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}
)

add_test(
    NAME test_ops
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_ops.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)



