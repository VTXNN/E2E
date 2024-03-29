# Install script for directory: /home/cebrown/Documents/Trigger/E2E/Ops

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/cebrown/Documents/Trigger/E2E/Ops/release")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/libKDEHistogram.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/libKDEHistogram.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/libKDEHistogram.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/libKDEHistogram.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops" TYPE MODULE FILES "/home/cebrown/Documents/Trigger/E2E/Ops/libKDEHistogram.so")
  if(EXISTS "$ENV{DESTDIR}/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/libKDEHistogram.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/libKDEHistogram.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/libKDEHistogram.so"
         OLD_RPATH "/home/cebrown/anaconda3/envs/qtf/lib/python3.9/site-packages/tensorflow:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/libKDEHistogram.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/libHistogramMaxSample.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/libHistogramMaxSample.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/libHistogramMaxSample.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/libHistogramMaxSample.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops" TYPE MODULE FILES "/home/cebrown/Documents/Trigger/E2E/Ops/libHistogramMaxSample.so")
  if(EXISTS "$ENV{DESTDIR}/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/libHistogramMaxSample.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/libHistogramMaxSample.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/libHistogramMaxSample.so"
         OLD_RPATH "/home/cebrown/anaconda3/envs/qtf/lib/python3.9/site-packages/tensorflow:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/libHistogramMaxSample.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/__init__.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops" TYPE FILE FILES "/home/cebrown/Documents/Trigger/E2E/Ops/__init__.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/test_ops.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops" TYPE FILE FILES "/home/cebrown/Documents/Trigger/E2E/Ops/test_ops.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/kde_histogram.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops" TYPE FILE FILES "/home/cebrown/Documents/Trigger/E2E/Ops/kde_histogram.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops/histogram_max.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/cebrown/Documents/Trigger/E2E/Ops/release/vtxops" TYPE FILE FILES "/home/cebrown/Documents/Trigger/E2E/Ops/histogram_max.py")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/cebrown/Documents/Trigger/E2E/Ops/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
