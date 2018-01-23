# --------------------------------------------------------------------------
# SmartPeak
# --------------------------------------------------------------------------
# TODO: license
# --------------------------------------------------------------------------
# $Maintainer: Douglas McCloseky
# $Author: Douglas McCloseky
# --------------------------------------------------------------------------


cmake_minimum_required(VERSION 3.0.2 FATAL_ERROR)
project ("smartPeak_host")

# The version number.
set (smartPeak_VERSION_MAJOR 1)
set (smartPeak_VERSION_MINOR 0)
set (smartPeak_VERSION_PATCH 0)
 
# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/TutorialConfig.h.in"
  "${PROJECT_BINARY_DIR}/TutorialConfig.h"
  )
 
# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")
 
# add the executable
add_executable(Tutorial tutorial.cxx)