# --------------------------------------------------------------------------
# SmartPeak
# --------------------------------------------------------------------------
# TODO: license
# --------------------------------------------------------------------------
# $Maintainer: Douglas McCloseky
# $Author: Douglas McCloseky
# --------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.8.2 FATAL_ERROR)

#------------------------------------------------------------------------------
# Handle superbuild first
#------------------------------------------------------------------------------
option (USE_SUPERBUILD "Whether or not a superbuild should be invoked" ON)

if (USE_SUPERBUILD)
  project ("SuperBuild" NONE)
  #include(external/boost.cmake)
  include(external/eigen.cmake)
  include(external/cereal.cmake)
  include(external/cub.cmake)
  return() # stop processing this file further
else()
  project ("SmartPeak_host")
endif()
 
#------------------------------------------------------------------------------
# General CMake definitions & helper
#------------------------------------------------------------------------------
SET(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

# Set C++ version
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

#------------------------------------------------------------------------------
## CMake sanity check: sometimes CMAKE_SIZEOF_VOID_P just vanishes when
## updating CMake.
if (NOT CMAKE_SIZEOF_VOID_P)
	message(FATAL_ERROR "'CMAKE_SIZEOF_VOID_P' is undefined. Thus you should"
                      " delete CMakeFiles (the directory) and the"
                      " CMakeCache.txt and rerun CMake again! This is some"
                      " weird CMake bug that seems to appear when updating the"
                      " CMake version.")
endif()

#------------------------------------------------------------------------------
# Include the standard Dart testing module
#------------------------------------------------------------------------------
# for Nightly Build log and Valgrind for memcheck
option (BUILD_TESTS "Whether or not build unit tests" ON)
if(BUILD_TESTS)
  include(CTest)
  include(Dart)
endif()

#------------------------------------------------------------------------------
option(MT_ENABLE_TBB "Enable Intel TBB support" OFF)
option(MT_ENABLE_OPENMP "Enable OpenMP support" ON)
option(BOOST_USE_STATIC "Use Boost static libraries." ON)

#------------------------------------------------------------------------------
# Extend module path with our modules
# (create an empty list if CMAKE_MODULE_PATH does not exist)
if(NOT CMAKE_MODULE_PATH)
	set(CMAKE_MODULE_PATH)
endif()
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
## CMake looks for the NSIS script in the module path.
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Windows")

#------------------------------------------------------------------------------
# SmartPeak Version Information
#------------------------------------------------------------------------------
set (smartPeak_VERSION_MAJOR 1)
set (smartPeak_VERSION_MINOR 0)
set (smartPeak_VERSION_PATCH 0)

#------------------------------------------------------------------------------
# retrieve detailed informations on the working copy (git)
include(GetGitRevisionDescription)
option(GIT_TRACKING "Embedding of Git version control information into the SmartPeak library (and SmartPeakInfo tool). Note that if activated, every branch-switch or commit will trigger re-generation of the build system (MakeFiles, VS Solutions etc)." ON)
if (GIT_TRACKING)
  ## will configure an include file
  git_short_info(SMARTPEAK_GIT_SHORT_REFSPEC SMARTPEAK_GIT_SHORT_SHA1 SMARTPEAK_GIT_LC_DATE)
endif()

message(STATUS "Building SmartPeak ${SMARTPEAK_PACKAGE_VERSION_MAJOR}.${SMARTPEAK_PACKAGE_VERSION_MINOR}.${SMARTPEAK_PACKAGE_VERSION_PATCH}")
if(NOT GIT_TRACKING)
  set(SMARTPEAK_GIT_SHORT_SHA1 "disabled")
  set(SMARTPEAK_GIT_SHORT_REFSPEC "disabled")
  message(STATUS "  [CMake is not tracking Git commits and branching. To enable use '-D GIT_TRACKING=ON'.]")
elseif(SMARTPEAK_GIT_SHORT_REFSPEC EQUAL "GIT-NOTFOUND" OR ${SMARTPEAK_GIT_SHORT_REFSPEC} EQUAL "HEAD-HASH-NOTFOUND")
  set(SMARTPEAK_GIT_SHORT_SHA1 "exported")
  set(SMARTPEAK_GIT_SHORT_REFSPEC "exported")
  message(STATUS "  [CMake is not tracking Git commits and branching. Git not found.]")
else()
  # everything found, print some status information
  message(STATUS "  - Repository revision ${SMARTPEAK_GIT_SHORT_SHA1}")
  message(STATUS "  - Repository branch ${SMARTPEAK_GIT_SHORT_REFSPEC}")
  message(STATUS "  - Repository last change date ${SMARTPEAK_GIT_LC_DATE}")
  message(STATUS "  [CMake is tracking Git commits and branching. To disable use '-D GIT_TRACKING=OFF'.]")
endif()

#------------------------------------------------------------------------------
message(STATUS "Building SmartPeak ${SMARTPEAK_PACKAGE_VERSION_MAJOR}.${SMARTPEAK_PACKAGE_VERSION_MINOR}.${SMARTPEAK_PACKAGE_VERSION_PATCH}")

#------------------------------------------------------------------------------
# Unsupported systems and other nuances
#------------------------------------------------------------------------------
if (MINGW OR MSYS)
  message(FATAL_ERROR "MSYS and/or MinGW are not supported! Please use a Visual Studio environment! See Windows build instructions for further information!")
endif()

# Handle large template libraries with WIN32
IF(WIN32)
  #ADD_DEFINITIONS(/bigobj)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
  SET(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /bigobj")
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /bigobj")
  SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /bigobj")
ENDIF(WIN32)

#------------------------------------------------------------------------------
# Multi-threading stuff (OpenMP, CUDA, TBB)
#------------------------------------------------------------------------------
include(external/multithreading.cmake)
if (EVONET_CUDA)
  find_package(CUDA REQUIRED)
  if(CUDA_FOUND)
    message(STATUS "Found CUDA Toolkit version ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" )
  endif()
  enable_language(CUDA)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr" )
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-extended-lambda" )
  if(WIN32)
    #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options;/bigobj") #see: https://cmake.org/pipermail/cmake/2017-June/065615.html
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"/bigobj\"")
  endif()
  add_compile_definitions(COMPILE_WITH_CUDA=true)
endif()

#------------------------------------------------------------------------------
# Check architecture 32bit vs. 64bit
#------------------------------------------------------------------------------

if (CMAKE_SIZEOF_VOID_P MATCHES "8")
	set(SMARTPEAK_64BIT_ARCHITECTURE 1 CACHE INTERNAL "Architecture-bits")
	message(STATUS "Architecture: 64 bit")
else()
	set(SMARTPEAK_64BIT_ARCHITECTURE 0 CACHE INTERNAL "Architecture-bits")
	message(STATUS "Architecture: 32 bit")
endif()

#------------------------------------------------------------------------------
# Handle build type
#------------------------------------------------------------------------------
# Set default build type (if not set by user on command line)
if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

# Force build type into the cache (needs to be set beforehand)
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)

if (CMAKE_BUILD_TYPE EQUAL Debug)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
  set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
elseif (CMAKE_BUILD_TYPE EQUAL Release)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")
endif()

#------------------------------------------------------------------------------
# Enable STL debug mode (GCC only)
#------------------------------------------------------------------------------
option(STL_DEBUG "[GCC only] Enable STL-DEBUG mode (very slow)." OFF)
if(STL_DEBUG)
  include(cmake/stl_debug.cmake)
endif()

#------------------------------------------------------------------------------
# Enable AddressSanitizer
#------------------------------------------------------------------------------
option(ADDRESS_SANITIZER "[Clang/GCC only] Enable AddressSanitizer mode (quite slow)." OFF)
if(ADDRESS_SANITIZER)
  include(cmake/AddressSanitizer.cmake)
endif()

#------------------------------------------------------------------------------
# Host directory for referencing from subprojects
set(SMARTPEAK_HOST_DIRECTORY "${PROJECT_SOURCE_DIR}")
set(SMARTPEAK_HOST_BINARY_DIRECTORY "${PROJECT_BINARY_DIR}")

#------------------------------------------------------------------------------
# CMake Utitlities
#------------------------------------------------------------------------------
# include some SmartPeak specific macros
include (${PROJECT_SOURCE_DIR}/cmake/build_system_macros.cmake)
# .. and some to ease the installation process
include (${PROJECT_SOURCE_DIR}/cmake/install_macros.cmake)
# .. and some to ease preparing the documentation builds
include (${PROJECT_SOURCE_DIR}/cmake/doc_macros.cmake)
# .. and some to ease the installation process
include (${PROJECT_SOURCE_DIR}/cmake/add_library_macros.cmake)
# .. and some to ease the export process
include (${PROJECT_SOURCE_DIR}/cmake/export_macros.cmake)

#------------------------------------------------------------------------------
# Output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${SMARTPEAK_HOST_BINARY_DIRECTORY}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${SMARTPEAK_HOST_BINARY_DIRECTORY}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${SMARTPEAK_HOST_BINARY_DIRECTORY}/lib")
set(SMARTPEAK_BINARY_DIR "${SMARTPEAK_HOST_BINARY_DIRECTORY}/bin")
set(SMARTPEAK_WIN32_DLL_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

#------------------------------------------------------------------------------
# define installation subdirectories to allow for custom installations
# note that all those directories are below CMAKE_INSTALL_PREFIX
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")
set(INSTALL_DOC_DIR share/doc CACHE PATH "Installation directory for documentation")
set(INSTALL_SHARE_DIR share/SmartPeak CACHE PATH "Installation directory for shared data")

if(WIN32)
  set(DEF_INSTALL_CMAKE_DIR CMake)
  set(DEF_INSTALL_LIB_DIR bin)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/cmake/SmartPeak)
  set(DEF_INSTALL_LIB_DIR lib)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH "Installation directory for CMake files")
set(INSTALL_LIB_DIR ${DEF_INSTALL_LIB_DIR} CACHE PATH "Installation directory for libraries")

# Never use RPATH. We have other means on every platform.
# E.g. not yet compatible with our fix_mac_dependencies script.
SET(CMAKE_MACOSX_RPATH FALSE)
# macOS-specific variable used when installing own libraries/bins. Use relative paths here.
SET(CMAKE_INSTALL_NAME_DIR "@executable_path/../${INSTALL_LIB_DIR}")

#------------------------------------------------------------------------------
# lib naming when building with msvc & convenience location of build.bat
if (MSVC)
	## use SmartPeakd.dll in debug mode
	SET(CMAKE_DEBUG_POSTFIX d)
	## copy build.bat to root of binary dir to enable convenient invokation (instead of typing path to source dir all the time)
	if(NOT ("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}"))
		file(COPY "${PROJECT_SOURCE_DIR}/tools/build.bat" DESTINATION "${PROJECT_BINARY_DIR}")
	endif()
endif()

#------------------------------------------------------------------------------
# The actual SmartPeak code (headers and source files)
#------------------------------------------------------------------------------
add_subdirectory(src)

#------------------------------------------------------------------------------
# Documentation [TODO]
#------------------------------------------------------------------------------
#add_subdirectory(doc)

#------------------------------------------------------------------------------
# Handle export of targets and install them
smartpeak_export_targets()
install_export_targets()

#------------------------------------------------------------------------------
# Installation and packaging:
#------------------------------------------------------------------------------
# packaging routines:
if(NOT "${PACKAGE_TYPE}" STREQUAL "none")
  include(cmake/package_general.cmake)
  if("${PACKAGE_TYPE}" STREQUAL "dmg") # .. MacOSX
    include(cmake/package_dragndrop_dmg.cmake)
  elseif("${PACKAGE_TYPE}" STREQUAL "rpm") # .. rpms
    include(cmake/package_rpm.cmake)
  elseif("${PACKAGE_TYPE}" STREQUAL "deb") # .. debs
    include(cmake/package_deb.cmake)
  elseif("${PACKAGE_TYPE}" STREQUAL "nsis") # .. Windows nsis
    # For now we use the old NSIS script AS IS
    # i.e. although CMake runs make install etc. NSIS
    # will use everything from the usual build folder like before.
    # TODO automate more and let CMake generate install commands
    # in the script. Unfortunately we have a lot of specific code in the script.
    include(cmake/package_nsis.cmake)
  endif()
  include(CPack)
  include(cmake/package_components.cmake)
endif()