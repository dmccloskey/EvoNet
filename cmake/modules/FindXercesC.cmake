# - Try to find Xerces-C++
# Once done this will define
#
#  XercesC_FOUND - system has XercesC
#  XercesC_INCLUDE_DIR - the XercesC include directory
#  XercesC_LIBRARIES - Link these to use XercesC
#  XercesC_VERSION_STRING - the version of XercesC found
#
# Inspired by Ben Morgan, <Ben.Morgan@warwick.ac.uk>
# http://geant4.cern.ch/support/source/geant4/cmake/Modules/FindXercesC.cmake
#

# additional search paths
set(_XercesC_PATHS
  "[HKEY_CURRENT_USER\\software\\xerces-c\\src]"
  "[HKEY_CURRENT_USER\\xerces-c\\src]"
)

set(_XercesC_INCLUDE_TARGET "xercesc/util/XercesVersion.hpp")

# Find Xerce-C include path
find_path(
    XercesC_INCLUDE_DIRS
    PATHS ${_XercesC_PATHS}
    NAMES ${_XercesC_INCLUDE_TARGET}
)

# Find the xerces libraries
if (NOT XercesC_LIBRARIES)
    find_library(XercesC_LIBRARY_RELEASE NAMES xerces-c xerces-c_3 xerces-c_3_1 xerces-c-3.1 xerces-c_3_2 xerces-c-3.2 ${_XercesC_PATHS} PATH_SUFFIXES lib)
    find_library(XercesC_LIBRARY_DEBUG NAMES xerces-c xerces-c_3D xerces-c_3_1D xerces-c-3.1D xerces-c_3_2D xerces-c-3.2D ${_XercesC_PATHS} PATH_SUFFIXES lib)

    include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)
    select_library_configurations(XercesC)
endif ()

# identify xerces version
if (XercesC_INCLUDE_DIRS AND EXISTS "${XercesC_INCLUDE_DIRS}/${_XercesC_INCLUDE_TARGET}")
  file(STRINGS "${XercesC_INCLUDE_DIRS}/${_XercesC_INCLUDE_TARGET}" _XercesC_H REGEX "^#define XERCES_VERSION_.* [0-9]+")
  #define XERCES_VERSION_MAJOR 3
  string(REGEX REPLACE ".*\#define XERCES_VERSION_MAJOR ([0-9]+).*" "\\1" XercesC_VERSION_MAJOR "${_XercesC_H}")
  #define XERCES_VERSION_MINOR 1
  string(REGEX REPLACE ".*\#define XERCES_VERSION_MINOR ([0-9]+).*" "\\1" XercesC_VERSION_MINOR "${_XercesC_H}")
  #define XERCES_VERSION_REVISION 1
  string(REGEX REPLACE ".*\#define XERCES_VERSION_REVISION ([0-9]+).*" "\\1" XercesC_VERSION_REVISION "${_XercesC_H}")

  set(XercesC_VERSION_STRING "${XercesC_VERSION_MAJOR}.${XercesC_VERSION_MINOR}.${XercesC_VERSION_REVISION}")
endif ()

# handle the QUIETLY and REQUIRED arguments and set XercesC_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XercesC
                                  REQUIRED_VARS XercesC_LIBRARIES XercesC_INCLUDE_DIRS
                                  VERSION_VAR XercesC_VERSION_STRING)
