cmake_minimum_required(VERSION 3.0.2 FATAL_ERROR)
project("EvoNet")

#------------------------------------------------------------------------------
# naming conventions:
#
# prefix a variable with 'CF_' if it is used to configure a file!
# e.g., CF_LibEvoNetExport
set(CF_EVONET_PACKAGE_VERSION "${EVONET_PACKAGE_VERSION_MAJOR}.${EVONET_PACKAGE_VERSION_MINOR}.${EVONET_PACKAGE_VERSION_PATCH}" CACHE INTERNAL "EvoNet VERSION" FORCE)

#------------------------------------------------------------------------------
# En/disable assertions
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
	set(CF_EVONET_ASSERTIONS 1)
else()
	set(CF_EVONET_ASSERTIONS 0)
endif()
set(CF_EVONET_ASSERTIONS ${CF_EVONET_ASSERTIONS} CACHE INTERNAL "Enables debug messages (precondition and postconditions are enabled, a bit slower) - this is NOT changing any compiler flags!" FORCE)

#------------------------------------------------------------------------------
# external libs (contrib or system)
#------------------------------------------------------------------------------
include(${PROJECT_SOURCE_DIR}/cmake_findExternalLibs.cmake)

#------------------------------------------------------------------------------
# At this point make a summary of where data and doc will be located:
message(STATUS "Info: CF_EVONET_DATA_PATH: ${CF_EVONET_DATA_PATH}")
message(STATUS "Info: CF_EVONET_DOC_PATH: ${CF_EVONET_DOC_PATH}")

#------------------------------------------------------------------------------
# configure config.h
#------------------------------------------------------------------------------
include(${PROJECT_SOURCE_DIR}/configh.cmake)

#------------------------------------------------------------------------------
# big include file for headers and cpp files, that fills the EvoNet_sources variable
include (${PROJECT_SOURCE_DIR}/includes.cmake)

#------------------------------------------------------------------------------
# all the dependency libraries are linked into libEvoNet.so
set(EVONET_DEP_LIBRARIES  ${Boost_LIBRARIES})

# xerces requires linking against CoreFoundation&CoreServices
if(APPLE)
  find_library(CoreFoundation_LIBRARY CoreFoundation )
  find_library(CoreServices_LIBRARY CoreServices )
  set(EVONET_DEP_LIBRARIES ${EVONET_DEP_LIBRARIES}
                           ${CoreFoundation_LIBRARY}
                           ${CoreServices_LIBRARY})
endif()

if (TBB_FOUND)
 list(APPEND EVONET_DEP_LIBRARIES ${TBB_LIBRARIES})
endif()
if (MSVC)
	list(APPEND EVONET_DEP_LIBRARIES opengl32.lib)
endif()

evonet_add_library(TARGET_NAME  EvoNet
                   SOURCE_FILES  ${EvoNet_sources}
                   HEADER_FILES  ${EvoNet_sources_h}
                                 ${EvoNet_configured_headers}
                   INTERNAL_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include
                                     ${CMAKE_CURRENT_BINARY_DIR}/include
                   EXTERNAL_INCLUDES ${EIGEN3_INCLUDE_DIR}
                                     ${CEREAL_INCLUDE_DIRS}
                   LINK_LIBRARIES ${APPLE_EXTRA_LIBS}
                                  ${EVONET_DEP_LIBRARIES}
                   DLL_EXPORT_PATH "EvoNet/")

#------------------------------------------------------------------------------
# since the share basically belongs to EvoNet core we control its installation
# here
# Note: that slash(/) is important here, otherwise the whole directory
#       (not its content) will be copied!
install_directory(${EVONET_HOST_DIRECTORY}/share/EvoNet/ ${INSTALL_SHARE_DIR} share)

#------------------------------------------------------------------------------
# register relevant paths for the doxygen doc generation
evonet_doc_path("${PROJECT_SOURCE_DIR}/include")
