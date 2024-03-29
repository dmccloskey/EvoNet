

# required modules
include(CMakeParseArguments)
include(GenerateExportHeader)
include(CheckLibArchitecture)

#------------------------------------------------------------------------------
## export a single option indicating if libraries should be build as unity
## build
option(ENABLE_UNITYBUILD "Enables unity builds for all libraries." OFF)

#------------------------------------------------------------------------------
## Unity Build of a set of cpp files
## i.e., make one large compilation unit which usually compiles a lot faster
##
## pros:
##    - compiles a lot faster
##    - reveals double definitions of internal classes in different cpp files
##    - reveals weird variable names (like 'IN'), which are defined as macro in certain headers which are now part of the compilation unit
## cons:
##    - not desirable for developing, since a small change will trigger recreation of the whole lib
function(convert_to_unity_build UB_SUFFIX SOURCE_FILES_NAME)
   set(files ${${SOURCE_FILES_NAME}})
   # generate a unique file name for the unity build translation unit
   set(unit_build_file ${CMAKE_CURRENT_BINARY_DIR}/ub_${UB_SUFFIX}.cpp)
   # exclude all translation units from compilation
   set_source_files_properties(${files} PROPERTIES HEADER_FILE_ONLY true)
   # open the unity build file
   file(WRITE ${unit_build_file} "// Unity Build generated by CMake\n")
   # add include statement for each translation unit
   foreach(source_file ${files})
     # we have headers in there as well, which should not be included explicitly
     if (${source_file} MATCHES "\\.cpp|\\.cxx") # cxx for moc's;
       if (IS_ABSOLUTE ${source_file})
         file( APPEND ${unit_build_file} "#include<${source_file}>\n")
       else()
         file( APPEND ${unit_build_file} "#include<${CMAKE_CURRENT_SOURCE_DIR}/${source_file}>\n")
       endif()
     endif()
   endforeach(source_file)
   # add unity build aggregate as source file
   set(${SOURCE_FILES_NAME} ${${SOURCE_FILES_NAME}} ${unit_build_file} PARENT_SCOPE)
endfunction(convert_to_unity_build)

#------------------------------------------------------------------------------
## Copy the dll produced by the given target to the test/doc binary path.
## @param targetname The target to modify.
## @note This macro will do nothing with non MSVC generators.
macro(copy_dll_to_extern_bin targetname)
  if(MSVC)
    file(TO_NATIVE_PATH "${EVONET_HOST_BINARY_DIRECTORY}/src/tests/class_tests/bin/$(ConfigurationName)/$(TargetFileName)" DLL_TEST_TARGET)
    file(TO_NATIVE_PATH "${EVONET_HOST_BINARY_DIRECTORY}/src/tests/class_tests/bin/$(ConfigurationName)" DLL_TEST_TARGET_PATH)

    file(TO_NATIVE_PATH "${EVONET_HOST_BINARY_DIRECTORY}/doc/doxygen/parameters/$(ConfigurationName)/$(TargetFileName)" DLL_DOC_TARGET)
    file(TO_NATIVE_PATH "${EVONET_HOST_BINARY_DIRECTORY}/doc/doxygen/parameters/$(ConfigurationName)" DLL_DOC_TARGET_PATH)


    add_custom_command(TARGET ${targetname}
                      POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E make_directory "${DLL_TEST_TARGET_PATH}"
                      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${targetname}> ${DLL_TEST_TARGET}
                      COMMAND ${CMAKE_COMMAND} -E make_directory "${DLL_DOC_TARGET_PATH}"
                      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${targetname}> ${DLL_DOC_TARGET})
  endif(MSVC)
endmacro()

#------------------------------------------------------------------------------
# evonet_add_library()
# Create an EvoNet library, install it, register for export of targets, and
# export all required variables for later usage in the build system.
#
# Signature:
# evonet_add_library(TARGET_NAME  EvoNet
#                    SOURCE_FILES  <source files to build the library>
#                    HEADER_FILES  <header files associated to the library>
#                                  (will be installed with the library)
#                    INTERNAL_INCLUDES <list of internal include directories for the library>
#                    PRIVATE_INCLUDES <list of include directories that will used for compilate but that will not be exposed to other libraries>
#                    EXTERNAL_INCLUDES <list of external include directories for the library>
#                                      (will be added with -isystem if available)
#                    LINK_LIBRARIES <list of libraries used when linking the library>
#                    DLL_EXPORT_PATH <path to the dll export header>)
function(evonet_add_library)
  #------------------------------------------------------------------------------
  # parse arguments to function
  set(options )
  set(oneValueArgs TARGET_NAME DLL_EXPORT_PATH)
  set(multiValueArgs INTERNAL_INCLUDES PRIVATE_INCLUDES EXTERNAL_INCLUDES SOURCE_FILES HEADER_FILES LINK_LIBRARIES)
  cmake_parse_arguments(evonet_add_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

  #------------------------------------------------------------------------------
  # Status message for configure output
  message(STATUS "Adding library ${evonet_add_library_TARGET_NAME}")

  #------------------------------------------------------------------------------
  # merge into global exported includes
  set(${evonet_add_library_TARGET_NAME}_INCLUDE_DIRECTORIES ${evonet_add_library_INTERNAL_INCLUDES}
                                                            ${evonet_add_library_EXTERNAL_INCLUDES}
      CACHE INTERNAL "${evonet_add_library_TARGET_NAME} include directories" FORCE)

  #------------------------------------------------------------------------------
  # Include directories
  include_directories(${evonet_add_library_INTERNAL_INCLUDES})
  include_directories(SYSTEM ${evonet_add_library_EXTERNAL_INCLUDES})
  include_directories(SYSTEM ${evonet_add_library_PRIVATE_INCLUDES})

  #------------------------------------------------------------------------------
  # Check if we want a unity build
  if (ENABLE_UNITYBUILD)
  	message(STATUS "Enabled Unity Build for ${evonet_add_library_TARGET_NAME}")
  	convert_to_unity_build(${evonet_add_library_TARGET_NAME}_UnityBuild evonet_add_library_SOURCE_FILES)
  endif()

  #------------------------------------------------------------------------------
  # Add the library
  add_library(${evonet_add_library_TARGET_NAME} ${evonet_add_library_SOURCE_FILES})

  #------------------------------------------------------------------------------
  # Generate export header if requested
  if(NOT ${evonet_add_library_DLL_EXPORT_PATH} STREQUAL "")
    set(_CONFIG_H "include/${evonet_add_library_DLL_EXPORT_PATH}${evonet_add_library_TARGET_NAME}Config.h")
    string(TOUPPER ${evonet_add_library_TARGET_NAME} _TARGET_UPPER_CASE)
    include(GenerateExportHeader)
    generate_export_header(${evonet_add_library_TARGET_NAME}
                          EXPORT_MACRO_NAME ${_TARGET_UPPER_CASE}_DLLAPI
                          EXPORT_FILE_NAME ${_CONFIG_H})

    string(REGEX REPLACE "/" "\\\\" _fixed_path ${evonet_add_library_DLL_EXPORT_PATH})

    # add generated header to visual studio
    source_group("Header Files\\${_fixed_path}" FILES ${_CONFIG_H})
  endif()

  #------------------------------------------------------------------------------
  # Link library against other libraries
  if(evonet_add_library_LINK_LIBRARIES)
    ## check for consistent lib arch (e.g. all 64bit)?
    check_lib_architecture(evonet_add_library_LINK_LIBRARIES)
    target_link_libraries(${evonet_add_library_TARGET_NAME} ${evonet_add_library_LINK_LIBRARIES})
    list(LENGTH evonet_add_library_LINK_LIBRARIES _library_count)
  endif()

  #------------------------------------------------------------------------------
  # Export libraries (self + dependencies)
  set(${evonet_add_library_TARGET_NAME}_LIBRARIES
        ${evonet_add_library_TARGET_NAME}
        ${evonet_add_library_LINK_LIBRARIES}
        CACHE
        INTERNAL "${evonet_add_library_TARGET_NAME} libraries" FORCE)

  #------------------------------------------------------------------------------
  # we also want to install the library
  install_library(${evonet_add_library_TARGET_NAME})
  install_headers("${evonet_add_library_HEADER_FILES};${PROJECT_BINARY_DIR}/${_CONFIG_H}" ${evonet_add_library_TARGET_NAME})

  #------------------------------------------------------------------------------
  # register for export
  evonet_register_export_target(${evonet_add_library_TARGET_NAME})

  #------------------------------------------------------------------------------
  # copy dll to test/doc bin folder on MSVC systems
  copy_dll_to_extern_bin(${evonet_add_library_TARGET_NAME})

  #------------------------------------------------------------------------------
  # Status message for configure output
  message(STATUS "Adding library ${evonet_add_library_TARGET_NAME} - SUCCESS")
endfunction()
