

cmake_minimum_required(VERSION 3.8.2 FATAL_ERROR)
project("EvoNet_class_tests_evonet")

#------------------------------------------------------------------------------
# Configure test file to get the TEST_DATA_PATH into the tests
set(CF_EVONET_TEST_DATA_PATH "${PROJECT_SOURCE_DIR}/data/")
set (CONFIGURED_TEST_CONFIG_H "${PROJECT_BINARY_DIR}/include/EvoNet/test_config.h")
configure_file(${PROJECT_SOURCE_DIR}/include/EvoNet/test_config.h.in ${CONFIGURED_TEST_CONFIG_H})

#------------------------------------------------------------------------------
# Boost
#------------------------------------------------------------------------------
find_package(boost 1.64 COMPONENTS unit_test_framework)

if(Boost_FOUND)
  message(STATUS "Found Boost version ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}" )
  set(CF_EVONET_BOOST_VERSION_MAJOR ${Boost_MAJOR_VERSION})
	set(CF_EVONET_BOOST_VERSION_MINOR ${Boost_MINOR_VERSION})
  set(CF_EVONET_BOOST_VERSION_SUBMINOR ${Boost_SUBMINOR_VERSION})
	set(CF_EVONET_BOOST_VERSION ${Boost_VERSION})
else()
  message(FATAL_ERROR "Boost or one of its components not found!")
endif()

#------------------------------------------------------------------------------
# get the test executables
include(executables.cmake)

#------------------------------------------------------------------------------
# Include directories for tests
set(EVONET_CLASS_TESTS_INTERNAL_INCLUDE_DIRECTORIES "${PROJECT_BINARY_DIR}/include/")
# add EvoNet directories
set(EVONET_CLASS_TESTS_EXTERNAL_INCLUDE_DIRECTORIES "${EvoNet_INCLUDE_DIRECTORIES};${Boost_INCLUDE_DIRS}")
include_directories(${EVONET_CLASS_TESTS_INTERNAL_INCLUDE_DIRECTORIES})
include_directories(SYSTEM ${EVONET_CLASS_TESTS_EXTERNAL_INCLUDE_DIRECTORIES})

#------------------------------------------------------------------------------
# disable optimization for tests for gcc like compilers
if (CMAKE_COMPILER_IS_INTELCXX OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANG)
  set(_TMP_CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  set(CMAKE_CXX_FLAGS_RELEASE "-O0")
endif()

#------------------------------------------------------------------------------
# core
add_custom_target(CORE_TEST)
add_dependencies(CORE_TEST ${core_executables_list})

# io
add_custom_target(IO_TEST)
add_dependencies(IO_TEST ${io_executables_list})

# ml
add_custom_target(ML_TEST)
add_dependencies(ML_TEST ${ml_executables_list})

# ml
add_custom_target(MODELS_TEST)
add_dependencies(MODELS_TEST ${models_executables_list})

# algorithm
add_custom_target(SIMULATOR_TEST)
add_dependencies(SIMULATOR_TEST ${simulator_executables_list})

# algorithm
add_custom_target(GRAPH_TEST)
add_dependencies(GRAPH_TEST ${graph_executables_list})

#------------------------------------------------------------------------------
# Add the actual tests
foreach(_class_test ${TEST_executables})
  add_executable(${_class_test} source/${_class_test})
  target_link_libraries(${_class_test} ${EvoNet_LIBRARIES} ${Boost_LIBRARIES})
  add_test(${_class_test} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${_class_test})
  # only add OPENMP flags to gcc linker (execpt Mac OS X, due to compiler bug
  # see https://sourceforge.net/apps/trac/open-ms/ticket/280 for details)
  if (OPENMP_FOUND AND NOT MSVC AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_target_properties(${_class_test} PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
  endif()
  if (EVONET_CUDA)
    target_link_libraries(${_class_test} ${CUDA_LIBRARIES})
    set_property(TARGET ${_class_test} PROPERTY CUDA_STANDARD 11)
  endif()
endforeach(_class_test)

#------------------------------------------------------------------------------
# restore old compiler flags
if (CMAKE_COMPILER_IS_INTELCXX OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANG)
  set(${CMAKE_CXX_FLAGS_RELEASE} ${_TMP_CMAKE_CXX_FLAGS_RELEASE})
endif()

#------------------------------------------------------------------------------
# add filenames to Visual Studio solution tree
set(sources_VS)
foreach(i ${TEST_executables})
  list(APPEND sources_VS "${i}")
endforeach(i)
source_group("" FILES ${sources_VS})
