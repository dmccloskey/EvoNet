#------------------------------------------------------------------------------
# This cmake file handles finding external libs for SmartPeak
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# set which library extensions are preferred (we prefer shared libraries)
if(NOT MSVC)
	set(CMAKE_FIND_LIBRARY_SUFFIXES ".so;.a")
endif()
if (APPLE)
	set(CMAKE_FIND_LIBRARY_SUFFIXES ".dylib;.a")
endif()


#------------------------------------------------------------------------------
# find libs (for linking)
# On Windows:
#   * on windows we need the *.lib versions (dlls alone won't do for linking)
#   * never mix Release/Debug versions of libraries. Leads to strange segfaults,
#     stack corruption etc, due to different runtime libs ...

#------------------------------------------------------------------------------
# BOOST
find_boost(iostreams date_time math_c99 regex)

if(Boost_FOUND)
  message(STATUS "Found Boost version ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}" )
  set(CF_SMARTPEAK_BOOST_VERSION_MAJOR ${Boost_MAJOR_VERSION})
	set(CF_SMARTPEAK_BOOST_VERSION_MINOR ${Boost_MINOR_VERSION})
  set(CF_SMARTPEAK_BOOST_VERSION_SUBMINOR ${Boost_SUBMINOR_VERSION})
	set(CF_SMARTPEAK_BOOST_VERSION ${Boost_VERSION})
else()
  message(FATAL_ERROR "Boost or one of its components not found!")
endif()

#------------------------------------------------------------------------------
# COIN-OR
set(CF_USECOINOR 1)
find_package(COIN REQUIRED)

#------------------------------------------------------------------------------
# GLPK
find_package(GLPK REQUIRED)
if (GLPK_FOUND)
	set(CF_SMARTPEAK_GLPK_VERSION_MAJOR ${GLPK_VERSION_MAJOR})
	set(CF_SMARTPEAK_GLPK_VERSION_MINOR ${GLPK_VERSION_MINOR})
	set(CF_SMARTPEAK_GLPK_VERSION ${GLPK_VERSION_STRING})
endif()

#------------------------------------------------------------------------------
# SQLITE
find_package(SQLITE 3.15.0 REQUIRED)

#------------------------------------------------------------------------------
# Done finding contrib libraries
#------------------------------------------------------------------------------

#except for the contrib libs, prefer shared libraries
if(NOT MSVC AND NOT APPLE)
	set(CMAKE_FIND_LIBRARY_SUFFIXES ".so;.a")
endif()

#------------------------------------------------------------------------------
