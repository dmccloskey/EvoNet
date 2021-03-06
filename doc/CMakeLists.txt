project("SmartPeak_doc")
cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

#------------------------------------------------------------------------------
# helper macro
#------------------------------------------------------------------------------
macro(make_doxygen_path doxygen_style_list_var doc_path_list_var)
  set(${doxygen_style_list_var} "")
  foreach(_doc_path ${${doc_path_list_var}})
    set(${doxygen_style_list_var} "\"${_doc_path}\" ${${doxygen_style_list_var}}")
  endforeach()
endmacro()

#------------------------------------------------------------------------------
# decide which doc target is part of all
#------------------------------------------------------------------------------
if (HAS_XSERVER)
  set(_DOC_ALL "ALL")
  set(_DOC_MINIMAL_ALL "")
  set(_DOC_MINIMAL_EXTRA_MESSAGE )
else ()
  set(_DOC_ALL "")
  set(_DOC_MINIMAL_ALL "ALL")
  set(_DOC_MINIMAL_EXTRA_MESSAGE
      COMMAND ${CMAKE_COMMAND} -E echo "We build only a minimal doc since SmartPeak was configured without HAS_XSERVER=Off."
      COMMAND ${CMAKE_COMMAND} -E echo "If you want the full doc, either execute cmake --build . --target doc manually"
      COMMAND ${CMAKE_COMMAND} -E echo "or rerun cmake with -DHAS_XSERVER=On."
  )
endif ()

#------------------------------------------------------------------------------
# doc programs
#------------------------------------------------------------------------------
set(SmartPeak_doc_executables)
include(doxygen/parameters/executables.cmake)

set(SMARTPEAK_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "doxygen/parameters")

# check whether SmartPeak_GUI is available or not
set(_doc_progs_include )
set(_doc_progs_link_libraries )
if(WITH_GUI)
  set(_doc_progs_include ${SmartPeak_GUI_INCLUDE_DIRECTORIES})
  set(_doc_progs_link_libraries ${SmartPeak_GUI_LIBRARIES})
else()
  set(_doc_progs_include ${SmartPeak_INCLUDE_DIRECTORIES})
  set(_doc_progs_link_libraries ${SmartPeak_LIBRARIES})
endif()

include_directories(SYSTEM ${_doc_progs_include})
add_definitions(/DBOOST_ALL_NO_LIB)

# build doc executables
foreach(i ${SmartPeak_doc_executables})
  add_executable(${i} EXCLUDE_FROM_ALL doxygen/parameters/${i}.cpp)
  target_link_libraries(${i} ${_doc_progs_link_libraries})

  # let the doc progs decide how to handle a missing SmartPeak_GUI library
  if(WITH_GUI)
    set_target_properties(${i} PROPERTIES COMPILE_DEFINITIONS "WITH_GUI=1")
  endif()
endforeach(i)

# collection target
add_custom_target(doc_progs)
add_dependencies(doc_progs ${SmartPeak_doc_executables} TOPP UTILS)

if(WITH_GUI)
  add_dependencies(doc_progs GUI)
endif()

#------------------------------------------------------------------------------
# Initialize variables needed for a proper doxygen configuration
#------------------------------------------------------------------------------
set(CF_SMARTPEAK_BIN_PATH ${PROJECT_BINARY_DIR})
set(CF_SMARTPEAK_SRC_PATH ${SMARTPEAK_HOST_DIRECTORY})

# make it a doxygen conform list
make_doxygen_path(CF_SMARTPEAK_DOCUMENTATION_DIRECTORIES SMARTPEAK_DOCUMENTATION_DIRECTORIES)
make_doxygen_path(CF_SMARTPEAK_DOCUMENTATION_STRIP_INCLUDES SMARTPEAK_DOCUMENTATION_DIRECTORIES)
make_doxygen_path(CF_STRIP_PATH SMARTPEAK_DOCUMENTATION_DIRECTORIES)

#------------------------------------------------------------------------------
# find the necessary packages
find_package(Doxygen)
find_package(LATEX)
# Doxygen with formulas built with LaTeX actually also needs Ghostscript but there is no official find module for it, yet.
#find_package(Ghostscript)

#------------------------------------------------------------------------------
# the doc targets
#------------------------------------------------------------------------------
# As soon as there is a TeX formula in the html documentation we need LaTeX, dvips and ghostscript.
# Probably only affects class documentation.

if (DOXYGEN_FOUND)
  #------------------------------------------------------------------------------
  # configure doxygen configuration files
  configure_file(${PROJECT_SOURCE_DIR}/doxygen/Doxyfile.in ${PROJECT_BINARY_DIR}/doxygen/Doxyfile)
  configure_file(${PROJECT_SOURCE_DIR}/doxygen/Doxyfile_dot.in ${PROJECT_BINARY_DIR}/doxygen/Doxyfile_dot)
  configure_file(${PROJECT_SOURCE_DIR}/doxygen/Doxyfile_noclass.in ${PROJECT_BINARY_DIR}/doxygen/Doxyfile_noclass)
  configure_file(${PROJECT_SOURCE_DIR}/doxygen/Doxyfile_xml.in ${PROJECT_BINARY_DIR}/doxygen/Doxyfile_xml)

  #------------------------------------------------------------------------------
  # create refman files for PDF tutorials
  configure_file(${PROJECT_SOURCE_DIR}/SmartPeak_tutorial/refman_overwrite.tex.in ${PROJECT_BINARY_DIR}/SmartPeak_tutorial/refman_overwrite.tex)
  configure_file(${PROJECT_SOURCE_DIR}/TOPP_tutorial/refman_overwrite.tex.in ${PROJECT_BINARY_DIR}/TOPP_tutorial/refman_overwrite.tex)

  #------------------------------------------------------------------------------
  # doc paths (bin path, topp documenter, defaultparamhandler documenter)
  #------------------------------------------------------------------------------
  # Checks for Multiconfiguration generators like MSVC or XCode
  if(NOT CMAKE_CONFIGURATION_TYPES)
    set(_TOPPDOCUMENTER_EXECUTABLE "${PROJECT_BINARY_DIR}/doxygen/parameters/TOPPDocumenter")
    set(_DEFAULTPARAMHANDLERDOCUMENTER_EXECUTABLE "${PROJECT_BINARY_DIR}/doxygen/parameters/DefaultParamHandlerDocumenter")
    set(_BINARY_PATH "${SMARTPEAK_RUNTIME_OUTPUT_DIRECTORY}")
  else()
    set(_TOPPDOCUMENTER_EXECUTABLE "${PROJECT_BINARY_DIR}/doxygen/parameters/${CMAKE_CFG_INTDIR}/TOPPDocumenter")
    set(_DEFAULTPARAMHANDLERDOCUMENTER_EXECUTABLE "${PROJECT_BINARY_DIR}/doxygen/parameters/${CMAKE_CFG_INTDIR}/DefaultParamHandlerDocumenter")
    set(_BINARY_PATH "${SMARTPEAK_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}")
  endif()

  file(TO_NATIVE_PATH "${_TOPPDOCUMENTER_EXECUTABLE}" TOPPDOCUMENTER_EXECUTABLE)
  file(TO_NATIVE_PATH "${_DEFAULTPARAMHANDLERDOCUMENTER_EXECUTABLE}" DEFAULTPARAMHANDLERDOCUMENTER_EXECUTABLE)
  file(TO_NATIVE_PATH "${_BINARY_PATH}" BINARY_PATH)

  #------------------------------------------------------------------------------
  # doc_param_internal targets
  add_custom_target(doc_param_internal
                    COMMAND ${CMAKE_COMMAND} -E echo ""
                    COMMAND ${CMAKE_COMMAND} -E echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                    COMMAND ${CMAKE_COMMAND} -E echo "Creating the algorithm parameter and TOPP parameter documentation"
                    COMMAND ${CMAKE_COMMAND} -E echo ""
                    COMMAND ${CMAKE_COMMAND} -E echo "Note: A functioning SmartPeak/TOPP installation and a running X-server (Unix) is required for this step!"
                    COMMAND ${CMAKE_COMMAND} -E echo "      Windows only: SmartPeak and Open_GUI.dll's need to be accessible by the Documenter executables in"
                    COMMAND ${CMAKE_COMMAND} -E echo "      doxygen/parameters/. If the automatic CMake post build copy step did not work, try to add them"
                    COMMAND ${CMAKE_COMMAND} -E echo "      to your Path variable."
                    COMMAND ${CMAKE_COMMAND} -E echo "      If this step fails, use the target 'doc_minimal'."
                    COMMAND ${CMAKE_COMMAND} -E echo ""
                    COMMAND ${CMAKE_COMMAND} -E echo "Building SmartPeak parameter docu:"
                    COMMAND ${CMAKE_COMMAND} -E echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                    COMMAND ${CMAKE_COMMAND} -E make_directory doxygen/parameters/output/
                    COMMAND ${CMAKE_COMMAND} -E chdir doxygen/parameters/ ${DEFAULTPARAMHANDLERDOCUMENTER_EXECUTABLE}
                    COMMAND ${CMAKE_COMMAND} -E echo ""
                    COMMAND ${CMAKE_COMMAND} -E echo "Building TOPP/UTILS docu:"
                    COMMAND ${CMAKE_COMMAND} -E echo "~~~~~~~~~~~~~~~~~~~~~~~~~"
                    COMMAND ${CMAKE_COMMAND} -E chdir doxygen/parameters/ ${TOPPDOCUMENTER_EXECUTABLE} ${BINARY_PATH}
                    COMMAND ${CMAKE_COMMAND} -E echo ""
                    COMMENT "Build the parameters documentation"
                    VERBATIM)
  add_dependencies(doc_param_internal doc_progs)

  #------------------------------------------------------------------------------
  # doc(_html) generation code reused in two independent targets
  set(_DOC_HTML_CODE
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    COMMAND ${CMAKE_COMMAND} -E echo "Creating html documentation"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E remove_directory html
    COMMAND ${CMAKE_COMMAND} -E chdir ${PROJECT_BINARY_DIR} "${DOXYGEN_EXECUTABLE}" doxygen/Doxyfile
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/index.html index.html
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/doxygen/common/style_ini.css html/style_ini.css
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    COMMAND ${CMAKE_COMMAND} -E echo "The documentation has been successfully created."
    COMMAND ${CMAKE_COMMAND} -E echo "You can now open 'doc/index.html' in a web browser."
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMENT "Build the doxygen documentation"
  )

  # regular doc target executed to generate full class and tool documentation (without tutorials)
  add_custom_target(doc ${_DOC_ALL}
                    ${_DOC_HTML_CODE}
                    DEPENDS doc_param_internal
                    VERBATIM)

  # doc_html_only to generate only the doxygen documentation (e.g., when writing new documentation)
  add_custom_target(doc_class_only
                    ${_DOC_HTML_CODE}
                    COMMAND ${CMAKE_COMMAND} -E echo "NOTE: The algorithm/TOPP parameter documentation was not generated/updated."
                    COMMAND ${CMAKE_COMMAND} -E echo "      You will only see updates to the class documentation/doxygen files."
                    COMMAND ${CMAKE_COMMAND} -E echo "      To build the full documentation execute the doc target."
                    VERBATIM)

  #------------------------------------------------------------------------------
  # doc_internal target
  add_custom_target(doc_xml
                    COMMAND ${CMAKE_COMMAND} -E echo ""
                    COMMAND ${CMAKE_COMMAND} -E echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                    COMMAND ${CMAKE_COMMAND} -E echo "Creating XML documentation"
                    COMMAND ${CMAKE_COMMAND} -E echo ""
                    COMMAND ${CMAKE_COMMAND} -E remove_directory xml_output
                    COMMAND ${CMAKE_COMMAND} -E chdir ${PROJECT_BINARY_DIR} "${DOXYGEN_EXECUTABLE}" doxygen/Doxyfile_xml
                    COMMAND ${CMAKE_COMMAND} -E echo ""
                    COMMAND ${CMAKE_COMMAND} -E echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                    COMMAND ${CMAKE_COMMAND} -E echo "The XML documentation has been successfully created."
                    COMMAND ${CMAKE_COMMAND} -E echo ""
                    COMMENT "Build the doxygen documentation"
                    VERBATIM)

  #------------------------------------------------------------------------------
  # doc_noclass target
  add_custom_target(doc_noclass
                    COMMAND ${CMAKE_COMMAND} -E echo ""
                    COMMAND ${CMAKE_COMMAND} -E echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                    COMMAND ${CMAKE_COMMAND} -E echo "Creating html documentation without class documentation"
                    COMMAND ${CMAKE_COMMAND} -E echo ""
                    COMMAND ${CMAKE_COMMAND} -E remove_directory html
                    COMMAND ${CMAKE_COMMAND} -E chdir ${PROJECT_BINARY_DIR} "${DOXYGEN_EXECUTABLE}" doxygen/Doxyfile_noclass
                    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/index.html index.html
                    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/doxygen/common/style_ini.css html/style_ini.css
                    COMMAND ${CMAKE_COMMAND} -E echo ""
                    COMMAND ${CMAKE_COMMAND} -E echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                    COMMAND ${CMAKE_COMMAND} -E echo "The documentation has been successfully created."
                    COMMAND ${CMAKE_COMMAND} -E echo "You can now open 'doc/index.html' in a web browser."
                    COMMAND ${CMAKE_COMMAND} -E echo ""
                    COMMENT "Build the doxygen documentation"
                    VERBATIM)
  add_dependencies(doc_noclass doc_param_internal)

  #------------------------------------------------------------------------------
  # doc_minimal target
  add_custom_target(doc_minimal ${_DOC_MINIMAL_ALL}
                    COMMAND ${CMAKE_COMMAND} -E echo ""
                    COMMAND ${CMAKE_COMMAND} -E echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                    COMMAND ${CMAKE_COMMAND} -E echo "Creating html documentation without class/TOPP/UTILS documentation"
                    COMMAND ${CMAKE_COMMAND} -E echo ""
                    COMMAND ${CMAKE_COMMAND} -E remove_directory html
                    COMMAND ${CMAKE_COMMAND} -E chdir ${PROJECT_BINARY_DIR} "${DOXYGEN_EXECUTABLE}" doxygen/Doxyfile_noclass
                    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/index.html index.html
                    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/doxygen/common/style_ini.css html/style_ini.css
                    COMMAND ${CMAKE_COMMAND} -E echo ""
                    COMMAND ${CMAKE_COMMAND} -E echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                    COMMAND ${CMAKE_COMMAND} -E echo "The documentation has been successfully created."
                    COMMAND ${CMAKE_COMMAND} -E echo "You can now open 'doc/index.html' in a web browser."
                    ${_DOC_MINIMAL_EXTRA_MESSAGE}
                    COMMAND ${CMAKE_COMMAND} -E echo ""
                    COMMENT "Build the doxygen documentation"
                    VERBATIM)
  #------------------------------------------------------------------------------
  # add virtual dependency of doc_minimal on TOPP & UTILS
  # this is not necessary but defers the generation of doc_minimal to a later
  # stage to avoid confusion if doc_minimal is build first
  add_dependencies(doc_minimal TOPP UTILS)

  if (DOXYGEN_DOT_FOUND OR DOXYGEN_DOT_EXECUTABLE)
    #------------------------------------------------------------------------------
    # doc_dot target
    add_custom_target(doc_dot
                      COMMAND ${CMAKE_COMMAND} -E echo ""
                      COMMAND ${CMAKE_COMMAND} -E echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                      COMMAND ${CMAKE_COMMAND} -E echo "Creating DOT html documentation"
                      COMMAND ${CMAKE_COMMAND} -E echo ""
                      COMMAND ${CMAKE_COMMAND} -E remove_directory html-dot
                      COMMAND ${CMAKE_COMMAND} -E chdir ${PROJECT_BINARY_DIR} "${DOXYGEN_EXECUTABLE}" doxygen/Doxyfile_dot
                      COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/index.html index.html
                      COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/doxygen/common/style_ini.css html/style_ini.css
                      COMMAND ${CMAKE_COMMAND} -E echo ""
                      COMMAND ${CMAKE_COMMAND} -E echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                      COMMAND ${CMAKE_COMMAND} -E echo "The documentation has been successfully created."
                      COMMAND ${CMAKE_COMMAND} -E echo "You can now open 'doc/index.html' in a web browser."
                      COMMAND ${CMAKE_COMMAND} -E echo ""
                      COMMENT "Build the doxygen documentation"
                      VERBATIM)
    add_dependencies(doc_dot doc_param_internal)
  else()
    message(STATUS "DOT not found. Disabling target 'doc_dot'!")
  endif()

  #------------------------------------------------------------------------------
  # Install documentation
  install_directory(${PROJECT_BINARY_DIR}/html ${INSTALL_DOC_DIR} doc)
  install_file(${PROJECT_BINARY_DIR}/index.html ${INSTALL_DOC_DIR} doc)

  #------------------------------------------------------------------------------
  # PDF Tutorials
  #------------------------------------------------------------------------------
  # DOXYGEN_FOUND is true. Otherwise we would not list any doc targets at all.
  if (PDFLATEX_COMPILER)

    if (ENABLE_TUTORIALS)
      set(DOC_TUTORIALS_ACTIVE TRUE)

      # configure doxyfiles for tutorials
      configure_file(${PROJECT_SOURCE_DIR}/SmartPeak_tutorial/Doxyfile.in ${PROJECT_BINARY_DIR}/SmartPeak_tutorial/Doxyfile)
      configure_file(${PROJECT_SOURCE_DIR}/TOPP_tutorial/Doxyfile.in ${PROJECT_BINARY_DIR}/TOPP_tutorial/Doxyfile)

      # check doxygen for bug with generated latex files
      set(DOXYGEN_START_BUGGY "1.6.3")
      set(DOXYGEN_END_BUGGY "1.7.2")
      exec_program(${DOXYGEN_EXECUTABLE}
        ARGS "--version"
        OUTPUT_VARIABLE DOXYGEN_VERSION)

      if (DOXYGEN_VERSION STRGREATER DOXYGEN_START_BUGGY AND DOXYGEN_VERSION STRLESS DOXYGEN_END_BUGGY )
        message(ERROR "Warning, DoxygenBug ( 1.6.? < vers. installed < 1.7.3 ) disguises generated tex inputfiles and files will not be recognized")
      endif()

      #------------------------------------------------------------------------------
      # doc_tutorials target
      add_custom_target(doc_tutorials ALL
                        COMMAND ${CMAKE_COMMAND} -E echo ""
                        COMMAND ${CMAKE_COMMAND} -E echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                        COMMAND ${CMAKE_COMMAND} -E echo "Creating SmartPeak pdf tutorial"
                        COMMAND ${CMAKE_COMMAND} -E echo ""
                        COMMAND ${CMAKE_COMMAND} -E remove_directory SmartPeak_tutorial/latex_output
                        COMMAND ${CMAKE_COMMAND} -E chdir SmartPeak_tutorial/ "${DOXYGEN_EXECUTABLE}" Doxyfile
                        COMMAND ${CMAKE_COMMAND} -E copy SmartPeak_tutorial/refman_overwrite.tex SmartPeak_tutorial/latex_output/refman.tex
                        COMMAND ${CMAKE_COMMAND} -E chdir SmartPeak_tutorial/latex_output/  "${PDFLATEX_COMPILER}" refman.tex
                        COMMAND ${CMAKE_COMMAND} -E chdir SmartPeak_tutorial/latex_output/  "${MAKEINDEX_COMPILER}" refman.idx
                        COMMAND ${CMAKE_COMMAND} -E chdir SmartPeak_tutorial/latex_output/  "${PDFLATEX_COMPILER}" refman.tex
                        COMMAND ${CMAKE_COMMAND} -E chdir SmartPeak_tutorial/latex_output/  "${PDFLATEX_COMPILER}" refman.tex
                        COMMAND ${CMAKE_COMMAND} -E copy SmartPeak_tutorial/latex_output/refman.pdf SmartPeak_tutorial.pdf
                        COMMAND ${CMAKE_COMMAND} -E echo ""
                        COMMAND ${CMAKE_COMMAND} -E echo "The SmartPeak tutorial in PDF format has been successfully created:"
                        COMMAND ${CMAKE_COMMAND} -E echo "doc/SmartPeak_tutorial.pdf"
                        COMMAND ${CMAKE_COMMAND} -E echo ""
                        COMMAND ${CMAKE_COMMAND} -E echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                        COMMAND ${CMAKE_COMMAND} -E echo "Creating TOPP/TOPPView pdf tutorial"
                        COMMAND ${CMAKE_COMMAND} -E echo ""
                        COMMAND ${CMAKE_COMMAND} -E remove_directory TOPP_tutorial/latex_output
                        COMMAND ${CMAKE_COMMAND} -E chdir TOPP_tutorial/ "${DOXYGEN_EXECUTABLE}" Doxyfile
                        COMMAND ${CMAKE_COMMAND} -E copy TOPP_tutorial/refman_overwrite.tex TOPP_tutorial/latex_output/refman.tex
                        COMMAND ${CMAKE_COMMAND} -E chdir TOPP_tutorial/latex_output/  "${PDFLATEX_COMPILER}" refman.tex
                        COMMAND ${CMAKE_COMMAND} -E chdir TOPP_tutorial/latex_output/  "${MAKEINDEX_COMPILER}" refman.idx
                        COMMAND ${CMAKE_COMMAND} -E chdir TOPP_tutorial/latex_output/  "${PDFLATEX_COMPILER}" refman.tex
                        COMMAND ${CMAKE_COMMAND} -E chdir TOPP_tutorial/latex_output/  "${PDFLATEX_COMPILER}" refman.tex
                        COMMAND ${CMAKE_COMMAND} -E copy TOPP_tutorial/latex_output/refman.pdf TOPP_tutorial.pdf
                        COMMAND ${CMAKE_COMMAND} -E echo ""
                        COMMAND ${CMAKE_COMMAND} -E echo "The SmartPeak tutorial in PDF format has been successfully created:"
                        COMMAND ${CMAKE_COMMAND} -E echo "doc/TOPP_tutorial.pdf"
                        COMMAND ${CMAKE_COMMAND} -E echo ""
                        COMMENT "Build the SmartPeak/TOPP pdf tutorial"
                        VERBATIM)

      #------------------------------------------------------------------------------
      # add virtual dependency of tutorials on TOPP & UTILS
      # this is not necessary but defers the generation of the tutorials to a later
      # stage to avoid confusion if tutorials are build first
      add_dependencies(doc_tutorials TOPP UTILS)

      #------------------------------------------------------------------------------
      # install also the tutorial pdfs
      install_file(${PROJECT_BINARY_DIR}/TOPP_tutorial.pdf ${INSTALL_DOC_DIR} doc)
      install_file(${PROJECT_BINARY_DIR}/SmartPeak_tutorial.pdf ${INSTALL_DOC_DIR} doc)

      #------------------------------------------------------------------------------
      # inform the user
      message(STATUS "Enabled tutorials (-D ENABLE_TUTORIALS=On).")
    else()
      message(STATUS "Disabled tutorials (-D ENABLE_TUTORIALS=Off).")
    endif()
  else()
    set(DOC_TUTORIALS_ACTIVE FALSE)
    message(STATUS "PDFLaTeX missing. Disabling 'doc_tutorials' target!")
  endif()
else()
  message(STATUS "Doxygen not found. Disabling all documentation targets!")
  message(STATUS "Note that no documentation will be installed alongside SmartPeak.")
endif()

#------------------------------------------------------------------------------
# Examples / Tutorials
#------------------------------------------------------------------------------
option(BUILD_EXAMPLES "Compile SmartPeak code examples" ON)
if(BUILD_EXAMPLES AND "${PACKAGE_TYPE}" STREQUAL "none")
  add_subdirectory(code_examples)
endif(BUILD_EXAMPLES AND "${PACKAGE_TYPE}" STREQUAL "none")
