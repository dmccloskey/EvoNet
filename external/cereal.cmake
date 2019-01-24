include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

set_property (DIRECTORY PROPERTY EP_BASE Dependencies)

set (DEPENDENCIES)

set (CEREAL_VERSION master)
message (STATUS "Adding Eigen ${CEREAL_VERSION} as an external project.")

ExternalProject_Add(cereal
  GIT_REPOSITORY "https://github.com/USCiLab/cereal.git"
  #GIT_TAG ${CEREAL_VERSION} # Need the dev branch to compile use MSVC
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_IN_SOURCE 1
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  #INSTALL_COMMAND ${CEREAL_INSTALL_CMD}
  #INSTALL_DIR include
)