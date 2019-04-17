include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

set_property (DIRECTORY PROPERTY EP_BASE Dependencies)

set (DEPENDENCIES)

set (CUB_VERSION 1.8.0)
message (STATUS "Adding Cub ${CUB_VERSION} as an external project.")

ExternalProject_Add(cub
  GIT_REPOSITORY "https://github.com/NVlabs/cub.git"
  GIT_TAG ${CUB_VERSION}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_IN_SOURCE 1
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  #INSTALL_COMMAND ${CUB_INSTALL_CMD}
  #INSTALL_DIR include
)