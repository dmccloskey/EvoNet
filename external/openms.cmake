include (ExternalProject)

# set(INCLUDE_DIR_OPENMS ${DEPENDENCIES_BIN_INCLUDE_DIR}/OpenMS)

ExternalProject_Add(openms
    GIT_REPOSITORY https://github.com/OpenMS/OpenMS
    GIT_TAG develop
    CMAKE_ARGS 
        -DCMAKE_INSTALL_PREFIX=${INCLUDE_DIR_OPENMS}
        -DPYOPENMS=OFF
        -DCMAKE_PREFIX_PATH="/usr/local/contrib-build/;/usr/local/contrib/;/usr/;/usr/local"
        -DBOOST_USE_STATIC=OFF
        -DHAS_XSERVER=OFF
        -DWITH_GUI=OFF
)
