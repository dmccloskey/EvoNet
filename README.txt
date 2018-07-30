# build docker images
docker build -t dmccloskey/docker-openms-vscode .

# run docker container
docker run -it --name=cpp_openms_1 -v //C/Users/dmccloskey/Documents/GitHub/smartPeak_cpp/:/home/user/code/ --privileged --security-opt seccomp:unconfined dmccloskey/docker-openms-vscode /bin/bash
docker run -it --name=cpp_openms_1 -v //C/Users/domccl/GitHub/smartPeak_cpp/:/home/user/code/ --privileged --security-opt seccomp:unconfined dmccloskey/docker-openms-vscode /bin/bash

# TMP until finished testing cuda
docker run -it --name=cpp_openms_1 -v //C/Users/domccl/GitHub/smartPeak_cpp/:/home/user/code/ --privileged --security-opt seccomp:unconfined dmccloskey/docker-openms-vscode:mast /bin/bash

# MNST
docker run -it --name=cpp_openms_1 -v //C/Users/domccl/GitHub/smartPeak_cpp/:/home/user/code/ -v //C/Users/domccl/GitHub/mnist/:/home/user/data/ --privileged --security-opt seccomp:unconfined dmccloskey/docker-openms-vscode /bin/bash
docker run -it --name=cpp_openms_1 -v //C/Users/dmccloskey/Documents/GitHub/smartPeak_cpp/:/home/user/code/ -v //C/Users/dmccloskey/Documents/GitHub/mnist/:/home/user/data/ --privileged --security-opt seccomp:unconfined dmccloskey/docker-openms-vscode /bin/bash

# build code 
## Docker pre-requisites only
cd /home/user/code/build

##Linux
cmake -DCMAKE_PREFIX_PATH="/usr/local/contrib-build/;/usr/local/contrib/;/usr/;/usr/local" -DBOOST_USE_STATIC=OFF -DHAS_XSERVER=Off ..
cmake -DCMAKE_PREFIX_PATH="/usr/local/dependencies-build/;/usr/local/smartPeak_dependencies/;/usr/;/usr/local" -DBOOST_USE_STATIC=OFF -DHAS_XSERVER=Off ..
cmake -DEIGEN_USE_GPU=ON -DCUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda-9.2" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="/usr/local/contrib-build/;/usr/local/contrib-build/include/;/usr/local/smartPeak_dependencies/;/usr/;/usr/local" -DBOOST_USE_STATIC=OFF -DHAS_XSERVER=Off ..
cmake -DEIGEN_USE_GPU=OFF -DUSE_SUPERBUILD=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="/usr/local/contrib-build/;/usr/local/contrib-build/include/;/usr/local/smartPeak_dependencies/;/usr/;/usr/local" -DBOOST_USE_STATIC=OFF -DHAS_XSERVER=Off ..

### code prifiling with valgrind and gprof
LIBRARY_PATH=/usr/lib/pcc/x86_64-alpine-linux-musl/1.2.0.DEVEL/lib:$LIBRARY_PATH
cmake -DEIGEN_USE_GPU=OFF -DUSE_SUPERBUILD=OFF -DCMAKE_CXX_FLAGS_DEBUG=-pg -DCMAKE_EXE_LINKER_FLAGS_DEBUG=-pg -DCMAKE_SHARED_LINKER_FLAGS=-pg -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="/usr/local/contrib-build/;/usr/local/contrib-build/include/;/usr/local/smartPeak_dependencies/;/usr/;/usr/local" -DBOOST_USE_STATIC=OFF -DHAS_XSERVER=Off ..
ctest -DCTEST_MEMORYCHECK_COMMAND="/usr/bin/valgrind" -DMemoryCheckCommand="/usr/bin/valgrind" -T memcheck -R Model_DAG_test -V
./Model_DAG_test

##Windows (MSVC or Cygwin64)
###dmccloskey
cmake -DEIGEN_USE_GPU=OFF -DBOOST_USE_STATIC=OFF -G "Visual Studio 15 2017 Win64" -T host=x64 -DUSE_SUPERBUILD=OFF -DBOOST_ROOT=C:/local/boost_1_67_0 -DEIGEN3_INCLUDE_DIR=C:/Users/dmccloskey/Documents/GitHub/smartPeak_cpp/build_external/Dependencies/Source/eigen ..

###domccl
cmake -G "Visual Studio 15 2017 Win64" -T host=x64 -DUSE_SUPERBUILD=ON ..
cmake -G "Unix Makefiles" -DUSE_SUPERBUILD=ON ..
cmake -DEIGEN_USE_GPU=ON -DBOOST_USE_STATIC=OFF -G "Visual Studio 15 2017 Win64" -T host=x64 -DUSE_SUPERBUILD=OFF -DEIGEN3_INCLUDE_DIR=C:/Users/domccl/GitHub/smartPeak_cpp/build_external/Dependencies/Source/eigen ..
cmake -DEIGEN_USE_GPU=OFF -DBOOST_USE_STATIC=OFF -G "Unix Makefiles" -DBOOST_ROOT=C:/Users/domccl/GitHub/smartPeak_cpp/build2/Dependencies/Source/boost -DUSE_SUPERBUILD=OFF -DEIGEN3_INCLUDE_DIR=C:/Users/domccl/GitHub/smartPeak_cpp/build2/Dependencies/Source/eigen ..

