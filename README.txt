# Build from source
## Linux (Outdated...)
cmake -DCMAKE_PREFIX_PATH="/usr/local/contrib-build/;/usr/local/contrib/;/usr/;/usr/local" -DBOOST_USE_STATIC=OFF -DHAS_XSERVER=Off ..
cmake -DCMAKE_PREFIX_PATH="/usr/local/dependencies-build/;/usr/local/smartPeak_dependencies/;/usr/;/usr/local" -DBOOST_USE_STATIC=OFF -DHAS_XSERVER=Off ..
cmake -DEVONET_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda-9.2" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="/usr/local/contrib-build/;/usr/local/contrib-build/include/;/usr/local/smartPeak_dependencies/;/usr/;/usr/local" -DBOOST_USE_STATIC=OFF -DHAS_XSERVER=Off ..
cmake -DEVONET_CUDA=OFF -DUSE_SUPERBUILD=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="/usr/local/contrib-build/;/usr/local/contrib-build/include/;/usr/local/smartPeak_dependencies/;/usr/;/usr/local" -DEIGEN3_INCLUDE_DIR=/home/user/code/build2/Dependencies/Source/eigen -DCEREAL_ROOT=/home/user/code/build2/Dependencies/Source/cereal -DBOOST_USE_STATIC=OFF -DHAS_XSERVER=Off ..
cmake -DEVONET_CUDA=OFF -DUSE_SUPERBUILD=ON -DHAS_XSERVER=Off ..

## Windows (MSVC or Cygwin64)
cmake -DEVONET_CUDA=ON -DEVONET_CUDA_ARCHITECTURES=50-real -DBOOST_USE_STATIC=OFF -DBoost_NO_SYSTEM_PATHS=ON -DBOOST_INCLUDEDIR="C:/local/boost_1_72_0/boost" -DBOOST_ROOT="C:/local/boost_1_72_0" -DBOOST_LIBRARYDIR="C:/local/boost_1_72_0/lib64-msvc-14.2" -G "Visual Studio 16 2019" -T host=x64 -DUSE_SUPERBUILD=OFF -DEIGEN3_INCLUDE_DIR=C:/Users/domccl/GitHub/EvoNet/build_external/Dependencies/Source/eigen -DCEREAL_ROOT=C:/Users/domccl/GitHub/EvoNet/build_external/Dependencies/Source/cereal ..

# SuperBuild
## Windows
cmake -G "Visual Studio 16 2019" -T host=x64 -DUSE_SUPERBUILD=ON ..

## Linux
cmake -G "Unix Makefiles" -DUSE_SUPERBUILD=ON ..

# Other
## code prifiling with valgrind and gprof
LIBRARY_PATH=/usr/lib/pcc/x86_64-alpine-linux-musl/1.2.0.DEVEL/lib:$LIBRARY_PATH
cmake -DEVONET_CUDA=OFF -DUSE_SUPERBUILD=OFF -DCMAKE_CXX_FLAGS_DEBUG=-pg -DCMAKE_EXE_LINKER_FLAGS_DEBUG=-pg -DCMAKE_SHARED_LINKER_FLAGS=-pg -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="/usr/local/contrib-build/;/usr/local/contrib-build/include/;/usr/local/smartPeak_dependencies/;/usr/;/usr/local" -DBOOST_USE_STATIC=OFF -DHAS_XSERVER=Off ..
ctest -DCTEST_MEMORYCHECK_COMMAND="/usr/bin/valgrind" -DMemoryCheckCommand="/usr/bin/valgrind" -T memcheck -R Model_DAG_test -V
./Model_DAG_test

