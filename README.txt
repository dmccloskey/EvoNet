# build docker images
docker build -t dmccloskey/docker-openms-vscode .

# run docker container
docker run -it --name=cpp_openms_1 -v //C/Users/dmccloskey/Documents/GitHub/smartPeak_cpp/:/home/user/code/ --privileged --security-opt seccomp:unconfined dmccloskey/docker-openms-vscode /bin/bash
docker run -it --name=cpp_openms_1 -v //C/Users/domccl/GitHub/smartPeak_cpp/:/home/user/code/ --privileged --security-opt seccomp:unconfined dmccloskey/docker-openms-vscode /bin/bash

# MNST
docker run -it --name=cpp_openms_1 -v //C/Users/domccl/GitHub/smartPeak_cpp/:/home/user/code/ -v //C/Users/domccl/GitHub/mnist/:/home/user/data/ --privileged --security-opt seccomp:unconfined dmccloskey/docker-openms-vscode /bin/bash

# build code
cd /home/user/code/build
cmake -DCMAKE_PREFIX_PATH="/usr/local/contrib-build/;/usr/local/contrib/;/usr/;/usr/local" -DBOOST_USE_STATIC=OFF -DHAS_XSERVER=Off ..
cmake -DCMAKE_PREFIX_PATH="/usr/local/dependencies-build/;/usr/local/smartPeak_dependencies/;/usr/;/usr/local" -DBOOST_USE_STATIC=OFF -DHAS_XSERVER=Off ..
cmake -DEIGEN_USE_GPU=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="/usr/local/contrib-build/;/usr/local/contrib-build/include/;/usr/local/smartPeak_dependencies/;/usr/;/usr/local" -DBOOST_USE_STATIC=OFF -DHAS_XSERVER=Off ..
ctest -DCTEST_MEMORYCHECK_COMMAND="/usr/bin/valgrind" -DMemoryCheckCommand="/usr/bin/valgrind" -T memcheck -R Model_DAG_test -V