docker build -t dmccloskey/docker-openms-vscode .
docker run -it --name=cpp_openms_1 -v //C/Users/dmccloskey/Documents/GitHub/smartPeak_cpp/:/home/user/code/ --privileged --security-opt seccomp:unconfined dmccloskey/docker-openms-vscode /bin/bash
docker run -it --name=cpp_openms_1 -v //C/Users/dmccloskey/Documents/GitHub/smartPeak_cpp/src:/home/user/code/src `
    -v //C/Users/dmccloskey/Documents/GitHub/smartPeak_cpp/build:/home/user/code/build `
    -v //C/Users/dmccloskey/Documents/GitHub/smartPeak_cpp/cmake:/home/user/code/cmake `
    -v //C/Users/dmccloskey/Documents/GitHub/smartPeak_cpp/doc:/home/user/code/doc `
    -v //C/Users/dmccloskey/Documents/GitHub/smartPeak_cpp/CMakeLists.txt:/home/user/code/CMakeLists.txt `
    -v //C/Users/dmccloskey/Documents/GitHub/smartPeak_cpp/CTestConfig.cmake:/home/user/code/CTestConfig.cmake `
    --privileged --security-opt seccomp:unconfined dmccloskey/docker-openms-vscode /bin/bash
docker run -it --name=cpp_openms_1 -v //C/Users/domccl/GitHub/smartPeak_cpp/:/home/user/code/ --privileged --security-opt seccomp:unconfined dmccloskey/docker-openms-vscode /bin/bash
cd /home/user/code/build
cmake -DCMAKE_PREFIX_PATH="/usr/local/contrib-build/;/usr/local/contrib/;/usr/;/usr/local" -DBOOST_USE_STATIC=OFF -DHAS_XSERVER=Off ..
cmake -DCMAKE_PREFIX_PATH="/usr/local/dependencies-build/;/usr/local/smartPeak_dependencies/;/usr/;/usr/local" -DBOOST_USE_STATIC=OFF -DHAS_XSERVER=Off ..
cmake -DCMAKE_PREFIX_PATH="/usr/local/contrib-build/;/usr/local/contrib-build/include/;/usr/local/smartPeak_dependencies/;/usr/;/usr/local" -DBOOST_USE_STATIC=OFF -DHAS_XSERVER=Off ..