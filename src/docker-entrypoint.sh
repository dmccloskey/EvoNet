#!/bin/bash
cd /home/user/code
##debug external code
g++ -L/usr/local/openms-build/lib -I/home/user/code/OpenMS/include -I/usr/local/OpenMS/src/openms/include -I/usr/local/openms-build/src/openms/include -I/usr/local/contrib-build/include -I/usr/local/contrib-build/include/WildMagic -I/usr/include/qt5 -fPIC -g main.cpp /home/user/code/OpenMS/source/TransformationModel.cpp /home/user/code/OpenMS/source/TransformationModelLinear.cpp -lOpenMS
##debug tests
# g++ -L/usr/local/openms-build/lib -I/home/user/code/OpenMS/include -I/usr/local/OpenMS/src/openms/include -I/usr/local/openms-build/src/openms/include -I/usr/local/contrib-build/include -I/usr/include/qt5 -fPIC -g /home/user/code/OpenMS/source/TransformationModel.cpp /home/user/code/OpenMS/tests/TransformationModel_test.cpp -lOpenMS
g++ -L/usr/local/openms-build/lib -I/home/user/code/OpenMS/include -I/usr/local/OpenMS/src/openms/include -I/usr/local/openms-build/src/openms/include -I/usr/local/contrib-build/include -I/usr/local/contrib-build/include/WildMagic -I/usr/include/qt5 -fPIC -g /home/user/code/OpenMS/source/TransformationModel.cpp /home/user/code/OpenMS/source/TransformationModelLinear.cpp /home/user/code/OpenMS/tests/TransformationModelLinear_test.cpp -lOpenMS
##debug setup
# g++ -g main.cpp

#manually keep the container running
sleep infinity

##Instructions
#cd ...GitHub/smartPeak/cpp
#docker-compose up
#docker exec -it cpp_openms_1 /bin/bash
#$cd code
#$"debug external code/debut tests"
#$...
#$exit
#docker-compose down