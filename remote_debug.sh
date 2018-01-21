#!/usr/bin/env bash
ssh rootr@systemsbiochemistry.ucsd.edu:3000 "
cd  C:\\Users\\domccl\\GitHub\\smartPeak\\cpp  > /dev/null
g++ -g main.cpp > /dev/null
gdb --interpreter=mi a.out "