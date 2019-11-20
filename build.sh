#!/bin/bash

dir="${PWD}/obj"   #This directory will contain the objects of the different translation units
dirTest="${PWD}/test/obj" #This directory will also contain the object files, but the ones of the test.cpp main translation unit.
dirData="${PWD}/data" #This directory will store the different results produced by the program. As for the obj directories, this directory should remain in place after creation.

read -p "clang or gcc compiler?: " OPT

echo $OPT

# Creating the important directories...
if [ ! -d $dir ]; then
        mkdir ${dir}
else
        echo -e "Directory $dir already exists.\n"
fi
if [ ! -d $dirTest ]; then
        mkdir ${dirTest}
else
        echo -e "Directory $dirTest already exists.\n"
fi
if [ ! -d $dirData ]; then
        mkdir ${dirData}
else
        echo -e "Directory $dirData already exists.\n"
fi


# Which compiler's instructions to launch...
if [ ${OPT} == "clang" ]; then # Makefile are mainly for debugging with VSCode.
        echo "clang"
        if [ ! -d "./build" ]; then
                mkdir build && cd build && cmake .. && make -j 2
        else
                rm -rf build && mkdir build && cd build && cmake .. && make -j 2
        fi
elif [ ${OPT} == "gcc" ]; then
        echo "gcc"
        if [ ! -d "./build" ]; then
                mkdir build && cd build && cmake -DCMPLR:STRING=${OPT} .. && make -j 2
        else
                rm -rf build && cd build && cmake -DCMPLR:STRING=${OPT} .. && make -j 2
        fi
        #make --file=Makefile_GCC -j 2
else
        echo -e "You must choose between clang or gcc compilers. Not tested against other compilers."
fi
