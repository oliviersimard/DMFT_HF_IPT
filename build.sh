#!/bin/bash

buildDir="build2"
dir="${PWD}/obj"   #This directory will contain the objects of the different translation units
dirTest="${PWD}/test/obj" #This directory will also contain the object files, but the ones of the test.cpp main translation unit.
dirData="${PWD}/data" #This directory will store the different results produced by the program. As for the obj directories, this directory should remain in place after creation.

read -p "clang or gcc compiler?: " OPT
read -p "dimension: 1, 2 or 3?: " DIM
read -p "parallelized calculations? (y or n): " PRLL

# Creating the important directories...
# if [ ! -d $dir ]; then
#         mkdir ${dir}
# else
#         echo -e "Directory $dir already exists."
# fi
# if [ ! -d $dirTest ]; then
#         mkdir ${dirTest}
# else
#         echo -e "Directory $dirTest already exists."
# fi
# if [ ! -d $dirData ]; then
#         mkdir ${dirData}
# else
#         echo -e "Directory $dirData already exists."
# fi


# Which compiler's instructions to launch...
if [[ ${OPT} == "clang" ]]; then # Makefile are mainly for debugging with VSCode.
        if [[ "$DIM" -eq "1" ]]; then
                if [[ ${PRLL} == "y" ]]; then
                        if [ ! -d "./$buildDir" ]; then
                                mkdir ${buildDir} && cd ${buildDir} && cmake -DDIM:STRING="oneD" -DPRLL:BOOL=ON .. && make -j 2
                        else
                                rm -rf ${buildDir} && mkdir ${buildDir} && cd build && cmake -DDIM:STRING="oneD" -DPRLL:BOOL=ON .. && make -j 2
                        fi
                elif [[ ${PRLL} == "n" ]]; then
                        if [ ! -d "./$buildDir" ]; then
                                mkdir ${buildDir} && cd ${buildDir} && cmake -DDIM:STRING="oneD" .. && make -j 2
                        else
                                rm -rf ${buildDir} && mkdir ${buildDir} && cd build && cmake -DDIM:STRING="oneD" .. && make -j 2
                        fi
                else
                        echo -e "ERROR: You must choose between yes (y) or no (n).\n"
                        exit 1    
                fi
        elif [[ "$DIM" -eq "2" ]]; then
                if [[ ${PRLL} == "y" ]]; then
                        if [ ! -d "./$buildDir" ]; then
                                mkdir ${buildDir} && cd ${buildDir} && cmake -DDIM:STRING="twoD" -DPRLL:BOOL=ON .. && make -j 2
                        else
                                rm -rf ${buildDir} && mkdir ${buildDir} && cd build && cmake -DDIM:STRING="twoD" -DPRLL:BOOL=ON .. && make -j 2
                        fi
                elif [[ ${PRLL} == "n" ]]; then
                        if [ ! -d "./$buildDir" ]; then
                                mkdir ${buildDir} && cd ${buildDir} && cmake -DDIM:STRING="twoD" .. && make -j 2
                        else
                                rm -rf ${buildDir} && mkdir ${buildDir} && cd build && cmake -DDIM:STRING="twoD" .. && make -j 2
                        fi
                else
                        echo -e "ERROR: You must choose between yes (y) or no (n).\n"
                        exit 1   
                fi
        elif [[ "$DIM" -eq "3" ]]; then
                if [[ ${PRLL} == "y" ]]; then
                        if [ ! -d "./$buildDir" ]; then
                                mkdir ${buildDir} && cd ${buildDir} && cmake -DDIM:STRING="threeD" -DPRLL:BOOL=ON .. && make -j 2
                        else
                                rm -rf ${buildDir} && mkdir ${buildDir} && cd build && cmake -DDIM:STRING="threeD" -DPRLL:BOOL=ON .. && make -j 2
                        fi
                elif [[ ${PRLL} == "n" ]]; then
                        if [ ! -d "./$buildDir" ]; then
                                mkdir ${buildDir} && cd ${buildDir} && cmake -DDIM:STRING="threeD" .. && make -j 2
                        else
                                rm -rf ${buildDir} && mkdir ${buildDir} && cd build && cmake -DDIM:STRING="threeD" .. && make -j 2
                        fi
                else
                        echo -e "ERROR: You must choose between yes (y) or no (n).\n"
                        exit 1   
                fi
        else
                echo -e "ERROR: You must choose between 1D (1), 2D (2) or 3D (3).\n"
                exit 1   
        fi

elif [[ ${OPT} == "gcc" ]]; then
        if [[ "$DIM" -eq "1" ]]; then
                if [[ ${PRLL} == "y" ]]; then
                        if [ ! -d "./$buildDir" ]; then
                                mkdir ${buildDir} && cd ${buildDir} && cmake -DCMPLR:STRING=${OPT} -DDIM:STRING="oneD" -DPRLL:BOOL=ON .. && make -j 2
                        else
                                rm -rf ${buildDir} && cd ${buildDir} && cmake -DCMPLR:STRING=${OPT} -DDIM:STRING="oneD" -DPRLL:BOOL=ON .. && make -j 2
                        fi
                elif [[ ${PRLL} == "n" ]]; then
                        if [ ! -d "./$buildDir" ]; then
                                mkdir ${buildDir} && cd ${buildDir} && cmake -DCMPLR:STRING=${OPT} -DDIM:STRING="oneD" .. && make -j 2
                        else
                                rm -rf ${buildDir} && cd ${buildDir} && cmake -DCMPLR:STRING=${OPT} -DDIM:STRING="oneD" .. && make -j 2
                        fi
                else
                        echo -e "ERROR: You must choose between yes (y) or no (n).\n"
                        exit 1 
                fi
        elif [[ "$DIM" -eq "2" ]]; then
                if [[ ${PRLL} == "y" ]]; then
                        if [ ! -d "./$buildDir" ]; then
                                mkdir ${buildDir} && cd ${buildDir} && cmake -DCMPLR:STRING=${OPT} -DDIM:STRING="twoD" -DPRLL:BOOL=ON .. && make -j 2
                        else
                                rm -rf ${buildDir} && cd ${buildDir} && cmake -DCMPLR:STRING=${OPT} -DDIM:STRING="twoD" -DPRLL:BOOL=ON .. && make -j 2
                        fi
                elif [[ ${PRLL} == "n" ]]; then
                        if [ ! -d "./$buildDir" ]; then
                                mkdir ${buildDir} && cd ${buildDir} && cmake -DCMPLR:STRING=${OPT} -DDIM:STRING="twoD" .. && make -j 2
                        else
                                rm -rf ${buildDir} && cd ${buildDir} && cmake -DCMPLR:STRING=${OPT} -DDIM:STRING="twoD" .. && make -j 2
                        fi
                else
                        echo -e "ERROR: You must choose between yes (y) or no (n).\n"
                        exit 1
                fi
        elif [[ "$DIM" -eq "3" ]]; then
                if [[ ${PRLL} == "y" ]]; then
                        if [ ! -d "./$buildDir" ]; then
                                mkdir ${buildDir} && cd ${buildDir} && cmake -DCMPLR:STRING=${OPT} -DDIM:STRING="threeD" -DPRLL:BOOL=ON .. && make -j 2
                        else
                                rm -rf ${buildDir} && cd ${buildDir} && cmake -DCMPLR:STRING=${OPT} -DDIM:STRING="threeD" -DPRLL:BOOL=ON .. && make -j 2
                        fi
                elif [[ ${PRLL} == "n" ]]; then
                        if [ ! -d "./$buildDir" ]; then
                                mkdir ${buildDir} && cd ${buildDir} && cmake -DCMPLR:STRING=${OPT} -DDIM:STRING="threeD" .. && make -j 2
                        else
                                rm -rf ${buildDir} && cd ${buildDir} && cmake -DCMPLR:STRING=${OPT} -DDIM:STRING="threeD" .. && make -j 2
                        fi
                else
                        echo -e "ERROR: You must choose between yes (y) or no (n).\n"
                        exit 1
                fi
        else
                echo -e "ERROR: You must choose between 1D (1), 2D (2) or 3D (3).\n"
                exit 1
        fi
        #make --file=Makefile_GCC -j 2
else
        echo -e "You must choose between clang or gcc compilers. Not tested against other compilers."
        exit 1
fi
