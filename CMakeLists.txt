cmake_minimum_required(VERSION 3.02 FATAL_ERROR)

project(HubbardIPT)

# Setting c++ custom preprocessing flags
set(CMPLR "clang" CACHE STRING "Determines the compiler.") # Use for instance cmake -DCMPLR:STRING=gcc ..
message(STATUS "Compiler chosen is: " ${CMPLR})
set(DIM "oneD" CACHE STRING "Determines the dimensions for calculations.") # Use for instance cmake -DDIM:STRING=twoD ..
message(STATUS "Dimension: " ${DIM})
set(PRLL OFF CACHE STRING "Determines whether the program is parallelized or not.") # Use for instance cmake -DDIM:BOOL=ON ..
message(STATUS "Parallelization: " ${PRLL})

if (${CMPLR} STREQUAL "clang")
        if (${DIM} STREQUAL "oneD")
                if (${PRLL})
                        set(CMAKE_CXX_FLAGS "-Wall -std=c++14 -DDIM=1 -DPARALLEL")
                else()
                        set(CMAKE_CXX_FLAGS "-Wall -std=c++14 -DDIM=1")
                endif()
        elseif (${DIM} STREQUAL "twoD")
                if (${PRLL})
                        set(CMAKE_CXX_FLAGS "-Wall -std=c++14 -DDIM=2 -DPARALLEL")
                else()
                        set(CMAKE_CXX_FLAGS "-Wall -std=c++14 -DDIM=2")
                endif()
        endif()
elseif (${CMPLR} STREQUAL "gcc")
        if (${DIM} STREQUAL "oneD")
                if (${PRLL})
                        set(CMAKE_CXX_FLAGS "-Wall -std=c++14 -fpermissive -DDIM=1 -DPARALLEL")
                else()
                        set(CMAKE_CXX_FLAGS "-Wall -std=c++14 -fpermissive -DDIM=1")
                endif()
        elseif (${DIM} STREQUAL "twoD")
                if (${PRLL})
                        set(CMAKE_CXX_FLAGS "-Wall -std=c++14 -fpermissive -DDIM=2 -DPARALLEL")
                else()
                        set(CMAKE_CXX_FLAGS "-Wall -std=c++14 -fpermissive -DDIM=2")
                endif()
        endif()
endif()
set(CMAKE_SOURCE_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/cmake_source")
message("CMAKE_SOURCE_PREFIX for GMock setup: " ${CMAKE_SOURCE_PREFIX})

if (${CMPLR} STREQUAL "clang")
        set(LIBRARIES_CUSTOM fftw3 armadillo json_spirit)
elseif (${CMPLR} STREQUAL "gcc")
        set(LIBRARIES_CUSTOM fftw3 armadillo json_spirit pthread)
endif()

set(SOURCES_COMMON
        src/IPT2nd3rdorderSingle2
        src/integral_utils
        src/green_utils
        src/file_utils
        src/json_utils
        src/thread_utils
        src/susceptibilities)

set(SOURCES_MAIN mainIPT ${SOURCES_COMMON})
set(SOURCES_TEST test/src/test_utils)

# Locate GSL
set(GSL_ROOT_DIR $ENV{HOME}/gsl) # This is required before calling in find_package...
find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})
message(STATUS "gsl found: " ${GSL_FOUND})

#------Executables
add_executable(MAIN ${SOURCES_MAIN})
target_link_libraries(MAIN ${GSL_LIBRARIES} ${LIBRARIES_CUSTOM})

# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
message(STATUS "gtest found: " ${GTEST_FOUND})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_PREFIX})

#Locate gmock
find_package(GMock REQUIRED)
include_directories(${GMOCK_INCLUDE_DIRS})
message(STATUS "gmock found: " ${GMOCK_FOUND})

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
endif()

# # Now simply link against gtest or gtest_main as needed. Eg
enable_testing()
add_executable(IPT_test test/test.cpp ${SOURCES_COMMON} ${SOURCES_TEST})
target_link_libraries(IPT_test ${GSL_LIBRARIES} ${GTEST_LIBRARIES} ${GMOCK_LIBRARIES} ${LIBRARIES_CUSTOM} gtest gmock)
add_test(NAME IPT_test COMMAND ./IPT_test) ## Allows one to enter "make test" and launch the tests!
