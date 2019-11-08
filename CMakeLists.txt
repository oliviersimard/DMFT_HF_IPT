cmake_minimum_required(VERSION 3.02 FATAL_ERROR)

project(HubbardIPT)

# important for the use of tuples
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_SOURCE_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/cmake_source")
message("CMAKE_SOURCE_PREFIX for GMock setup: " ${CMAKE_SOURCE_PREFIX})

set(LIBRARIES_CUSTOM fftw3 armadillo json_spirit)

set(SOURCES_COMMON
        src/IPT2nd3rdorderSingle2
        src/integral_utils
        src/green_utils
        src/file_utils
        src/json_utils)

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
