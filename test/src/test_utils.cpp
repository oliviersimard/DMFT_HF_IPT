#include "test_utils.hpp"

std::vector<double> vecK; // Need to declare the external variables here to the test file to link properly
std::vector< std::complex<double> > iwnArr_l;
std::vector< std::complex<double> > iqnArr_l;

std::vector<float> vecFloat={ 1., 2., 3., 4. };

std::string hello_world(){
    return "Hello World!";
}

void say_hello() {
    std::cout << "Hello, World!";
}


