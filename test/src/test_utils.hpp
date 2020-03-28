#ifndef test_H_
#define test_H_

// #include<string>
// #include<iostream>
#include<utility> // pair
// #include<vector>
#include<ctime>
#include<cstdlib> // srand and rand
#include "../../src/IPT2nd3rdorderSingle2.hpp"
#include "../../src/integral_utils.hpp"

extern std::vector<float> vecFloat;

std::string hello_world();
void say_hello();

template<typename T>
std::vector< T > generate_random_numbers(size_t arr_size, T min, T max);

// Struct to redirect stdout/stderr to string. rdbuf associates a new buffer with the current one.
// If functions return on the std output, it is captured to be compared. Relevant espacially for functions returning void or std::string vals.
struct cout_redirect {
    cout_redirect( std::stringbuf* new_buffer ) : old( std::cout.rdbuf( new_buffer ) ){}

    ~cout_redirect() {
        std::cout.rdbuf( old );
    }

private:
    std::streambuf* old;
};

struct cerr_redirect {
    cerr_redirect( std::stringbuf* new_buffer ) : old( std::cerr.rdbuf( new_buffer ) ){}

    ~cerr_redirect() {
        std::cerr.rdbuf( old );
    }

private:
    std::streambuf* old;
};

template<typename T>
std::vector< T > generate_random_numbers(size_t arr_size, T min, T max){
    srand(time(0));
    std::vector< T > rand_num_container(arr_size);
    T random_number;
    for (size_t i=0; i<arr_size; i++){
        random_number = min + (T)( ( (T)rand() ) / ( (T)RAND_MAX ) * (max - min) );
        rand_num_container[i] = random_number;
    }
    return rand_num_container;
}

#endif /* end of test_H_ */
