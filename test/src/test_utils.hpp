#ifndef test_H_
#define test_H_

#include<string>
#include<iostream>
#include<utility> // pair
#include<vector>
#include<ctime>
#include "../../src/IPT2nd3rdorderSingle2.hpp"
#include "../../src/integral_utils.hpp"

extern std::vector<float> vecFloat;

std::string hello_world();

void say_hello();

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

#endif /* end of test_H_ */
