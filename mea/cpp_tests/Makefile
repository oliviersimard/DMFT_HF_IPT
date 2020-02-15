SRC=$(PWD)/../../src
OBJ=$(PWD)/obj

CXX=g++ -std=c++11 -Wall -g
INC=${HOME}/gsl/include
LIB=${HOME}/gsl/lib

CXXFLAGS=-L$(LIB) -lm -lgsl -lgslcblas -larmadillo -lfftw3 #-lgtest -lgmock -lpthread

PROG=CPP-TESTS

all: $(PROG)

$(PROG): $(OBJ)/green_utils.o $(OBJ)/cubic_spline.o
	$(CXX) $(OBJ)/green_utils.o $(OBJ)/cubic_spline.o $(CXXFLAGS) -o spl.out

$(OBJ)/green_utils.o: $(SRC)/green_utils.cpp
	$(CXX) -c -I$(INC) $(SRC)/green_utils.cpp -o $(OBJ)/green_utils.o

$(OBJ)/cubic_spline.o: $(PWD)/spline_cpp_tests.cpp
	$(CXX) -c -I$(INC) $(PWD)/spline_cpp_tests.cpp -o $(OBJ)/cubic_spline.o

clean:
	rm -f $(OBJ)/* $(PWD)/spl.out
