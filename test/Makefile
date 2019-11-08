SRC=$(PWD)/src
OBJ=$(PWD)/obj
UPPEROBJ=$(PWD)/../obj

CXX=g++ -std=c++14 -Wall -g
INC=${HOME}/gsl/include
LIB=${HOME}/gsl/lib

CXXFLAGS=-L$(LIB) -lgtest -lgmock -lfftw3 -larmadillo -lm -lgsl -lgslcblas

PROG=IPT_test

all: $(PROG)

$(PROG): $(OBJ)/test.o $(OBJ)/test_utils.o $(UPPEROBJ)/IPT2nd3rdorderSingle2.o $(UPPEROBJ)/integral_utils.o
	$(CXX) $(OBJ)/test.o $(OBJ)/test_utils.o $(UPPEROBJ)/IPT2nd3rdorderSingle2.o $(UPPEROBJ)/integral_utils.o $(CXXFLAGS) -o TEST.out

$(OBJ)/test.o: $(PWD)/test.cpp
	$(CXX) -c -I$(INC) $(PWD)/test.cpp -o $(OBJ)/test.o

$(OBJ)/test_utils.o: $(SRC)/test_utils.cpp
	$(CXX) -c -I$(INC) $(SRC)/test_utils.cpp -o $(OBJ)/test_utils.o

clean:
	rm -f $(OBJ)/* $(PWD)/TEST.out


