SRC=$(PWD)/../src
OBJ=$(PWD)/obj

CXX=g++ -std=c++11 -Wall -g
INC=${HOME}/gsl/include
LIB=${HOME}/gsl/lib

CXXFLAGS=-L$(LIB) -lm -lgsl -lgslcblas -larmadillo #-lgtest -lgmock -lpthread

PROG=IPT-DMFT-OLI

all: $(PROG)

$(PROG): $(OBJ)/read.o $(OBJ)/green_utils.o $(OBJ)/thread_utils.o
	$(CXX) $(OBJ)/read.o $(OBJ)/green_utils.o $(OBJ)/thread_utils.o $(CXXFLAGS) -o read.out

$(OBJ)/read.o: $(PWD)/readFile2.cpp
	$(CXX) -c -I$(INC) $(PWD)/readFile2.cpp -o $(OBJ)/read.o

$(OBJ)/green_utils.o: $(SRC)/green_utils.cpp
	$(CXX) -c -I$(INC) $(SRC)/green_utils.cpp -o $(OBJ)/green_utils.o

$(OBJ)/thread_utils.o: $(SRC)/thread_utils.cpp
	$(CXX) -c -I$(INC) $(SRC)/thread_utils.cpp -o $(OBJ)/thread_utils.o

clean:
	rm -f $(OBJ)/* $(PWD)/read.out
