SRC=$(PWD)/src
OBJ=$(PWD)/obj

CXX=g++ -std=c++11 -Wall -g -DDEBUG
INC=${HOME}/gsl/include
LIB=${HOME}/gsl/lib

CXXFLAGS=-L$(LIB) -lm -lgsl -lgslcblas -larmadillo -lfftw3 -ljson_spirit #-lgtest -lgmock -lpthread

PROG=IPT-DMFT-OLI

all: $(PROG)

$(PROG): $(OBJ)/mainIPT.o $(OBJ)/green_utils.o $(OBJ)/integral_utils.o $(OBJ)/IPT2nd3rdorderSingle2.o $(OBJ)/json_utils.o $(OBJ)/file_utils.o $(OBJ)/thread_utils.o $(OBJ)/susceptibilities.o
	$(CXX) $(OBJ)/mainIPT.o $(OBJ)/green_utils.o $(OBJ)/integral_utils.o $(OBJ)/IPT2nd3rdorderSingle2.o $(OBJ)/json_utils.o $(OBJ)/file_utils.o $(OBJ)/thread_utils.o $(OBJ)/susceptibilities.o $(CXXFLAGS) -o mainIPT.out

$(OBJ)/mainIPT.o: $(PWD)/mainIPT.cpp
	$(CXX) -c -I$(INC) $(PWD)/mainIPT.cpp -o $(OBJ)/mainIPT.o

$(OBJ)/green_utils.o: $(SRC)/green_utils.cpp
	$(CXX) -c -I$(INC) $(SRC)/green_utils.cpp -o $(OBJ)/green_utils.o

$(OBJ)/integral_utils.o: $(SRC)/integral_utils.cpp
	$(CXX) -c -I$(INC) $(SRC)/integral_utils.cpp -o $(OBJ)/integral_utils.o

$(OBJ)/IPT2nd3rdorderSingle2.o: $(SRC)/IPT2nd3rdorderSingle2.cpp
	$(CXX) -c -I$(INC) $(SRC)/IPT2nd3rdorderSingle2.cpp -o $(OBJ)/IPT2nd3rdorderSingle2.o

$(OBJ)/json_utils.o: $(SRC)/json_utils.cpp
	$(CXX) -c -I$(INC) $(SRC)/json_utils.cpp -o $(OBJ)/json_utils.o

$(OBJ)/file_utils.o: $(SRC)/file_utils.cpp
	$(CXX) -c -I$(INC) $(SRC)/file_utils.cpp -o $(OBJ)/file_utils.o

$(OBJ)/thread_utils.o: $(SRC)/thread_utils.cpp
	$(CXX) -c -I$(INC) $(SRC)/thread_utils.cpp -o $(OBJ)/thread_utils.o

$(OBJ)/susceptibilities.o: $(SRC)/susceptibilities.cpp
	$(CXX) -c -I$(INC) $(SRC)/susceptibilities.cpp -o $(OBJ)/susceptibilities.o

clean:
	rm -f $(OBJ)/* $(PWD)/mainIPT.out && cd test && make clean

.PHONY: test # To avoid conflict with test directory

test:
	cd test; make all; sleep 1; ./TEST.out
