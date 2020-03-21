#ifndef File_Utils_H_
#define File_Utils_H_

#define VERBOSE 0

#include <sys/stat.h> // for posix stat()
//#include<regex> // for regular expressions
extern "C" {
#include <glob.h> // glob() and globfree()
}
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <cstring> // For memset()
#include <stdexcept>
#include <regex>
#include <complex>
#include <armadillo>
#include "H5Cpp.h"

typedef struct cplx_t{ // Custom data holder for the HDF5 handling
    double re;
    double im;
} cplx_t;

struct FileData{
    std::vector<double> iwn;
    std::vector<double> re;
    std::vector<double> im;
};

std::vector<std::string> glob(const std::string& pattern) noexcept(false);
void mkdirTree(std::string sub, std::string dir) noexcept(false);
void check_file_content(const std::vector< std::string >& filenamesToSave, std::string pathToDir1, std::string pathToDir2) noexcept(false);
int extractIntegerLastWords(std::string str);
FileData get_data(const std::string& strName, const unsigned int& Ntau) noexcept(false);
std::vector<std::string> get_info_from_filename(const std::string& strName,const std::vector<std::string>& fetches,const char* separation_char="_") noexcept(false);
void writeInHDF5File(std::vector< std::complex<double> >& GG_iqn_q, H5::H5File* file, const unsigned int& DATA_SET_DIM, const std::string& DATASET_NAME) noexcept(false);
arma::Mat< std::complex<double> > readFromHDF5File(H5::H5File* file, const std::string& DATASET_NAME) noexcept(false);
void save_matrix_in_HDF5(const arma::Mat< std::complex<double> >& mat_to_save, double k_bar, double k_tilde, H5::H5File* file) noexcept(false);
void save_matrix_in_HDF5(std::complex<double>* mat_to_save, double k_bar, double k_tilde, H5::H5File* file, size_t NX, size_t NY) noexcept(false);

inline bool file_exists(const std::string& name){
  struct stat buffer;   
  return (stat(name.c_str(), &buffer) == 0); // string null terminates
}

inline std::string eraseAllSubStr(std::string mainStr, const std::string& toErase){
	size_t pos = std::string::npos;
	// Search for the substring in string in a loop until nothing is found
	while( (pos = mainStr.find(toErase) ) != std::string::npos){
		// If found then erase it from string
		mainStr.erase(pos, toErase.length());
	}
    return mainStr;
}

#endif /* end of File_Utils_ */