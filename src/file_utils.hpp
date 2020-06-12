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
    typedef std::vector<double> dvec;
    dvec iwn{0};
    dvec re{0};
    dvec im{0};
    FileData()=default;
    FileData(dvec iwn, dvec re, dvec im) : iwn(iwn), re(re), im(im){};
    FileData(const FileData&)=delete;
    FileData(FileData&& src) : iwn(std::move(src.iwn)), re(std::move(src.re)), im(std::move(src.im)){}
    FileData& operator=(FileData&& src){
        this->iwn.clear();
        this->re.clear();
        this->im.clear();
        this->iwn=std::move(src.iwn);
        this->re=std::move(src.re);
        this->im=std::move(src.im);

        return *this;
    }
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
template<typename... Ts> void check_file_content(std::string pathToDir1, std::string pathToDir2, Ts &... filenamesToSave) noexcept(false);

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

template<typename... Ts> void check_file_content(std::string pathToDir1, std::string pathToDir2, Ts &... filenamesToSave) noexcept(false){
    mkdirTree(pathToDir1,"");
    mkdirTree(pathToDir2,"");
    // Now looking if files exist
    // Now looking if files exist
    // const int size = sizeof...(filenamesToSave);
    std::vector< std::string > search;
    std::vector<std::string> strFilesToSerach={(filenamesToSave+"*")...};
    for (auto el : strFilesToSerach){
        std::cout << el << "\n";
        try{
            search=glob(el);
        }
        catch(const std::runtime_error& e){
            std::cerr << e.what() << "\n";
        }
        if ( search.size() == 0 ){
            throw std::runtime_error("File "+el+" do not exist!!");
        }
    }
    
}

#endif /* end of File_Utils_ */