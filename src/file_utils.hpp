#ifndef File_Utils_H_
#define File_Utils_H_

#include<sys/stat.h> // for posix stat()
//#include<regex> // for regular expressions
extern "C" {
#include<glob.h> // glob() and globfree()
}
#include<fstream>
#include<vector>
#include<string>
#include<iostream>
#include<sstream>

inline bool file_exists(const std::string& name);
std::vector<std::string> glob(const std::string& pattern) noexcept(false);
void mkdirTree(std::string sub, std::string dir) noexcept(false);
void check_file_content(const std::vector< std::string >& filenamesToSave, std::string pathToDir) noexcept(false);
int extractIntegerWords(std::string str);
std::string eraseAllSubStr(std::string,const std::string&);


#endif /* end of File_Utils_ */