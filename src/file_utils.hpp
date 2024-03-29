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
#include<cstring> // For memset()

inline bool file_exists(const std::string& name);
std::vector<std::string> glob(const std::string& pattern) noexcept(false);
void mkdirTree(std::string sub, std::string dir) noexcept(false);
void check_file_content(const std::vector< std::string >& filenamesToSave, std::string pathToDir1, std::string pathToDir2) noexcept(false);
int extractIntegerLastWords(std::string str);
std::string eraseAllSubStr(std::string,const std::string&);

inline bool file_exists(const std::string& name) {
  struct stat buffer;   
  return (stat(name.c_str(), &buffer) == 0); // string null terminateds
}

#endif /* end of File_Utils_ */