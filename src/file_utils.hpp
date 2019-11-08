#ifndef File_Utils_
#define File_Utils_

#include<sys/stat.h> // for posix stat()
//#include<regex> // for regular expressions
#include<glob.h> // glob() and globfree()
#include<fstream>
#include<vector>
#include<string>
#include<iostream>
#include<sstream>

inline bool file_exists(const std::string& name);
std::vector<std::string> glob(const std::string& pattern) noexcept(false);
void mkdirTree(std::string sub, std::string dir) noexcept(false);
void check_file_content(const std::vector< std::string >& filenamesToSave, std::string pathToDir) noexcept(false);


#endif /* end of File_Utils_ */