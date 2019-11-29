#include "file_utils.hpp"

std::vector<std::string> glob(const std::string& pattern) noexcept(false){
    using namespace std;

    // glob struct resides on the stack
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    // do the glob operation
    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if(return_value == GLOB_NOMATCH){
        globfree(&glob_result);
        stringstream ss;
        ss << "glob() failed with return_value " << return_value << " for "+pattern+"; pattern was not found." << "\n";
        if (VERBOSE > 0)
            std::cerr << ss.str() << "\n";
    } else if (return_value == 0){
        if (VERBOSE > 0)
            std::cerr << "The file "+pattern+" was found!!" << "\n";
    } else{
        throw std::runtime_error("Glob issued an error: "+std::to_string(return_value));
    }
    // collect all the filenames into a std::list<std::string>
    vector<string> filenames;
    for(size_t i = 0; i < glob_result.gl_pathc; ++i) {
        filenames.push_back(string(glob_result.gl_pathv[i]));
    }

    // cleanup
    globfree(&glob_result);

    // done
    return filenames;
}

void mkdirTree(std::string sub, std::string dir) noexcept(false){
    struct stat info;
    if (sub.length()==0) return;

    int i=0;
    for (; i<sub.length(); i++){
        dir+=sub[i];
        if (sub[i] == '/')
            break;
    }
    if( stat( dir.c_str() , &info ) != 0 ){
        const int dir_err = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err){
            throw std::runtime_error("Could not create directory hosting the data produced by the program.");
        }
    }
    else if( info.st_mode & S_IFDIR )  // S_ISDIR() doesn't exist on my windows 
        printf( "%s is a directory\n", dir.c_str() );

    if (i+1<sub.length()){
        mkdirTree(sub.substr(i+1),dir);
    }
}

void check_file_content(const std::vector< std::string >& filenamesToSave, std::string pathToDir1, std::string pathToDir2) noexcept(false){
    mkdirTree(pathToDir1,"");
    mkdirTree(pathToDir2,"");
    // Now looking if files exist
    std::string globsearch0(filenamesToSave[0]+"*");
    std::string globsearch1(filenamesToSave[1]+"*");
    std::string globsearch2(filenamesToSave[2]+"*");
    std::vector< std::string > search0, search1, search2;
    try{
        search0=glob(globsearch0);
        search1=glob(globsearch1);
        search2=glob(globsearch2);
    }
    catch(const std::runtime_error& e){
        std::cerr << e.what() << "\n";
    }

    if ( search0.size() > 0 || search1.size() > 0 || search2.size() > 0 ){
        #ifndef DEBUG
        throw std::runtime_error("Files "+filenamesToSave[0]+" and "+filenamesToSave[1]+" and "+filenamesToSave[2]+" already exist!!");
        #else
        std::cerr << "Warning: Files "+filenamesToSave[0]+" and "+filenamesToSave[1]+" and "+filenamesToSave[2]+" already exist!!" << "\n";
        #endif
    }
}

int extractIntegerLastWords(std::string str){ 
    //std::regex r("[0-9]*\\.[0-9]+|[0-9]+");
    std::vector<double> intContainer;
    std::string toDel(".dat");
    str = eraseAllSubStr(str,toDel);
    size_t last_index = str.find_last_not_of("0123456789");
    std::string result = str.substr(last_index + 1);
    double integerStr = static_cast<int>(atof(result.c_str()));
    return integerStr;
} 
