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
        if (VERBOSE > 0) {
            ss << "glob() failed with return_value " << return_value << " for "+pattern+"; pattern was not found." << "\n";
            std::cerr << ss.str() << "\n";
        } else { // Mainly for testing 
            ss << GLOB_NOMATCH;
            std::cerr << ss.str() << "\n";
        }
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
        if (EACCES == dir_err){ // In case access is refused, especially on cluster...
            throw std::runtime_error("Could not create directory hosting the data produced by the program; access permission required.");
        }
    }
    else if( info.st_mode & S_IFDIR )  // S_ISDIR() doesn't exist on my windows
        if (VERBOSE > 0)
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

FileData get_data(const std::string& strName, const unsigned int& Ntau) noexcept(false){
    /*  This method fetches the data (self-energy) contained inside a file named "strName". The data has to be laid out 
    the following way: 
        1. 1st column are the fermionic Matsubara frequencies.
        2. 2nd column are the real parts of the self-energy.
        3. 3rd column are the imaginary parts of the self-energy. 
        
        Parameters:
            strName (const std::string&): Filename containeing the self-energy.
            Ntau (const unsigned int&): Number of fermionic Matsubara frequencies (length of the columns).
        
        Returns:
            fileDataObj (struct FileData): struct containing a vector of data for each column in the data file.
    */

    std::vector<double> iwn(Ntau,0.0);
    std::vector<double> re(Ntau,0.0);
    std::vector<double> im(Ntau,0.0);

    std::ifstream inputFile(strName);
    std::string increment("");
    unsigned int idx = 0;
    size_t pos=0;
    std::string token;
    const std::string delimiter = "\t\t";
    if (inputFile.fail()){
        std::cerr << "Error loading the file..." << "\n";
        throw std::ios_base::failure("Check loading file procedure..");
    } 
    std::vector<double> tmp_vec;
    while (getline(inputFile,increment)){
        if (increment[0]=='/'){
            continue;
        }
        while ((pos = increment.find(delimiter)) != std::string::npos) {
            token = increment.substr(0, pos);
            increment.erase(0, pos + delimiter.length());
            tmp_vec.push_back(std::atof(token.c_str()));
        }
        tmp_vec.push_back(std::atof(increment.c_str()));
        iwn[idx] = tmp_vec[0];
        re[idx] = tmp_vec[1];
        im[idx] = tmp_vec[2];

        increment.clear();
        tmp_vec.clear();
        idx++;
    }
    
    inputFile.close();

    FileData fileDataObj={iwn,re,im};
    return fileDataObj;
}

std::vector<std::string> get_info_from_filename(const std::string& strName,const std::vector<std::string>& fetches,const char* separation_char) noexcept(false){
    /*  This function returns a vector of strings representing the matches to a given regex expression request. This regex expression
    is fixed, i.e local within the function scope. 
        
        Parameters:
            strName (std::string&): the string to probe for given regex expressions.
            fetches (const std::vector<std::string>&): The set of strings after which the numbers are fetched.
            separation_char (const char*): character separating the elements of fetches and the numbers. Defaults to "_".
        
        Returns:
            results (std::vector<std::string>): vector containing the numbers fetched from the string strName.
    */

    std::vector<std::string> results(fetches.size());
    // The group to search inside strName for all fetches...
    const std::string regex_group("(\\d*\\.\\d+|\\d+)");
    std::smatch match;
    for (int i=0; i<fetches.size(); i++){
        std::string re_request = std::string(".*")+fetches[i]+separation_char+regex_group;
        try{
            std::regex re(re_request.c_str());
            if (std::regex_search(strName,match,re) && match.size() > 1){
                results[i] = match.str(1);
            } else{
                results[i] = std::string("");
            }
        } catch (std::regex_error& e){
            std::cerr << e.what() << "\n";
            throw std::runtime_error("Problem encountered in the regex subroutine.");
        }

    }

    return results;

}

void writeInHDF5File(std::vector< std::complex<double> >& GG_iqn_q, H5::H5File* file, const unsigned int& DATA_SET_DIM, const int& RANK, const H5std_string& MEMBER1, const H5std_string& MEMBER2, const std::string& DATASET_NAME) noexcept(false){
    /*  This method writes in an HDF5 file the data passed in the first entry "GG_iqn_q". The data has to be complex-typed. This function hinges on the
    the existence of a custom complex structure "cplx_t" to parse in the data:
    
        typedef struct cplx_t{ // Custom data holder for the HDF5 handling
            double re;
            double im;
        } cplx_t;
        
        Parameters:
            GG_iqn_q (std::vector< std::complex<double> >&): function mesh over (iqn,q)-space.
            file (H5::H5File*): pointer to file object.
            DATA_SET_DIM (const unsigned int&): corresponds to the number of bosonic Matsubara frequencies and therefore to the length of columns in HDF5 file.
            RANK (const int&): rank of the object to be saved. Should be 1.
            MEMBER1 (const H5std_string&): name designating the internal metadata to label the first member variable of cplx_t structure.
            MEMBER2 (const H5std_string&): name designating the internal metadata to label the second member variable of cplx_t structure.
            DATASET_NAME (const std::string&): name of the dataset to be saved.
        
        Returns:
            (void)
    */
    try{
        H5::Exception::dontPrint();
        // Casting all the real values into the following array to get around the custom type design. Also easier to read out using Python.
        std::vector<cplx_t> custom_cplx_GG_iqn_q(DATA_SET_DIM);
        std::transform(GG_iqn_q.begin(),GG_iqn_q.end(),custom_cplx_GG_iqn_q.begin(),[](std::complex<double> d){ return cplx_t{d.real(),d.imag()}; });

        hsize_t dimsf[1];
        dimsf[0] = DATA_SET_DIM;
        H5::DataSpace dataspace( RANK, dimsf );

        // H5::CompType std_cmplx_type( sizeof(std::complex<double>) );
        // H5::FloatType datatype( H5::PredType::NATIVE_DOUBLE );
        // datatype.setOrder( H5T_ORDER_LE );
        // size_t size_real_cmplx = sizeof(((std::complex<double> *)0)->real());
        // size_t size_imag_cmplx = sizeof(((std::complex<double> *)0)->imag());
        // std_cmplx_type.insertMember( MEMBER1, size_real_cmplx, H5::PredType::NATIVE_DOUBLE);
        // std_cmplx_type.insertMember( MEMBER2, size_imag_cmplx, H5::PredType::NATIVE_DOUBLE);

        H5::CompType mCmplx_type( sizeof(cplx_t) );
        mCmplx_type.insertMember( MEMBER1, HOFFSET(cplx_t, re), H5::PredType::NATIVE_DOUBLE);
        mCmplx_type.insertMember( MEMBER2, HOFFSET(cplx_t, im), H5::PredType::NATIVE_DOUBLE);

        // Create the dataset.
        H5::DataSet* dataset;
        // dataset = new H5::DataSet(file->createDataSet(DATASET_NAME, std_cmplx_type, dataspace));
        dataset = new H5::DataSet(file->createDataSet(DATASET_NAME, mCmplx_type, dataspace));
        // Write data to dataset
        dataset->write( custom_cplx_GG_iqn_q.data(), mCmplx_type );

        delete dataset;

    } catch( H5::FileIException err ){
        err.printErrorStack();
        throw std::runtime_error("H5::FileIException thrown!");
    }
    // catch failure caused by the DataSet operations
    catch( H5::DataSetIException error ){
        error.printErrorStack();
        throw std::runtime_error("H5::DataSetIException!");
    }
    // catch failure caused by the DataSpace operations
    catch( H5::DataSpaceIException error ){
        error.printErrorStack();
        throw std::runtime_error("H5::DataSpaceIException!");
    }
    // catch failure caused by the DataSpace operations
    catch( H5::DataTypeIException error ){
        error.printErrorStack();
        throw std::runtime_error("H5::DataTypeIException!");
    }

}
