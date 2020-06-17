#include "file_utils.hpp"

std::vector<std::string> glob(const std::string& pattern) noexcept(false){
    /* This function fetches all the files with a given pattern. This function returns a vector of strings containing 
    the results of the query.
        
        Parameters:
            pattern (const std::string&): pattern of the filenames that one seeks. Needs to terminate with the wild card symbol "*".
        
        Returns:
            filenames (std::vector<std::string>): vector of strings representing the found results based on the pattern fed in.
    */
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
            cerr << ss.str() << "\n";
        } else { // Mainly for testing 
            ss << GLOB_NOMATCH;
            cerr << ss.str() << "\n";
        }
    } else if (return_value == 0){
        if (VERBOSE > 0)
            cerr << "The file "+pattern+" was found!!" << "\n";
    } else{
        throw runtime_error("Glob issued an error: "+to_string(return_value));
    }
    // collect all the filenames into a std::vector<std::string>
    vector<string> filenames;
    for(size_t i = 0; i < glob_result.gl_pathc; ++i) {
        auto tmp_str = string(glob_result.gl_pathv[i]);
        if ( !( tmp_str.find(string("AFM")) != string::npos ) ){ // to avoid mixing the AFM and PM data in data/
            filenames.push_back(tmp_str);
        }
    }
    // cleanup
    globfree(&glob_result);

    // done
    return filenames;
}

void mkdirTree(std::string sub, std::string dir) noexcept(false){
    struct stat info;
    if (sub.length()==0) return;

    int i=0; // This integer keeps track of the chars passed by.
    // This recursive routine stops when substring's length is shorter than integer i.
    for (; i<sub.length(); i++){
        dir+=sub[i];
        if (sub[i] == '/')
            break;
    }
    // Info on stat(): https://linux.die.net/man/2/stat
    if( stat( dir.c_str() , &info ) != 0 ){ // if stat() returns errno, because dir doesn't exist..
        // see https://techoverflow.net/2013/04/05/how-to-use-mkdir-from-sysstat-h/ for mode_t field.
        const int dir_err = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (EACCES == dir_err){ // In case access is refused, especially on cluster...
            throw std::runtime_error("Could not create directory hosting the data produced by the program; access permission required.");
        }
    }
    else if( info.st_mode & S_IFDIR ){  // then directory already exists..
        if (VERBOSE > 0)
            printf( "%s is a directory\n", dir.c_str() );
    }

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
    /* This function extracts the last integers of the filenames corresponding to the iteration number (N_it). It returns 
    the largest number N_it corresponding to the converged solution. Assumes that the filename finishes with ".dat" extension.
        
        Parameters:
            str (std::string): filename containing data of calculations carried out by the impurity solver (N_it).
        
        Returns:
            integerStr (int): integers at the end of the string representing the filename.
    */
    //std::regex r("[0-9]*\\.[0-9]+|[0-9]+");
    std::vector<double> intContainer;
    std::string toDel(".dat");
    str = eraseAllSubStr(str,toDel);
    size_t last_index = str.find_last_not_of("0123456789");
    std::string result = str.substr(last_index + 1);
    int integerStr = static_cast<int>(atof(result.c_str()));
    return integerStr;
}

FileData get_data(const std::string& strName, const unsigned int& Ntau) noexcept(false){
    /*  This method fetches the data (self-energy) contained inside a file named "strName". The data has to be laid out 
    the following way: 
        1. 1st column are the fermionic Matsubara frequencies.
        2. 2nd column are the real parts of the self-energy.
        3. 3rd column are the imaginary parts of the self-energy. 
        
        Parameters:
            strName (const std::string&): Filename containing the self-energy.
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

    return FileData(iwn,re,im);
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

void writeInHDF5File(std::vector< std::complex<double> >& GG_iqn_q_jj, std::vector< std::complex<double> >& GG_iqn_q_szsz, H5::H5File* file, const unsigned int& DATA_SET_DIM, const std::string& DATASET_NAME) noexcept(false){
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
    const H5std_string MEMBER1( "RE" );
    const H5std_string MEMBER2( "IM" );
    const int RANK = 1;
    H5::CompType mCmplx_type( sizeof(cplx_t) );
    mCmplx_type.insertMember( MEMBER1, HOFFSET(cplx_t, re), H5::PredType::NATIVE_DOUBLE);
    mCmplx_type.insertMember( MEMBER2, HOFFSET(cplx_t, im), H5::PredType::NATIVE_DOUBLE);
    try{
        H5::Exception::dontPrint();
        // Casting all the real values into the following array to get around the custom type design. Also easier to read out using Python.
        std::vector<cplx_t> custom_cplx_GG_iqn_q_jj(DATA_SET_DIM), custom_cplx_GG_iqn_q_szsz(DATA_SET_DIM);
        std::transform(GG_iqn_q_jj.begin(),GG_iqn_q_jj.end(),custom_cplx_GG_iqn_q_jj.begin(),[](std::complex<double> d){ return cplx_t{d.real(),d.imag()}; });
        std::transform(GG_iqn_q_szsz.begin(),GG_iqn_q_szsz.end(),custom_cplx_GG_iqn_q_szsz.begin(),[](std::complex<double> d){ return cplx_t{d.real(),d.imag()}; });

        hsize_t dimsf[1];
        dimsf[0] = DATA_SET_DIM;
        H5::DataSpace dataspace( RANK, dimsf );

        /*
        * Create a group in the file
        */
        H5::Group* group = new H5::Group( file->createGroup( "/"+DATASET_NAME ) );

        // H5::CompType std_cmplx_type( sizeof(std::complex<double>) );
        // H5::FloatType datatype( H5::PredType::NATIVE_DOUBLE );
        // datatype.setOrder( H5T_ORDER_LE );
        // size_t size_real_cmplx = sizeof(((std::complex<double> *)0)->real());
        // size_t size_imag_cmplx = sizeof(((std::complex<double> *)0)->imag());
        // std_cmplx_type.insertMember( MEMBER1, size_real_cmplx, H5::PredType::NATIVE_DOUBLE);
        // std_cmplx_type.insertMember( MEMBER2, size_imag_cmplx, H5::PredType::NATIVE_DOUBLE);

        // Create the dataset for jj.
        H5::DataSet* datasetjj;
        // dataset = new H5::DataSet(file->createDataSet(DATASET_NAME, std_cmplx_type, dataspace));
        datasetjj = new H5::DataSet(group->createDataSet("/"+DATASET_NAME+"/"+"jj", mCmplx_type, dataspace));
        // Write data to dataset
        datasetjj->write( custom_cplx_GG_iqn_q_jj.data(), mCmplx_type );
        delete datasetjj;

        // Create the dataset for szsz.
        H5::DataSet* datasetszsz;
        // dataset = new H5::DataSet(file->createDataSet(DATASET_NAME, std_cmplx_type, dataspace));
        datasetszsz = new H5::DataSet(group->createDataSet("/"+DATASET_NAME+"/"+"szsz", mCmplx_type, dataspace));
        // Write data to dataset
        datasetszsz->write( custom_cplx_GG_iqn_q_szsz.data(), mCmplx_type );
        delete datasetszsz;

        delete group;

    } catch( H5::FileIException error ){
        //err.printErrorStack();
        throw std::runtime_error("H5::FileIException thrown in writeInHDF5File!");
    }
    // catch failure caused by the DataSet operations
    catch( H5::DataSetIException error ){
        //error.printErrorStack();
        throw std::runtime_error("H5::DataSetIException thrown in writeInHDF5File!");
    }
    // catch failure caused by the DataSpace operations
    catch( H5::DataSpaceIException error ){
        //error.printErrorStack();
        throw std::runtime_error("H5::DataSpaceIException thrown in writeInHDF5File!");
    }
    // catch failure caused by the DataSpace operations
    catch( H5::DataTypeIException error ){
        //error.printErrorStack();
        throw std::runtime_error("H5::DataTypeIException thrown in writeInHDF5File!");
    }

}

void save_matrix_in_HDF5(const arma::Mat< std::complex<double> >& mat_to_save, double k_bar, double k_tilde, H5::H5File* file) noexcept(false){
    /* This method saves the denominator of the single ladder contribution inside an HDF5 file for later use - especially in the
    case where one wants to compute the infinite ladder contribution.
        
        Parameters:
            k (double): k-point.

        Returns:
            (double): current vertex.

    */
    const size_t NX = mat_to_save.n_cols;
    const size_t NY = mat_to_save.n_rows;
    const H5std_string MEMBER1 = std::string( "RE" );
    const H5std_string MEMBER2 = std::string( "IM" );
    const H5std_string  DATASET_NAME( std::string("kbar_")+std::to_string(k_bar)+std::string("ktilde_")+std::to_string(k_tilde) );
    const int RANK = 2;
    try{
        cplx_t cplx_mat_to_save[NY][NX];
        // casting data into custom complex struct..
        for (size_t i=0; i<NY; i++){
            for (size_t j=0; j<NX; j++){
                cplx_mat_to_save[i][j] = cplx_t{mat_to_save(i,j).real(),mat_to_save(i,j).imag()};
            }
        }
        /*
        * Turn off the auto-printing when failure occurs so that we can
        * handle the errors appropriately
        */
        H5::Exception::dontPrint();
        /*
        * Define the size of the array and create the data space for fixed
        * size dataset.
        */
        hsize_t dimsf[2];              // dataset dimensions
        dimsf[0] = NX;
        dimsf[1] = NY;
        H5::DataSpace dataspace( RANK, dimsf );
        H5::CompType mCmplx_type( sizeof(cplx_t) );
        mCmplx_type.insertMember( MEMBER1, HOFFSET(cplx_t, re), H5::PredType::NATIVE_DOUBLE);
        mCmplx_type.insertMember( MEMBER2, HOFFSET(cplx_t, im), H5::PredType::NATIVE_DOUBLE);

        // Create the dataset.
        H5::DataSet* dataset;
        dataset = new H5::DataSet(file->createDataSet(DATASET_NAME, mCmplx_type, dataspace));
        // Write data to dataset
        dataset->write( cplx_mat_to_save, mCmplx_type );

        delete dataset;
    } catch( H5::FileIException err ){
        //err.printErrorStack();
        throw std::runtime_error("H5::FileIException thrown in save_matrix_in_HDF5!");
    }
    // catch failure caused by the DataSet operations
    catch( H5::DataSetIException error ){
        //error.printErrorStack();
        throw std::runtime_error("H5::DataSetIException thrown in save_matrix_in_HDF5!");
    }
    // catch failure caused by the DataSpace operations
    catch( H5::DataSpaceIException error ){
        //error.printErrorStack();
        throw std::runtime_error("H5::DataSpaceIException thrown in save_matrix_in_HDF5!");
    }
    // catch failure caused by the DataSpace operations
    catch( H5::DataTypeIException error ){
        //error.printErrorStack();
        throw std::runtime_error("H5::DataTypeIException thrown in save_matrix_in_HDF5!");
    }

}

void save_matrix_in_HDF5(const arma::Cube< std::complex<double> >& cube_to_save, const std::vector< std::complex<double> >& iqn, H5std_string DATASET_NAME, H5::H5File* file) noexcept(false){
    /* This method saves the denominator of the single ladder contribution inside an HDF5 file for later use - especially in the
    case where one wants to compute the infinite ladder contribution.
        
        Parameters:
            k (double): k-point.

        Returns:
            (double): current vertex.

    */
    const size_t NX = cube_to_save.n_cols;
    const size_t NY = cube_to_save.n_rows;
    const size_t NZ = cube_to_save.n_slices;
    const H5std_string  MEMBER1 = std::string( "RE" );
    const H5std_string  MEMBER2 = std::string( "IM" );
    // const H5std_string  DATASET_NAME( std::string("ktilde_m_bar_")+std::to_string(k_tilde_m_bar) );
    const int RANK = 2;
    
    try{
        /*
        * Define the size of the array and create the data space for fixed
        * size dataset.
        */
        hsize_t dimsf[2];              // dataset dimensions
        dimsf[0] = NY;
        dimsf[1] = NX;
        H5::DataSpace dataspace( RANK, dimsf );
        H5::CompType mCmplx_type( sizeof(cplx_t) );
        mCmplx_type.insertMember( MEMBER1, HOFFSET(cplx_t, re), H5::PredType::NATIVE_DOUBLE);
        mCmplx_type.insertMember( MEMBER2, HOFFSET(cplx_t, im), H5::PredType::NATIVE_DOUBLE);

        /*
        * Create a group in the file
        */
        H5::Group* group = new H5::Group( file->createGroup( "/"+DATASET_NAME ) );

        // Attributes 
        hsize_t dimatt[1]={1};
        H5::DataSpace attr_dataspace = H5::DataSpace(1, dimatt);

        /*
        * Turn off the auto-printing when failure occurs so that we can
        * handle the errors appropriately
        */
        H5::Exception::dontPrint();

        cplx_t cplx_mat_to_save[NY][NX];
        for (size_t k=0; k<NZ; k++){
            H5std_string  ATTR_NAME( std::string("iqn") );
            // casting data into custom complex struct..
            for (size_t i=0; i<NY; i++){
                for (size_t j=0; j<NX; j++){
                    cplx_mat_to_save[i][j] = cplx_t{cube_to_save.slice(k)(i,j).real(),cube_to_save.slice(k)(i,j).imag()};
                }
            }

            // Create the dataset.
            H5::DataSet dataset;
            dataset = H5::DataSet(group->createDataSet("/"+DATASET_NAME+"/"+ATTR_NAME+"_"+std::to_string(iqn[k].imag()), mCmplx_type, dataspace));

            // Write data to dataset
            dataset.write( cplx_mat_to_save, mCmplx_type );
            
            // Create a dataset attribute. 
            H5::Attribute attribute = dataset.createAttribute( ATTR_NAME, H5::PredType::NATIVE_DOUBLE, 
                                                attr_dataspace);
        
            // Write the attribute data.
            double attr_data[1] = { iqn[k].imag() };
            attribute.write(H5::PredType::NATIVE_DOUBLE, attr_data);

        }
        
        delete group;

    } catch( H5::FileIException err ){
        //err.printErrorStack();
        throw std::runtime_error("H5::FileIException thrown in save_matrix_in_HDF5 2!");
    }
    // catch failure caused by the DataSet operations
    catch( H5::DataSetIException error ){
        //error.printErrorStack();
        throw std::runtime_error("H5::DataSetIException thrown in save_matrix_in_HDF5 2!");
    }
    // catch failure caused by the DataSpace operations
    catch( H5::DataSpaceIException error ){
        //error.printErrorStack();
        throw std::runtime_error("H5::DataSpaceIException thrown in save_matrix_in_HDF5 2!");
    }
    // catch failure caused by the DataSpace operations
    catch( H5::DataTypeIException error ){
        //error.printErrorStack();
        throw std::runtime_error("H5::DataTypeIException thrown in save_matrix_in_HDF5 2!");
    }

}

// void save_matrix_in_HDF5(std::complex<double>* mat_to_save, double k_bar, double k_tilde, H5::H5File* file, size_t NX, size_t NY) noexcept(false){
//     /* This method saves the denominator of the single ladder contribution inside an HDF5 file for later use - especially in the
//     case where one wants to compute the infinite ladder contribution.
        
//         Parameters:
//             k (double): k-point.

//         Returns:
//             (double): current vertex.

//     */
//     const H5std_string MEMBER1 = std::string( "RE" );
//     const H5std_string MEMBER2 = std::string( "IM" );
//     const H5std_string  DATASET_NAME( std::string("kbar_")+std::to_string(k_bar)+std::string("ktilde_")+std::to_string(k_tilde) );
//     const int RANK = 2;
//     try{
//         cplx_t cplx_mat_to_save[NY][NX];
//         // casting data into custom complex struct..
//         for (size_t i=0; i<NY; i++){
//             std::transform(&(mat_to_save[i*NX]),&(mat_to_save[i*NX+NX]),cplx_mat_to_save[i],[](std::complex<double> d){ return cplx_t{d.real(),d.imag()}; });
//         }
//         /*
//         * Turn off the auto-printing when failure occurs so that we can
//         * handle the errors appropriately
//         */
//         H5::Exception::dontPrint();
//         /*
//         * Define the size of the array and create the data space for fixed
//         * size dataset.
//         */
//         hsize_t dimsf[2];              // dataset dimensions
//         dimsf[0] = NX;
//         dimsf[1] = NY;
//         H5::DataSpace dataspace( RANK, dimsf );
//         H5::CompType mCmplx_type( sizeof(cplx_t) );
//         mCmplx_type.insertMember( MEMBER1, HOFFSET(cplx_t, re), H5::PredType::NATIVE_DOUBLE);
//         mCmplx_type.insertMember( MEMBER2, HOFFSET(cplx_t, im), H5::PredType::NATIVE_DOUBLE);

//         // Create the dataset.
//         H5::DataSet* dataset;
//         dataset = new H5::DataSet(file->createDataSet(DATASET_NAME, mCmplx_type, dataspace));
//         // Write data to dataset
//         dataset->write( cplx_mat_to_save, mCmplx_type );

//         // delete[] cplx_mat_to_save[0];
//         // delete[] cplx_mat_to_save;
//         delete dataset;
//     } catch( H5::FileIException err ){
//         //err.printErrorStack();
//         throw std::runtime_error("H5::FileIException thrown!");
//     }
//     // catch failure caused by the DataSet operations
//     catch( H5::DataSetIException error ){
//         //error.printErrorStack();
//         throw std::runtime_error("H5::DataSetIException!");
//     }
//     // catch failure caused by the DataSpace operations
//     catch( H5::DataSpaceIException error ){
//         //error.printErrorStack();
//         throw std::runtime_error("H5::DataSpaceIException!");
//     }
//     // catch failure caused by the DataSpace operations
//     catch( H5::DataTypeIException error ){
//         //error.printErrorStack();
//         throw std::runtime_error("H5::DataTypeIException!");
//     }

// }

arma::Mat< std::complex<double> > readFromHDF5File(H5::H5File* file, const std::string& DATASET_NAME) noexcept(false){

    const H5std_string MEMBER1( "RE" );
    const H5std_string MEMBER2( "IM" );
    const int RANK_OUT = 2;
    /*
    * Open the specified file and the specified dataset in the file.
    */
    std::cout << "DATASET_NAME: " << DATASET_NAME << std::endl; 
    H5::DataSet dataset_open = file->openDataSet(DATASET_NAME);
    /*
    * Get dataspace of the dataset.
    */
    H5::DataSpace dataspace_open = dataset_open.getSpace();
    hsize_t dims_out[2];
    int n_dims = dataspace_open.getSimpleExtentDims( dims_out, nullptr );
    const size_t NY = dims_out[0];
    const size_t NX = dims_out[1];
    arma::Mat< std::complex<double> > ret_mat(NY,NX);
    try{
        /*
        * Get the class of the datatype that is used by the dataset.
        */
        H5T_class_t type_class = dataset_open.getTypeClass();
        H5::CompType custom_cplx( sizeof(cplx_t) );
        custom_cplx.insertMember( MEMBER1, HOFFSET(cplx_t,re), H5::PredType::NATIVE_DOUBLE );
        custom_cplx.insertMember( MEMBER2, HOFFSET(cplx_t,im), H5::PredType::NATIVE_DOUBLE );
        std::cout << "NX: " << NX << " and NY: " << NY << std::endl;
        
        cplx_t data_out[NY][NX];
        H5::DataSpace memspace_out( RANK_OUT, dims_out );
        dataset_open.read(data_out, custom_cplx, memspace_out);
        for (size_t i=0; i<NY; i++){
            for (size_t j=0; j<NX; j++){
                ret_mat(i,j) = std::complex<double>( data_out[i][j].re,data_out[i][j].im );
            }
        }
        // for (size_t j=0; j<NX; j++){
        //     std::cout << ret_mat_ptr->at(0,j) << std::endl;
        // }
        // for (size_t j=0; j<NX; j++){
        //     std::cout << data_out[0][j].re << " " << data_out[0][j].im << std::endl;
        // }

    } catch( H5::FileIException err ){
        //err.printErrorStack();
        throw std::runtime_error("H5::FileIException thrown in readFromHDF5File!");
    }
    // catch failure caused by the DataSet operations
    catch( H5::DataSetIException error ){
        //error.printErrorStack();
        throw std::runtime_error("H5::DataSetIException thrown in readFromHDF5File!");
    }
    // catch failure caused by the DataSpace operations
    catch( H5::DataSpaceIException error ){
        //error.printErrorStack();
        throw std::runtime_error("H5::DataSpaceIException thrown in readFromHDF5File!");
    }
    // catch failure caused by the DataSpace operations
    catch( H5::DataTypeIException error ){
        //error.printErrorStack();
        throw std::runtime_error("H5::DataTypeIException thrown in readFromHDF5File!");
    }

    return ret_mat;

}

arma::Mat< std::complex<double> > readFromHDF5FileCube(H5::H5File* file, const std::string& DATASET_NAME, std::complex<double> iqn) noexcept(false){

    const H5std_string MEMBER1( "RE" );
    const H5std_string MEMBER2( "IM" );
    H5::CompType custom_cplx( sizeof(cplx_t) );
    custom_cplx.insertMember( MEMBER1, HOFFSET(cplx_t,re), H5::PredType::NATIVE_DOUBLE );
    custom_cplx.insertMember( MEMBER2, HOFFSET(cplx_t,im), H5::PredType::NATIVE_DOUBLE );
    const int RANK_OUT = 2;
    /*
    * Open the specified file and the specified dataset in the file.
    */
    H5::DataSet dataset_open;

    // std::cout << "Group fetched: " << "iqn_"+std::to_string(iqn.imag()) << std::endl;
    H5::Group* group = new H5::Group(file->openGroup(DATASET_NAME));
    try {  // to determine if the dataset exists in the group
        dataset_open = H5::DataSet( group->openDataSet( "iqn_"+std::to_string(iqn.imag()) ) );
    } catch( H5::GroupIException not_found_error ) {
        std::cerr << " Dataset is not found." << "\n";
    }
    /*
    * Get dataspace of the dataset.
    */
    H5::DataSpace dataspace_open = dataset_open.getSpace();
    hsize_t dims_out[2];
    int n_dims = dataspace_open.getSimpleExtentDims( dims_out, nullptr );
    const size_t NY = dims_out[0];
    const size_t NX = dims_out[1];
    arma::Mat< std::complex<double> > ret_mat(NY,NX);
    try{
        /*
        * Get the class of the datatype that is used by the dataset.
        */
        H5T_class_t type_class = dataset_open.getTypeClass();
        
        cplx_t data_out[NY][NX];
        H5::DataSpace memspace_out( RANK_OUT, dims_out );
        dataset_open.read(data_out, custom_cplx, memspace_out);
        for (size_t i=0; i<NY; i++){
            for (size_t j=0; j<NX; j++){
                ret_mat(i,j) = std::complex<double>( data_out[i][j].re,data_out[i][j].im );
            }
        }
        // for (size_t j=0; j<NX; j++){
        //     std::cout << ret_mat_ptr->at(0,j) << std::endl;
        // }
        // for (size_t j=0; j<NX; j++){
        //     std::cout << data_out[0][j].re << " " << data_out[0][j].im << std::endl;
        // }
        delete group;
    } catch( H5::FileIException err ){
        //err.printErrorStack();
        throw std::runtime_error("H5::FileIException thrown in readFromHDF5FileCube!");
    }
    // catch failure caused by the DataSet operations
    catch( H5::DataSetIException error ){
        //error.printErrorStack();
        throw std::runtime_error("H5::DataSetIException thrown in readFromHDF5FileCube!");
    }
    // catch failure caused by the DataSpace operations
    catch( H5::DataSpaceIException error ){
        //error.printErrorStack();
        throw std::runtime_error("H5::DataSpaceIException thrown in readFromHDF5FileCube!");
    }
    // catch failure caused by the DataSpace operations
    catch( H5::DataTypeIException error ){
        //error.printErrorStack();
        throw std::runtime_error("H5::DataTypeIException thrown in readFromHDF5FileCube!");
    }

    return ret_mat;

}
