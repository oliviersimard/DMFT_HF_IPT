#include "sus_vertex_corrections.hpp"
#include "../../src/json_utils.hpp"

std::complex<double> get_denom(const std::vector< std::complex<double> >& iqn, const std::vector< std::complex<double> >& iwn, const std::vector<double>& k_arr, const std::vector< std::complex<double> >& SE, size_t n_tilde, size_t n_bar, size_t k_t_b, double mu) noexcept;
void save_matrix_in_HDF5(const arma::Cube< std::complex<double> >& cube_to_save, const std::vector<double>& k_arr, H5std_string DATASET_NAME, H5::H5File* file) noexcept(false);

int main(int argc, char** argv){
    
    std::string inputFilename("../../data/1D_U_8.000000_beta_20.000000_n_0.500000_N_tau_1024/Self_energy_1D_U_8.000000_beta_20.000000_n_0.500000_N_tau_1024_Nit_19.dat");
    // Fetching results from string
    std::vector<std::string> results;
    std::vector<std::string> fetches = {"U", "beta", "N_tau"};
    
    results = get_info_from_filename(inputFilename,fetches);

    const size_t NCA_Ntau = 2*(unsigned int)atoi(results[2].c_str()); // size of the full NCA calculation
    const size_t Ntau = 2*128; // One has to assume that the number of Matsubara frequencies defining the self-energy is sufficient.
    // Has to be a power of two as well: this is no change from IPT.
    assert(Ntau%2==0);
    const unsigned int N_k = 25;
    const double beta = atof(results[1].c_str());
    const double U = atof(results[0].c_str());
    const double mu = U/2.0; // Half-filling. Depending whether AFM-PM solution is loaded or not, mu=U/2 in PM only scenario and mu=0.0 in AFM-PM scenario.
    std::cout << "beta: " << beta << " U: " << U << "\n";
    // beta array constructed
    std::vector<double> beta_array;
    for (size_t j=0; j<=Ntau; j++){
        beta_array.push_back( j*beta/(Ntau) );
    }
    // k_t_b_array constructed
    std::vector<double> k_t_b_array;
    double k_tmp;
    for (size_t l=0; l<N_k; l++){
        k_tmp = -M_PI + l*2.0*M_PI/(double)(N_k-1);
        k_t_b_array.push_back(k_tmp);
    }

    // HDF5 business
    H5::H5File* file = nullptr;
    #ifdef INFINITE
    #ifdef NCA
    std::string filename("div_"+std::to_string(DIM)+"D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_Ntau_"+std::to_string(Ntau)+"_Nk_"+std::to_string(N_k)+"_NCA_infinite_ladder_sum.hdf5");
    #else
    std::string filename("div_"+std::to_string(DIM)+"D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_Ntau_"+std::to_string(Ntau)+"_Nk_"+std::to_string(N_k)+"_infinite_ladder_sum.hdf5");
    #endif
    #else
    #ifdef NCA
    std::string filename("div_"+std::to_string(DIM)+"D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_Ntau_"+std::to_string(Ntau)+"_Nk_"+std::to_string(N_k)+"_single_ladder_sum.hdf5");
    #else
    std::string filename("div_"+std::to_string(DIM)+"D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_Ntau_"+std::to_string(Ntau)+"_Nk_"+std::to_string(N_k)+"_single_ladder_sum.hdf5");
    #endif
    #endif
    const H5std_string FILE_NAME( filename );
    // The different processes cannot create more than once the file to be written in.
    file = new H5::H5File( FILE_NAME, H5F_ACC_TRUNC );
    // Getting the data of the self-energy
    FileData dataFromFile;  
    dataFromFile = get_data(inputFilename,NCA_Ntau);
    std::vector<double> wn = std::move(dataFromFile.iwn);
    std::vector<double> re = std::move(dataFromFile.re);
    std::vector<double> im = std::move(dataFromFile.im);

    std::vector< std::complex<double> > sigma_iwn(4*Ntau);
    std::vector< std::complex<double> > iwn(Ntau); // this size has got to be 1/4 the size of the self-energy loaded...
    try{
        if (wn.size()<4*Ntau)
            throw std::out_of_range("Problem in the truncation of the self-energy in loading process.");
        
        for (size_t i=0; i<4*Ntau; i++){
            sigma_iwn[i] = std::complex<double>(re[i+wn.size()/2-2*Ntau],im[i+wn.size()/2-2*Ntau]);
        }
    } catch(const std::out_of_range& err){
        std::cerr << err.what() << "\n";
        exit(1);
    }

    for (auto el : sigma_iwn){
        std::cout << el << std::endl;
    }

    std::transform(wn.data()+wn.size()/2-Ntau/2,wn.data()+wn.size()/2+Ntau/2,iwn.data(),[](double d){ return std::complex<double>(0.0,d); });

    // Bosonic Matsubara array
    std::vector< std::complex<double> > iqn; // for the total susceptibility
    std::vector< std::complex<double> > iqn_tilde; // for the inner loop inside Gamma.
    for (size_t j=0; j<Ntau; j++){ // change Ntau for lower value to decrease time when testing...
        iqn.push_back( std::complex<double>( 0.0, (2.0*j)*M_PI/beta ) );
    }

    for (signed int j=(-static_cast<int>(Ntau/2))+1; j<(signed int)static_cast<int>(Ntau/2); j++){ // Bosonic frequencies.
        iqn_tilde.push_back( std::complex<double>(0.0 , (2.0*(double)j)*M_PI/beta ) );
    }

    std::vector<double> initVec(NCA_Ntau,0.0); // This data contains twice as much data to perform the interpolation
    // Spline object used to interpolate the self-energy
    IPT2::SplineInline< std::complex<double> > splInlineObj(NCA_Ntau/2,initVec,k_t_b_array,iwn);
    splInlineObj.loadFileSpline(inputFilename,IPT2::spline_type::linear);
    
    // One-ladder calculations
    #ifndef INFINITE
    IPT2::OneLadder< std::complex<double> > one_ladder_obj(splInlineObj,sigma_iwn,iqn,k_t_b_array,iqn_tilde,mu,U,beta);
    #else
    IPT2::InfiniteLadders< std::complex<double> > inf_ladder_obj(splInlineObj,sigma_iwn,iqn,k_t_b_array,iqn_tilde,mu,U,beta);
    #endif
    
    #if DIM == 1
    // Should be a loop over the external momentum from this point on...
    arma::Cube< std::complex<double> > denom_sl(iwn.size(),iwn.size(),k_t_b_array.size());

    // Computing the derivatives for given (k_tilde,k_bar) tuple. Used inside the cubic spline algorithm...
    for (size_t n_tilde=0; n_tilde<iwn.size(); n_tilde++){
        std::cout << "n_tilde: " << n_tilde << std::endl;
        for (size_t n_bar=0; n_bar<iwn.size(); n_bar++){
            for (size_t l=0; l<k_t_b_array.size(); l++){
                denom_sl(n_tilde,n_bar,l) = get_denom(iqn_tilde,iwn,k_t_b_array,sigma_iwn,n_tilde,n_bar,l,mu)/beta*U;
            }
        }
    }
    H5std_string DATASET_NAME("div_sus");
    save_matrix_in_HDF5(denom_sl,k_t_b_array,DATASET_NAME,file);

    #elif DIM == 2
    // Should be a loop over the external momentum from this point on...
    arma::Cube< std::complex<double> > G_k_bar_q_tilde_iwn(k_t_b_array.size(),k_t_b_array.size(),iwn.size());
    arma::Cube<double> G_k_bar_q_tilde_tau(k_t_b_array.size(),k_t_b_array.size(),Ntau+1);

    // Computing the derivatives for given (k_tilde,k_bar) tuple. Used inside the cubic spline algorithm...
    for (size_t n_bar=0; n_bar<iwn.size(); n_bar++){
        // Building the lattice Green's function from the local DMFT self-energy
        for (size_t l=0; l<k_t_b_array.size(); l++){
            for (size_t m=0; m<k_t_b_array.size(); m++){
                G_k_bar_q_tilde_iwn.at(l,m,n_bar) = 1.0/( ( iwn[n_bar] ) + mu - epsilonk(k_t_b_array[l]+k_bar,k_t_b_array[m]+k_bar) - splInlineObj.calculateSpline(iwn[n_bar].imag()) );
            }
        }
    }

    for (size_t n_bar=0; n_bar<iwn.size(); n_bar++){
        for (size_t l=0; l<k_t_b_array.size(); l++){
            for (size_t m=0; m<k_t_b_array.size(); m++){
                // Substracting the tail of the Green's function
                G_k_bar_q_tilde_iwn.at(l,m,n_bar) -= 1.0/(iwn[n_bar]) + epsilonk(k_t_b_array[l]+k_bar,k_t_b_array[m]+k_bar)/iwn[n_bar]/iwn[n_bar]; //+ ( U*U/4.0 + epsilonk(q_tilde_array[l])*epsilonk(q_tilde_array[l]) )/iwn[j]/iwn[j]/iwn[j];
            }
        }
    }

    // FFT of G(iwn) --> G(tau)
    for (size_t l=0; l<q_tilde_array.size(); l++){
        for (size_t m=0; m<q_tilde_array.size(); m++){
            std::vector< std::complex<double> > G_iwn_k_slice(G_k_bar_q_tilde_iwn(arma::span(l,l),arma::span(m,m),arma::span::all).begin(),G_k_bar_q_tilde_iwn(arma::span(l,l),arma::span(m,m),arma::span::all).end());
            FFT_k_bar_q_tilde_tau = IPT2::get_iwn_to_tau(G_iwn_k_slice,beta); // beta_arr.back() is beta
            for (size_t i=0; i<beta_array.size(); i++){
                G_k_bar_q_tilde_tau(l,m,i) = FFT_k_bar_q_tilde_tau[i] - 0.5 - 0.25*(beta-2.0*beta_array[i])*epsilonk(q_tilde_array[l]+k_bar,q_tilde_array[m]+k_bar); //+ 0.25*beta_array[i]*(beta-beta_array[i])*(U*U/4.0 + epsilonk(q_tilde_array[l])*epsilonk(q_tilde_array[l]));
            }
        }
    }
    #endif

    delete file;

    return 0;

}

std::complex<double> get_denom(const std::vector< std::complex<double> >& iqn, const std::vector< std::complex<double> >& iwn, const std::vector<double>& k_arr, const std::vector< std::complex<double> >& SE, size_t n_tilde, size_t n_bar, size_t n_k_t_b, double mu) noexcept{
    std::complex<double> tmp_val{0};
    std::function<std::complex<double>(double)> k_integrand;
    const Integrals intObj;
    for (size_t ii=0; ii<iqn.size(); ii++){
        k_integrand = [&](double k){
            return 1.0/( ( iwn[n_bar] - iqn[ii] ) + mu - epsilonk(k+k_arr[n_k_t_b]) - SE[(2*iwn.size()-1)+n_bar-ii] )*1.0/( ( iwn[n_tilde] - iqn[ii] ) + mu - epsilonk(k) - SE[(2*iwn.size()-1)+n_tilde-ii] );
        };
        tmp_val += 1.0/(2.0*M_PI)*intObj.gauss_quad_1D(k_integrand,0.0,2.0*M_PI);
    }
    return tmp_val;
}

void save_matrix_in_HDF5(const arma::Cube< std::complex<double> >& cube_to_save, const std::vector<double>& k_arr, H5std_string DATASET_NAME, H5::H5File* file) noexcept(false){
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

        cplx_t* cplx_mat_to_save = new cplx_t[NX*NY];
        for (size_t k=0; k<NZ; k++){
            H5std_string  ATTR_NAME( std::string("k") );
            // casting data into custom complex struct..
            for (size_t j=0; j<NX; j++){
                for (size_t i=0; i<NY; i++){
                    cplx_mat_to_save[j*NY+i] = cplx_t{cube_to_save.at(i,j,k).real(),cube_to_save.at(i,j,k).imag()};
                }
            }

            // Create the dataset.
            H5::DataSet dataset;
            dataset = H5::DataSet(group->createDataSet("/"+DATASET_NAME+"/"+ATTR_NAME+"_"+std::to_string(k_arr[k]), mCmplx_type, dataspace));

            // Write data to dataset
            dataset.write( cplx_mat_to_save, mCmplx_type );
            
            // Create a dataset attribute. 
            H5::Attribute attribute = dataset.createAttribute( ATTR_NAME, H5::PredType::NATIVE_DOUBLE, 
                                                attr_dataspace );
        
            // Write the attribute data.
            double attr_data[1] = { k_arr[k] };
            attribute.write(H5::PredType::NATIVE_DOUBLE, attr_data);

        }
        delete[] cplx_mat_to_save;

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
