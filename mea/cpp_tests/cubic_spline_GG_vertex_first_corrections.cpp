//#include "../../src/tridiagonal.hpp"
//#include "../../src/integral_utils.hpp"
#include "../../src/IPT2nd3rdorderSingle2.hpp"
#include <fftw3.h>
#include <fstream>
#include "H5Cpp.h" 
/*
Info HDF5:
https://support.hdfgroup.org/HDF5/doc/cpplus_RM/compound_8cpp-example.html
https://www.hdfgroup.org/downloads/hdf5/source-code/
*/

struct FileData;

FileData get_data(const std::string& strName, const unsigned int& Ntau) noexcept(false);
std::vector<double> get_iwn_to_tau(const std::vector< std::complex<double> >& F_iwn, double beta, std::string obj="Green");
template<typename T> std::vector< T > generate_random_numbers(size_t arr_size, T min, T max) noexcept;
arma::Mat<double> get_derivative_FFT(arma::Mat< std::complex<double> > G_k_iwn, const std::vector< std::complex<double> >& iwn, const std::vector<double>& k_arr, const std::vector<double>& beta_arr, double U, double mu, double q=0.0, std::string opt="positive");
inline double velocity(double k) noexcept;
void writeInHDF5File(std::vector< std::complex<double> >& GG_iqn_q, H5::H5File* file, const unsigned int& DATA_SET_DIM, const int& RANK, const H5std_string& MEMBER1, const H5std_string& MEMBER2, const std::string& DATASET_NAME) noexcept(false);
inline void sum_rule_iqn_0(double n_k, double k, double& sum_rule_total, unsigned int N_q) noexcept;
std::complex<double> getGreen(double k, double mu, std::complex<double> iwn, IPT2::SplineInline< std::complex<double> >& splInlineobj);

struct FileData{
    std::vector<double> iwn;
    std::vector<double> re;
    std::vector<double> im;
};

typedef struct cplx_t{ // Custom data holder for the HDF5 handling
    double re;
    double im;
} cplx_t;

int main(void){
    
    std::string inputFilename("../data/Self_energy_1D_U_10.000000_beta_50.000000_n_0.500000_N_tau_1024_Nit_32.dat");
    std::string inputFilenameLoad("../data/Self_energy_1D_U_10.000000_beta_50.000000_n_0.500000_N_tau_2048");
    // Choose whether current-current or spin-spin correlation function is computed.
    const bool is_jj = true; 
    const unsigned int Ntau = 2*1024;
    const unsigned int N_q = 500;
    const double beta = 50.0;
    const double U = 10.0;
    const double mu = U/2.0; // Half-filling
    // k_tilde and k_bar momenta
    const double k_tilde = M_PI;
    const double k_bar = M_PI;
    const double qq = 0.0;

    // beta array constructed
    std::vector<double> beta_array;
    for (size_t j=0; j<=Ntau; j++){
        double beta_tmp = j*beta/(Ntau);
        beta_array.push_back(beta_tmp);
    }
    // q_tilde_array constructed
    std::vector<double> q_tilde_array;
    for (size_t l=0; l<N_q; l++){
        double q_tilde_tmp = l*2.0*M_PI/(double)(N_q-1);
        q_tilde_array.push_back(q_tilde_tmp);
    }
    // HDF5 business
    std::string filename(std::string("bb_1D_U_")+std::to_string(U)+std::string("_beta_")+std::to_string(beta)+std::string("_Ntau_")+std::to_string(Ntau)+std::string("_Nk_")+std::to_string(N_q)+std::string("_isjj_")+std::to_string(is_jj)+std::string(".hdf5"));
    const H5std_string FILE_NAME( filename );
    const int RANK = 1;
    const unsigned int DATA_SET_DIM = Ntau;
    const H5std_string MEMBER1( "RE" );
    const H5std_string MEMBER2( "IM" );
    
    H5::H5File* file = new H5::H5File( FILE_NAME, H5F_ACC_TRUNC );
    // Getting the data
    FileData dataFromFile = get_data(inputFilename,Ntau);

    std::vector<double> wn = dataFromFile.iwn;
    std::vector<double> re = dataFromFile.re;
    std::vector<double> im = dataFromFile.im;
    std::vector< std::complex<double> > sigma_iwn;
    for (size_t i=0; i<wn.size(); i++){
        sigma_iwn.push_back(std::complex<double>(re[i],im[i]));
    }

    std::vector< std::complex<double> > iwn(wn.size());
    std::transform(wn.begin(),wn.end(),iwn.begin(),[](double d){ return std::complex<double>(0.0,d); }); // casting into array of double for cubic spline.
    // Bosonic Matsubara array
    std::vector< std::complex<double> > iqn;
    for (size_t j=0; j<iwn.size(); j++){
        iqn.push_back( std::complex<double>( 0.0, (2.0*j)*M_PI/beta ) );
    }

    spline<double> splObj;
    std::vector<double> initVec(2*Ntau,0.0); // This data contains twice as much data to perform the interpolation
    IPT2::SplineInline< std::complex<double> > splInlineObj(Ntau,initVec,q_tilde_array,iwn,iqn);
    splInlineObj.loadFileSpline(inputFilenameLoad,IPT2::spline_type::linear);

    
    // Should be a loop over the external momentum from this point on...
    arma::Mat< std::complex<double> > G_k_bar_q_tilde_iwn(q_tilde_array.size(),iwn.size());
    arma::Mat< std::complex<double> > G_k_tilde_q_tilde_iwn(q_tilde_array.size(),iwn.size());
    arma::Mat<double> G_k_bar_q_tilde_tau(q_tilde_array.size(),Ntau+1);
    arma::Mat<double> G_k_tilde_q_tilde_tau(q_tilde_array.size(),Ntau+1);
    std::vector<double> FFT_k_bar_q_tilde_tau;
    std::vector<double> FFT_k_tilde_q_tilde_tau;
    arma::Mat<double> dGG_tau_for_k(q_tilde_array.size(),beta_array.size());
    std::vector< std::complex<double> > cubic_spline_GG_iqn(iqn.size());

    // Computing the derivatives for given (k_tilde,k_bar) tuple. Used inside the cubic spline algorithm...
    for (size_t n_bar=0; n_bar<iwn.size(); n_bar++){
        // Building the lattice Green's function from the local DMFT self-energy
        for (size_t l=0; l<q_tilde_array.size(); l++){
            G_k_bar_q_tilde_iwn(l,n_bar) = 1.0/( ( iwn[n_bar] ) + mu - epsilonk(k_bar-q_tilde_array[l]) - sigma_iwn[n_bar] );
        }
    }
    arma::Mat<double> dG_dtau_m_FFT_k_bar = get_derivative_FFT(G_k_bar_q_tilde_iwn,iwn,q_tilde_array,beta_array,U,mu,k_bar,std::string("negative"));

    for (size_t n_bar=0; n_bar<iwn.size(); n_bar++){
        for (size_t l=0; l<q_tilde_array.size(); l++){
            // Substracting the tail of the Green's function
            G_k_bar_q_tilde_iwn(l,n_bar) -= 1.0/(iwn[n_bar]); //+ epsilonk(q_tilde_array[l])/iwn[j]/iwn[j]; //+ ( U*U/4.0 + epsilonk(q_tilde_array[l])*epsilonk(q_tilde_array[l]) )/iwn[j]/iwn[j]/iwn[j];
        }
    }

    for (size_t n_tilde=0; n_tilde<iwn.size(); n_tilde++){
        // Building the lattice Green's function from the local DMFT self-energy
        for (size_t l=0; l<q_tilde_array.size(); l++){
            G_k_tilde_q_tilde_iwn(l,n_tilde) = 1.0/( ( iwn[n_tilde] ) + mu - epsilonk(k_tilde-q_tilde_array[l]) - sigma_iwn[n_tilde] );
            // Substracting the tail of the Green's function
            G_k_tilde_q_tilde_iwn(l,n_tilde) -= 1.0/(iwn[n_tilde]); //+ epsilonk(q_tilde_array[l])/iwn[j]/iwn[j]; //+ ( U*U/4.0 + epsilonk(q_tilde_array[l])*epsilonk(q_tilde_array[l]) )/iwn[j]/iwn[j]/iwn[j];
        }
    }
    arma::Mat<double> dG_dtau_FFT_k_tilde = get_derivative_FFT(G_k_tilde_q_tilde_iwn,iwn,q_tilde_array,beta_array,U,mu,k_tilde);

    for (size_t n_tilde=0; n_tilde<iwn.size(); n_tilde++){
        for (size_t l=0; l<q_tilde_array.size(); l++){
            // Substracting the tail of the Green's function
            G_k_tilde_q_tilde_iwn(l,n_tilde) -= 1.0/(iwn[n_tilde]); //+ epsilonk(q_tilde_array[l])/iwn[j]/iwn[j]; //+ ( U*U/4.0 + epsilonk(q_tilde_array[l])*epsilonk(q_tilde_array[l]) )/iwn[j]/iwn[j]/iwn[j];
        }
    }

    /* TEST dG(-tau)/dtau */
    std::ofstream test1("test_1_corr.dat", std::ios::out);
    for (size_t j=0; j<beta_array.size(); j++){
        test1 << beta_array[j] << "  " << dG_dtau_m_FFT_k_bar(52,j) << "\n";
    }
    test1.close();

    // FFT of G(iwn) --> G(tau)
    for (size_t l=0; l<q_tilde_array.size(); l++){
        std::vector< std::complex<double> > G_iwn_k_slice(G_k_bar_q_tilde_iwn(l,arma::span::all).begin(),G_k_bar_q_tilde_iwn(l,arma::span::all).end());
        FFT_k_bar_q_tilde_tau = get_iwn_to_tau(G_iwn_k_slice,beta); // beta_arr.back() is beta
        for (size_t i=0; i<beta_array.size(); i++){
            G_k_bar_q_tilde_tau(l,i) = FFT_k_bar_q_tilde_tau[i] - 0.5; //- 0.25*(beta-2.0*beta_array[i])*epsilonk(q_tilde_array[l]); //+ 0.25*beta_array[i]*(beta-beta_array[i])*(U*U/4.0 + epsilonk(q_tilde_array[l])*epsilonk(q_tilde_array[l]));
        }
    }

    // FFT of G(iwn) --> G(tau)
    for (size_t l=0; l<q_tilde_array.size(); l++){
        std::vector< std::complex<double> > G_iwn_k_slice(G_k_tilde_q_tilde_iwn(l,arma::span::all).begin(),G_k_tilde_q_tilde_iwn(l,arma::span::all).end());
        FFT_k_tilde_q_tilde_tau = get_iwn_to_tau(G_iwn_k_slice,beta); // beta_arr.back() is beta
        for (size_t i=0; i<beta_array.size(); i++){
            G_k_tilde_q_tilde_tau(l,i) = FFT_k_tilde_q_tilde_tau[i] - 0.5; //- 0.25*(beta-2.0*beta_array[i])*epsilonk(q_tilde_array[l]); //+ 0.25*beta_array[i]*(beta-beta_array[i])*(U*U/4.0 + epsilonk(q_tilde_array[l])*epsilonk(q_tilde_array[l]));
        }
    }

    /* TEST G(-tau) */
    std::ofstream test2("test_2_corr.dat", std::ios::out);
    for (size_t j=0; j<beta_array.size(); j++){
        test2 << beta_array[j] << "  " << -1.0*G_k_bar_q_tilde_tau(52,Ntau-j) << "\n";
    }
    test2.close();

    arma::Cube< std::complex<double> > cub_spl_GG_n_bar_vs_n_tilde(iwn.size(),iwn.size(),q_tilde_array.size());
    for (size_t l=0; l<q_tilde_array.size(); l++){
        spline<double> splObj_GG;
        arma::Cube<double> GG_tau_for_k(2,2,beta_array.size());

        for (size_t i=0; i<beta_array.size(); i++){
            dGG_tau_for_k(l,i) = (-2.0)*( G_k_tilde_q_tilde_tau(l,i)*dG_dtau_m_FFT_k_bar(l,i) - dG_dtau_FFT_k_tilde(l,i)*G_k_bar_q_tilde_tau(l,Ntau-i) );
            GG_tau_for_k.slice(i)(0,0) = (-2.0)*(-1.0)*( G_k_tilde_q_tilde_tau(l,i)*G_k_bar_q_tilde_tau(l,Ntau-i) );
        }

        // Taking the derivative for boundary conditions
        double left_der_GG = dGG_tau_for_k(l,0);
        double right_der_GG = dGG_tau_for_k(l,Ntau);

        splObj_GG.set_boundary(spline<double>::bd_type::first_deriv,left_der_GG,spline<double>::bd_type::first_deriv,right_der_GG);
        splObj_GG.set_points(beta_array,GG_tau_for_k);

        for (size_t n_bar=0; n_bar<iwn.size(); n_bar++){
            std::vector< std::complex<double> > cub_spl_GG = splObj_GG.bosonic_corr_single_ladder(iwn,beta,n_bar);
            cub_spl_GG_n_bar_vs_n_tilde.slice(l)(n_bar,arma::span::all) = arma::Row< std::complex<double> >(cub_spl_GG);
        }
    }
    for (auto el : cub_spl_GG_n_bar_vs_n_tilde.slice(0)(0,arma::span::all)){
        std::cout << el << std::endl;
    }
    exit(0);

    const Integrals integralsObj;
    arma::Mat< std::complex<double> > GG_n_bar_n_tilde(iwn.size(),iwn.size());
    constexpr double delta = 2.0*M_PI/(double)(N_q-1);
    for (size_t em=0; em<iqn.size(); em++){
        for (size_t n_bar=0; n_bar<iwn.size(); n_bar++){
            for (size_t n_tilde=0; n_tilde<iwn.size(); n_tilde++){
                std::vector< std::complex<double> > GG_n_bar_n_tilde_k_tmp(cub_spl_GG_n_bar_vs_n_tilde(arma::span(n_bar,n_bar),arma::span(n_tilde,n_tilde),arma::span::all).begin(),cub_spl_GG_n_bar_vs_n_tilde(arma::span(n_bar,n_bar),arma::span(n_tilde,n_tilde),arma::span::all).end());
                // This part remains to be done....
                GG_n_bar_n_tilde(n_bar,n_tilde) = getGreen(k_tilde,mu,iwn[n_tilde],splInlineObj)*getGreen(k_tilde+qq,mu,iwn[n_tilde]+iqn[em],splInlineObj)*( 1.0 / ( 1.0 + U/(2.0*M_PI)*integralsObj.I1D_CPLX(GG_n_bar_n_tilde_k_tmp,delta) ) )*getGreen(k_bar,mu,iwn[n_bar],splInlineObj)*getGreen(k_bar+qq,mu,iwn[n_bar]+iqn[em],splInlineObj);
            }
        }
        cubic_spline_GG_iqn[em] = (U/beta/beta)*arma::accu(GG_n_bar_n_tilde); // summing over the internal ikn_tilde and ikn_bar
    }

    std::string DATASET_NAME("q_"+std::to_string(qq));
    writeInHDF5File(cubic_spline_GG_iqn, file, DATA_SET_DIM, RANK, MEMBER1, MEMBER2, DATASET_NAME);

    delete file;

}

inline void sum_rule_iqn_0(double n_k, double k, double& sum_rule_total, unsigned int N_q) noexcept{
    /*  This method computes the sum rule that determines the value of the optical conductivity at iqn=0. This sum rule reads:

        sum_rule = 1/N_q*sum_{k,sigma} d^2\epsilon_k/dk^2 <n_{k,sigma}>.
        
        Parameters:
            n_k (double): electron density at a given k-point.
            k (double): k-point.
            sum_rule_total (double&): total of sum rule, that is the expression above as final result after whole k-summation.
            N_q (unsigned int): Number of k-points spanning the original Brillouin zone.
        
        Returns:
            (void).
    */
    sum_rule_total += 1.0/N_q*4.0*std::cos(k)*n_k;
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

inline std::complex<double> getGreen(double k, double mu, std::complex<double> iwn, IPT2::SplineInline< std::complex<double> >& splInlineobj){
    return 1.0 / ( iwn + mu - epsilonk(k) - splInlineobj.calculateSpline( iwn.imag() ) );
}

inline double velocity(double k) noexcept{
    /* This method computes the current vertex in the case of a 1D nearest-neighbour dispersion relation.

    Parameters:
        k (double): k-point.

    Returns:
        (double): current vertex.
    */
    return 2.0*std::sin(k);
}

arma::Mat<double> get_derivative_FFT(arma::Mat< std::complex<double> > G_k_iwn, const std::vector< std::complex<double> >& iwn, const std::vector<double>& q_arr, const std::vector<double>& beta_arr, double U, double mu, double k, std::string opt){
    /*  The main difference here from the previous method is that the k momenta are switched, i.e. the k-vector summed over is subtracted now.
    */
    
    arma::Mat<double> dG_tau_for_k(q_arr.size(),beta_arr.size());
    const double beta = beta_arr.back();
    // Subtracting the leading moments of the Green's function...
    std::complex<double> moments;
    for (size_t n_k=0; n_k<iwn.size(); n_k++){
        for (size_t l=0; l<q_arr.size(); l++){
            moments = 1.0/( iwn[n_k] ) + epsilonk(k-q_arr[l])/( iwn[n_k] )/( iwn[n_k] ) + ( U*U/4.0 + epsilonk(k-q_arr[l])*epsilonk(k-q_arr[l]) )/( iwn[n_k] )/( iwn[n_k] )/( iwn[n_k] );
            G_k_iwn(l,n_k) -= moments;
            if (opt.compare("negative")==0){
                G_k_iwn(l,n_k) = std::conj(G_k_iwn(l,n_k));
            }
            G_k_iwn(l,n_k) *= -1.0*iwn[n_k];
        }
    }
    // Calculating the imaginary-time derivative of the Green's function.
    std::vector<double> FFT_iwn_tau;
    for (size_t l=0; l<q_arr.size(); l++){
        std::vector< std::complex<double> > G_iwn_k_slice(G_k_iwn(l,arma::span::all).begin(),G_k_iwn(l,arma::span::all).end());
        FFT_iwn_tau = get_iwn_to_tau(G_iwn_k_slice,beta,std::string("Derivative")); // beta_arr.back() is beta
        for (size_t i=0; i<beta_arr.size()-1; i++){
            if (opt.compare("negative")==0){
                dG_tau_for_k(l,i) = FFT_iwn_tau[i] + 0.5*epsilonk(k-q_arr[l]) + 0.5*( beta_arr[i] - beta/2.0 )*( U*U/4.0 + epsilonk(k-q_arr[l])*epsilonk(k-q_arr[l]) );
            } else if (opt.compare("positive")==0){
                dG_tau_for_k(l,i) = FFT_iwn_tau[i] + 0.5*epsilonk(k-q_arr[l]) + 0.5*( beta/2.0 - beta_arr[i] )*( U*U/4.0 + epsilonk(k-q_arr[l])*epsilonk(k-q_arr[l]) );
            }
        }
        dG_tau_for_k(l,arma::span::all)(beta_arr.size()-1) = epsilonk(k-q_arr[l])-mu+U*0.5 - 1.0*dG_tau_for_k(l,0); // Assumed half-filling
    }

    return dG_tau_for_k;
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

std::vector<double> get_iwn_to_tau(const std::vector< std::complex<double> >& F_iwn, double beta, std::string obj){
    /*  This method computes the imaginary-time object related to "F_iwn". Prior to calling in this method, one needs
    to subtract the leading moments of the object, be it the self-energy or the Green's function. 
        
        Parameters:
            F_iwn (const std::vector< std::complex<double> >&): Object defined in fermionic Matsubara frequencies to translate to imaginary time.
            beta (double): Inverse temperature.
            obj (std::string): string specifying the nature of the object to be transformed. Defaults to "Green". Anything else chosen,
            i.e "Derivative", means that the last component to the imaginary-time object (tau=beta) is left for later. This is particularly important for the derivative.
        
        Returns:
            tau_final_G (std::vector<double>): vector containing the imaginary-time definition of the inputted object "F_iwn".
    */

    size_t MM = F_iwn.size();
    //std::cout << "size: " << MM << std::endl;
    const std::complex<double> im(0.0,1.0);
    std::vector<double> tau_final_G(MM+1,0.0);
    std::complex<double>* output = new std::complex<double>[MM];
    std::complex<double>* input = new std::complex<double>[MM];
    for (size_t j=0; j<MM; j++){
        input[j] = F_iwn[j];
    }
    // FFT
    fftw_plan plan;
    plan=fftw_plan_dft_1d(MM, reinterpret_cast<fftw_complex*>(input), reinterpret_cast<fftw_complex*>(output), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    for (int i=0; i<MM; i++){
        tau_final_G[i] = ( ( 1.0/beta*std::exp( -im*(double)i*M_PI*(1.0/MM - 1.0) ) )*output[i] ).real();
    }
    if (obj.compare("Green")==0){
        for (int i=0; i<MM; i++){
            tau_final_G[MM] += ( 1.0/beta*std::exp(-im*M_PI*(1.0-MM))*F_iwn[i] ).real();
        }
    }
    
    delete[] output;
    delete[] input;

    return tau_final_G;
}

template<typename T>
std::vector< T > generate_random_numbers(size_t arr_size, T min, T max) noexcept{
    /*  This template (T) method generates an array of random numbers. 
        
        Parameters:
            arr_size (size_t): size of the vector to be generated randomly.
            min (T): minimum value of the randomly generated vector.
            max (T): maximum value of the randomly generated vector.
        
        Returns:
            rand_num_container (std::vector< T >): vector containing the numbers generated randomly.
    */

    srand(time(0));
    std::vector< T > rand_num_container(arr_size);
    T random_number;
    for (size_t i=0; i<arr_size; i++){
        random_number = min + (T)( ( (T)rand() ) / ( (T)RAND_MAX ) * (max - min) );
        rand_num_container[i] = random_number;
    }
    return rand_num_container;
}