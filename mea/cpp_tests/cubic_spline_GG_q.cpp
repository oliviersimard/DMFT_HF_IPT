#include "../../src/tridiagonal.hpp"
#include "../../src/integral_utils.hpp"
#include <fftw3.h>
#include <fstream>
#include "H5Cpp.h" 
/*
Info HDF5:
https://support.hdfgroup.org/HDF5/doc/cpplus_RM/compound_8cpp-example.html
https://www.hdfgroup.org/downloads/hdf5/source-code/
*/

struct FileData;

std::vector<double> get_derivative(std::vector<double> x_arr, std::vector<double> y_arr);
FileData get_data(const std::string& strName, const unsigned int& Ntau);
std::vector<double> get_iwn_to_tau(const std::vector< std::complex<double> >& F_iwn, double beta, std::string obj="Green");
template<typename T> std::vector< T > generate_random_numbers(size_t arr_size, T min, T max);
arma::Mat<double> get_derivative_FFT(arma::Mat< std::complex<double> > G_k_iwn, const std::vector< std::complex<double> >& iwn, const std::vector<double>& k_arr, const std::vector<double>& beta_arr, double U, double mu, double q=0.0, std::string opt="positive");
double velocity(double k);
std::complex<double> I1D_CPLX(std::vector< std::complex<double> >& vecfunct,double delta);

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

    std::string inputFilename("../data/Self_energy_1D_U_2.000000_beta_80.000000_n_0.500000_N_tau_4096_Nit_5.dat");
    // Choose whether current-current or spin-spin correlation function is computed.
    const bool is_jj = true; 
    const unsigned int Ntau = 2*4096;
    const unsigned int N_k = 600;
    const unsigned int N_q = 51;
    const double beta = 80.0;
    const double U = 2.0;
    const double mu = U/2.0; // Half-filling
    spline<double> splObj;
    // beta array constructed
    std::vector<double> beta_array;
    for (size_t j=0; j<=Ntau; j++){
        double beta_tmp = j*beta/(Ntau);
        beta_array.push_back(beta_tmp);
    }
    // k_array constructed
    std::vector<double> k_array;
    for (size_t l=0; l<N_k; l++){
        double k_tmp = l*2.0*M_PI/(double)(N_k-1);
        k_array.push_back(k_tmp);
    }
    // q_array constructed
    std::vector<double> q_array;
    for (size_t l=0; l<N_q; l++){
        double q_tmp = l*2.0*M_PI/(double)(N_q-1);
        q_array.push_back(q_tmp);
    }
    // HDF5 business
    std::string filename(std::string("bb_1D_U_")+std::to_string(U)+std::string("_beta_")+std::to_string(beta)+std::string("_Ntau_")+std::to_string(Ntau)+std::string("_Nk_")+std::to_string(N_k)+std::string(".hdf5"));
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

    // Building the lattice Green's function from the local DMFT self-energy
    arma::Mat< std::complex<double> > G_k_iwn(k_array.size(),iwn.size());
    for (size_t l=0; l<k_array.size(); l++){
        for (size_t j=0; j<iwn.size(); j++){
            G_k_iwn(l,j) = 1.0/( iwn[j] + mu - epsilonk(k_array[l]) - sigma_iwn[j] );
        }
    }

    arma::Mat<double> dG_dtau_m_FFT = get_derivative_FFT(G_k_iwn,iwn,k_array,beta_array,U,mu,0.0,std::string("negative"));
    
    // Substracting the tail of the Green's function
    for (size_t l=0; l<k_array.size(); l++){
        for (size_t j=0; j<iwn.size(); j++){
            G_k_iwn(l,j) -= 1.0/(iwn[j]) + epsilonk(k_array[l])/iwn[j]/iwn[j] + ( U*U/4.0 + epsilonk(k_array[l])*epsilonk(k_array[l]) )/iwn[j]/iwn[j]/iwn[j];
        }
    }
    // FFT of G(iwn) --> G(tau)
    arma::Mat<double> G_k_tau(k_array.size(),Ntau+1);
    std::vector<double> FFT_k_tau;
    for (size_t l=0; l<k_array.size(); l++){
        std::vector< std::complex<double> > G_iwn_k_slice(G_k_iwn(l,arma::span::all).begin(),G_k_iwn(l,arma::span::all).end());
        FFT_k_tau = get_iwn_to_tau(G_iwn_k_slice,beta); // beta_arr.back() is beta
        for (size_t i=0; i<beta_array.size(); i++){
            G_k_tau(l,i) = FFT_k_tau[i] - 0.5 - 0.25*(beta-2.0*beta_array[i])*epsilonk(k_array[l]) + 0.25*beta_array[i]*(beta-beta_array[i])*(U*U/4.0 + epsilonk(k_array[l])*epsilonk(k_array[l]));
        }
    }

    // Should be a loop over the external momentum from this point on...
    arma::Mat< std::complex<double> > G_k_q_iwn(k_array.size(),iwn.size());
    arma::Mat<double> G_k_q_tau(k_array.size(),Ntau+1);
    std::vector<double> FFT_k_q_tau;
    arma::Mat<double> dGG_tau_for_k(k_array.size(),beta_array.size());
    arma::Mat< std::complex<double> > cubic_spline_GG_iqn_k(iqn.size(),k_array.size());
    arma::Cube<double> GG_tau_for_k(2,2,beta_array.size());
    for (size_t em=0; em<q_array.size(); em++){
        std::cout << "q: " << q_array[em] << "\n";
        // Building the lattice Green's function from the local DMFT self-energy
        for (size_t l=0; l<k_array.size(); l++){
            for (size_t j=0; j<iwn.size(); j++){
                G_k_q_iwn(l,j) = 1.0/( iwn[j] + mu - epsilonk(k_array[l]+q_array[em]) -sigma_iwn[j] );
            }
        }

        arma::Mat<double> dG_dtau_FFT_q = get_derivative_FFT(G_k_q_iwn,iwn,k_array,beta_array,U,mu,q_array[em]);

        // Substracting the tail of the Green's function
        for (size_t l=0; l<k_array.size(); l++){
            for (size_t j=0; j<iwn.size(); j++){
                G_k_q_iwn(l,j) -= 1.0/(iwn[j]) + epsilonk(k_array[l]+q_array[em])/iwn[j]/iwn[j] + ( U*U/4.0 + epsilonk(k_array[l]+q_array[em])*epsilonk(k_array[l]+q_array[em]) )/iwn[j]/iwn[j]/iwn[j];
            }
        }
        // FFT of G(iwn) --> G(tau)
        for (size_t l=0; l<k_array.size(); l++){
            std::vector< std::complex<double> > G_iwn_k_q_slice(G_k_q_iwn(l,arma::span::all).begin(),G_k_q_iwn(l,arma::span::all).end());
            FFT_k_q_tau = get_iwn_to_tau(G_iwn_k_q_slice,beta); // beta_arr.back() is beta
            for (size_t i=0; i<beta_array.size(); i++){
                G_k_q_tau(l,i) = FFT_k_q_tau[i] - 0.5 - 0.25*(beta-2.0*beta_array[i])*epsilonk(k_array[l]+q_array[em]) + 0.25*beta_array[i]*(beta-beta_array[i])*( U*U/4.0 + epsilonk(k_array[l]+q_array[em])*epsilonk(k_array[l]+q_array[em]) );
            }
        }

        for (size_t l=0; l<k_array.size(); l++){
            spline<double> splObj_GG;
            if (is_jj){
                for (size_t i=0; i<beta_array.size(); i++){
                    dGG_tau_for_k(l,i) = velocity(k_array[l])*velocity(k_array[l])*(-2.0)*( dG_dtau_m_FFT(l,i)*G_k_q_tau(l,i) - G_k_tau(l,Ntau-i)*dG_dtau_FFT_q(l,i) );
                    GG_tau_for_k.slice(i)(0,0) = velocity(k_array[l])*velocity(k_array[l])*(-2.0)*(-1.0)*( G_k_q_tau(l,i)*G_k_tau(l,Ntau-i) );
                }
            }else{
                for (size_t i=0; i<beta_array.size(); i++){
                    dGG_tau_for_k(l,i) = (-2.0)*( dG_dtau_m_FFT(l,i)*G_k_q_tau(l,i) - G_k_tau(l,Ntau-i)*dG_dtau_FFT_q(l,i) );
                    GG_tau_for_k.slice(i)(0,0) = (-2.0)*(-1.0)*( G_k_q_tau(l,i)*G_k_tau(l,Ntau-i) );
                }
            }

            // Taking the derivative for boundary conditions
            double left_der_GG = dGG_tau_for_k(l,0);
            double right_der_GG = dGG_tau_for_k(l,Ntau);

            splObj_GG.set_boundary(spline<double>::bd_type::first_deriv,left_der_GG,spline<double>::bd_type::first_deriv,right_der_GG);
            splObj_GG.set_points(beta_array,GG_tau_for_k);

            std::vector< std::complex<double> > cub_spl_GG = splObj_GG.bosonic_corr(iqn,beta);
            for (size_t i=0; i<cub_spl_GG.size(); i++){
                cubic_spline_GG_iqn_k(i,l) = cub_spl_GG[i];
            }
        }

        const Integrals integralsObj;
        std::vector< std::complex<double> > GG_iqn_q(iqn.size());
        const double delta = 2.0*M_PI/(double)(N_k-1);
        for (size_t j=0; j<iqn.size(); j++){
            std::vector< std::complex<double> > GG_iqn_k_tmp(cubic_spline_GG_iqn_k(j,arma::span::all).begin(),cubic_spline_GG_iqn_k(j,arma::span::all).end());
            GG_iqn_q[j] = 1.0/(2.0*M_PI)*integralsObj.I1D_CPLX(GG_iqn_k_tmp,delta);
        }

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
            std::string DATASET_NAME("q_"+std::to_string(q_array[em]));
            H5::DataSet* dataset;
            // dataset = new H5::DataSet(file->createDataSet(DATASET_NAME, std_cmplx_type, dataspace));
            dataset = new H5::DataSet(file->createDataSet(DATASET_NAME, mCmplx_type, dataspace));
            // Write data to dataset
            dataset->write( custom_cplx_GG_iqn_q.data(), mCmplx_type );

            delete dataset;

        } catch( H5::FileIException err ){
            err.printErrorStack();
            return -1;
        }
        // catch failure caused by the DataSet operations
        catch( H5::DataSetIException error ){
            error.printErrorStack();
            return -1;
        }
        // catch failure caused by the DataSpace operations
        catch( H5::DataSpaceIException error ){
            error.printErrorStack();
            return -1;
        }
        // catch failure caused by the DataSpace operations
        catch( H5::DataTypeIException error ){
            error.printErrorStack();
            return -1;
        }
    }

    delete file;

}

inline double velocity(double k){
    return 2.0*std::sin(k);
}

arma::Mat<double> get_derivative_FFT(arma::Mat< std::complex<double> > G_k_iwn, const std::vector< std::complex<double> >& iwn, const std::vector<double>& k_arr, const std::vector<double>& beta_arr, double U, double mu, double q, std::string opt){
    /*  This method computes the derivatives of G(tau) at boundaries for the cubic spline. Recall that only positive imaginary time
        is used, i.e 0 < tau < beta. It takes care of subtracting the leading moments for smoother results.
        
        Parameters:
            G_k_iwn (complex np.ndarray): Green's function mesh over (k,iwn)-space.
            funct_dispersion (function): dispersion relation of the tight-binding model.
            iwn_arr (complex np.array): Fermionic Matsubara frequencies.
            k_arr (float np.array): k-space array.
            U (float): Hubbard local interaction.
            beta (float): Inverse temperature.
            mu (float): chemical potential.
            q (float): incoming momentum. Defaults q=0.0.
            opt (str): positive or negative imaginary-time Green's function derivated. Takes in "positive" of "negative". 
            Defaults opt="postitive".
        
        Returns:
            dG_tau_for_k (float np.ndarray): imaginary-time Green's function derivative mesh over (k,iwn)-space.
    */
    
    arma::Mat<double> dG_tau_for_k(k_arr.size(),beta_arr.size());
    const double beta = beta_arr.back();
    // Subtracting the leading moments of the Green's function...
    std::complex<double> moments;
    for (size_t j=0; j<iwn.size(); j++){
        for (size_t l=0; l<k_arr.size(); l++){
            moments = 1.0/iwn[j] + epsilonk(k_arr[l]+q)/iwn[j]/iwn[j] + ( U*U/4.0 + epsilonk(k_arr[l]+q)*epsilonk(k_arr[l]+q) )/iwn[j]/iwn[j]/iwn[j];
            G_k_iwn(l,j) -= moments;
            if (opt.compare("negative")==0){
                G_k_iwn(l,j) = std::conj(G_k_iwn(l,j));
            }
            G_k_iwn(l,j) *= -1.0*iwn[j];
        }
    }
    // Calculating the imaginary-time derivative of the Green's function.
    std::vector<double> FFT_iwn_tau;
    for (size_t l=0; l<k_arr.size(); l++){
        std::vector< std::complex<double> > G_iwn_k_slice(G_k_iwn(l,arma::span::all).begin(),G_k_iwn(l,arma::span::all).end());
        FFT_iwn_tau = get_iwn_to_tau(G_iwn_k_slice,beta,std::string("Derivative")); // beta_arr.back() is beta
        for (size_t i=0; i<beta_arr.size()-1; i++){
            if (opt.compare("negative")==0){
                dG_tau_for_k(l,i) = FFT_iwn_tau[i] + 0.5*epsilonk(k_arr[l]+q) + 0.5*( beta_arr[i] - beta/2.0 )*( U*U/4.0 + epsilonk(k_arr[l]+q)*epsilonk(k_arr[l]+q) );
            } else if (opt.compare("positive")==0){
                dG_tau_for_k(l,i) = FFT_iwn_tau[i] + 0.5*epsilonk(k_arr[l]+q) + 0.5*( beta/2.0 - beta_arr[i] )*( U*U/4.0 + epsilonk(k_arr[l]+q)*epsilonk(k_arr[l]+q) );
            }
        }
        dG_tau_for_k(l,arma::span::all)(beta_arr.size()-1) = epsilonk(k_arr[l]+q)-mu+U*0.5 - 1.0*dG_tau_for_k(l,0); // Assumed half-filling
    }

    return dG_tau_for_k;
}


FileData get_data(const std::string& strName, const unsigned int& Ntau){

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
std::vector< T > generate_random_numbers(size_t arr_size, T min, T max){
    srand(time(0));
    std::vector< T > rand_num_container(arr_size);
    T random_number;
    for (size_t i=0; i<arr_size; i++){
        random_number = min + (T)( ( (T)rand() ) / ( (T)RAND_MAX ) * (max - min) );
        rand_num_container[i] = random_number;
    }
    return rand_num_container;
}

std::complex<double> I1D_CPLX(std::vector< std::complex<double> >& vecfunct,double delta){
    std::complex<double> result, resultSumNk(0.0,0.0);

    for (unsigned int i=1; i<vecfunct.size()-1; i++){
        resultSumNk+=vecfunct[i];
    }
    result = 0.5*delta*vecfunct[0]+0.5*delta*vecfunct.back()+delta*resultSumNk;
    
    return result;
}