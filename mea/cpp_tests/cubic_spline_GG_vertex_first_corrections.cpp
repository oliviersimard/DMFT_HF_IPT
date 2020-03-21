#include "../../src/IPT2nd3rdorderSingle2.hpp"
#include <ctime>
/*
Info HDF5:
https://support.hdfgroup.org/HDF5/doc/cpplus_RM/compound_8cpp-example.html
https://www.hdfgroup.org/downloads/hdf5/source-code/
*/

std::vector<double> get_iwn_to_tau(const std::vector< std::complex<double> >& F_iwn, double beta, std::string obj="Green");
template<typename T> std::vector< T > generate_random_numbers(size_t arr_size, T min, T max) noexcept;
arma::Mat<double> get_derivative_FFT(arma::Mat< std::complex<double> > G_k_iwn, const std::vector< std::complex<double> >& iwn, const std::vector<double>& k_arr, const std::vector<double>& beta_arr, double U, double mu, double q=0.0, std::string opt="positive");
inline double velocity(double k) noexcept;
std::complex<double> getGreen(double k, double mu, std::complex<double> iwn, IPT2::SplineInline< std::complex<double> >& splInlineobj);
template<typename T> T summ(std::vector< T >) noexcept;

int main(void){
    
    std::string inputFilename("../data/Self_energy_1D_U_10.000000_beta_50.000000_n_0.500000_N_tau_512_Nit_32.dat");
    std::string inputFilenameLoad("../data/Self_energy_1D_U_10.000000_beta_50.000000_n_0.500000_N_tau_1024");
    // Choose whether current-current or spin-spin correlation function is computed.
    const bool is_jj = true; 
    const unsigned int Ntau = 2*512;
    const unsigned int N_q = 500;
    const double beta = 50.0;
    const double U = 10.0;
    const double mu = U/2.0; // Half-filling
    // k_tilde and k_bar momenta
    const double k_tilde = -M_PI/2.0;
    const double k_bar = M_PI/2.0;
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
    std::string filename(std::string("bb_1D_U_")+std::to_string(U)+std::string("_beta_")+std::to_string(beta)+std::string("_Ntau_")+std::to_string(Ntau)+std::string("_Nk_")+std::to_string(N_q)+std::string("_isjj_")+std::to_string(is_jj)+std::string("_kbar_")+std::to_string(k_bar)+std::string("_ktilde_")+std::to_string(k_tilde)+std::string("_1_moment.hdf5"));
    const H5std_string FILE_NAME( filename );
    const unsigned int DATA_SET_DIM = Ntau;
    
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
    IPT2::SplineInline< std::complex<double> > splInlineObj(Ntau,initVec,q_tilde_array,iwn);
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
            G_k_bar_q_tilde_iwn(l,n_bar) -= 1.0/(iwn[n_bar]); //+ epsilonk(q_tilde_array[l])/iwn[n_bar]/iwn[n_bar]; //+ ( U*U/4.0 + epsilonk(q_tilde_array[l])*epsilonk(q_tilde_array[l]) )/iwn[j]/iwn[j]/iwn[j];
        }
    }

    for (size_t n_tilde=0; n_tilde<iwn.size(); n_tilde++){
        // Building the lattice Green's function from the local DMFT self-energy
        for (size_t l=0; l<q_tilde_array.size(); l++){
            G_k_tilde_q_tilde_iwn(l,n_tilde) = 1.0/( ( iwn[n_tilde] ) + mu - epsilonk(k_tilde-q_tilde_array[l]) - sigma_iwn[n_tilde] );
        }
    }
    arma::Mat<double> dG_dtau_FFT_k_tilde = get_derivative_FFT(G_k_tilde_q_tilde_iwn,iwn,q_tilde_array,beta_array,U,mu,k_tilde);

    for (size_t n_tilde=0; n_tilde<iwn.size(); n_tilde++){
        for (size_t l=0; l<q_tilde_array.size(); l++){
            // Substracting the tail of the Green's function
            G_k_tilde_q_tilde_iwn(l,n_tilde) -= 1.0/(iwn[n_tilde]); //+ epsilonk(q_tilde_array[l])/iwn[n_tilde]/iwn[n_tilde]; //+ ( U*U/4.0 + epsilonk(q_tilde_array[l])*epsilonk(q_tilde_array[l]) )/iwn[j]/iwn[j]/iwn[j];
        }
    }

    /* TEST dG(-tau)/dtau */
    std::ofstream test1("test_1_corr.dat", std::ios::out);
    for (size_t j=0; j<beta_array.size(); j++){
        test1 << beta_array[j] << "  " << dG_dtau_m_FFT_k_bar(0,j) << "\n";
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
        test2 << beta_array[j] << "  " << -1.0*G_k_bar_q_tilde_tau(0,Ntau-j) << "\n";
    }
    test2.close();

    arma::Cube< std::complex<double> > cub_spl_GG_n_bar_vs_n_tilde(iwn.size(),iwn.size(),q_tilde_array.size());
    for (size_t l=0; l<q_tilde_array.size(); l++){
        std::cout << "q_tilde: " << l << "\n";
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

    // Computing Gamma
    const Integrals integralsObj;
    constexpr double delta = 2.0*M_PI/(double)(N_q-1);
    arma::Mat< std::complex<double> > Gamma_n_bar_n_tilde(iwn.size(),iwn.size()); // Doesn't depend on iq_n
    for (size_t n_bar=0; n_bar<iwn.size(); n_bar++){
        for (size_t n_tilde=0; n_tilde<iwn.size(); n_tilde++){
            std::vector< std::complex<double> > GG_n_bar_n_tilde_k_tmp(cub_spl_GG_n_bar_vs_n_tilde(arma::span(n_bar,n_bar),arma::span(n_tilde,n_tilde),arma::span::all).begin(),cub_spl_GG_n_bar_vs_n_tilde(arma::span(n_bar,n_bar),arma::span(n_tilde,n_tilde),arma::span::all).end());
            // This part remains to be done....
            Gamma_n_bar_n_tilde(n_bar,n_tilde) = ( U / ( 1.0 + U/(2.0*M_PI)*integralsObj.I1D_CPLX(GG_n_bar_n_tilde_k_tmp,delta) ) );
            //Gamma_n_bar_n_tilde(n_bar,n_tilde) = ( U / ( 1.0 + U/(N_q)*summ(GG_n_bar_n_tilde_k_tmp) ) );
        }
    }

    arma::Mat< std::complex<double> > GG_n_bar_n_tilde(iwn.size(),iwn.size());
    for (size_t em=0; em<iqn.size(); em++){
        clock_t begin = clock();
        for (size_t n_bar=0; n_bar<iwn.size(); n_bar++){
            for (size_t n_tilde=0; n_tilde<iwn.size(); n_tilde++){
                // This part remains to be done....
                GG_n_bar_n_tilde(n_bar,n_tilde) = getGreen(k_tilde,mu,iwn[n_tilde],splInlineObj)*getGreen(k_tilde+qq,mu,iwn[n_tilde]+iqn[em],splInlineObj)*Gamma_n_bar_n_tilde(n_bar,n_tilde)*getGreen(k_bar,mu,iwn[n_bar],splInlineObj)*getGreen(k_bar+qq,mu,iwn[n_bar]+iqn[em],splInlineObj);
            }
        }
        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "outer loop em: " << em << " done in " << elapsed_secs << " secs.." << "\n";
        cubic_spline_GG_iqn[em] = (1.0/beta/beta)*arma::accu(GG_n_bar_n_tilde); // summing over the internal ikn_tilde and ikn_bar
    }
    std::cout << "After the loop.." << std::endl;

    std::string DATASET_NAME("q_"+std::to_string(qq));
    writeInHDF5File(cubic_spline_GG_iqn, file, DATA_SET_DIM, DATASET_NAME);

    delete file;

    return 0;

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

template<typename T> 
T summ(std::vector< T > vec) noexcept{
    T tot {0.0};
    for (size_t l=0; l<vec.size(); l++){
        if ( (l==0) || (l==(vec.size()-1)) ){
            tot+=0.5*vec[l];
        } else{
            tot+=vec[l];
        }
    }

    return tot;
}