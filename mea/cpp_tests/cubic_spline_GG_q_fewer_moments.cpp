#include "../../src/IPT2nd3rdorderSingle2.hpp"
/*
Info HDF5:
https://support.hdfgroup.org/HDF5/doc/cpplus_RM/compound_8cpp-example.html
https://www.hdfgroup.org/downloads/hdf5/source-code/
*/

std::vector<double> get_iwn_to_tau(const std::vector< std::complex<double> >& F_iwn, double beta, std::string obj="Green");
template<typename T> std::vector< T > generate_random_numbers(size_t arr_size, T min, T max) noexcept;
arma::Mat<double> get_derivative_FFT(arma::Mat< std::complex<double> > G_k_iwn, const std::vector< std::complex<double> >& iwn, const std::vector<double>& k_arr, const std::vector<double>& beta_arr, double U, double mu, double qx=0.0, std::string opt="positive", size_t ii=0, double qy=0.0);
inline double velocity(double k) noexcept;
inline void sum_rule_iqn_0(double n_k, double k, double& sum_rule_total, unsigned int N_k) noexcept;
template<typename T> inline T eps(T);
template<typename T, typename... Ts> inline T eps(T,Ts...);


int main(void){
    
    std::string inputFilename("../../data/1D_U_8.000000_beta_7.000000_n_0.500000_N_tau_1024/Self_energy_1D_U_8.000000_beta_7.000000_n_0.500000_N_tau_1024_Nit_25.dat");
    // Choose whether current-current or spin-spin correlation function is computed.
    std::vector<std::string> results;
    std::vector<std::string> fetches = {"U", "beta", "N_tau"};
    
    results = get_info_from_filename(inputFilename,fetches);
    const bool is_jj = false; 
    const unsigned int Ntau = 2*(unsigned int)atof(results[2].c_str());
    const unsigned int N_k = 501;
    const unsigned int N_q = 21;
    const double beta = atof(results[1].c_str());
    const double U = atof(results[0].c_str());
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
    std::string filename(std::string("bb_")+std::to_string(DIM)+std::string("D_U_")+std::to_string(U)+std::string("_beta_")+std::to_string(beta)+std::string("_Ntau_")+std::to_string(Ntau)+std::string("_Nk_")+std::to_string(N_k)+std::string("_isjj_")+std::to_string(is_jj)+std::string(".hdf5"));
    const H5std_string FILE_NAME( filename );
    const unsigned int DATA_SET_DIM = Ntau;
    H5::H5File* file = new H5::H5File( FILE_NAME, H5F_ACC_TRUNC );
    // Getting the data
    FileData dataFromFile;
    dataFromFile = get_data(inputFilename,Ntau);
    
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
    #if DIM == 1
    arma::Mat< std::complex<double> > G_k_iwn(k_array.size(),iwn.size());
    for (size_t l=0; l<k_array.size(); l++){
        for (size_t j=0; j<iwn.size(); j++){
            G_k_iwn(l,j) = 1.0/( iwn[j] + mu - epsilonk(k_array[l]) - sigma_iwn[j] );
        }
    }
    #elif DIM == 2
    arma::Cube< std::complex<double> > G_k_iwn(k_array.size(),iwn.size(),k_array.size());
    for (size_t l=0; l<k_array.size(); l++){
        for (size_t m=0; m<k_array.size(); m++){
            for (size_t j=0; j<iwn.size(); j++){
                G_k_iwn(l,j,m) = 1.0/( iwn[j] + mu - epsilonk(k_array[l],k_array[m]) - sigma_iwn[j] );
            }
        }
    }
    #endif

    #if DIM == 1
    arma::Mat<double> dG_dtau_m_FFT;
    dG_dtau_m_FFT = get_derivative_FFT(G_k_iwn,iwn,k_array,beta_array,U,mu,0.0,std::string("negative"));

    /* TEST dG(-tau)/dtau */
    std::ofstream test1("test_1_fewer_moments.dat", std::ios::out);
    for (size_t j=0; j<beta_array.size(); j++){
        test1 << beta_array[j] << "  " << dG_dtau_m_FFT(52,j) << "\n";
    }
    test1.close();
    #elif DIM == 2
    arma::Cube<double> dG_dtau_m_FFT(k_array.size(),Ntau+1,k_array.size()); // (k,tau,k)
    for (size_t ii=0; ii<k_array.size(); ii++){
        dG_dtau_m_FFT.slice(ii) = get_derivative_FFT(G_k_iwn.slice(ii),iwn,k_array,beta_array,U,mu,0.0,std::string("negative"),ii,0.0);
    }
    
    /* TEST dG(-tau)/dtau */
    std::ofstream test1("test_1_fewer_moments_2D.dat", std::ios::out);
    for (size_t j=0; j<beta_array.size(); j++){
        test1 << beta_array[j] << "  " << dG_dtau_m_FFT(50,j,50) << "\n"; // (k,tau,k)
    }
    test1.close();
    #endif
    
    // Substracting the tail of the Green's function
    #if DIM == 1
    for (size_t l=0; l<k_array.size(); l++){
        for (size_t j=0; j<iwn.size(); j++){
            G_k_iwn(l,j) -= 1.0/(iwn[j]) + epsilonk(k_array[l])/iwn[j]/iwn[j]; //+ ( U*U/4.0 + epsilonk(k_array[l])*epsilonk(k_array[l]) )/iwn[j]/iwn[j]/iwn[j];
        }
    }
    #elif DIM == 2
    for (size_t l=0; l<k_array.size(); l++){
        for (size_t m=0; m<k_array.size(); m++){
            for (size_t j=0; j<iwn.size(); j++){
                G_k_iwn(l,j,m) -= 1.0/(iwn[j]); //+ epsilonk(k_array[l])/iwn[j]/iwn[j]; //+ ( U*U/4.0 + epsilonk(k_array[l])*epsilonk(k_array[l]) )/iwn[j]/iwn[j]/iwn[j];
            }
        }
    }
    #endif
    // FFT of G(iwn) --> G(tau)
    #if DIM == 1
    arma::Mat<double> G_k_tau(k_array.size(),Ntau+1);
    std::vector<double> FFT_k_tau;
    double sum_rule_val = 0.0;
    for (size_t l=0; l<k_array.size(); l++){
        std::vector< std::complex<double> > G_iwn_k_slice(G_k_iwn(l,arma::span::all).begin(),G_k_iwn(l,arma::span::all).end());
        FFT_k_tau = get_iwn_to_tau(G_iwn_k_slice,beta); // beta_arr.back() is beta
        for (size_t i=0; i<beta_array.size(); i++){
            G_k_tau(l,i) = FFT_k_tau[i] - 0.5 - 0.25*(beta-2.0*beta_array[i])*epsilonk(k_array[l]); //+ 0.25*beta_array[i]*(beta-beta_array[i])*(U*U/4.0 + epsilonk(k_array[l])*epsilonk(k_array[l]));
        }
        sum_rule_iqn_0(-1.0*G_k_tau(l,Ntau),k_array[l],sum_rule_val,N_k);
    }

    if (is_jj)
        std::cout << "The sum rule gives: " << sum_rule_val << "\n";

    /* TEST G(-tau) */
    std::ofstream test2("test_2_fewer_moments.dat", std::ios::out);
    for (size_t j=0; j<beta_array.size(); j++){
        test2 << beta_array[j] << "  " << -1.0*G_k_tau(3,Ntau-j) << "\n";
    }
    test2.close();
    #elif DIM == 2
    
    arma::Cube<double> G_k_tau(k_array.size(),Ntau+1,k_array.size());
    std::vector<double> FFT_k_tau;
    double sum_rule_val = 0.0;
    for (size_t l=0; l<k_array.size(); l++){
        for (size_t m=0; m<k_array.size(); m++){
            std::vector< std::complex<double> > G_iwn_k_slice(G_k_iwn(arma::span(l,l),arma::span::all,arma::span(m,m)).begin(),G_k_iwn(arma::span(l,l),arma::span::all,arma::span(m,m)).end());
            FFT_k_tau = get_iwn_to_tau(G_iwn_k_slice,beta); // beta_arr.back() is beta
            for (size_t i=0; i<beta_array.size(); i++){
                G_k_tau(l,i,m) = FFT_k_tau[i] - 0.5; //- 0.25*(beta-2.0*beta_array[i])*epsilonk(k_array[l]); //+ 0.25*beta_array[i]*(beta-beta_array[i])*(U*U/4.0 + epsilonk(k_array[l])*epsilonk(k_array[l]));
            }
            sum_rule_iqn_0(-1.0*G_k_tau(l,Ntau,m),k_array[l],sum_rule_val,N_k*N_k); // Homogeneous, so could be either m or l
        }
    }
    std::cout << "The sum rule gives: " << sum_rule_val << "\n";

    /* TEST G(-tau) */
    std::ofstream test2("test_2_fewer_moments_2D.dat", std::ios::out);
    for (size_t j=0; j<beta_array.size(); j++){
        test2 << beta_array[j] << "  " << -1.0*G_k_tau(50,Ntau-j,50) << "\n";
    }
    test2.close();
    #endif
    
    // Should be a loop over the external momentum from this point on...
    #if DIM == 1
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
                G_k_q_iwn(l,j) = 1.0/( iwn[j] + mu - epsilonk(k_array[l]+q_array[em]) - sigma_iwn[j] );
            }
        }

        arma::Mat<double> dG_dtau_FFT_q;
        dG_dtau_FFT_q = get_derivative_FFT(G_k_q_iwn,iwn,k_array,beta_array,U,mu,q_array[em]);

        // Substracting the tail of the Green's function
        for (size_t l=0; l<k_array.size(); l++){
            for (size_t j=0; j<iwn.size(); j++){
                G_k_q_iwn(l,j) -= 1.0/(iwn[j]) + epsilonk(k_array[l]+q_array[em])/iwn[j]/iwn[j]; //+ ( U*U/4.0 + epsilonk(k_array[l]+q_array[em])*epsilonk(k_array[l]+q_array[em]) )/iwn[j]/iwn[j]/iwn[j];
            }
        }
        // FFT of G(iwn) --> G(tau)
        for (size_t l=0; l<k_array.size(); l++){
            std::vector< std::complex<double> > G_iwn_k_q_slice(G_k_q_iwn(l,arma::span::all).begin(),G_k_q_iwn(l,arma::span::all).end());
            FFT_k_q_tau = get_iwn_to_tau(G_iwn_k_q_slice,beta); // beta_arr.back() is beta
            for (size_t i=0; i<beta_array.size(); i++){
                G_k_q_tau(l,i) = FFT_k_q_tau[i] - 0.5 - 0.25*(beta-2.0*beta_array[i])*epsilonk(k_array[l]+q_array[em]); //+ 0.25*beta_array[i]*(beta-beta_array[i])*( U*U/4.0 + epsilonk(k_array[l]+q_array[em])*epsilonk(k_array[l]+q_array[em]) );
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

            std::vector< std::complex<double> > cub_spl_GG(splObj_GG.bosonic_corr(iqn,beta)); //move ctor
            for (size_t i=0; i<cub_spl_GG.size(); i++){
                cubic_spline_GG_iqn_k(i,l) = cub_spl_GG[i];
            }
        }

        const Integrals integralsObj;
        std::vector< std::complex<double> > GG_iqn_q(iqn.size());
        const double delta = 2.0*M_PI/(double)(N_k-1);
        for (size_t j=0; j<iqn.size(); j++){
            std::vector< std::complex<double> > GG_iqn_k_tmp(cubic_spline_GG_iqn_k(j,arma::span::all).begin(),cubic_spline_GG_iqn_k(j,arma::span::all).end());
            GG_iqn_q[j] = 1.0/(2.0*M_PI)*integralsObj.I1D_VEC(std::move(GG_iqn_k_tmp),delta,"simpson");
        }

        std::string DATASET_NAME("q_"+std::to_string(q_array[em]));
        writeInHDF5File(GG_iqn_q, file, DATA_SET_DIM, DATASET_NAME);

    }
    #elif DIM == 2
    arma::Cube< std::complex<double> > G_k_q_iwn(k_array.size(),iwn.size(),k_array.size());
    arma::Cube<double> G_k_q_tau(k_array.size(),Ntau+1,k_array.size()), dG_dtau_FFT_q(k_array.size(),Ntau+1,k_array.size());
    std::vector<double> FFT_k_q_tau;
    arma::Cube<double> dGG_tau_for_k(k_array.size(),beta_array.size(),k_array.size());
    arma::Cube< std::complex<double> > cubic_spline_GG_iqn_k(iqn.size(),k_array.size(),k_array.size());
    arma::Cube<double> GG_tau_for_k(2,2,beta_array.size());
    for (size_t emx=0; emx<q_array.size(); emx++){
        for (size_t emy=0; emy<q_array.size(); emy++){
            std::cout << "qx: " << q_array[emx] << " qy: " << q_array[emy] << "\n";
            // Building the lattice Green's function from the local DMFT self-energy
            for (size_t l=0; l<k_array.size(); l++){
                for (size_t m=0; m<k_array.size(); m++){
                    for (size_t j=0; j<iwn.size(); j++){
                        G_k_q_iwn(l,j,m) = 1.0/( iwn[j] + mu - epsilonk(k_array[l]+q_array[emx],k_array[m]+q_array[emy]) - sigma_iwn[j] );
                    }
                }
            }
            
            for (size_t mm=0; mm<k_array.size(); mm++){
                dG_dtau_FFT_q.slice(mm) = get_derivative_FFT(G_k_q_iwn.slice(mm),iwn,k_array,beta_array,U,mu,q_array[emx],std::string("positive"),mm,q_array[emy]);
            }

            // Substracting the tail of the Green's function
            for (size_t l=0; l<k_array.size(); l++){
                for (size_t j=0; j<iwn.size(); j++){
                    for (size_t m=0; m<k_array.size(); m++){
                        G_k_q_iwn(l,j,m) -= 1.0/(iwn[j]); //+ epsilonk(k_array[l]+q_array[em])/iwn[j]/iwn[j]; //+ ( U*U/4.0 + epsilonk(k_array[l]+q_array[em])*epsilonk(k_array[l]+q_array[em]) )/iwn[j]/iwn[j]/iwn[j];
                    }
                }
            }
            
            // FFT of G(iwn) --> G(tau)
            for (size_t l=0; l<k_array.size(); l++){
                for (size_t m=0; m<k_array.size(); m++){
                    std::vector< std::complex<double> > G_iwn_k_q_slice(G_k_q_iwn(arma::span(l,l),arma::span::all,arma::span(m,m)).begin(),G_k_q_iwn(arma::span(l,l),arma::span::all,arma::span(m,m)).end());
                    FFT_k_q_tau = get_iwn_to_tau(G_iwn_k_q_slice,beta); // beta_arr.back() is beta
                    for (size_t i=0; i<beta_array.size(); i++){
                        G_k_q_tau(l,i,m) = FFT_k_q_tau[i] - 0.5; //- 0.25*(beta-2.0*beta_array[i])*epsilonk(k_array[l]+q_array[em]); //+ 0.25*beta_array[i]*(beta-beta_array[i])*( U*U/4.0 + epsilonk(k_array[l]+q_array[em])*epsilonk(k_array[l]+q_array[em]) );
                    }
                }
            }

            for (size_t l=0; l<k_array.size(); l++){
                for (size_t m=0; m<k_array.size(); m++){
                    spline<double> splObj_GG;
                    if (is_jj){
                        for (size_t i=0; i<beta_array.size(); i++){
                            dGG_tau_for_k(l,i,m) = velocity(k_array[l])*velocity(k_array[l])*(-2.0)*( dG_dtau_m_FFT(l,i,m)*G_k_q_tau(l,i,m) - G_k_tau(l,Ntau-i,m)*dG_dtau_FFT_q(l,i,m) );
                            GG_tau_for_k.slice(i)(0,0) = velocity(k_array[l])*velocity(k_array[l])*(-2.0)*(-1.0)*( G_k_q_tau(l,i,m)*G_k_tau(l,Ntau-i,m) );
                        }
                    }else{
                        for (size_t i=0; i<beta_array.size(); i++){
                            dGG_tau_for_k(l,i,m) = (-2.0)*( dG_dtau_m_FFT(l,i,m)*G_k_q_tau(l,i,m) - G_k_tau(l,Ntau-i,m)*dG_dtau_FFT_q(l,i,m) );
                            GG_tau_for_k.slice(i)(0,0) = (-2.0)*(-1.0)*( G_k_q_tau(l,i,m)*G_k_tau(l,Ntau-i,m) );
                        }
                    }
                    
                    // Taking the derivative for boundary conditions
                    double left_der_GG = dGG_tau_for_k(l,0,m);
                    double right_der_GG = dGG_tau_for_k(l,Ntau,m);

                    splObj_GG.set_boundary(spline<double>::bd_type::first_deriv,left_der_GG,spline<double>::bd_type::first_deriv,right_der_GG);
                    splObj_GG.set_points(beta_array,GG_tau_for_k);

                    std::vector< std::complex<double> > cub_spl_GG = splObj_GG.bosonic_corr(iqn,beta);
                    for (size_t i=0; i<cub_spl_GG.size(); i++){
                        cubic_spline_GG_iqn_k(i,l,m) = cub_spl_GG[i];
                    }
                }
            }
            
            const Integrals integralsObj;
            std::vector< std::complex<double> > GG_iqn_q(iqn.size());
            arma::Mat< std::complex<double> > tmp_int_iqn_k(iqn.size(),k_array.size());
            const double delta = 2.0*M_PI/(double)(N_k-1);
            for (size_t j=0; j<iqn.size(); j++){
                for (size_t i=0; i<k_array.size(); i++){
                    std::vector< std::complex<double> > GG_iqn_k_tmp(cubic_spline_GG_iqn_k(arma::span(j,j),arma::span(i,i),arma::span::all).begin(),cubic_spline_GG_iqn_k(arma::span(j,j),arma::span(i,i),arma::span::all).end());
                    tmp_int_iqn_k(j,i) = 1.0/(2.0*M_PI)*integralsObj.I1D_CPLX(GG_iqn_k_tmp,delta);
                }
                std::vector< std::complex<double> > GG_iqn_tmp(tmp_int_iqn_k(j,arma::span::all).begin(),tmp_int_iqn_k(j,arma::span::all).end());
                GG_iqn_q[j] = 1.0/(2.0*M_PI)*integralsObj.I1D_CPLX(GG_iqn_tmp,delta);
            }

            std::string DATASET_NAME("qx_"+std::to_string(q_array[emx])+"_qy_"+std::to_string(q_array[emy]));
            writeInHDF5File(GG_iqn_q, file, DATA_SET_DIM, DATASET_NAME);
        }
    }
    #endif

    delete file;

}

template<typename T> inline T eps(T k){
    return -2.0*std::cos(k);
}

template<typename T, typename... Ts> inline T eps(T k,Ts... ks){
    return -2.0*std::cos(k) + eps(ks...);
}

inline void sum_rule_iqn_0(double n_k, double k, double& sum_rule_total, unsigned int N_k) noexcept{
    /*  This method computes the sum rule that determines the value of the optical conductivity at iqn=0. This sum rule reads:

        sum_rule = 1/N_k*sum_{k,sigma} d^2\epsilon_k/dk^2 <n_{k,sigma}>.
        
        Parameters:
            n_k (double): electron density at a given k-point.
            k (double): k-point.
            sum_rule_total (double&): total of sum rule, that is the expression above as final result after whole k-summation.
            N_k (unsigned int): Number of k-points spanning the original Brillouin zone.
        
        Returns:
            (void).
    */
    sum_rule_total += 1.0/N_k*4.0*std::cos(k)*n_k;
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

arma::Mat<double> get_derivative_FFT(arma::Mat< std::complex<double> > G_k_iwn, const std::vector< std::complex<double> >& iwn, const std::vector<double>& k_arr, const std::vector<double>& beta_arr, double U, double mu, double qx, std::string opt, size_t ii, double qy){
    /*  This method computes the derivatives of G(tau) (or G(-tau)) at boundaries for the cubic spline. Recall that only positive imaginary time
        is used, i.e 0 < tau < beta. It takes care of subtracting the leading moments for smoother results.
        
        Parameters:
            G_k_iwn (arma::Mat< std::complex<double> >): Green's function mesh over (k,iwn)-space.
            iwn (const std::vector< std::complex<double> >&): Fermionic Matsubara frequencies.
            k_arr (const std::vector<double>&): k-space vector.
            beta_arr (const std::vector<double>&): imaginary-time vector.
            U (double): Hubbard local interaction.
            mu (double): chemical potential.
            q (double): incoming momentum. Defaults q=0.0.
            opt (std::string): positive or negative imaginary-time Green's function derivated. Takes in "positive" of "negative". Defaults opt="postitive".
        
        Returns:
            dG_tau_for_k (arma::Mat<double>): imaginary-time Green's function derivative mesh over (k,tau)-space.
    */
    arma::Mat<double> dG_tau_for_k(k_arr.size(),beta_arr.size());
    const double beta = beta_arr.back();
    // Subtracting the leading moments of the Green's function...
    std::complex<double> moments;
    for (size_t j=0; j<iwn.size(); j++){
        for (size_t l=0; l<k_arr.size(); l++){
            #if DIM == 1
            moments = 1.0/iwn[j] + epsilonk(k_arr[l]+qx)/iwn[j]/iwn[j] + ( U*U/4.0 + epsilonk(k_arr[l]+qx)*epsilonk(k_arr[l]+qx) )/iwn[j]/iwn[j]/iwn[j];
            #elif DIM == 2
            moments = 1.0/iwn[j] + epsilonk(k_arr[l]+qx,k_arr[ii]+qy)/iwn[j]/iwn[j] + ( U*U/4.0 + epsilonk(k_arr[l]+qx,k_arr[ii]+qy)*epsilonk(k_arr[l]+qx,k_arr[ii]+qy) )/iwn[j]/iwn[j]/iwn[j];
            #endif
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
                #if DIM == 1
                dG_tau_for_k(l,i) = FFT_iwn_tau[i] + 0.5*epsilonk(k_arr[l]+qx) + 0.5*( beta_arr[i] - beta/2.0 )*( U*U/4.0 + epsilonk(k_arr[l]+qx)*epsilonk(k_arr[l]+qx) );
                #elif DIM == 2
                dG_tau_for_k(l,i) = FFT_iwn_tau[i] + 0.5*epsilonk(k_arr[l]+qx,k_arr[ii]+qy) + 0.5*( beta_arr[i] - beta/2.0 )*( U*U/4.0 + epsilonk(k_arr[l]+qx,k_arr[ii]+qy)*epsilonk(k_arr[l]+qx,k_arr[ii]+qy) );
                #endif
            } else if (opt.compare("positive")==0){
                #if DIM == 1 
                dG_tau_for_k(l,i) = FFT_iwn_tau[i] + 0.5*epsilonk(k_arr[l]+qx) + 0.5*( beta/2.0 - beta_arr[i] )*( U*U/4.0 + epsilonk(k_arr[l]+qx)*epsilonk(k_arr[l]+qx) );
                #elif DIM == 2
                dG_tau_for_k(l,i) = FFT_iwn_tau[i] + 0.5*epsilonk(k_arr[l]+qx,k_arr[ii]+qy) + 0.5*( beta/2.0 - beta_arr[i] )*( U*U/4.0 + epsilonk(k_arr[l]+qx,k_arr[ii]+qy)*epsilonk(k_arr[l]+qx,k_arr[ii]+qy) );
                #endif
            }
        }
        #if DIM == 1
        dG_tau_for_k(l,arma::span::all)(beta_arr.size()-1) = epsilonk(k_arr[l]+qx)-mu+U*0.5 - 1.0*dG_tau_for_k(l,0); // Assumed half-filling
        #elif DIM == 2
        dG_tau_for_k(l,arma::span::all)(beta_arr.size()-1) = epsilonk(k_arr[l]+qx,k_arr[ii]+qy)-mu+U*0.5 - 1.0*dG_tau_for_k(l,0); // Assumed half-filling 
        #endif
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
