#include "../../src/IPT2nd3rdorderSingle2.hpp"
/*
Info HDF5:
https://support.hdfgroup.org/HDF5/doc/cpplus_RM/compound_8cpp-example.html
https://www.hdfgroup.org/downloads/hdf5/source-code/
*/

std::vector<double> get_iwn_to_tau(const std::vector< std::complex<double> >& F_iwn, double beta, std::string obj="Green");
template<typename T> std::vector< T > generate_random_numbers(size_t arr_size, T min, T max) noexcept;
arma::Mat<double> get_derivative_FFT(arma::Mat< std::complex<double> > G_k_iwn, const std::vector< std::complex<double> >& iwn, const std::vector<double>& k_arr, const std::vector<double>& beta_arr, double U, double mu, double qx=0.0, std::string opt="positive", size_t ii=0, double qy=0.0);
std::vector<double> get_derivative_FFT(std::vector< std::complex<double> > G_k_iwn, const std::vector< std::complex<double> >& iwn, const std::vector<double>& beta_arr, double U, double mu, double kx, double qx, double ky=0.0, double qy=0.0, double kz=0.0, double qz=0.0, std::string opt="positive");
inline double velocity(double k) noexcept;
inline double sum_rule_iqn_0(double n_k, double k) noexcept;
template<typename T> inline T eps(T);
template<typename T, typename... Ts> inline T eps(T,Ts...);


int main(void){
    
    #ifndef NCA
    std::string inputFilename("../../build10/data/1D_U_3.000000_beta_6.500000_n_0.500000_N_tau_2048/Self_energy_1D_U_3.000000_beta_6.500000_n_0.500000_N_tau_2048_Nit_6.dat");
    #else
    std::string inputFilename("../../../NCA_OCA/data_2D_test_NCA_damping_0.000000/2D_U_14.000000_beta_3.500000_n_0.500000_Ntau_4096/SE_2D_NCA_AFM_U_14.000000_beta_3.500000_N_tau_4096_h_0.000000_Nit_111.dat");
    #endif
    // Choose whether current-current or spin-spin correlation function is computed.
    std::vector<std::string> results;
    std::vector<std::string> fetches = {"U", "beta", "N_tau"};
    
    results = get_info_from_filename(inputFilename,fetches);
    #ifndef NCA
    const unsigned int Ntau = 2*(unsigned int)atof(results[2].c_str());
    #else
    const unsigned int NCA_Ntau = 2*(unsigned int)atof(results[2].c_str());
    const unsigned int Ntau = 2*1024;
    #endif
    const unsigned int N_k = 301;
    const unsigned int N_q = 3;
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
    // q_array constructed for injected momentum
    std::vector<double> q_array;
    for (size_t l=0; l<N_q; l++){
        double q_tmp = l*2.0*M_PI/(double)(N_q-1);
        q_array.push_back(q_tmp);
    }
    // HDF5 business
    #ifndef NCA
    std::string filename(std::string("bb_")+std::to_string(DIM)+std::string("D_U_")+std::to_string(U)+std::string("_beta_")+std::to_string(beta)+std::string("_Ntau_")+std::to_string(Ntau)+std::string("_Nk_")+std::to_string(N_k)+".hdf5");
    #else
    std::string filename(std::string("bb_")+std::to_string(DIM)+std::string("D_U_")+std::to_string(U)+std::string("_beta_")+std::to_string(beta)+std::string("_Ntau_")+std::to_string(Ntau)+std::string("_Nk_")+std::to_string(N_k)+".hdf5");
    #endif
    const H5std_string FILE_NAME( filename );
    const unsigned int DATA_SET_DIM = Ntau;
    H5::H5File* file = new H5::H5File( FILE_NAME, H5F_ACC_TRUNC );
    // Getting the data
    FileData dataFromFile;
    #ifndef NCA
    dataFromFile = get_data(inputFilename,Ntau);
    #else
    dataFromFile = get_data(inputFilename,NCA_Ntau);
    #endif
    
    std::vector<double> wn = std::move(dataFromFile.iwn);
    std::vector<double> re = std::move(dataFromFile.re);
    std::vector<double> im = std::move(dataFromFile.im);

    std::vector< std::complex<double> > sigma_iwn(Ntau);
    std::vector< std::complex<double> > iwn(Ntau);
    #ifdef NCA
    try{
        if (wn.size()<Ntau)
            throw std::out_of_range("Problem in the truncation of the self-energy in loading process.");
        
        for (size_t i=0; i<Ntau; i++){
            sigma_iwn[i] = std::complex<double>(re[i+wn.size()/2-Ntau/2+1],im[i+wn.size()/2-Ntau/2+1]);
        }
    } catch(const std::out_of_range& err){
        std::cerr << err.what() << "\n";
        exit(1);
    }
    #else
    for (size_t i=0; i<wn.size(); i++){
        sigma_iwn[i]=std::complex<double>(re[i],im[i]);
    }
    #endif
    #ifndef NCA
    std::transform(wn.data()+wn.size()/2-Ntau/2,wn.data()+wn.size()/2+Ntau/2,iwn.data(),[](double d){ return std::complex<double>(0.0,d); });
    #else
    std::transform(wn.data()+wn.size()/2-Ntau/2+1,wn.data()+wn.size()/2+Ntau/2+1,iwn.data(),[](double d){ return std::complex<double>(0.0,d); });
    #endif

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
        test1 << beta_array[j] << "  " << dG_dtau_m_FFT(2,j) << "\n";
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
    const Integrals integralsObj;
    const double delta = 2.0*M_PI/(double)(N_k-1);
    arma::Mat<double> G_k_tau(k_array.size(),Ntau+1);
    std::vector<double> FFT_k_tau, sum_rule_kx(k_array.size());
    for (size_t l=0; l<k_array.size(); l++){
        std::vector< std::complex<double> > G_iwn_k_slice(G_k_iwn(l,arma::span::all).begin(),G_k_iwn(l,arma::span::all).end());
        FFT_k_tau = get_iwn_to_tau(G_iwn_k_slice,beta); // beta_arr.back() is beta
        for (size_t i=0; i<beta_array.size(); i++){
            G_k_tau(l,i) = FFT_k_tau[i] - 0.5 - 0.25*(beta-2.0*beta_array[i])*epsilonk(k_array[l]); //+ 0.25*beta_array[i]*(beta-beta_array[i])*(U*U/4.0 + epsilonk(k_array[l])*epsilonk(k_array[l]));
        }
        sum_rule_kx[l] = sum_rule_iqn_0(-1.0*G_k_tau(l,Ntau),k_array[l]);
    }
    double sum_rule_val = 2.0/(2.0*M_PI)*integralsObj.I1D_VEC(sum_rule_kx,delta,"simpson");

    std::cout << "The sum rule gives: " << sum_rule_val << "\n";

    /* TEST G(-tau) */
    std::ofstream test2("test_2_fewer_moments.dat", std::ios::out);
    for (size_t j=0; j<beta_array.size(); j++){
        test2 << beta_array[j] << "  " << -1.0*G_k_tau(3,Ntau-j) << "\n";
    }
    test2.close();
    #elif DIM == 2
    const Integrals integralsObj;
    const double delta = 2.0*M_PI/(double)(N_k-1);
    arma::Cube<double> G_k_tau(k_array.size(),Ntau+1,k_array.size());
    std::vector<double> FFT_k_tau;
    std::vector<double> sum_rule_kx(k_array.size()), sum_rule_ky(k_array.size());
    for (size_t l=0; l<k_array.size(); l++){
        for (size_t m=0; m<k_array.size(); m++){
            std::vector< std::complex<double> > G_iwn_k_slice(G_k_iwn(arma::span(l,l),arma::span::all,arma::span(m,m)).begin(),G_k_iwn(arma::span(l,l),arma::span::all,arma::span(m,m)).end());
            FFT_k_tau = get_iwn_to_tau(G_iwn_k_slice,beta); // beta_arr.back() is beta
            for (size_t i=0; i<beta_array.size(); i++){
                G_k_tau(l,i,m) = FFT_k_tau[i] - 0.5; //- 0.25*(beta-2.0*beta_array[i])*epsilonk(k_array[l]); //+ 0.25*beta_array[i]*(beta-beta_array[i])*(U*U/4.0 + epsilonk(k_array[l])*epsilonk(k_array[l]));
            }
            sum_rule_ky[m] = sum_rule_iqn_0(-1.0*G_k_tau(l,Ntau,m),k_array[l]); // Homogeneous, so could be either m or l
        }
        sum_rule_kx[l] = integralsObj.I1D_VEC(sum_rule_ky,delta,"simpson");
    }
    double sum_rule_val = 2.0/(2.0*M_PI)/(2.0*M_PI)*integralsObj.I1D_VEC(sum_rule_kx,delta,"simpson");
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
    arma::Mat<double> G_k_q_tau(k_array.size(),Ntau+1),GG_tau(k_array.size(),Ntau+1);
    std::vector<double> FFT_k_q_tau;
    arma::Mat<double> dGG_tau_for_k_szsz(k_array.size(),beta_array.size()), dGG_tau_for_k_jj(k_array.size(),beta_array.size());
    arma::Mat< std::complex<double> > cubic_spline_GG_iqn_k_jj(iqn.size(),k_array.size()), cubic_spline_GG_iqn_k_szsz(iqn.size(),k_array.size());
    arma::Cube<double> GG_tau_for_k_szsz(2,2,beta_array.size()), GG_tau_for_k_jj(2,2,beta_array.size());
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
            spline<double> splObj_GG_jj, splObj_GG_szsz;
            
            for (size_t i=0; i<beta_array.size(); i++){
                dGG_tau_for_k_jj(l,i) = velocity(k_array[l])*velocity(k_array[l])*(-2.0)*( dG_dtau_m_FFT(l,i)*G_k_q_tau(l,i) - G_k_tau(l,Ntau-i)*dG_dtau_FFT_q(l,i) );
                dGG_tau_for_k_szsz(l,i) = (-2.0)*( dG_dtau_m_FFT(l,i)*G_k_q_tau(l,i) - G_k_tau(l,Ntau-i)*dG_dtau_FFT_q(l,i) );
                GG_tau_for_k_jj.slice(i)(0,0) = velocity(k_array[l])*velocity(k_array[l])*(-2.0)*(-1.0)*( G_k_q_tau(l,i)*G_k_tau(l,Ntau-i) );
                GG_tau_for_k_szsz.slice(i)(0,0) = (-2.0)*(-1.0)*( G_k_q_tau(l,i)*G_k_tau(l,Ntau-i) );
                // To output GG(tau)
                GG_tau(l,i) = GG_tau_for_k_jj.slice(i)(0,0);
            }

            // Taking the derivative for boundary conditions
            double left_der_GG_jj = dGG_tau_for_k_jj(l,0);
            double right_der_GG_jj = dGG_tau_for_k_jj(l,Ntau);
            double left_der_GG_szsz = dGG_tau_for_k_szsz(l,0);
            double right_der_GG_szsz = dGG_tau_for_k_szsz(l,Ntau);

            splObj_GG_jj.set_boundary(spline<double>::bd_type::first_deriv,left_der_GG_jj,spline<double>::bd_type::first_deriv,right_der_GG_jj);
            splObj_GG_jj.set_points(beta_array,GG_tau_for_k_jj);
            splObj_GG_szsz.set_boundary(spline<double>::bd_type::first_deriv,left_der_GG_szsz,spline<double>::bd_type::first_deriv,right_der_GG_szsz);
            splObj_GG_szsz.set_points(beta_array,GG_tau_for_k_szsz);

            std::vector< std::complex<double> > cub_spl_GG_jj(splObj_GG_jj.bosonic_corr(iqn,beta)); //move ctor
            std::vector< std::complex<double> > cub_spl_GG_szsz(splObj_GG_szsz.bosonic_corr(iqn,beta)); //move ctor
            for (size_t i=0; i<iqn.size(); i++){
                cubic_spline_GG_iqn_k_jj(i,l) = cub_spl_GG_jj[i];
                cubic_spline_GG_iqn_k_szsz(i,l) = cub_spl_GG_szsz[i];
            }
        }

        // To output GG(tau)
        if (em==0){
            std::ofstream outP("vGvG_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_q_"+std::to_string(q_array[em])+".dat",std::ios::out);
            double GG_integrated_for_tau;
            for (size_t i=0; i<beta_array.size(); i++){
                std::vector<double> GG_tmp_tau(GG_tau(arma::span::all,i).begin(),GG_tau(arma::span::all,i).end());
                GG_integrated_for_tau = 1.0/(2.0*M_PI)*integralsObj.I1D_VEC(std::move(GG_tmp_tau),delta,"simpson");
                outP << beta_array[i] << "  " << GG_integrated_for_tau << "\n";
            }   
            outP.close();
        }

        std::vector< std::complex<double> > GG_iqn_q_jj(iqn.size()), GG_iqn_q_szsz(iqn.size());
        for (size_t j=0; j<iqn.size(); j++){
            std::vector< std::complex<double> > GG_iqn_k_tmp_jj(cubic_spline_GG_iqn_k_jj(j,arma::span::all).begin(),cubic_spline_GG_iqn_k_jj(j,arma::span::all).end());
            std::vector< std::complex<double> > GG_iqn_k_tmp_szsz(cubic_spline_GG_iqn_k_szsz(j,arma::span::all).begin(),cubic_spline_GG_iqn_k_szsz(j,arma::span::all).end());
            GG_iqn_q_jj[j] = 1.0/(2.0*M_PI)*integralsObj.I1D_VEC(std::move(GG_iqn_k_tmp_jj),delta,"simpson");
            GG_iqn_q_szsz[j] = 1.0/(2.0*M_PI)*integralsObj.I1D_VEC(std::move(GG_iqn_k_tmp_szsz),delta,"simpson");
        }

        std::string DATASET_NAME("q_"+std::to_string(q_array[em]));
        writeInHDF5File(GG_iqn_q_jj, GG_iqn_q_szsz, file, DATA_SET_DIM, DATASET_NAME);

    }
    #elif DIM == 2
    arma::Cube< std::complex<double> > G_k_q_iwn(k_array.size(),iwn.size(),k_array.size());
    arma::Cube<double> G_k_q_tau(k_array.size(),Ntau+1,k_array.size()), dG_dtau_FFT_q(k_array.size(),Ntau+1,k_array.size());
    std::vector<double> FFT_k_q_tau;
    arma::Cube<double> dGG_tau_for_k_jj(k_array.size(),beta_array.size(),k_array.size()), dGG_tau_for_k_szsz(k_array.size(),beta_array.size(),k_array.size());
    arma::Cube< std::complex<double> > cubic_spline_GG_iqn_k_jj(iqn.size(),k_array.size(),k_array.size()), cubic_spline_GG_iqn_k_szsz(iqn.size(),k_array.size(),k_array.size());
    arma::Cube<double> GG_tau_for_k_jj(2,2,beta_array.size()), GG_tau_for_k_szsz(2,2,beta_array.size());
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
                        G_k_q_iwn(l,j,m) -= 1.0/(iwn[j]) + epsilonk(k_array[l]+q_array[emx],k_array[m]+q_array[emy])/iwn[j]/iwn[j]; //+ ( U*U/4.0 + epsilonk(k_array[l]+q_array[em])*epsilonk(k_array[l]+q_array[em]) )/iwn[j]/iwn[j]/iwn[j];
                    }
                }
            }
            
            // FFT of G(iwn) --> G(tau)
            for (size_t l=0; l<k_array.size(); l++){
                for (size_t m=0; m<k_array.size(); m++){
                    std::vector< std::complex<double> > G_iwn_k_q_slice(G_k_q_iwn(arma::span(l,l),arma::span::all,arma::span(m,m)).begin(),G_k_q_iwn(arma::span(l,l),arma::span::all,arma::span(m,m)).end());
                    FFT_k_q_tau = get_iwn_to_tau(G_iwn_k_q_slice,beta); // beta_arr.back() is beta
                    for (size_t i=0; i<beta_array.size(); i++){
                        G_k_q_tau(l,i,m) = FFT_k_q_tau[i] - 0.5 - 0.25*(beta-2.0*beta_array[i])*epsilonk(k_array[l]+q_array[emx],k_array[m]+q_array[emy]); //+ 0.25*beta_array[i]*(beta-beta_array[i])*( U*U/4.0 + epsilonk(k_array[l]+q_array[em])*epsilonk(k_array[l]+q_array[em]) );
                    }
                }
            }

            for (size_t l=0; l<k_array.size(); l++){
                for (size_t m=0; m<k_array.size(); m++){
                    spline<double> splObj_GG_jj, splObj_GG_szsz;
                    
                    for (size_t i=0; i<beta_array.size(); i++){
                        dGG_tau_for_k_jj(l,i,m) = velocity(k_array[l])*velocity(k_array[l])*(-2.0)*( dG_dtau_m_FFT(l,i,m)*G_k_q_tau(l,i,m) - G_k_tau(l,Ntau-i,m)*dG_dtau_FFT_q(l,i,m) );
                        dGG_tau_for_k_szsz(l,i,m) = (-2.0)*( dG_dtau_m_FFT(l,i,m)*G_k_q_tau(l,i,m) - G_k_tau(l,Ntau-i,m)*dG_dtau_FFT_q(l,i,m) );
                        GG_tau_for_k_jj.slice(i)(0,0) = velocity(k_array[l])*velocity(k_array[l])*(-2.0)*(-1.0)*( G_k_q_tau(l,i,m)*G_k_tau(l,Ntau-i,m) );
                        GG_tau_for_k_szsz.slice(i)(0,0) = (-2.0)*(-1.0)*( G_k_q_tau(l,i,m)*G_k_tau(l,Ntau-i,m) );
                    }
                
                    // Taking the derivative for boundary conditions
                    double left_der_GG_jj = dGG_tau_for_k_jj(l,0,m);
                    double right_der_GG_jj = dGG_tau_for_k_jj(l,Ntau,m);
                    double left_der_GG_szsz = dGG_tau_for_k_szsz(l,0,m);
                    double right_der_GG_szsz = dGG_tau_for_k_szsz(l,Ntau,m);

                    splObj_GG_jj.set_boundary(spline<double>::bd_type::first_deriv,left_der_GG_jj,spline<double>::bd_type::first_deriv,right_der_GG_jj);
                    splObj_GG_szsz.set_boundary(spline<double>::bd_type::first_deriv,left_der_GG_szsz,spline<double>::bd_type::first_deriv,right_der_GG_szsz);
                    splObj_GG_jj.set_points(beta_array,GG_tau_for_k_jj);
                    splObj_GG_szsz.set_points(beta_array,GG_tau_for_k_szsz);

                    std::vector< std::complex<double> > cub_spl_GG_jj = splObj_GG_jj.bosonic_corr(iqn,beta);
                    std::vector< std::complex<double> > cub_spl_GG_szsz = splObj_GG_szsz.bosonic_corr(iqn,beta);
                    for (size_t i=0; i<iqn.size(); i++){
                        cubic_spline_GG_iqn_k_jj(i,l,m) = cub_spl_GG_jj[i];
                        cubic_spline_GG_iqn_k_szsz(i,l,m) = cub_spl_GG_szsz[i];
                    }
                }
            }
            
            std::vector< std::complex<double> > GG_iqn_q_jj(iqn.size()), GG_iqn_q_szsz(iqn.size());
            arma::Mat< std::complex<double> > tmp_int_iqn_k_jj(iqn.size(),k_array.size()), tmp_int_iqn_k_szsz(iqn.size(),k_array.size());
            for (size_t j=0; j<iqn.size(); j++){
                for (size_t i=0; i<k_array.size(); i++){
                    std::vector< std::complex<double> > GG_iqn_k_tmp_jj(cubic_spline_GG_iqn_k_jj(arma::span(j,j),arma::span(i,i),arma::span::all).begin(),cubic_spline_GG_iqn_k_jj(arma::span(j,j),arma::span(i,i),arma::span::all).end());
                    std::vector< std::complex<double> > GG_iqn_k_tmp_szsz(cubic_spline_GG_iqn_k_szsz(arma::span(j,j),arma::span(i,i),arma::span::all).begin(),cubic_spline_GG_iqn_k_szsz(arma::span(j,j),arma::span(i,i),arma::span::all).end());
                    tmp_int_iqn_k_jj(j,i) = 1.0/(2.0*M_PI)*integralsObj.I1D_VEC(std::move(GG_iqn_k_tmp_jj),delta,"simpson");
                    tmp_int_iqn_k_szsz(j,i) = 1.0/(2.0*M_PI)*integralsObj.I1D_VEC(std::move(GG_iqn_k_tmp_szsz),delta,"simpson");
                }
                std::vector< std::complex<double> > GG_iqn_tmp_jj(tmp_int_iqn_k_jj(j,arma::span::all).begin(),tmp_int_iqn_k_jj(j,arma::span::all).end());
                std::vector< std::complex<double> > GG_iqn_tmp_szsz(tmp_int_iqn_k_szsz(j,arma::span::all).begin(),tmp_int_iqn_k_szsz(j,arma::span::all).end());
                GG_iqn_q_jj[j] = 2.0*2.0/(2.0*M_PI)*integralsObj.I1D_VEC(std::move(GG_iqn_tmp_jj),delta,"simpson");
                GG_iqn_q_szsz[j] = 2.0*2.0/(2.0*M_PI)*integralsObj.I1D_VEC(std::move(GG_iqn_tmp_szsz),delta,"simpson");
            }

            std::string DATASET_NAME("qx_"+std::to_string(q_array[emx])+"_qy_"+std::to_string(q_array[emy]));
            writeInHDF5File(GG_iqn_q_jj, GG_iqn_q_szsz, file, DATA_SET_DIM, DATASET_NAME);
        }
    }
    #elif DIM == 3
    const Integrals integralsObj;
    std::function<std::vector< std::complex<double> >(double,double,double)> int_sus_3D_jj, int_sus_3D_szsz;
    std::vector< std::complex<double> > G_k_q_iwn(iwn.size()), G_k_iwn(iwn.size());
    std::vector<double> FFT_k_q_tau, FFT_k_tau, dG_dtau_FFT_q(Ntau+1), dG_dtau_m_FFT(Ntau+1), G_k_q_tau(Ntau+1), dGG_tau_for_k_jj(beta_array.size()), dGG_tau_for_k_szsz(beta_array.size());
    std::vector<double> GG_tau_for_k_jj(beta_array.size()), GG_tau_for_k_szsz(beta_array.size()), G_k_tau(Ntau+1);
    arma::Cube<double> GG_tau_for_k_jj_conversion(2,2,Ntau+1), GG_tau_for_k_szsz_conversion(2,2,Ntau+1);
    for (size_t emx=0; emx<q_array.size(); emx++){
        for (size_t emy=0; emy<q_array.size(); emy++){
            for (size_t emz=0; emz<q_array.size(); emz++){
                std::cout << "qx: " << q_array[emx] << " qy: " << q_array[emy] << " qz: " << q_array[emz] << "\n";
                // jj
                int_sus_3D_jj = [&](double kx, double ky, double kz){
                    // Building the lattice Green's function from the local DMFT self-energy
                    for (size_t j=0; j<iwn.size(); j++){
                        G_k_q_iwn[j] = 1.0/( iwn[j] + mu - epsilonk(kx+q_array[emx],ky+q_array[emy],kz+q_array[emz]) - sigma_iwn[j] );
                    }
            
                    if ( (q_array[emx]==0.0) && (q_array[emy]==0.0) && (q_array[emz]==0.0) ){
                        G_k_iwn = G_k_q_iwn;
                        dG_dtau_m_FFT = get_derivative_FFT(G_k_iwn,iwn,beta_array,U,mu,kx,0.0,ky,0.0,kz,0.0,std::string("negative"));
                        for (size_t j=0; j<iwn.size(); j++){
                            G_k_iwn[j] -= 1.0/(iwn[j]) + epsilonk(kx,ky,kz)/iwn[j]/iwn[j]; //+ ( U*U/4.0 + epsilonk(k_array[l])*epsilonk(k_array[l]) )/iwn[j]/iwn[j]/iwn[j];
                        }
                        FFT_k_tau = get_iwn_to_tau(G_k_iwn,beta); // beta_arr.back() is beta
                        for (size_t i=0; i<beta_array.size(); i++){
                            G_k_tau[i] = FFT_k_tau[i] - 0.5 - 0.25*(beta-2.0*beta_array[i])*epsilonk(kx,ky,kz); //+ 0.25*beta_array[i]*(beta-beta_array[i])*(U*U/4.0 + epsilonk(k_array[l])*epsilonk(k_array[l]));
                        }
                    }
                    
                    dG_dtau_FFT_q = get_derivative_FFT(G_k_q_iwn,iwn,beta_array,U,mu,kx,q_array[emx],ky,q_array[emy],kz,q_array[emz],std::string("positive"));

                    // Substracting the tail of the Green's function
                    
                    for (size_t j=0; j<iwn.size(); j++){ 
                        G_k_q_iwn[j] -= 1.0/(iwn[j]) + epsilonk(kx+q_array[emx],ky+q_array[emy],kz+q_array[emz])/iwn[j]/iwn[j]; //+ ( U*U/4.0 + epsilonk(k_array[l]+q_array[em])*epsilonk(k_array[l]+q_array[em]) )/iwn[j]/iwn[j]/iwn[j];
                    }

                    // FFT of G(iwn) --> G(tau)
                    FFT_k_q_tau = get_iwn_to_tau(G_k_q_iwn,beta); // beta_arr.back() is beta
                    for (size_t i=0; i<beta_array.size(); i++){
                        G_k_q_tau[i] = FFT_k_q_tau[i] - 0.5 - 0.25*(beta-2.0*beta_array[i])*epsilonk(kx+q_array[emx],ky+q_array[emy],kz+q_array[emz]); //+ 0.25*beta_array[i]*(beta-beta_array[i])*( U*U/4.0 + epsilonk(k_array[l]+q_array[em])*epsilonk(k_array[l]+q_array[em]) );
                    }

                    // std::ofstream ofN("lol.dat",std::ios::out);
                    // for (int iii=0; iii<=Ntau; iii++){
                    //     ofN << beta_array[iii] << "  " << G_k_q_tau[iii] << "\n";
                    // }
                    // ofN.close();
                    
                    spline<double> splObj_GG_jj;
                    
                    for (size_t i=0; i<beta_array.size(); i++){
                        dGG_tau_for_k_jj[i] = velocity(kx)*velocity(kx)*(-2.0)*( dG_dtau_m_FFT[i]*G_k_q_tau[i] - G_k_tau[Ntau-i]*dG_dtau_FFT_q[i] );
                        GG_tau_for_k_jj[i] = velocity(kx)*velocity(kx)*(-2.0)*(-1.0)*( G_k_q_tau[i]*G_k_tau[Ntau-i] );
                    }
                
                    // Taking the derivative for boundary conditions
                    double left_der_GG_jj = dGG_tau_for_k_jj[0];
                    double right_der_GG_jj = dGG_tau_for_k_jj[Ntau];

                    splObj_GG_jj.set_boundary(spline<double>::bd_type::first_deriv,left_der_GG_jj,spline<double>::bd_type::first_deriv,right_der_GG_jj);
                    std::transform(GG_tau_for_k_jj.begin(),GG_tau_for_k_jj.end(),GG_tau_for_k_jj_conversion(arma::span(0,0),arma::span(0,0),arma::span::all).begin(),[](double el){return el;});
                    splObj_GG_jj.set_points(beta_array,GG_tau_for_k_jj_conversion);

                    std::vector< std::complex<double> > cub_spl_GG_jj = splObj_GG_jj.bosonic_corr(iqn,beta);
                    
                    return cub_spl_GG_jj;
                };
                // szsz
                int_sus_3D_szsz = [&](double kx, double ky, double kz){
                    // Building the lattice Green's function from the local DMFT self-energy
                    for (size_t j=0; j<iwn.size(); j++){
                        G_k_q_iwn[j] = 1.0/( iwn[j] + mu - epsilonk(kx+q_array[emx],ky+q_array[emy],kz+q_array[emz]) - sigma_iwn[j] );
                    }
                    
                    dG_dtau_FFT_q = get_derivative_FFT(G_k_q_iwn,iwn,beta_array,U,mu,kx,q_array[emx],ky,q_array[emy],kz,q_array[emz],std::string("positive"));

                    // Substracting the tail of the Green's function
                    
                    for (size_t j=0; j<iwn.size(); j++){ 
                        G_k_q_iwn[j] -= 1.0/(iwn[j]) + epsilonk(kx+q_array[emx],ky+q_array[emy],kz+q_array[emz])/iwn[j]/iwn[j]; //+ ( U*U/4.0 + epsilonk(k_array[l]+q_array[em])*epsilonk(k_array[l]+q_array[em]) )/iwn[j]/iwn[j]/iwn[j];
                    }

                    // FFT of G(iwn) --> G(tau)
                    FFT_k_q_tau = get_iwn_to_tau(G_k_q_iwn,beta); // beta_arr.back() is beta
                    for (size_t i=0; i<beta_array.size(); i++){
                        G_k_q_tau[i] = FFT_k_q_tau[i] - 0.5 - 0.25*(beta-2.0*beta_array[i])*epsilonk(kx+q_array[emx],ky+q_array[emy],kz+q_array[emz]); //+ 0.25*beta_array[i]*(beta-beta_array[i])*( U*U/4.0 + epsilonk(k_array[l]+q_array[em])*epsilonk(k_array[l]+q_array[em]) );
                    }
                    
                    spline<double> splObj_GG_szsz;
                    
                    for (size_t i=0; i<beta_array.size(); i++){
                        dGG_tau_for_k_szsz[i] = (-2.0)*( dG_dtau_m_FFT[i]*G_k_q_tau[i] - G_k_tau[Ntau-i]*dG_dtau_FFT_q[i] );
                        GG_tau_for_k_szsz[i] = (-2.0)*(-1.0)*( G_k_q_tau[i]*G_k_tau[Ntau-i] );
                    }
                
                    // Taking the derivative for boundary conditions
                    double left_der_GG_szsz = dGG_tau_for_k_szsz[0];
                    double right_der_GG_szsz = dGG_tau_for_k_szsz[Ntau];

                    splObj_GG_szsz.set_boundary(spline<double>::bd_type::first_deriv,left_der_GG_szsz,spline<double>::bd_type::first_deriv,right_der_GG_szsz);
                    std::transform(GG_tau_for_k_szsz.begin(),GG_tau_for_k_szsz.end(),GG_tau_for_k_szsz_conversion(arma::span(0,0),arma::span(0,0),arma::span::all).begin(),[](double el){return el;});
                    splObj_GG_szsz.set_points(beta_array,GG_tau_for_k_szsz_conversion);

                    std::vector< std::complex<double> > cub_spl_GG_szsz = splObj_GG_szsz.bosonic_corr(iqn,beta);
                    
                    return cub_spl_GG_szsz;
                };
                
                std::vector< std::complex<double> > GG_iqn_q_jj(iqn.size()), GG_iqn_q_szsz(iqn.size());
                GG_iqn_q_jj = integralsObj.gauss_quad_3D(int_sus_3D_jj,0.0,2.0*M_PI,0.0,2.0*M_PI,0.0,2.0*M_PI);
                GG_iqn_q_szsz = integralsObj.gauss_quad_3D(int_sus_3D_szsz,0.0,2.0*M_PI,0.0,2.0*M_PI,0.0,2.0*M_PI);

                std::string DATASET_NAME("qx_"+std::to_string(q_array[emx])+"_qy_"+std::to_string(q_array[emy])+"_qz_"+std::to_string(q_array[emz]));
                writeInHDF5File(GG_iqn_q_jj, GG_iqn_q_szsz, file, DATA_SET_DIM, DATASET_NAME);
            }
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

inline double sum_rule_iqn_0(double n_k, double k) noexcept{
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
    return 4.0*std::cos(k)*n_k;
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

std::vector<double> get_derivative_FFT(std::vector< std::complex<double> > G_k_iwn, const std::vector< std::complex<double> >& iwn, const std::vector<double>& beta_arr, double U, double mu, double kx, double qx, double ky, double qy, double kz, double qz, std::string opt){
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
    std::vector<double> dG_tau_for_k(beta_arr.size());
    const double beta = beta_arr.back();
    // Subtracting the leading moments of the Green's function...
    std::complex<double> moments;
    for (size_t j=0; j<iwn.size(); j++){
        #if DIM == 1
        moments = 1.0/iwn[j] + epsilonk(kx+qx)/iwn[j]/iwn[j] + ( U*U/4.0 + epsilonk(kx+qx)*epsilonk(kx+qx) )/iwn[j]/iwn[j]/iwn[j];
        #elif DIM == 2
        moments = 1.0/iwn[j] + epsilonk(kx+qx,ky+qy)/iwn[j]/iwn[j] + ( U*U/4.0 + epsilonk(kx+qx,ky+qy)*epsilonk(kx+qx,ky+qy) )/iwn[j]/iwn[j]/iwn[j];
        #elif DIM == 3
        moments = 1.0/iwn[j] + epsilonk(kx+qx,ky+qy,kz+qz)/iwn[j]/iwn[j] + ( U*U/4.0 + epsilonk(kx+qx,ky+qy,kz+qz)*epsilonk(kx+qx,ky+qy,kz+qz) )/iwn[j]/iwn[j]/iwn[j];
        #endif
        G_k_iwn[j] -= moments;
        if (opt.compare("negative")==0){
            G_k_iwn[j] = std::conj(G_k_iwn[j]);
        }
        G_k_iwn[j] *= -1.0*iwn[j];
    }
    // Calculating the imaginary-time derivative of the Green's function.
    std::vector<double> FFT_iwn_tau;
    FFT_iwn_tau = get_iwn_to_tau(G_k_iwn,beta,std::string("Derivative")); // beta_arr.back() is beta
    for (size_t i=0; i<beta_arr.size()-1; i++){
        if (opt.compare("negative")==0){
            #if DIM == 1
            dG_tau_for_k[i] = FFT_iwn_tau[i] + 0.5*epsilonk(kx+qx) + 0.5*( beta_arr[i] - beta/2.0 )*( U*U/4.0 + epsilonk(kx+qx)*epsilonk(kx+qx) );
            #elif DIM == 2
            dG_tau_for_k[i] = FFT_iwn_tau[i] + 0.5*epsilonk(kx+qx,ky+qy) + 0.5*( beta_arr[i] - beta/2.0 )*( U*U/4.0 + epsilonk(kx+qx,ky+qy)*epsilonk(kx+qx,ky+qy) );
            #elif DIM == 3
            dG_tau_for_k[i] = FFT_iwn_tau[i] + 0.5*epsilonk(kx+qx,ky+qy,kz+qz) + 0.5*( beta_arr[i] - beta/2.0 )*( U*U/4.0 + epsilonk(kx+qx,ky+qy,kz+qz)*epsilonk(kx+qx,ky+qy,kz+qz) );
            #endif
        } else if (opt.compare("positive")==0){
            #if DIM == 1 
            dG_tau_for_k[i] = FFT_iwn_tau[i] + 0.5*epsilonk(kx+qx) + 0.5*( beta/2.0 - beta_arr[i] )*( U*U/4.0 + epsilonk(kx+qx)*epsilonk(kx+qx) );
            #elif DIM == 2
            dG_tau_for_k[i] = FFT_iwn_tau[i] + 0.5*epsilonk(kx+qx,ky+qy) + 0.5*( beta/2.0 - beta_arr[i] )*( U*U/4.0 + epsilonk(kx+qx,ky+qy)*epsilonk(kx+qx,ky+qy) );
            #elif DIM == 3
            dG_tau_for_k[i] = FFT_iwn_tau[i] + 0.5*epsilonk(kx+qx,ky+qy,kz+qz) + 0.5*( beta/2.0 - beta_arr[i] )*( U*U/4.0 + epsilonk(kx+qx,ky+qy,kz+qz)*epsilonk(kx+qx,ky+qy,kz+qz) );
            #endif
        }
    }
    #if DIM == 1
    dG_tau_for_k[beta_arr.size()-1] = epsilonk(kx+qx)-mu+U*0.5 - 1.0*dG_tau_for_k[0]; // Assumed half-filling
    #elif DIM == 2
    dG_tau_for_k[beta_arr.size()-1] = epsilonk(kx+qx,ky+qy)-mu+U*0.5 - 1.0*dG_tau_for_k[0]; // Assumed half-filling 
    #elif DIM == 3
    dG_tau_for_k[beta_arr.size()-1] = epsilonk(kx+qx,ky+qy,kz+qz)-mu+U*0.5 - 1.0*dG_tau_for_k[0]; // Assumed half-filling 
    #endif
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
