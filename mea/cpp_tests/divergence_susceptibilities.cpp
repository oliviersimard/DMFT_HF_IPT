#include "sus_vertex_corrections.hpp"
#include "../../src/json_utils.hpp"

static const int Ntau_division_factor = 8;
static arma::Mat< std::complex<double> > _ladder;

typedef struct Div{
    double beta;
    double dnom_re;
} Div;

void writeInHDF5File(std::vector< Div >& arr_to_save, H5::H5File* file, const unsigned int& DATA_SET_DIM, const std::string& DATASET_NAME) noexcept(false);
std::complex<double> get_denom(const std::vector< std::complex<double> >& iqn, const std::vector< std::complex<double> >& iwn, const std::vector<double>& k_arr, const std::vector< std::complex<double> >& SE, size_t n_tilde, size_t n_bar, size_t k_t_b, double mu) noexcept;
#if DIM == 1
std::complex<double> ladder(size_t n_iqpn, double qp, double mu, double beta, double U, const std::vector< std::complex<double> >& iwn, const std::vector< std::complex<double> >& iqn_tilde, const std::vector< std::complex<double> >& SE, int SE_shift) noexcept;
std::complex<double> ladder_2(std::complex<double> iqqn, double k, int n_iqqn, double mu, double beta, double U, const std::vector<std::complex<double>>& iwn, const std::vector<std::complex<double>>& SE) noexcept;
std::complex<double> chi_corr(size_t n_ikn_tilde, size_t k_tilde, size_t n_ikn_bar, size_t k_bar, size_t n_iqpn, size_t n_qp, size_t n_iqn, double qq, double mu, double beta, double U, const std::vector<std::complex<double>>& SE, const std::vector<std::complex<double>>& iqn_tilde, const std::vector<std::complex<double>>& iwn, const std::vector<std::complex<double>>& iqn, const std::vector<double>& k_arr) noexcept;
std::complex<double> ladder_merged_with_chi_corr(const std::vector< std::complex<double> >& iqn_tilde, const std::vector< std::complex<double> >& iwn, const std::vector< std::complex<double> >& iqn, const std::vector<double>& k_arr, const std::vector< std::complex<double> >& SE, size_t n_ikn_bar, size_t n_ikn_tilde, size_t n_k_bar, size_t n_k_tilde, size_t n_iqn, double qq, double mu, double U, double beta) noexcept;
#elif DIM == 2
std::complex<double> Gamma_merged_corr(const std::vector< std::complex<double> >& iwn, const std::vector< std::complex<double> >& iqn, const std::vector< std::complex<double> >& SE, size_t n_ikn_bar, double k_bar_x,double k_bar_y, size_t n_iqn, double qq_x, double qq_y, double mu, double U, double beta) noexcept(false);
std::complex<double> Gamma_correction_denominator(const std::vector< std::complex<double> >& iwn, const std::vector<std::complex<double>>& SE, double k_bar_x, double k_bar_y, double kpp_x, double kpp_y, double qq_x, double qq_y, size_t n_ikn_bar, size_t n_ikppn, double mu, double U, double beta) noexcept(false);
#endif
void save_matrix_in_HDF5(const arma::Cube< std::complex<double> >& cube_to_save, const std::vector<double>& k_arr, H5std_string DATASET_NAME, H5::H5File* file) noexcept(false);
std::vector< std::complex<double> > DMFTNCA(arma::Cube<double>& Hyb, std::vector<double>& SE_0, arma::Cube<double>& SE_1, std::vector<double>& SE_2, std::vector<double>& G_0, arma::Cube<double>& G_1, std::vector<double>& G_2,
    std::vector<double>& dG_0, arma::Cube<double>& dG_1, std::vector<double>& dG_2, arma::Cube< std::complex<double> >& SE_iwn, arma::Cube< std::complex<double> >& Gloc_iwn,
    double U, double beta, double hyb_c, double alpha, double delta_tau, int N_tau) noexcept(false);

enum spin : short { up, down };

int main(int argc, char** argv){
    
    const std::string filename_params("./../../params.json");
    Json_utils JsonObj;
    const MembCarrier params = JsonObj.JSONLoading(filename_params);
    const double n_t_spin=params.db_arr[0];
    const unsigned int N_tau=(unsigned int)params.int_arr[0]; // Has to be a power of 2, i.e 512!
    // Has to be a power of two as well: this is no change from IPT.
    assert(N_tau%2==0);
    const unsigned int N_it=(unsigned int)params.int_arr[1];
    const unsigned int N_k=(unsigned int)params.int_arr[2];
    const double beta_init=params.db_arr[6], beta_step=params.db_arr[5], beta_max=params.db_arr[4];
    const double U_init=params.db_arr[3], U_step=params.db_arr[2], U_max=params.db_arr[1];
    // const char* solver_type=params.char_arr[0];
    // std::string solver_type_s(solver_type);
    const double alpha=0.0;
    // k_t_b_array constructed
    std::vector<double> k_t_b_array;
    double k_tmp;
    for (size_t l=0; l<N_k; l++){
        k_tmp = -M_PI + l*2.0*M_PI/(double)(N_k-1);
        k_t_b_array.push_back(k_tmp);
    }
    #if DIM == 1
    const double Hyb_c=2.0; // For the 1D chain with nearest-neighbor hopping, it is 2t.
    #elif DIM == 2
    const double Hyb_c=4.0; // For the 2D square lattice with nearest-neighbor hopping.
    #elif DIM == 3
    const double Hyb_c=6.0;
    #endif
    const double dividing_factor = 1.0;
    const double dividing_factor_corr = 1.0;
    #ifndef NCA
    // Saving on memory doing so. For DMFT loop, no parallelization is needed.
    arma::Cube<double> weiss_green_A_matsubara_t_pos(2,2,2*N_tau+1,arma::fill::zeros), weiss_green_A_matsubara_t_neg(2,2,2*N_tau+1,arma::fill::zeros); 
    arma::Cube<double> weiss_green_tmp_A_matsubara_t_pos(2,2,2*N_tau+1,arma::fill::zeros), weiss_green_tmp_A_matsubara_t_neg(2,2,2*N_tau+1,arma::fill::zeros); 
    arma::Cube<double> self_A_matsubara_t_pos(2,2,2*N_tau+1,arma::fill::zeros), self_A_matsubara_t_neg(2,2,2*N_tau+1,arma::fill::zeros);
    arma::Cube<double> local_green_A_matsubara_t_pos(2,2,2*N_tau+1,arma::fill::zeros), local_green_A_matsubara_t_neg(2,2,2*N_tau+1,arma::fill::zeros);
    //
    arma::Cube< std::complex<double> > weiss_green_A_matsubara_w(2,2,2*N_tau,arma::fill::zeros); 
    arma::Cube< std::complex<double> > weiss_green_tmp_A_matsubara_w(2,2,2*N_tau,arma::fill::zeros); 
    arma::Cube< std::complex<double> > self_A_matsubara_w(2,2,2*N_tau,arma::fill::zeros); 
    arma::Cube< std::complex<double> > local_green_A_matsubara_w(2,2,2*N_tau,arma::fill::zeros);

    // To save the derivative of G(tau) and G(-tau)
    arma::Cube<double> data_dG_dtau_pos_A(2,2,2*N_tau+1,arma::fill::zeros); 
    arma::Cube<double> data_dG_dtau_neg_A(2,2,2*N_tau+1,arma::fill::zeros);
    #else
    arma::Cube<double> Hyb(2,2,2*N_tau+1,arma::fill::zeros), SE_1(2,2,2*N_tau+1,arma::fill::zeros);
    std::vector<double> SE_0(2*N_tau+1,0), SE_2(2*N_tau+1,0);
    arma::Cube<double> G_1(2,2,2*N_tau+1,arma::fill::zeros), dG_1(2,2,2*N_tau+1,arma::fill::zeros), G(2,2,2*N_tau+1,arma::fill::zeros); // Physical Green's function
    std::vector<double> G_0(2*N_tau+1,0), G_2(2*N_tau+1,0), dG_0(2*N_tau+1,0), dG_2(2*N_tau+1,0), G_tmp_up(2*N_tau+1), G_tmp_down(2*N_tau+1);
    arma::Cube< std::complex<double> > SE_iwn(2,2,2*N_tau,arma::fill::zeros), Gloc_iwn(2,2,2*N_tau,arma::fill::zeros);
    std::vector< std::complex<double> > G_up_iwn(2*N_tau), Hyb_up_iwn(2*N_tau), G_down_iwn(2*N_tau), Hyb_down_iwn(2*N_tau);
    #endif
    std::vector< std::complex<double> > iqn, iqn_tilde, iwn(2*N_tau/Ntau_division_factor), iqn_tilde_big; // for the inner loop inside Gamma.
    // HDF5 business
    H5::H5File* file = nullptr;
    #ifdef INFINITE
    #ifdef NCA
    std::string filename("div/div_"+std::to_string(DIM)+"D_U_"+std::to_string(U_init)+"_"+std::to_string(U_step)+"_"+std::to_string(U_max)+"_beta_"+std::to_string(beta_init)+"_"+std::to_string(beta_step)+"_"+std::to_string(beta_max)+"_Ntau_"+std::to_string(N_tau)+"_Nk_"+std::to_string(N_k)+"_NCA_infinite_ladder_sum.hdf5");
    #else
    std::string filename("div/div_"+std::to_string(DIM)+"D_U_"+std::to_string(U_init)+"_"+std::to_string(U_step)+"_"+std::to_string(U_max)+"_beta_"+std::to_string(beta_init)+"_"+std::to_string(beta_step)+"_"+std::to_string(beta_max)+"_Ntau_"+std::to_string(N_tau)+"_Nk_"+std::to_string(N_k)+"_DIV_"+std::to_string(dividing_factor)+"_DIVCORR_"+std::to_string(dividing_factor_corr)+"_infinite_ladder_sum.hdf5");
    #endif
    #else
    #ifdef NCA
    std::string filename("div/div_"+std::to_string(DIM)+"D_U_"+std::to_string(U_init)+"_"+std::to_string(U_step)+"_"+std::to_string(U_max)+"_beta_"+std::to_string(beta_init)+"_"+std::to_string(beta_step)+"_"+std::to_string(beta_max)+"_Ntau_"+std::to_string(N_tau)+"_Nk_"+std::to_string(N_k)+"_NCA_single_ladder_sum.hdf5");
    #else
    std::string filename("div/div_"+std::to_string(DIM)+"D_U_"+std::to_string(U_init)+"_"+std::to_string(U_step)+"_"+std::to_string(U_max)+"_beta_"+std::to_string(beta_init)+"_"+std::to_string(beta_step)+"_"+std::to_string(beta_max)+"_Ntau_"+std::to_string(N_tau)+"_Nk_"+std::to_string(N_k)+"_DIV_"+std::to_string(dividing_factor)+"_single_ladder_sum.hdf5");
    #endif
    #endif
    const H5std_string FILE_NAME( filename );
    // The different processes cannot create more than once the file to be written in.
    file = new H5::H5File( FILE_NAME, H5F_ACC_TRUNC );
    std::vector< Div > div_arr;
    for (double U=U_init; U<=U_max; U+=U_step){
        for (double beta=beta_init; beta<=beta_max; beta+=( (beta>=40.0) ? 2.0 : beta_step ) ){
            const double delta_tau = beta/(double)(2*N_tau);
            // Bosonic Matsubara array
            for (size_t j=0; j<N_tau/2; j++){ // change Ntau for lower value to decrease time when testing...
                iqn.push_back( std::complex<double>( 0.0, (2.0*j)*M_PI/beta ) );
            }
            for (signed int j=(-(signed int)N_tau/Ntau_division_factor)+1; j<(signed int)N_tau/Ntau_division_factor; j++){ // Bosonic frequencies.
                iqn_tilde.push_back( std::complex<double>( 0.0, (2.0*(double)j)*M_PI/beta ) );
            }
            for (signed int j=(-(signed int)N_tau); j<(signed int)N_tau; j++){ // Fermionic frequencies.
                iwnArr_l.push_back( std::complex<double>( 0.0, (2.0*(double)j+1.0)*M_PI/beta ) );
            }
            for (signed int j=(-3*(signed int)N_tau/4)+1; j<3*(signed int)N_tau/4; j++){ // Fermionic frequencies.
                iqn_tilde_big.push_back( std::complex<double>( 0.0, (2.0*(double)j)*M_PI/beta ) );
            }
            std::transform(iwnArr_l.data()+iwnArr_l.size()/2-N_tau/Ntau_division_factor,iwnArr_l.data()+iwnArr_l.size()/2+N_tau/Ntau_division_factor,iwn.data(),[](std::complex<double> d){ return d; });

            const double mu = U/2.0; // Half-filling.
            #ifndef NCA
            GreenStuff WeissGreenA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,weiss_green_A_matsubara_t_pos,weiss_green_A_matsubara_t_neg,weiss_green_A_matsubara_w);
            GreenStuff HybA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,weiss_green_tmp_A_matsubara_t_pos,weiss_green_tmp_A_matsubara_t_neg,weiss_green_tmp_A_matsubara_w);
            GreenStuff SelfEnergyA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,self_A_matsubara_t_pos,self_A_matsubara_t_neg,self_A_matsubara_w);
            GreenStuff LocalGreenA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,local_green_A_matsubara_t_pos,local_green_A_matsubara_t_neg,local_green_A_matsubara_w);
        
            IPT2::DMFTproc EqDMFTA(WeissGreenA,HybA,LocalGreenA,SelfEnergyA,data_dG_dtau_pos_A,data_dG_dtau_neg_A,k_t_b_array,n_t_spin);
            
            auto sigma_iwn = DMFTloop(EqDMFTA,N_it); // The self-energy exited is 4 times the length of the iwn array
            // std::ofstream outP("lol_test.dat",std::ios::out);
            // for (size_t ii=0; ii<iqn_tilde.size(); ii++){
            //     outP << (iwn[N_tau/4]-iqn_tilde[ii]).imag() << " " << sigma_iwn[(2*iwn.size()-1)+N_tau/4-ii].imag() << "\n";
            // }
            // outP.close();
            // std::ofstream outS("lol_SE_tot.dat",std::ios::out);
            // for (size_t ii=0; ii<sigma_iwn.size(); ii++){
            //     outS << iwnArr_l[ii].imag() << " " << sigma_iwn[ii].imag() << "\n";
            // }
            // outS.close();
            // std::ofstream outP("lol_test.dat",std::ios::out);
            // std::complex<double> denom_val{0.0};
            // const Integrals intObj;
            // const size_t NI = iwn.size();
            // std::function<std::complex<double>(double)> int_k_1D;
            // const size_t n_ikn_tilde = N_tau/4, n_ikppn = N_tau/4, n_iqn = 0;
            // std::complex<double> iqppn, ikn_tilde = iwn[n_ikn_tilde], ikppn = iwn[n_ikppn], iiqn = iqn[n_iqn];
            // for (size_t n_k_tilde=0; n_k_tilde<k_t_b_array.size(); n_k_tilde++){
            //     for (size_t n_k_pp=0; n_k_pp<k_t_b_array.size(); n_k_pp++){
            //         for (size_t n_q_pp=0; n_q_pp<iqn_tilde.size(); n_q_pp++){
            //             iqppn = iqn_tilde[n_q_pp];
            //             int_k_1D = [&](double q_pp){
            //                 return 1.0/( ( ikn_tilde - iqppn + iiqn ) + mu - epsilonk(k_t_b_array[n_k_tilde]+0.0-q_pp) - sigma_iwn[(2*NI-1)+n_ikn_tilde+n_iqn-n_q_pp] 
            //                 )*1.0/( ( ikppn - iqppn ) + mu - epsilonk(k_t_b_array[n_k_pp]-q_pp) - sigma_iwn[(2*NI-1)+n_ikppn-n_q_pp] );
            //             };
            //             denom_val += intObj.gauss_quad_1D(int_k_1D,0.0,2.0*M_PI);
            //         }
            //         denom_val *= U/beta/(2.0*M_PI);
            //         outP << denom_val.real() << " ";
            //     }
            //     outP << "\n";
            // }
            // outP.close();
            // exit(0);
            #else
            auto sigma_iwn = DMFTNCA(Hyb,SE_0,SE_1,SE_2,G_0,G_1,G_2,dG_0,dG_1,dG_2,SE_iwn,Gloc_iwn,U,beta,Hyb_c,alpha,delta_tau,N_tau);
            #endif

            // N_tau/4 corresponds to the lowest positive fermionic Matsubara freq.
            const unsigned int n_bar = N_tau/Ntau_division_factor, n_tilde = N_tau/Ntau_division_factor;
            #ifndef INFINITE
            auto val_denom = U/dividing_factor*get_denom(iqn_tilde,iwn,k_t_b_array,sigma_iwn,n_tilde,n_bar,N_k-1,mu)/beta;
            #else
            #if DIM == 1
            _ladder = arma::Mat<std::complex<double>>(iqn_tilde_big.size(),k_t_b_array.size());
            // auto val_denom_sl = U/dividing_factor*get_denom(iqn_tilde,iwn,k_t_b_array,sigma_iwn,n_tilde,n_bar,N_k-1,mu)/beta;
            std::cout << "iqn size: " << iqn.size() << " iqn_tilde size " << iqn_tilde.size() << " SE size " << sigma_iwn.size() << " _ladder n_rows " << _ladder.n_rows << std::endl;
            for (size_t n_iqpn=0; n_iqpn<iqn_tilde_big.size(); n_iqpn++){
                for (size_t n_qp=0; n_qp<k_t_b_array.size(); n_qp++){
                    _ladder(n_iqpn,n_qp) = ladder(n_iqpn,k_t_b_array[n_qp],mu,beta,U,iwn,iqn_tilde_big,sigma_iwn,3*(int)N_tau/4-1);
                }
            }
            auto val_denom = ladder_merged_with_chi_corr(iqn_tilde,iwn,iqn,k_t_b_array,sigma_iwn,n_bar,n_tilde,N_k/4,3*N_k/4,0,0.0,mu,U/dividing_factor_corr,beta); // N_k/4, 3*N_k/4
            #elif DIM == 2
            auto val_denom_sl = U/dividing_factor*get_denom(iqn_tilde,iwn,k_t_b_array,sigma_iwn,n_tilde,n_bar,N_k-1,mu)/beta;
            auto val_denom_corr = Gamma_merged_corr(iwn,iqn,sigma_iwn,n_bar,k_t_b_array[N_k-1],k_t_b_array[N_k-1],0,0.0,0.0,mu,U/dividing_factor,beta);
            #endif
            //auto val_denom = val_denom_sl-val_denom_corr;
            #endif
            div_arr.push_back( Div{1.0/beta,val_denom.real()} );


            iqn.clear();
            iqn_tilde.clear();
            iqn_tilde_big.clear();
            iwnArr_l.clear(); // Clearing to not append at each iteration over previous set.
        }
        H5std_string DATASET_NAME("U_"+std::to_string(U));
        writeInHDF5File(div_arr,file,div_arr.size(),DATASET_NAME);
        div_arr.clear();
    }

    delete file;

    return 0;

}

std::complex<double> get_denom(const std::vector< std::complex<double> >& iqn_tilde, const std::vector< std::complex<double> >& iwn, const std::vector<double>& k_arr, const std::vector< std::complex<double> >& SE, size_t n_tilde, size_t n_bar, size_t n_k_t_b, double mu) noexcept{
    std::complex<double> tmp_val{0};
    #if DIM == 1
    std::function<std::complex<double>(double)> k_integrand;
    #elif DIM == 2
    std::function<std::complex<double>(double,double)> k_integrand;
    #endif
    const Integrals intObj;
    for (size_t ii=0; ii<iqn_tilde.size(); ii++){
        #if DIM == 1
        k_integrand = [&](double k){
            return 1.0/( ( iwn[n_bar] - iqn_tilde[ii] ) + mu - epsilonk(k+k_arr[n_k_t_b]) - SE[(2*iwn.size()-1)+n_bar-ii] )*1.0/( ( iwn[n_tilde] - iqn_tilde[ii] ) + mu - epsilonk(k) - SE[(2*iwn.size()-1)+n_tilde-ii] );
        };
        tmp_val += 1.0/(2.0*M_PI)*intObj.gauss_quad_1D(k_integrand,0.0,2.0*M_PI);
        #elif DIM == 2
        k_integrand = [&](double kx,double ky){
            return 1.0/( ( iwn[n_bar] - iqn_tilde[ii] ) + mu - epsilonk(kx+k_arr[n_k_t_b],ky+k_arr[n_k_t_b]) - SE[(2*iwn.size()-1)+n_bar-ii] )*1.0/( ( iwn[n_tilde] - iqn_tilde[ii] ) + mu - epsilonk(kx,ky) - SE[(2*iwn.size()-1)+n_tilde-ii] );
        };
        tmp_val += 1.0/(2.0*M_PI)/(2.0*M_PI)*intObj.gauss_quad_2D(k_integrand,0.0,2.0*M_PI,0.0,2.0*M_PI);
        #endif
    }
    return tmp_val;
}

#if DIM == 1
std::complex<double> ladder(size_t n_iqpn, double qp, double mu, double beta, double U, const std::vector< std::complex<double> >& iwn, const std::vector< std::complex<double> >& iqn_tilde, const std::vector< std::complex<double> >& SE, int SE_shift) noexcept{
    std::complex<double> lower_val{0.0};
    const Integrals intObj;
    const size_t NI = iwn.size(); // corresponds to half the size of the SE array
    std::function< std::complex<double>(double) > int_k_1D;
    std::complex<double> iqpn = iqn_tilde[n_iqpn], ikpn;
    const size_t starting_point_SE = ((double)Ntau_division_factor/2.0-1.0/2.0)*(int)NI;
    //const size_t q_starting_point_SE = ((double)Ntau_division_factor/2.0)*(int)NI - 1;
    for (size_t n_ikpn=0; n_ikpn<NI; n_ikpn++){
        ikpn = iwn[n_ikpn];
        int_k_1D = [&](double k){
            return ( 1.0 / ( ikpn + mu - epsilonk(k) - SE[starting_point_SE+n_ikpn] ) 
            )*( 1.0 / ( ikpn-iqpn + mu - epsilonk(k-qp) - SE[starting_point_SE+SE_shift+n_ikpn-n_iqpn] ) 
            );
        };
        lower_val += 1.0/(2.0*M_PI)*intObj.gauss_quad_1D(int_k_1D,0.0,2.0*M_PI);
    }
    
    lower_val *= U/beta;
    lower_val += 1.0;
    lower_val = U/lower_val;

    return lower_val;
};

std::complex<double> ladder_2(std::complex<double> iqqn, double k, int n_iqqn, double mu, double beta, double U, const std::vector<std::complex<double>>& iwn, const std::vector<std::complex<double>>& SE) noexcept{
    std::complex<double> lower_val{0.0};
    const Integrals intObj;
    const size_t NI = iwn.size(); // corresponds to half the size of the SE array
    const size_t starting_point_SE = ((double)Ntau_division_factor/2.0-1.0/2.0)*(int)NI;
    const size_t q_starting_point_SE = ((double)Ntau_division_factor/2.0+1.0/2.0)*(int)NI - 2;
    std::function<std::complex<double>(double)> int_k_1D;
    std::complex<double> ikpn;
    for (size_t n_ikpn=0; n_ikpn<NI; n_ikpn++){
        ikpn = iwn[n_ikpn];
        int_k_1D = [&](double kk){
            return ( 1.0 / ( ikpn + mu - epsilonk(kk) - SE[starting_point_SE+n_ikpn] ) 
            )*( 1.0 / ( ikpn-iqqn + mu - epsilonk(kk-k) - SE[q_starting_point_SE+n_ikpn-n_iqqn] ) 
            );
        };
        lower_val += 1.0/(2.0*M_PI)*intObj.gauss_quad_1D(int_k_1D,0.0,2.0*M_PI);
    }
    
    lower_val *= U/beta;
    lower_val += 1.0;
    lower_val = U/lower_val;

    return lower_val;
};

std::complex<double> chi_corr(size_t n_ikn_tilde, size_t n_k_tilde, size_t n_ikn_bar, size_t n_k_bar, size_t n_iqpn, size_t n_qp, size_t n_iqn, double qq, double mu, double beta, double U, const std::vector<std::complex<double>>& SE, const std::vector<std::complex<double>>& iqn_tilde, const std::vector<std::complex<double>>& iwn, const std::vector<std::complex<double>>& iqn, const std::vector<double>& k_arr) noexcept{
    std::complex<double> val{0.0};
    const Integrals intObj;
    const int size_k_arr = static_cast<int>(k_arr.size());
    auto k_resizing = [size_k_arr](int n_k_val) -> int {if (n_k_val>=0) return n_k_val%(size_k_arr-1); else return (size_k_arr-1)+n_k_val%(size_k_arr-1);};
    double delta = 2.0*M_PI/(double)(size_k_arr-1);
    const size_t NI = iwn.size();
    std::vector< std::complex<double> > int_k_1D(size_k_arr);
    std::complex<double> ikn_tilde = iwn[n_ikn_tilde], iqppn;//, ikn_bar = iwn[n_ikn_bar];
    std::complex<double> iqpn = iqn_tilde[n_iqpn], iqqn = iqn[n_iqn];
    double qpp, qp=k_arr[n_qp], k_tilde=k_arr[n_k_tilde];//, k_bar=k_arr[n_k_bar];
    for (size_t n_iqppn=0; n_iqppn<iqn_tilde.size(); n_iqppn++){
        iqppn = iqn_tilde[n_iqppn];
        for (size_t n_qpp=0; n_qpp<k_arr.size(); n_qpp++){
            //std::cout << (int)ladder_shift_complex+(int)n_ikn_tilde+(int)n_iqn+(int)n_iqpn+(int)n_iqppn+(int)n_ikn_bar << " " << k_resizing((int)n_k_tilde-(int)n_qp-(int)n_qpp-(int)n_k_bar) << std::endl;
            qpp = k_arr[n_qpp];
            // int_k_1D[n_qpp] = ( 1.0 / ( ikn_tilde+iqqn-iqpn + mu - epsilonk(k_tilde+qq-qp) - SE[(Ntau_division_factor/2*NI-1)+n_ikn_tilde+n_iqn-n_iqpn] ) 
            // )*( 1.0 / ( ikn_tilde-iqpn + mu - epsilonk(k_tilde-qp) - SE[(Ntau_division_factor/2*NI-1)+n_ikn_tilde-n_iqpn] ) 
            // )*ladder(n_iqppn,k_arr[n_qpp],mu,beta,U,iwn,iqn_tilde,SE,(int)NI/2-1)*( 1.0 / ( ikn_tilde+iqqn-iqpn-iqppn + mu - epsilonk(k_tilde+qq-qp-qpp) - SE[(Ntau_division_factor/2*NI+NI/2-2)+n_ikn_tilde+n_iqn-n_iqpn-n_iqppn] ) 
            // )*( 1.0 / ( ikn_tilde-iqpn-iqppn + mu - epsilonk(k_tilde-qp-qpp) - SE[(Ntau_division_factor/2*NI+NI/2-2)+n_ikn_tilde-n_iqpn-n_iqppn] ) 
            // )*ladder_2(ikn_tilde+iqqn-iqpn-iqppn-ikn_bar,-(int)n_ikn_tilde-(int)n_iqn+(int)n_iqpn+(int)n_iqppn+(int)n_ikn_bar,k_tilde-qp-qpp-k_bar,mu,beta,U,iwn,SE);
            int_k_1D[n_qpp] = ( 1.0 / ( ikn_tilde+iqqn-iqpn + mu - epsilonk(k_tilde+qq-qp) - SE[(Ntau_division_factor/2*NI-1)+n_ikn_tilde+n_iqn-n_iqpn] ) 
            )*( 1.0 / ( ikn_tilde-iqpn + mu - epsilonk(k_tilde-qp) - SE[(Ntau_division_factor/2*NI-1)+n_ikn_tilde-n_iqpn] ) 
            )*_ladder( n_iqppn+((int)_ladder.n_rows/2-(int)NI/2)+1, n_qpp )*( 1.0 / ( ikn_tilde+iqqn-iqpn-iqppn + mu - epsilonk(k_tilde+qq-qp-qpp) - SE[(Ntau_division_factor/2*NI+NI/2-2)+n_ikn_tilde+n_iqn-n_iqpn-n_iqppn] ) 
            )*( 1.0 / ( ikn_tilde-iqpn-iqppn + mu - epsilonk(k_tilde-qp-qpp) - SE[(Ntau_division_factor/2*NI+NI/2-2)+n_ikn_tilde-n_iqpn-n_iqppn] ) 
            )*_ladder( ((int)_ladder.n_rows/2-(int)NI)+2-(int)n_ikn_tilde-(int)n_iqn+(int)n_iqpn+(int)n_iqppn+(int)n_ikn_bar, k_resizing((int)n_k_tilde-(int)n_qp-(int)n_qpp-(int)n_k_bar) );
        }
        val += 1.0/(2.0*M_PI)*intObj.I1D_VEC(int_k_1D,delta,"simpson");
    }
    val *= -1.0/beta;
    val += 1.0;

    return 1.0/val;
}

std::complex<double> ladder_merged_with_chi_corr(const std::vector< std::complex<double> >& iqn_tilde, const std::vector< std::complex<double> >& iwn, const std::vector< std::complex<double> >& iqn, const std::vector<double>& k_arr, const std::vector< std::complex<double> >& SE, size_t n_ikn_bar, size_t n_ikn_tilde, size_t n_k_bar, size_t n_k_tilde, size_t n_iqn, double qq, double mu, double U, double beta) noexcept{
    std::complex<double> val{0.0};
    Integrals intObj;
    std::vector< std::complex<double> > int_k_1D(k_arr.size());
    double delta = 2.0*M_PI/(double)(k_arr.size()-1);
    const size_t NI = iwn.size();
    std::cout << "NI: " << NI << std::endl;
    std::cout << "N_k: " << k_arr.size() << std::endl;
    for (size_t n_iqpn=0; n_iqpn<iqn_tilde.size(); n_iqpn++){
        std::cout << "n_iqpn: " << n_iqpn << std::endl;
        for (size_t n_qp=0; n_qp<k_arr.size(); n_qp++){
            //std::cout << "n_qp: " << n_qp << std::endl;
            int_k_1D[n_qp] = _ladder( n_iqpn+((int)_ladder.n_rows/2-(int)NI/2)+1, n_qp )*chi_corr(n_ikn_tilde,n_k_tilde,n_ikn_bar,n_k_bar,n_iqpn,n_qp,n_iqn,qq,mu,beta,U,SE,iqn_tilde,iwn,iqn,k_arr);
        }
        val += 1.0/(2.0*M_PI)*intObj.I1D_VEC(int_k_1D,delta,"simpson");
    }
    
    val *= 1.0/beta;

    return val;
};
# elif DIM == 2
std::complex<double> Gamma_correction_denominator(const std::vector< std::complex<double> >& iwn, const std::vector<std::complex<double>>& SE, double k_bar_x, double k_bar_y, double kpp_x, double kpp_y, double qq_x, double qq_y, size_t n_ikn_bar, size_t n_ikppn, double mu, double U, double beta) noexcept(false){
    std::complex<double> denom_val{0.0};
    const Integrals intObj;
    const size_t NI = iwn.size();
    std::function<std::complex<double>(double,double)> int_k_2D;
    std::complex<double> ikpppn, ikn_bar = iwn[n_ikn_bar], ikppn = iwn[n_ikppn];
    for (size_t n_ppp=0; n_ppp<iwn.size(); n_ppp++){
        ikpppn = iwn[n_ppp];
        int_k_2D = [&](double k_ppp_x, double k_ppp_y){
            return ( 1.0 / ( ikpppn+ikppn-ikn_bar + mu - epsilonk(k_ppp_x+kpp_x-k_bar_x,k_ppp_y+kpp_y-k_bar_y) - SE[3*NI/2+n_ppp+n_ikppn-n_ikn_bar] ) 
                )*( 1.0 / ( ikpppn + mu - epsilonk(k_ppp_x,k_ppp_y) - SE[3*NI/2+n_ppp] ) 
                );
        };
        denom_val += intObj.gauss_quad_2D(int_k_2D,0.0,2.0*M_PI,0.0,2.0*M_PI);
    }
    denom_val *= U/beta/(2.0*M_PI)/(2.0*M_PI);
    denom_val += 1.0;

    return 1.0/denom_val;
}

std::complex<double> Gamma_merged_corr(const std::vector< std::complex<double> >& iwn, const std::vector< std::complex<double> >& iqn, const std::vector< std::complex<double> >& SE, size_t n_ikn_bar, double k_bar_x,double k_bar_y, size_t n_iqn, double qq_x, double qq_y, double mu, double U, double beta) noexcept(false){
    // This function method is an implicit function of k_bar and ikn_bar
    std::complex<double> tot_corr{0.0};
    const Integrals intObj;
    // const double delta = 2.0*M_PI/(double)(OneLadder< T >::_splInlineobj._k_array.size()-1);
    const size_t NI = iwn.size();
    std::function<std::complex<double>(double,double)> int_k_2D;
    std::complex<double> iqqn = iqn[n_iqn], ikppn;
    for (size_t n_pp=0; n_pp<NI; n_pp++){
        std::cout << n_pp << std::endl;
        ikppn = iwn[n_pp];
        int_k_2D = [&](double k_pp_x, double k_pp_y){
            return ( 1.0/( ikppn + mu - epsilonk(k_pp_x,k_pp_y) - SE[3*NI/2+n_pp] )
                )*Gamma_correction_denominator(iwn,SE,k_bar_x,k_bar_y,k_pp_x,k_pp_y,qq_x,qq_y,n_ikn_bar,n_pp,mu,U,beta
                )*( 1.0/( ikppn - iqqn + mu - epsilonk(k_pp_x-qq_x,k_pp_y-qq_y) - SE[3*NI/2+n_pp-n_iqn] )
                );
        };
        tot_corr += intObj.gauss_quad_2D(int_k_2D,0.0,2.0*M_PI,0.0,2.0*M_PI);
    }
    tot_corr *= U/beta/(2.0*M_PI)/(2.0*M_PI);
    
    return tot_corr; 
}
#endif

void writeInHDF5File(std::vector< Div >& arr_to_save, H5::H5File* file, const unsigned int& DATA_SET_DIM, const std::string& DATASET_NAME) noexcept(false){
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
    const H5std_string MEMBER1( "MEM1" );
    const H5std_string MEMBER2( "MEM2" );
    const int RANK = 1;
    H5::CompType m_type( sizeof(Div) );
    m_type.insertMember( MEMBER1, HOFFSET(Div, beta), H5::PredType::NATIVE_DOUBLE);
    m_type.insertMember( MEMBER2, HOFFSET(Div, dnom_re), H5::PredType::NATIVE_DOUBLE);
    try{
        H5::Exception::dontPrint();

        hsize_t dimsf[1];
        dimsf[0] = DATA_SET_DIM;
        H5::DataSpace dataspace( RANK, dimsf );

        /*
        * Create a group in the file
        */
        
        H5::DataSet* dataset;
        // dataset = new H5::DataSet(file->createDataSet(DATASET_NAME, std_cmplx_type, dataspace));
        dataset = new H5::DataSet(file->createDataSet("/"+DATASET_NAME, m_type, dataspace));
        // Write data to dataset
        dataset->write( arr_to_save.data(), m_type );
        delete dataset;

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

std::vector< std::complex<double> > partial_DFT(const std::vector< std::complex<double> >& input){
    size_t MM = input.size();
    constexpr std::complex<double> im(0.0,1.0);
    std::vector< std::complex<double> > output(MM,0.0);
    for (size_t n=0; n<MM; n++){  // For each output element
        std::complex<double> s(0.0);
        for (size_t l=1; l<MM; l++){  // For each input element, leaving out the first element in tau-defined array object
            std::complex<double> angle = 2.0*im * M_PI * (double)(l * n) / (double)(MM);
            s += input[l] * std::exp(angle);
        }
        output[n] = s;
    }
    
    return output;
}

std::vector< std::complex<double> > linear_spline_Sigma_tau_to_iwn(const std::vector<double>& SE_tau, const std::vector< std::complex<double> >& iwn_array, double beta, double delta_tau){
    constexpr std::complex<double> im(0.0,1.0);
    size_t M = SE_tau.size();
    assert(M%2==1); 
    size_t NN = static_cast<size_t>(M/2);
    double SE_0 = SE_tau[0], SE_beta = SE_tau[M-1];
    std::vector< std::complex<double> > S_p(M-1,0.0);
    std::vector< std::complex<double> > SE_iwn(M-1,0.0);
    // Filling up the kernel array that will enter FFT
    for (int ip=0; ip<(int)M-1; ip++){
        S_p[ip] =  std::exp( im * (M_PI * ip) / (double)(M-1) ) * SE_tau[ip];
    }
    
    // Fourier transforming
    auto IFFT_data_m = partial_DFT(S_p);
    fftw_plan plan_fft;
    std::vector< std::complex<double> > IFFT_data(M-1);
    plan_fft = fftw_plan_dft_1d(M-1, reinterpret_cast<fftw_complex*>(S_p.data()), reinterpret_cast<fftw_complex*>(IFFT_data.data()), FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan_fft);
    // Mirroring, because using negative Matsubara frequencies to start with
    std::vector< std::complex<double> > slicem1(IFFT_data_m.begin(),IFFT_data_m.begin()+NN), slicem2(IFFT_data_m.begin()+NN,IFFT_data_m.end());
    IFFT_data_m = std::vector< std::complex<double> >(slicem2.begin(),slicem2.end());
    IFFT_data_m.insert(IFFT_data_m.end(),slicem1.begin(),slicem1.end());
    std::vector< std::complex<double> > slice1(IFFT_data.begin(),IFFT_data.begin()+NN), slice2(IFFT_data.begin()+NN,IFFT_data.end());
    IFFT_data = std::vector< std::complex<double> >(slice2.begin(),slice2.end());
    IFFT_data.insert(IFFT_data.end(),slice1.begin(),slice1.end());

    for (size_t i=0; i<(M-1); i++){
        std::complex<double> iwn = iwn_array[i];
        SE_iwn[i] = ( -SE_beta - SE_0 )/iwn - (SE_beta*(std::exp(-iwn*delta_tau)-1.0))/( delta_tau * (iwn*iwn) )
        + (-1.0 + std::exp(-iwn*delta_tau))/( (iwn*iwn)*delta_tau ) * IFFT_data_m[i]
        + (std::exp(iwn*delta_tau)-1.0)/( delta_tau * (iwn*iwn) ) * IFFT_data[i];
    }
    fftw_destroy_plan(plan_fft);

    return SE_iwn;
}

std::vector< std::complex<double> > linear_spline_tau_to_iwn(const std::vector<double>& G_tau, const std::vector< std::complex<double> >& iwn_array, double beta, double delta_tau){
    constexpr std::complex<double> im(0.0,1.0);
    size_t M = G_tau.size();
    assert(M%2==1); 
    size_t NN = static_cast<size_t>(M/2);
    std::vector< std::complex<double> > S_p(M-1,0.0);
    std::vector< std::complex<double> > G_iwn(M-1,0.0);
    // Filling up the kernel array that will enter FFT
    for (int ip=0; ip<(int)M-1; ip++){
        S_p[ip] =  std::exp( im * (M_PI * ip) / (double)(M-1) ) * G_tau[ip];
    }
    
    // Fourier transforming
    fftw_plan plan_fft;
    std::vector< std::complex<double> > IFFT_data(M-1);
    plan_fft = fftw_plan_dft_1d(M-1, reinterpret_cast<fftw_complex*>(S_p.data()), reinterpret_cast<fftw_complex*>(IFFT_data.data()), FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan_fft);
    // Mirroring, because using negative Matsubara frequencies to start with
    std::vector< std::complex<double> > slice1(IFFT_data.begin(),IFFT_data.begin()+NN), slice2(IFFT_data.begin()+NN,IFFT_data.end());
    IFFT_data = std::vector< std::complex<double> >(slice2.begin(),slice2.end());
    IFFT_data.insert(IFFT_data.end(),slice1.begin(),slice1.end());

    for (size_t i=0; i<(M-1); i++){
        std::complex<double> iwn = iwn_array[i];
        G_iwn[i] = 1.0/iwn + (-1.0 + std::exp(-iwn*delta_tau))/( (iwn*iwn)*delta_tau )
             + 2.0/( delta_tau * (iwn*iwn) ) * ( std::cos(iwn.imag()*delta_tau) - 1.0 ) * IFFT_data[i];
    }
    fftw_destroy_plan(plan_fft);

    return G_iwn;
}

void fft_w2t(const std::vector< std::complex<double> >& data1, std::vector<double>& data2, double beta){
    assert(data1.size()==(data2.size()-1));
    const unsigned int N_tau = data2.size();
    std::cout << "size tau: " << N_tau << " size iwn: " << data1.size() << std::endl; 
    constexpr std::complex<double> im(0.0,1.0);
    std::complex<double>* inUp=new std::complex<double> [N_tau-1];
    std::complex<double>* outUp=new std::complex<double> [N_tau-1];
    fftw_plan pUp; //fftw_plan pDown;
    for(size_t k=0;k<N_tau-1;k++){
        inUp[k]=data1[k];
    }
    pUp=fftw_plan_dft_1d(N_tau-1, reinterpret_cast<fftw_complex*>(inUp), reinterpret_cast<fftw_complex*>(outUp), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(pUp); //fftw_execute(pDown);

    for(size_t j=0;j<N_tau-1;j++){
        data2[j]=(outUp[j]*std::exp(im*(double)(N_tau-2)*(double)j*M_PI/((double)N_tau-1.0))).real()/beta;
    }
    data2[N_tau-1]=0.0; // Up spin
    for(size_t k=0;k<N_tau-1;k++){
        data2[N_tau-1]+=(data1[k]*std::exp(im*(double)(N_tau-2)*M_PI)).real()/beta;
    }
    delete [] inUp;
    delete [] outUp;
    fftw_destroy_plan(pUp); //fftw_destroy_plan(pDown);
}


std::vector< std::complex<double> > DMFTNCA(arma::Cube<double>& Hyb, std::vector<double>& SE_0, arma::Cube<double>& SE_1, std::vector<double>& SE_2,
    std::vector<double>& G_0, arma::Cube<double>& G_1, std::vector<double>& G_2, std::vector<double>& dG_0, arma::Cube<double>& dG_1, std::vector<double>& dG_2,
    arma::Cube< std::complex<double> >& SE_iwn, arma::Cube< std::complex<double> >& Gloc_iwn, double U, double beta, double hyb_c, double alpha, double delta_tau, int N_tau) noexcept(false){
    
    const Integrals intObj;
    std::vector< std::complex<double> > G_up_iwn(2*N_tau), Hyb_up_iwn(2*N_tau), G_down_iwn(2*N_tau), Hyb_down_iwn(2*N_tau);
    std::vector<double> G_tmp_up(2*N_tau+1), G_tmp_down(2*N_tau+1);
    arma::Cube<double> G(2,2,2*N_tau+1,arma::fill::zeros);
    const double E_0 = 0.0;
    const double E_1 = -U/2.0;
    const double E_2 = 0.0; // half-filling
    double h = 0.0; // Staggered magnetization

    for (int i=0; i<=2*N_tau; i++) {
        // spin splitting between spins on impurity due to h in first iterations
        G_0[i] = -std::exp(-E_0*delta_tau*i);
        G_1.slice(i)(up,up) = -std::exp(-(E_1-h)*delta_tau*i);
        G_1.slice(i)(down,down) = -std::exp(-(E_1+h)*delta_tau*i); 
        G_2[i] = -std::exp(-E_2*delta_tau*i);
        
        Hyb.slice(i)(up,up) = -0.5*hyb_c;
        Hyb.slice(i)(down,down) = -0.5*hyb_c;
    }

    // normalization of the Green's function in the Q=1 pseudo-particle subspace
    double lambda0 = log( -G_0[2*N_tau] - G_1.slice(2*N_tau)(up,up) - G_1.slice(2*N_tau)(down,down) - G_2[2*N_tau] ) / beta; // G_0 and G_2 shouldn't depend on the spin
    std::cout << "lambda0 " << lambda0 << "\n";
    for (int i=0; i<=2*N_tau; i++) {
        G_0[i] *= std::exp(-i*lambda0*delta_tau);
        G_1.slice(i) *= std::exp(-i*lambda0*delta_tau); // Multiplication is done for all the elements in the 2X2 spin matrix at each tau
        G_2[i] *= std::exp(-i*lambda0*delta_tau);
    }
    double lambda = lambda0;
    unsigned int iter = 0;
    bool is_converged = false;
    double G_0_norm, G_1_up_norm, G_1_down_norm, G_2_norm;
    double rhs_0, rhs_1_up, rhs_1_down, rhs_2;
    double G_up_diff, G_down_diff;
    while (!is_converged && iter<150){
        if (iter>0){
            h=0.0;
        }
        std::cout << "********************************** iter : " << iter << " **********************************" << "\n";
        std::cout << "U: " << U << " beta: " << beta << " h: " << h << "\n";
        // NCA self-energy update in AFM case scenario
        for (size_t i=0; i<=2*N_tau; i++) {
            SE_0[i] = G_1.slice(i)(up,up)*Hyb.slice(2*N_tau-i)(up,up)*(-1) + G_1.slice(i)(down,down)*Hyb.slice(2*N_tau-i)(down,down)*(-1); // <0 Added twice because of spin degree of freedom
            SE_1.slice(i)(up,up) = G_2[i]*Hyb.slice(2*N_tau-i)(down,down)*(-1) + G_0[i]*Hyb.slice(i)(up,up)*(-1);
            SE_1.slice(i)(down,down) = G_2[i]*Hyb.slice(2*N_tau-i)(up,up)*(-1) + G_0[i]*Hyb.slice(i)(down,down)*(-1);
            SE_2[i] = G_1.slice(i)(up,up)*Hyb.slice(i)(down,down)*(-1) + G_1.slice(i)(down,down)*Hyb.slice(i)(up,up)*(-1);
        }

        G_0_norm = 1.0 + delta_tau*0.5*(0.5*delta_tau*SE_0[0] + (E_0 + lambda));
        G_1_up_norm = 1.0 + delta_tau*0.5*(0.5*delta_tau*SE_1.slice(0)(up,up) + (E_1 - h + lambda)); // Same chemical pot. for both spins
        G_1_down_norm = 1.0 + delta_tau*0.5*(0.5*delta_tau*SE_1.slice(0)(down,down) + (E_1 + h + lambda)); // Same chemical pot. for both spins
        G_2_norm = 1.0 + delta_tau*0.5*(0.5*delta_tau*SE_2[0] + (E_2 + lambda));

        // std::cout << "G_0_norm: " << G_0_norm << " G_1_up_norm: " << G_1_up_norm << " G_1_down_norm: " << G_1_down_norm << " G_2_norm: " << G_2_norm << "\n";

        G_0[0] = -1.0;
        G_1.slice(0)(up,up) = -1.0;
        G_1.slice(0)(down,down) = -1.0;
        G_2[0] = -1.0;

        // keeping the derivatives G' in memory for later usage
        dG_0[0] = -(E_0+lambda)*G_0[0] - delta_tau*0.5*SE_0[0]*G_0[0];
        dG_1.slice(0)(up,up) = -(E_1-h+lambda)*G_1.slice(0)(up,up) - delta_tau*0.5*SE_1.slice(0)(up,up)*G_1.slice(0)(up,up);
        dG_1.slice(0)(down,down) = -(E_1+h+lambda)*G_1.slice(0)(down,down) - delta_tau*0.5*SE_1.slice(0)(down,down)*G_1.slice(0)(down,down);
        dG_2[0] = -(E_2+lambda)*G_2[0] - delta_tau*0.5*SE_2[0]*G_2[0];

        // std::cout << "dG_0[0]: " << dG_0[0] << " dG_1_up[0]: " << dG_1.slice(0)(up,up) << " dG_1_down[0]: " << dG_1.slice(0)(down,down) << " dG_2[0]: " << dG_2[0] << "\n";

        double factor;
        for (size_t n=1; n<=2*N_tau; n++) {
            // std::cout << "n: " << n << " delta_tau: " << delta_tau << "\n";
            // computing G(t_j)
            rhs_0 = G_0[n-1] + delta_tau/2.0*dG_0[n-1];
            rhs_1_up = G_1.slice(n-1)(up,up) + delta_tau/2.0*dG_1.slice(n-1)(up,up);
            rhs_1_down = G_1.slice(n-1)(down,down) + delta_tau/2.0*dG_1.slice(n-1)(down,down);
            rhs_2 = G_2[n-1] + delta_tau/2.0*dG_2[n-1];

            for (size_t j=0; j<=n-1; j++){
                factor = (j==0) ? 0.25 : 0.5;
                rhs_0 += -factor*delta_tau*delta_tau*SE_0[n-j]*G_0[j];
                rhs_1_up += -factor*delta_tau*delta_tau*SE_1.slice(n-j)(up,up)*G_1.slice(j)(up,up);
                rhs_1_down += -factor*delta_tau*delta_tau*SE_1.slice(n-j)(down,down)*G_1.slice(j)(down,down);
                rhs_2 += -factor*delta_tau*delta_tau*SE_2[n-j]*G_2[j];
            }
            // std::cout << "rhs_0: " << rhs_0 << " rhs_1_up: " << rhs_1_up << " rhs_1_down: " << rhs_1_down << " rhs_2: " << rhs_2 << "\n";
            
            G_0[n] = rhs_0/G_0_norm;
            G_1.slice(n)(up,up) = rhs_1_up/G_1_up_norm;
            G_1.slice(n)(down,down) = rhs_1_down/G_1_down_norm;
            G_2[n] = rhs_2/G_2_norm;
            // std::cout << "G_0: " << G_0[n] << " G_1_up: " << G_1.slice(n)(up,up) << " G_1_down: " << G_1.slice(n)(down,down) << " G_2: " << G_2[n] << "\n";
            
            // computing G'(t_j) for next time step
            dG_0[n] = -(E_0+lambda)*G_0[n];
            dG_1.slice(n)(up,up) = -(E_1-h+lambda)*G_1.slice(n)(up,up);
            dG_1.slice(n)(down,down) = -(E_1+h+lambda)*G_1.slice(n)(down,down);
            dG_2[n] = -(E_2+lambda)*G_2[n];

            for (size_t j=0; j<=n; j++) {
                factor = (j==0 || j==n) ? 0.5 : 1.0;
                dG_0[n] -= delta_tau*factor*SE_0[n-j]*G_0[j];
                dG_1.slice(n)(up,up) -= delta_tau*factor*SE_1.slice(n-j)(up,up)*G_1.slice(j)(up,up);
                dG_1.slice(n)(down,down) -= delta_tau*factor*SE_1.slice(n-j)(down,down)*G_1.slice(j)(down,down);
                dG_2[n] -= delta_tau*factor*SE_2[n-j]*G_2[j];
            }
        }

        // Physical GF
        // std::cout << "log(...): " << -G_0[2*N_tau] - G_1.slice(2*N_tau)(up,up) - G_1.slice(2*N_tau)(down,down) - G_2[2*N_tau] << std::endl;
        double lambda_tmp = log( -G_0[2*N_tau] - G_1.slice(2*N_tau)(up,up) - G_1.slice(2*N_tau)(down,down) - G_2[2*N_tau] ) / beta;
        lambda += lambda_tmp;
        // if NANs pop up, the iterations are stopped at iter==10
        if ( (lambda != lambda) && iter==8){
            is_converged=true;
        } 
        std::cout << "lambda updated " << lambda << "\n";
        for (int i=0; i<=2*N_tau; i++) {
            G_0[i] *= std::exp(-i*lambda_tmp*delta_tau);
            G_1.slice(i) *= std::exp(-i*lambda_tmp*delta_tau);
            // G_1.slice(i)(down,down) *= std::exp(-i*lambda_tmp*delta_tau);
            G_2[i] *= std::exp(-i*lambda_tmp*delta_tau);
        }
        
        for (int i=0; i<=2*N_tau; i++) {    
            G.slice(i)(up,up) = (-1)*G_1.slice(i)(up,up)*G_0[2*N_tau-i]+G_1.slice(2*N_tau-i)(down,down)*(-1)*G_2[i];
            G.slice(i)(down,down) = (-1)*G_1.slice(i)(down,down)*G_0[2*N_tau-i]+G_1.slice(2*N_tau-i)(up,up)*(-1)*G_2[i];
            //std::cout << "Gnca " << i << " " << G[i] << " Hyb " << Hyb[i] << "\n"; 
        }

        // Get G and Hyb in terms of iwn...needs to transform into vectors
        std::vector<double> G_up( G(arma::span(up,up),arma::span(up,up),arma::span::all).begin(), G(arma::span(up,up),arma::span(up,up),arma::span::all).end() );
        std::vector<double> G_down( G(arma::span(down,down),arma::span(down,down),arma::span::all).begin(), G(arma::span(down,down),arma::span(down,down),arma::span::all).end() );
        std::vector<double> Hyb_up( Hyb(arma::span(up,up),arma::span(up,up),arma::span::all).begin(), Hyb(arma::span(up,up),arma::span(up,up),arma::span::all).end() );
        std::vector<double> Hyb_down( Hyb(arma::span(down,down),arma::span(down,down),arma::span::all).begin(), Hyb(arma::span(down,down),arma::span(down,down),arma::span::all).end() );
        //
        G_up_iwn = linear_spline_tau_to_iwn(G_up,iwnArr_l,beta,delta_tau);
        G_down_iwn = linear_spline_tau_to_iwn(G_down,iwnArr_l,beta,delta_tau);
        Hyb_up_iwn = linear_spline_Sigma_tau_to_iwn(Hyb_up,iwnArr_l,beta,delta_tau);
        Hyb_down_iwn = linear_spline_Sigma_tau_to_iwn(Hyb_down,iwnArr_l,beta,delta_tau);
        
        std::cout << "n_up " << -1.0*G_up[2*N_tau] << " n_down: " << -1.0*G_down[2*N_tau] << "\n"; 
        
        for (size_t i=0; i<2*N_tau; i++){
            SE_iwn.slice(i)(up,up) = iwnArr_l[i] - h + U/2.0 - Hyb_up_iwn[i] - 1.0/G_up_iwn[i]; // Physical SE
            SE_iwn.slice(i)(down,down) = iwnArr_l[i] + h + U/2.0 - Hyb_down_iwn[i] - 1.0/G_down_iwn[i]; // Physical SE
            // std::cout << "SE_up: " << SE_iwn.slice(i)(up,up) << "SE_down: " << SE_iwn.slice(i)(down,down) << "\n";
        }

        // Computing G_loc with the extracted physical self-energy. DMFT procedure
        #if DIM == 1
        std::function< std::complex<double>(double) > funct_k_integration_up, funct_k_integration_down;
        #elif DIM == 2
        std::function< std::complex<double>(double,double) > funct_k_integration_up, funct_k_integration_down;
        #endif
        for (size_t i=0; i<2*N_tau; i++){
            std::complex<double> iwn = iwnArr_l[i];
            #if DIM == 1
            // AA
            funct_k_integration_up = [&](double kx){
                return 1.0/( iwn - h + U/2.0 - SE_iwn.slice(i)(up,up) - epsilonk(kx)*epsilonk(kx) / ( iwn + h + U/2.0 - SE_iwn.slice(i)(down,down) ) );
            };
            Gloc_iwn.slice(i)(up,up) = 1.0/(2.0*M_PI)*intObj.gauss_quad_1D(funct_k_integration_up,0.0,2.0*M_PI);
            // BB
            funct_k_integration_down = [&](double kx){
                return 1.0/( iwn + h + U/2.0 - SE_iwn.slice(i)(down,down) - epsilonk(kx)*epsilonk(kx) / ( iwn - h + U/2.0 - SE_iwn.slice(i)(up,up) ) );
            };
            Gloc_iwn.slice(i)(down,down) = 1.0/(2.0*M_PI)*intObj.gauss_quad_1D(funct_k_integration_down,0.0,2.0*M_PI);
            #elif DIM == 2
            // AA
            funct_k_integration_up = [&](double kx,double ky){
                return 1.0/( iwn - h + U/2.0 - SE_iwn.slice(i)(up,up) - epsilonk(kx,ky)*epsilonk(kx,ky) / ( iwn + h + U/2.0 - SE_iwn.slice(i)(down,down) ) );
            };
            Gloc_iwn.slice(i)(up,up) = 1.0/(4.0*M_PI*M_PI)*intObj.gauss_quad_2D(funct_k_integration_up,0.0,2.0*M_PI,0.0,2.0*M_PI);
            // BB
            funct_k_integration_down = [&](double kx,double ky){
                return 1.0/( iwn + h + U/2.0 - SE_iwn.slice(i)(down,down) - epsilonk(kx,ky)*epsilonk(kx,ky) / ( iwn - h + U/2.0 - SE_iwn.slice(i)(up,up) ) );
            };
            Gloc_iwn.slice(i)(down,down) = 1.0/(4.0*M_PI*M_PI)*intObj.gauss_quad_2D(funct_k_integration_down,0.0,2.0*M_PI,0.0,2.0*M_PI);
            #endif
        }
        // Updating the hybridisation function DMFT
        for (size_t i=0; i<2*N_tau; i++){
            Hyb_up_iwn[i] = iwnArr_l[i] - h + U/2.0 - SE_iwn.slice(i)(up,up) - 1.0/Gloc_iwn.slice(i)(up,up) - hyb_c/iwnArr_l[i];
            Hyb_down_iwn[i] = iwnArr_l[i] + h + U/2.0 - SE_iwn.slice(i)(down,down) - 1.0/Gloc_iwn.slice(i)(down,down) - hyb_c/iwnArr_l[i];
            //std::cout << Hyb_iwn[i] << "\n";
        }
        // FFT and updating at the same time
        
        fft_w2t(Hyb_up_iwn,Hyb_up,beta);
        fft_w2t(Hyb_down_iwn,Hyb_down,beta);
        // Transferring into armadillo container the dumbest way
        for (size_t i=0; i<=2*N_tau; i++){
            Hyb.slice(i)(up,up) = (1.0-alpha)*(Hyb_up[i] - 0.5*hyb_c) + alpha*(Hyb.slice(i)(up,up));
            Hyb.slice(i)(down,down) = (1.0-alpha)*(Hyb_down[i] - 0.5*hyb_c) + alpha*(Hyb.slice(i)(down,down));
        }
        
        // std::ofstream outHyb2("Hyb_PP_tau_first_iter.dat",std::ios::out);
        // for (int i=0; i<=2*N_tau; i++){
        //     outHyb2 << delta_tau*i << "\t\t" << Hyb.slice(i)(up,up) << "\n";
        // }
        // outHyb2.close();

        // Hyb_up = std::vector<double>( Hyb(arma::span(up,up),arma::span(up,up),arma::span::all).begin(), Hyb(arma::span(up,up),arma::span(up,up),arma::span::all).end() );
        
        // Hyb_up_iwn = linear_spline_Sigma_tau_to_iwn(Hyb_up,iwn_array,beta,delta_tau);

        // std::ofstream outHyb3("Hyb_PP_iwn_first_iter_after.dat",std::ios::out);
        // for (int i=0; i<2*N_tau; i++){
        //     outHyb3 << iwn_array[i].imag() << "\t\t" << Hyb_up_iwn[i].real() << "\t\t" << Hyb_up_iwn[i].imag() << "\n";
        // }
        // outHyb3.close();

        if (iter>0){
            G_up_diff = 0.0; G_down_diff = 0.0;
            for (size_t l=0; l<=2*N_tau; l++){
                G_up_diff += std::abs(G_tmp_up[l]-G_up[l]);
                G_down_diff += std::abs(G_tmp_down[l]-G_down[l]);
            }
            std::cout << "G_diff up: " << G_up_diff << " and G_diff down: " << G_down_diff << "\n";
            if (G_up_diff<5e-4 && G_down_diff<5e-4)
                is_converged=true;
        }
        G_tmp_up = G_up;
        G_tmp_down = G_down;

        
        iter+=1;
    }
    std::vector< std::complex<double> > sigma_iwn( SE_iwn(arma::span(up,up),arma::span(up,up),arma::span::all).begin(), SE_iwn(arma::span(up,up),arma::span(up,up),arma::span::all).end() );
    return sigma_iwn;
}
