#include "src/susceptibilities.hpp"
#include "src/json_utils.hpp"


int main(int argc, char** argv){
    // Loading parameters from Json file
    const std::string filename("../params.json"); // ../ necessary because compiled inside build directory using CMake. For Makefile, set to params.json only (Debug mode).
    Json_utils JsonObj;
    const MembCarrier params = JsonObj.JSONLoading(filename);
    const double n_t_spin=params.db_arr[0];
    const unsigned int N_tau=(unsigned int)params.int_arr[0]; // 4096
    const unsigned int N_it=(unsigned int)params.int_arr[1];
    const unsigned int N_k=(unsigned int)params.int_arr[2];
    const double beta_init=params.db_arr[6], beta_step=params.db_arr[5], beta_max=params.db_arr[4];
    const double U_init=params.db_arr[3], U_step=params.db_arr[2], U_max=params.db_arr[1];
    #if DIM == 1
    const double Hyb_c=2; // For the 1D chain with nearest-neighbor hopping, it is 2t.
    #elif DIM == 2
    const double Hyb_c=4; // For the 2D square lattice with nearest-neighbor hopping. 
    #endif
    std::ofstream objSaveStreamGloc;
    std::ofstream objSaveStreamSE;
    std::ofstream objSaveStreamGW;

    for (size_t k=0; k<=N_k; k++){
        double epsilonk = -1.0*M_PI + 2.0*(double)k*M_PI/N_k;
        vecK.push_back(epsilonk);
    }
    
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
    arma::Cube<double> data_dG_dtau_pos(2,2,2*N_tau+1,arma::fill::zeros); 
    arma::Cube<double> data_dG_dtau_neg(2,2,2*N_tau+1,arma::fill::zeros);

    std::string pathToDir("./data/");
    std::string trailingStr("");
    for (double beta=beta_init; beta<=beta_max; beta+=beta_step){
        iwnArr_l.clear(); // Clearing to not append at each iteration over previous set.
        for (signed int j=(-(signed int)N_tau); j<(signed int)N_tau; j++){
            std::complex<double> matFreq(0.0 , (2.0*(double)j+1.0)*M_PI/beta );
            iwnArr_l.push_back( matFreq );
        }
        iqnArr_l.clear();
        for (size_t j=0; j<N_tau; j++){
            std::complex<double> matFreq(0.0 , (2.0*(double)j)*M_PI/beta );
            iqnArr_l.push_back( matFreq );
        }

        for (double U=U_init; U<=U_max; U+=U_step){

            #if DIM == 1
            std::string filenameToSaveGloc(pathToDir+"Green_loc_1D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_n_"+std::to_string(n_t_spin)+"_N_tau_"+std::to_string(N_tau)+trailingStr);
            std::string filenameToSaveSE(pathToDir+"Self_energy_1D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_n_"+std::to_string(n_t_spin)+"_N_tau_"+std::to_string(N_tau)+trailingStr);
            std::string filenameToSaveGW(pathToDir+"Weiss_Green_1D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_n_"+std::to_string(n_t_spin)+"_N_tau_"+std::to_string(N_tau)+trailingStr);
            #elif DIM == 2
            std::string filenameToSaveGloc(pathToDir+"Green_loc_2D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_n_"+std::to_string(n_t_spin)+"_N_tau_"+std::to_string(N_tau)+trailingStr);
            std::string filenameToSaveSE(pathToDir+"Self_energy_2D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_n_"+std::to_string(n_t_spin)+"_N_tau_"+std::to_string(N_tau)+trailingStr);
            std::string filenameToSaveGW(pathToDir+"Weiss_Green_2D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_n_"+std::to_string(n_t_spin)+"_N_tau_"+std::to_string(N_tau)+trailingStr);
            #endif
            std::vector< std::string > vecFiles={filenameToSaveGloc,filenameToSaveSE,filenameToSaveGW};
            try{
                check_file_content(vecFiles,pathToDir+"analytic_continuations"); // Checks whether files already exist to avoid overwritting.
            } catch (const std::exception& err){
                std::cerr << err.what();
                exit(1);
            }
            // Initializing the main Green's function objects.
            GreenStuff WeissGreenA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,weiss_green_A_matsubara_t_pos,weiss_green_A_matsubara_t_neg,weiss_green_A_matsubara_w);
            GreenStuff HybA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,weiss_green_tmp_A_matsubara_t_pos,weiss_green_tmp_A_matsubara_t_neg,weiss_green_tmp_A_matsubara_w);
            GreenStuff SelfEnergyA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,self_A_matsubara_t_pos,self_A_matsubara_t_neg,self_A_matsubara_w);
            GreenStuff LocalGreenA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,local_green_A_matsubara_t_pos,local_green_A_matsubara_t_neg,local_green_A_matsubara_w);

            IPT2::DMFTproc EqDMFTA(WeissGreenA,HybA,LocalGreenA,SelfEnergyA,data_dG_dtau_pos,data_dG_dtau_neg,vecK,n_t_spin);

            /* Performs the complete DMFT calculations */
            DMFTloop(EqDMFTA,objSaveStreamGloc,objSaveStreamSE,objSaveStreamGW,vecFiles,N_it);
            
        }
    }

    return 0;
}
