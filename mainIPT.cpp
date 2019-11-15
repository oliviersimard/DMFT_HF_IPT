#include "src/susceptibilities.hpp"
#include "src/json_utils.hpp"

#define MULT_N_TAU 4
//enum SolverType{ HF, IPT } SolverType;  // Include in params.json. OPT takes in "IPT" and "HF" as parameters.

int main(int argc, char** argv){
    // Loading parameters from Json file
    #ifndef DEBUG
    const std::string filename("./../params.json"); // ../ necessary because compiled inside build directory using CMake. For Makefile, set to params.json only (Debug mode).
    #else
    const std::string filename("./params.json");
    #endif
    Json_utils JsonObj;
    struct stat infoDir;
    const MembCarrier params = JsonObj.JSONLoading(filename);
    const double n_t_spin=params.db_arr[0];
    const unsigned int N_tau=(unsigned int)params.int_arr[0]; // Has to be a power of 2, i.e 512!
    const unsigned int N_it=(unsigned int)params.int_arr[1];
    const unsigned int N_k=(unsigned int)params.int_arr[2];
    const double beta_init=params.db_arr[6], beta_step=params.db_arr[5], beta_max=params.db_arr[4];
    const double U_init=params.db_arr[3], U_step=params.db_arr[2], U_max=params.db_arr[1];
    const bool is_full=params.boo_arr[0], load_first=params.boo_arr[1];
    #if DIM == 1
    const double Hyb_c=2; // For the 1D chain with nearest-neighbor hopping, it is 2t.
    #elif DIM == 2
    const double Hyb_c=4; // For the 2D square lattice with nearest-neighbor hopping. 
    #endif
    std::ofstream objSaveStreamGloc;
    std::ofstream objSaveStreamSE;
    std::ofstream objSaveStreamGW;
    // Parameters to search for N_tau
    //std::regex r("(?<=N_tau_)([0-9]*\\.[0-9]+|[0-9]+)");

    for (size_t k=0; k<=N_k; k++){ // Used when computing the susceptibilities.
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
    std::string pathToDirLoad("./../data/");
    std::string trailingStr("");
    for (double beta=beta_init; beta<=beta_max; beta+=beta_step){
        iwnArr_l.clear(); // Clearing to not append at each iteration over previous set.
        for (signed int j=(-(signed int)N_tau); j<(signed int)N_tau; j++){ // Fermionic frequencies.
            std::complex<double> matFreq(0.0 , (2.0*(double)j+1.0)*M_PI/beta );
            iwnArr_l.push_back( matFreq );
        }
        iqnArr_l.clear();
        for (size_t j=0; j<N_tau; j++){ // Bosonic frequencies.
            std::complex<double> matFreq(0.0 , (2.0*(double)j)*M_PI/beta );
            iqnArr_l.push_back( matFreq );
        }

        for (double U=U_init; U<=U_max; U+=U_step){

            std::string customDirName(std::to_string(DIM)+"D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_n_"+std::to_string(n_t_spin)+trailingStr+"_N_tau_"+std::to_string(N_tau));
            std::string filenameToSaveGloc(pathToDir+customDirName+"/Green_loc_"+customDirName);
            std::string filenameToSaveSE(pathToDir+customDirName+"/Self_energy_"+customDirName);
            std::string filenameToSaveGW(pathToDir+customDirName+"/Weiss_Green_"+customDirName);
            std::string filenameToLoad;
            if (load_first){ // This file has got to be containing the self-energy data. For at least twice the number N_tau for proper interpolation in formulae.
                std::string customDirNameLoad(std::to_string(DIM)+"D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_n_"+std::to_string(n_t_spin)+trailingStr+"_N_tau_"+std::to_string(MULT_N_TAU*N_tau));
                #ifndef DEBUG
                filenameToLoad=pathToDirLoad+customDirNameLoad+"/Self_energy_"+customDirNameLoad; // Stick to the self-energy.
                #else
                filenameToLoad=pathToDir+customDirNameLoad+"/Self_energy_"+customDirNameLoad; // Stick to the self-energy.
                #endif
            }
            std::vector< std::string > vecFiles={filenameToSaveGloc,filenameToSaveSE,filenameToSaveGW};
            try{ // Ensures that we don't overwrite any files within build/.
                check_file_content(vecFiles,pathToDir+customDirName+"/analytic_continuations",pathToDir+customDirName+"/susceptibilities"); // Checks whether files already exist to avoid overwritting. Also creates 
            } catch (const std::exception& err){                                                                                            // directory architecture
                std::cerr << err.what() << "\n";
                exit(1);
            }
            // Initializing the main Green's function objects.
            GreenStuff WeissGreenA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,weiss_green_A_matsubara_t_pos,weiss_green_A_matsubara_t_neg,weiss_green_A_matsubara_w);
            GreenStuff HybA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,weiss_green_tmp_A_matsubara_t_pos,weiss_green_tmp_A_matsubara_t_neg,weiss_green_tmp_A_matsubara_w);
            GreenStuff SelfEnergyA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,self_A_matsubara_t_pos,self_A_matsubara_t_neg,self_A_matsubara_w);
            GreenStuff LocalGreenA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,local_green_A_matsubara_t_pos,local_green_A_matsubara_t_neg,local_green_A_matsubara_w);

            IPT2::DMFTproc EqDMFTA(WeissGreenA,HybA,LocalGreenA,SelfEnergyA,data_dG_dtau_pos,data_dG_dtau_neg,vecK,n_t_spin);

            /* Performs the complete DMFT calculations if directory doesn't already exist */
            #ifndef DEBUG
            int message = stat( (pathToDirLoad+customDirName).c_str(), &infoDir );
            if ( !(infoDir.st_mode & S_IFDIR) && message!=0 ) // If the directory doesn't already exist in ../data/ ...
                DMFTloop(EqDMFTA,objSaveStreamGloc,objSaveStreamSE,objSaveStreamGW,vecFiles,N_it);
            else
                std::cout << "The DMFT loop has been skipped since according to the directories, it has already been created for this set of parameters." << std::endl;
            #endif

            if (load_first){ // The file containing wider Matsubara frequency domain is loaded for spline.
                std::vector<double> initVec(2*MULT_N_TAU*N_tau,0.0); // Important that it is 2*MULT_N_TAU
                IPT2::SplineInline< std::complex<double> > splInlineObj(MULT_N_TAU*N_tau,initVec);
                try{
                    splInlineObj.loadFileSpline(filenameToLoad); // Spline is ready for use by calling function calculateSpline()-
                } catch(const std::exception& err){ // If filenameToLoad is not found...
                    std::cerr << err.what() << "\n";
                    std::cerr << "Check if data with "+std::to_string(MULT_N_TAU)+" times the N_tau selected is available...\n";
                    exit(1);
                }
                //std::complex<double> test = splInlineObj.calculateSpline(0.4); // Should include this into test file with the proper file to load.
                //std::cout << test << std::endl;
                Susceptibility<IPT2::DMFTproc> susObj;
                std::ofstream outputChispspGamma, outputChispspWeights, outputChispspBubble, outputChispspBubbleCorr;
                std::string trailingStr = is_full ? "_full" : "";
                std::string strOutputChispspGamma(pathToDir+customDirName+"/susceptibilities/ChispspGamma_"+std::to_string(DIM)+"D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_N_tau_"+std::to_string(N_tau)+"_Nk_"+std::to_string(N_k)+trailingStr+".dat");
                std::string strOutputChispspWeights(pathToDir+customDirName+"/susceptibilities/ChispspWeights_"+std::to_string(DIM)+"D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_N_tau_"+std::to_string(N_tau)+"_Nk_"+std::to_string(N_k)+trailingStr+".dat");
                std::string strOutputChispspBubble(pathToDir+customDirName+"/susceptibilities/ChispspBubble_"+std::to_string(DIM)+"D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_N_tau_"+std::to_string(N_tau)+"_Nk_"+std::to_string(N_k)+trailingStr+".dat");
                std::string strOutputChispspBubbleCorr(pathToDir+customDirName+"/susceptibilities/ChispspBubbleCorr_"+std::to_string(DIM)+"D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_N_tau_"+std::to_string(N_tau)+"_Nk_"+std::to_string(N_k)+trailingStr+".dat");
                for (size_t ktilde=0; ktilde<vecK.size(); ktilde++){
                    std::cout << "ktilde: " << ktilde << "\n";
                    for (size_t kbar=0; kbar<vecK.size(); kbar++){
                        std::cout << "kbar: " << kbar << "\n";
                        std::complex<double> tmp_val_kt_kb(0.0,0.0), tmp_val_kt_kb_bubble(0.0,0.0), tmp_val_weights(0.0,0.0), tmp_val_bubble_corr(0.0,0.0);
                        for (size_t wtilde=static_cast<size_t>(iwnArr_l.size()/2); wtilde<iwnArr_l.size(); wtilde++){
                            for (size_t wbar=static_cast<size_t>(iwnArr_l.size()/2); wbar<iwnArr_l.size(); wbar++){
                                std::tuple< std::complex<double>, std::complex<double>, std::complex<double> > gammaStuffFull;
                                std::tuple< std::complex<double>, std::complex<double> > gammaStuff;
                                if ( (wtilde==static_cast<size_t>(iwnArr_l.size()/2)) && (wbar==static_cast<size_t>(iwnArr_l.size()/2)) ){
                                    if (!is_full){
                                        gammaStuff=susObj.gamma_oneD_spsp_plotting(EqDMFTA,vecK[ktilde],iwnArr_l[wtilde],vecK[kbar],iwnArr_l[wbar],HF::K_1D(0.0,std::complex<double>(0.0,0.0)),splInlineObj);
                                        tmp_val_kt_kb+=std::get<0>(gammaStuff);
                                        tmp_val_kt_kb_bubble+=std::get<1>(gammaStuff);
                                    } else{
                                        gammaStuffFull=susObj.gamma_oneD_spsp_full_middle_plotting(EqDMFTA,vecK[ktilde],iwnArr_l[wtilde],vecK[kbar],iwnArr_l[wbar],HF::K_1D(0.0,std::complex<double>(0.0,0.0)),splInlineObj);
                                        tmp_val_kt_kb+=std::get<0>(gammaStuffFull);
                                        tmp_val_kt_kb_bubble+=std::get<1>(gammaStuffFull);
                                        tmp_val_bubble_corr+=std::get<2>(gammaStuffFull);
                                    }
                                    tmp_val_weights+=susObj.get_weights(EqDMFTA,vecK[ktilde],iwnArr_l[wtilde],vecK[kbar],iwnArr_l[wbar],HF::K_1D(0.0,std::complex<double>(0.0,0.0)),splInlineObj);
                                }
                            }
                        }
                        outputChispspWeights.open(strOutputChispspWeights, std::ofstream::in | std::ofstream::app);
                        outputChispspGamma.open(strOutputChispspGamma, std::ofstream::in | std::ofstream::app);
                        outputChispspBubble.open(strOutputChispspBubble, std::ofstream::in | std::ofstream::app);
                        outputChispspWeights << tmp_val_weights << " ";
                        outputChispspGamma << tmp_val_kt_kb << " ";
                        outputChispspBubble << tmp_val_kt_kb_bubble << " ";
                        if (is_full){
                            outputChispspBubbleCorr.open(strOutputChispspBubbleCorr, std::ofstream::in | std::ofstream::app);
                            outputChispspBubbleCorr << tmp_val_bubble_corr << " ";
                            outputChispspBubbleCorr.close();
                        }
                        outputChispspWeights.close();
                        outputChispspGamma.close();
                        outputChispspBubble.close();
                    }
                    outputChispspWeights.open(strOutputChispspWeights, std::ofstream::in | std::ofstream::app);
                    outputChispspGamma.open(strOutputChispspGamma, std::ofstream::in | std::ofstream::app);
                    outputChispspBubble.open(strOutputChispspBubble, std::ofstream::in | std::ofstream::app);
                    outputChispspWeights << "\n";
                    outputChispspGamma << "\n";
                    outputChispspBubble << "\n";
                    if (is_full){
                        outputChispspBubbleCorr.open(strOutputChispspBubbleCorr, std::ofstream::in | std::ofstream::app);
                        outputChispspBubbleCorr << "\n";
                        outputChispspBubbleCorr.close();
                    }
                    outputChispspWeights.close();
                    outputChispspGamma.close();
                    outputChispspBubble.close();
                }
            }
        }
    }

    return 0;
}
