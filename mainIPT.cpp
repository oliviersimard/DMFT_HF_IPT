//#define PARALLEL
#ifdef PARALLEL
#include "src/thread_utils.hpp"
#else
#include "src/susceptibilities.hpp"
#endif
#include "src/json_utils.hpp"

int main(int argc, char** argv){
    #ifdef PARALLEL
    int world_rank;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    #endif
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
    const bool is_full=params.boo_arr[0], load_self=params.boo_arr[1], is_jj=params.boo_arr[2];
    const char* solver_type=params.char_arr[0];
    std::string solver_type_s(solver_type);
    #if DIM == 1
    const double Hyb_c=2; // For the 1D chain with nearest-neighbor hopping, it is 2t.
    #elif DIM == 2
    const double Hyb_c=4; // For the 2D square lattice with nearest-neighbor hopping. 
    #endif
    std::ofstream objSaveStreamGloc;
    std::ofstream objSaveStreamSE;
    std::ofstream objSaveStreamGW;

    for (size_t k=0; k<=N_k; k++){ // Used when computing the susceptibilities.
        double epsilonk = -1.0*M_PI + 2.0*(double)k*M_PI/N_k;
        vecK.push_back(epsilonk);
    }
    #ifdef PARALLEL
    matGamma = arma::Mat< std::complex<double> >(vecK.size(),vecK.size(),arma::fill::zeros); // Matrices used in case parallel.
    matWeigths = arma::Mat< std::complex<double> >(vecK.size(),vecK.size(),arma::fill::zeros);
    matTotSus = arma::Mat< std::complex<double> >(vecK.size(),vecK.size(),arma::fill::zeros);
    matCorr = arma::Mat< std::complex<double> >(vecK.size(),vecK.size(),arma::fill::zeros);
    matMidLev = arma::Mat< std::complex<double> >(vecK.size(),vecK.size(),arma::fill::zeros);
    // Allocating
    gamma_tensor = new std::complex<double>***[vecK.size()];
    for (size_t i=0; i<vecK.size(); i++){
        gamma_tensor[i] = new std::complex<double>**[N_tau];
        for (size_t j=0; j<N_tau; j++){
            gamma_tensor[i][j] = new std::complex<double>*[vecK.size()];
            for (size_t k=0; k<vecK.size(); k++){
                gamma_tensor[i][j][k] = new std::complex<double>[N_tau];
            }
        } 
    }
    // To be able to initialize the static variables important to IPT (GreenStuff), one has to instantiate a GreenStuff object.
    arma::Cube<double> initiate_double_slots; arma::Cube< std::complex<double> > initiate_cplx_double_slot;
    #else // Saving on memory doing so. For DMFT loop, no parallelization is needed.
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
    #endif /* PARALLEL */
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

            /* Computing according to solver chosen. */
            if ( (solver_type_s.compare("IPT")==0) ){
                std::string customDirName(std::to_string(DIM)+"D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_n_"+std::to_string(n_t_spin)+trailingStr+"_N_tau_"+std::to_string(N_tau));
                std::string filenameToSaveGloc(pathToDir+customDirName+"/Green_loc_"+customDirName);
                std::string filenameToSaveSE(pathToDir+customDirName+"/Self_energy_"+customDirName);
                std::string filenameToSaveGW(pathToDir+customDirName+"/Weiss_Green_"+customDirName);
                std::vector< std::string > vecFiles={filenameToSaveGloc,filenameToSaveSE,filenameToSaveGW};
                try{ // Ensures that we don't overwrite any files within build/.
                    check_file_content(vecFiles,pathToDir+customDirName+"/analytic_continuations",pathToDir+customDirName+"/susceptibilities"); // Checks whether files already exist to avoid overwritting. Also creates 
                } catch (const std::runtime_error& err){                                                                                            // directory architecture
                    std::cerr << err.what() << "\n";
                    exit(1);
                }
                std::string filenameToLoad;
                if (load_self){ // This file has got to be containing the self-energy data. For at least twice the number N_tau for proper interpolation in formulae.
                    std::string customDirNameLoad(std::to_string(DIM)+"D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_n_"+std::to_string(n_t_spin)+trailingStr+"_N_tau_"+std::to_string(MULT_N_TAU*N_tau));
                    #ifndef DEBUG
                    filenameToLoad=pathToDirLoad+customDirNameLoad+"/Self_energy_"+customDirNameLoad; // Stick to the self-energy.
                    #else
                    filenameToLoad=pathToDir+customDirNameLoad+"/Self_energy_"+customDirNameLoad; // Stick to the self-energy.
                    #endif
                }
                // Saving memory doing so. For DMFT loop, no parallelization is needed.
                #ifndef PARALLEL
                // Initializing the main Green's function objects.
                GreenStuff WeissGreenA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,weiss_green_A_matsubara_t_pos,weiss_green_A_matsubara_t_neg,weiss_green_A_matsubara_w);
                GreenStuff HybA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,weiss_green_tmp_A_matsubara_t_pos,weiss_green_tmp_A_matsubara_t_neg,weiss_green_tmp_A_matsubara_w);
                GreenStuff SelfEnergyA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,self_A_matsubara_t_pos,self_A_matsubara_t_neg,self_A_matsubara_w);
                GreenStuff LocalGreenA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,local_green_A_matsubara_t_pos,local_green_A_matsubara_t_neg,local_green_A_matsubara_w);

                IPT2::DMFTproc EqDMFTA(WeissGreenA,HybA,LocalGreenA,SelfEnergyA,data_dG_dtau_pos,data_dG_dtau_neg,vecK,n_t_spin);
            
                /* Performs the complete DMFT calculations if directory doesn't already exist */
                #ifndef DEBUG
                int message = stat( (pathToDirLoad+customDirName).c_str(), &infoDir );
                if ( !(infoDir.st_mode & S_IFDIR) && message!=0 ){ // If the directory doesn't already exist in ../data/ ...
                    DMFTloop(EqDMFTA,objSaveStreamGloc,objSaveStreamSE,objSaveStreamGW,vecFiles,N_it);
                }
                else
                    std::cout << "The DMFT loop has been skipped since according to the directories, it has already been created for this set of parameters." << std::endl;
                #endif /* DEBUG */
                #else
                // Initializing the static member variables to GreenStuff.
                GreenStuff IPTStaticVariables(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,initiate_double_slots,initiate_double_slots,initiate_cplx_double_slot);
                std::cout << " U: " << U << "\n";
                std::cout << " beta: " << beta << "\n";
                std::cout << " N_k: " << N_k << "\n";
                std::cout << " N_tau: " << N_tau << std::endl;
                #endif /* PARALLEL */
                if (load_self){ // The file containing wider Matsubara frequency domain is loaded for spline.
                    std::vector<double> initVec(2*MULT_N_TAU*N_tau,0.0); // Important that it is 2*MULT_N_TAU
                    IPT2::SplineInline< std::complex<double> > splInlineObj(MULT_N_TAU*N_tau,initVec,vecK,iwnArr_l,iqnArr_l);
                    try{
                        splInlineObj.loadFileSpline(filenameToLoad); // Spline is ready for use by calling function calculateSpline()-
                    } catch(const std::invalid_argument& err){ // If filenameToLoad is not found...
                        std::cerr << err.what() << "\n";
                        std::cerr << "Check if data with "+std::to_string(MULT_N_TAU)+" times the N_tau selected is available...\n";
                        exit(1);
                    } catch(const std::runtime_error& err){
                        std::cerr << err.what() << "\n";
                    }
                    //std::complex<double> test = splInlineObj.calculateSpline(0.4); // Should include this into test file with the proper file to load.
                    //std::cout << test << std::endl;

                    /* Calculation Susceptibilities. The second template argument specifies the type of the SplineInline object. */
                    #ifdef PARALLEL
                    MPI_Barrier(MPI_COMM_WORLD);
                    calculateSusceptibilitiesParallel<IPT2::DMFTproc>(splInlineObj,pathToDir,customDirName,is_full,is_jj,ThreadFunctor::solver_prototype::IPT2_prot);
                    #else
                    calculateSusceptibilities<IPT2::DMFTproc>(EqDMFTA,splInlineObj,pathToDir,customDirName,is_full,is_jj);
                    #endif
                } else{
                    std::cout << "To compute the susceptibilities, you must load a self-energy defined on a wider Matsubara frequency" << "\n";
                    std::cout << "range (MULT_N_TAU times larger) to be able to interpolate it using a cubic spline." << std::endl;
                }
            }
            else if ( (solver_type_s.compare("HF")==0) ){
                double mu_HF=U/2.0; // Half-filling
                double ndo=0.6;
                std::vector< std::complex<double> > Gk_up(N_tau,0.0);
                HF::FunctorBuildGk Gk(mu_HF,beta,U,ndo,vecK,N_it,N_k,Gk_up);
                /* Computing the corresponding HF self-energy. */
                #if DIM == 1
                Gk.update_ndo_1D();
                #elif DIM == 2
                Gk.update_ndo_2D();
                #endif
                double ndo_converged=Gk.get_ndo();
                std::cout << "The final density on impurity is: " << ndo_converged << "\n";
                std::string customDirName(std::to_string(DIM)+"D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_n_"+std::to_string(ndo_converged)+trailingStr+"_N_tau_"+std::to_string(N_tau));
                std::string filenameToSaveGloc(pathToDir+customDirName+"/Green_loc_"+customDirName);
                std::string filenameToSaveSE(pathToDir+customDirName+"/Self_energy_"+customDirName);
                std::string filenameToSaveGW(pathToDir+customDirName+"/Weiss_Green_"+customDirName);
                std::vector< std::string > vecFiles={filenameToSaveGloc,filenameToSaveSE,filenameToSaveGW};
                try{ // Ensures that we don't overwrite any files within build/.
                    check_file_content(vecFiles,pathToDir+customDirName+"/analytic_continuations",pathToDir+customDirName+"/susceptibilities"); // Checks whether files already exist to avoid overwritting. Also creates 
                } catch (const std::exception& err){                                                                                            // directory architecture
                    std::cerr << err.what() << "\n";
                    exit(1);
                }
                #ifdef PARALLEL
                MPI_Barrier(MPI_COMM_WORLD);
                calculateSusceptibilitiesParallel<HF::FunctorBuildGk>(Gk,pathToDir,customDirName,is_full,is_jj,ndo_converged,ThreadFunctor::solver_prototype::HF_prot);
                #else
                calculateSusceptibilities<HF::FunctorBuildGk>(Gk,pathToDir,customDirName,is_full,is_jj);
                #endif // PARALLEL
            }
        }
    }
    // Deallocating
    #ifdef PARALLEL
    for (size_t i=0; i<vecK.size(); i++){
        for (size_t j=0; j<N_tau; j++){
            for (size_t k=0; k<vecK.size(); k++){
                delete[] gamma_tensor[i][j][k];
            }
            delete[] gamma_tensor[i][j];
        }
        delete[] gamma_tensor[i];
    }
    delete[] gamma_tensor;
    #endif
    return 0;
}
