#include "src/IPT2nd3rdorderSingle2.hpp"
#include "src/integral_utils.hpp"


int main(int argc, char** argv){

    const double n_t_spin=0.5;
    const unsigned int N_tau=512; // 4096
    const unsigned int N_k=400;
    const unsigned int N_it=100;
    const double beta=5;
    const double U=14;
    #if DIM == 1
    const double Hyb_c=2; // For the 1D chain with nearest-neighbor hopping, it is 2t.
    #elif DIM == 2
    const double Hyb_c=4; // For the 2D square lattice with nearest-neighbor hopping. 
    #endif
    const Integrals integralsObj;
    std::ofstream objSaveStreamGloc;
    std::ofstream objSaveStreamSE;
    std::ofstream objSaveStreamGW;

    for (size_t k=0; k<=N_k; k++){
        double epsilonk = -1.0*M_PI + 2.0*(double)k*M_PI/N_k;
        vecK.push_back(epsilonk);
    }
    for (signed int j=(-(signed int)N_tau); j<(signed int)N_tau; j++){
        std::complex<double> matFreq(0.0 , (2.0*(double)j+1.0)*M_PI/beta );
        iwnArr_l.push_back( matFreq );
    }
    
    std::string pathToDir("./data/");
    std::string trailingStr("");
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

    // Initializing the main Green's function objects.
    GreenStuff WeissGreenA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,weiss_green_A_matsubara_t_pos,weiss_green_A_matsubara_t_neg,weiss_green_A_matsubara_w);
    GreenStuff HybA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,weiss_green_tmp_A_matsubara_t_pos,weiss_green_tmp_A_matsubara_t_neg,weiss_green_tmp_A_matsubara_w);
    GreenStuff SelfEnergyA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,self_A_matsubara_t_pos,self_A_matsubara_t_neg,self_A_matsubara_w);
    GreenStuff LocalGreenA(N_tau,N_k,beta,U,Hyb_c,iwnArr_l,local_green_A_matsubara_t_pos,local_green_A_matsubara_t_neg,local_green_A_matsubara_w);

    IPT2::DMFTproc EqDMFTA(WeissGreenA,HybA,LocalGreenA,SelfEnergyA,data_dG_dtau_pos,data_dG_dtau_neg,vecK,n_t_spin);

    /* DMFT loop */
    unsigned int iter=1;
    double n,n0,G0_diff=0.0;
    bool converged=false;
    arma::Cube< std::complex<double> > WeissGreenTmpA(2,2,iwnArr_l.size()); // Initialization of the hybridization function.
    arma::Cube< std::complex<double> > G0_density_mu0_m(2,2,iwnArr_l.size()), G_density_mu_m(2,2,iwnArr_l.size());
    std::function<double(double)> wrapped_density_mu, wrapped_density_mu0;
    for (size_t i=0; i<iwnArr_l.size(); i++) HybA.matsubara_w.slice(i)(0,0) = Hyb_c/iwnArr_l[i];
    while (iter<=N_it && !converged){ // Everything has been designed to accomodate a bipartite lattice.
        G0_diff=0.0; // resetting G0_diff
        objSaveStreamGloc.open(filenameToSaveGloc+"_Nit_"+std::to_string(iter)+".dat", std::ios::out | std::ios::app);
        objSaveStreamSE.open(filenameToSaveSE+"_Nit_"+std::to_string(iter)+".dat", std::ios::out | std::ios::app);
        objSaveStreamGW.open(filenameToSaveGW+"_Nit_"+std::to_string(iter)+".dat", std::ios::out | std::ios::app);
        for (size_t j=0; j<iwnArr_l.size(); j++){
            WeissGreenA.matsubara_w.slice(j)(0,0)=1.0/(iwnArr_l[j]+WeissGreenA.get_mu0()-HybA.matsubara_w.slice(j)(0,0));
        }
        EqDMFTA.update_impurity_self_energy(); // Spits out Sigma(iwn). To begin with, mu=U/2.0.
        for (size_t j=0; j<iwnArr_l.size(); j++){
            G0_density_mu0_m.slice(j)(0,0)=iwnArr_l[j]-HybA.matsubara_w.slice(j)(0,0);
            G_density_mu_m.slice(j)(0,0)=iwnArr_l[j]-HybA.matsubara_w.slice(j)(0,0)-SelfEnergyA.matsubara_w.slice(j)(0,0);
        }
        std::cout << "mu: " << SelfEnergyA.get_mu() << " and mu0 " << SelfEnergyA.get_mu0() << std::endl;
        // Compute the various chemical potentials to get n_target
        n=EqDMFTA.density_mu(G_density_mu_m);
        n0=EqDMFTA.density_mu0(G0_density_mu0_m);

        std::cout << "n0: " << n0 <<  " for mu0: " << WeissGreenA.get_mu0() << std::endl;
        std::cout << "n: " << n << " for mu: " << WeissGreenA.get_mu() << std::endl;

        if ( ( std::abs(n-n_t_spin)>ROOT_FINDING_TOL ) && iter>1 ){
            try{
                wrapped_density_mu = [&](double mu){ return EqDMFTA.density_mu(mu,G_density_mu_m)-n_t_spin; };
                double mu_new = integralsObj.falsePosMethod(wrapped_density_mu,-20.,20.);
                std::cout << "mu_new at iter " << iter << ": " << mu_new << std::endl;
                WeissGreenA.update_mu(mu_new); // Updates mu from instance, even though it is a static member variable.
            }catch (const std::exception& err){
                std::cerr << err.what() << std::endl;
            }
        }
        if ( ( std::abs(n0-n_t_spin)>ROOT_FINDING_TOL ) ){
            try{
                wrapped_density_mu0 = [&](double mu0){ return EqDMFTA.density_mu0(mu0,G0_density_mu0_m)-n_t_spin; };
                double mu0_new = integralsObj.falsePosMethod(wrapped_density_mu0,-20.,20.);
                std::cout << "mu0_new at iter " << iter << ": " << mu0_new << std::endl;
                WeissGreenA.update_mu0(mu0_new); // Updates mu from instance, even though it is a static member variable.
            }catch (const std::exception& err){
                std::cerr << err.what() << std::endl;
            }
        }
        // Determining G_loc
        for (size_t j=0; j<iwnArr_l.size(); j++){
            #if DIM == 1
            std::function<std::complex<double>(double,std::complex<double>)> G_latt = [&](double kx, std::complex<double> iwn){
                return 1.0/(iwn + WeissGreenA.get_mu() - epsilonk(kx) - SelfEnergyA.matsubara_w.slice(j)(0,0));
            };
            LocalGreenA.matsubara_w.slice(j)(0,0)=1./(2.*M_PI)*integralsObj.integrate_simps(G_latt,-M_PI,M_PI,iwnArr_l[j],0.001);
            #elif DIM == 2
            std::function<std::complex<double>(double,double,std::complex<double>)> G_latt = [&](double kx, double ky, std::complex<double> iwn){
                return 1.0/(iwn + WeissGreenA.get_mu() - epsilonk(kx,ky) - SelfEnergyA.matsubara_w.slice(j)(0,0));
            };
            LocalGreenA.matsubara_w.slice(j)(0,0)=1./(4.*M_PI*M_PI)*integralsObj.I2D(G_latt,-M_PI,M_PI,-M_PI,M_PI,iwnArr_l[j]);
            #endif
        }
        // Updating the hybridization function for next round.
        for (size_t j=0; j<iwnArr_l.size(); j++){
            HybA.matsubara_w.slice(j)(0,0) = iwnArr_l[j] + WeissGreenA.get_mu() - SelfEnergyA.matsubara_w.slice(j)(0,0) - 1.0/LocalGreenA.matsubara_w.slice(j)(0,0);
        }
        if (iter>2){
            for (size_t j=0; j<iwnArr_l.size(); j++) G0_diff+=std::abs(WeissGreenTmpA.slice(j)(0,0)-WeissGreenA.matsubara_w.slice(j)(0,0));
            std::cout << "G0_diff: " << G0_diff << std::endl;
            if (G0_diff<0.001 && std::abs(n-n_t_spin)<ROOT_FINDING_TOL && std::abs(n0-n_t_spin)<ROOT_FINDING_TOL) converged=true; 
        }
        if (iter>2){ // Assess the convergence process 
            for (size_t j=0; j<iwnArr_l.size(); j++){
                WeissGreenTmpA.slice(j)(0,0) = WeissGreenA.matsubara_w.slice(j)(0,0);
            }
        }
        std::cout << "new n0: " << EqDMFTA.density_mu0(G0_density_mu0_m) <<  " for mu0: " << WeissGreenA.get_mu0() << std::endl;
        std::cout << "new n: " << EqDMFTA.density_mu(G_density_mu_m) << " for mu: " << WeissGreenA.get_mu() << std::endl;
        std::cout << "double occupancy: " << EqDMFTA.double_occupancy() << std::endl;
        saveEachIt(EqDMFTA,objSaveStreamGloc,objSaveStreamSE,objSaveStreamGW);
        std::cout << "iteration #" << iter << std::endl;
        iter++;
    }

    return 0;
}
