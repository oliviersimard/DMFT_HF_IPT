#include "IPT2nd3rdorderSingle2.hpp"

//using namespace std;

/**************************************************************************************************/

namespace IPT2{

double DMFTproc::n=0.0, DMFTproc::n0=0.0;
unsigned int DMFTproc::objCount=0;

DMFTproc::DMFTproc(GreenStuff& WeissGreen_,GreenStuff& Hyb_,GreenStuff& LocalGreen_, GreenStuff& SelfEnergy_,
                                                arma::Cube<double>& dg_dtau_pos, arma::Cube<double>& dg_dtau_neg,
                                                const std::vector<double>& karr_l_, const double n_t_spin_) : WeissGreen(WeissGreen_),
                                                Hyb(Hyb_), LocalGreen(LocalGreen_), SelfEnergy(SelfEnergy_),
                                                data_dg_dtau_pos(dg_dtau_pos), data_dg_dtau_neg(dg_dtau_neg), karr_l(karr_l_){
    std::cout << "DMFTproc: N_k: " << GreenStuff::N_k << std::endl;
    std::cout << "DMFTproc: N_tau: " << GreenStuff::N_tau << std::endl;
    if (objCount<1){
        this->n=n_t_spin_;
    }
    objCount++;
}

void DMFTproc::update_parametrized_self_energy(FFTtools FFTObj){
    /* update_impurity_self_energy() has to be launched before this very method */
    // To compute the spline, the derivatives have to be known
    FFTObj.fft_spec(WeissGreen,WeissGreen,data_dg_dtau_pos,data_dg_dtau_neg,FFTObj.dG_dtau_positive);
    FFTObj.fft_spec(WeissGreen,WeissGreen,data_dg_dtau_pos,data_dg_dtau_neg,FFTObj.dG_dtau_negative);
    spline<double> s;
    // Splitting up the real and imaginary parts for the cubic spline procedure.
    double left_der=2.0*WeissGreen.matsubara_t_pos.slice(0)(0,0)*WeissGreen.matsubara_t_neg.slice(0)(0,0)*data_dg_dtau_pos.slice(0)(0,0) + WeissGreen.matsubara_t_pos.slice(0)(0,0)*WeissGreen.matsubara_t_pos.slice(0)(0,0)*data_dg_dtau_neg.slice(0)(0,0);
    double right_der=2.0*WeissGreen.matsubara_t_pos.slice(2.0*GreenStuff::N_tau)(0,0)*WeissGreen.matsubara_t_neg.slice(2.0*GreenStuff::N_tau)(0,0)*data_dg_dtau_pos.slice(2.0*GreenStuff::N_tau)(0,0) + WeissGreen.matsubara_t_pos.slice(2.0*GreenStuff::N_tau)(0,0)*WeissGreen.matsubara_t_pos.slice(2.0*GreenStuff::N_tau)(0,0)*data_dg_dtau_neg.slice(2.0*GreenStuff::N_tau)(0,0);
    s.set_boundary(spline<double>::bd_type::first_deriv,left_der,spline<double>::bd_type::first_deriv,right_der);
    // tau array
    std::vector<double> tau_arr;
    for (int j=0; j<=2.0*GreenStuff::N_tau; j++){
        tau_arr.push_back(j*GreenStuff::beta/(2.0*GreenStuff::N_tau));
    }
    s.set_points(tau_arr,SelfEnergy.matsubara_t_pos);
    s.iwn_tau_spl_extrm(SelfEnergy,GreenStuff::beta,GreenStuff::N_tau); // Sets up for the new Sigma(iwn) computed with the cubic spline.
}

void DMFTproc::update_impurity_self_energy(){ // Returns the full parametrized self-energy.
    FFTtools FFTObj;
    // parametrization of the self-energy allowing to wander off half-filling.
    double A = 2.0*n*(2.0-2.0*n)/( 2.0*n*(2.0-2.0*n) ); // IPTD becomes one because n_0 is set to be equal to n using right chemical potential mu_0.
    // Also, in the paramagnetic state for a single impurity site, n_up=n_down, hence the total electron density is 2.0*n_t_spin.
    double B = ( (1.0-n)*GreenStuff::U + GreenStuff::mu0 - GreenStuff::mu )/( n*(1.0-n)*GreenStuff::U*GreenStuff::U );
    std::cout << "U: " << GreenStuff::U << " and beta : " << GreenStuff::beta << std::endl; 
    // Compute the Fourier transformation to tau space for impurity self-energy calculation
    // Stores data in matsubara_t_pos and matsubara_t_neg
    FFTObj.fft_spec(WeissGreen,WeissGreen,data_dg_dtau_pos,data_dg_dtau_neg,FFTObj.plain_positive);
    FFTObj.fft_spec(WeissGreen,WeissGreen,data_dg_dtau_pos,data_dg_dtau_neg,FFTObj.plain_negative);
    for (size_t j=0; j<=2*GreenStuff::N_tau; j++){ // IPT2 self-energy diagram
        SelfEnergy.matsubara_t_pos.slice(j)(0,0)=WeissGreen.matsubara_t_pos.slice(j)(0,0)*WeissGreen.matsubara_t_pos.slice(j)(0,0)*WeissGreen.matsubara_t_neg.slice(j)(0,0);
    }
    update_parametrized_self_energy(FFTObj); // Updates Sigma(iwn) for IPT2::
    for (size_t i=0; i<2*GreenStuff::N_tau; i++){
        SelfEnergy.matsubara_w.slice(i)(0,0) = GreenStuff::U*n + A*GreenStuff::U*GreenStuff::U*SelfEnergy.matsubara_w.slice(i)(0,0)/(1.0-B*GreenStuff::U*GreenStuff::U*SelfEnergy.matsubara_w.slice(i)(0,0));
    }
}

double DMFTproc::density_mu(double mu, const arma::Cube< std::complex<double> >& G) const{
    double n=0.0;
    for (size_t j=0; j<iwnArr_l.size(); j++){
        n += ( 1.0/( G.slice(j)(0,0) + mu ) - 1./iwnArr_l[j] ).real();
    }
    n*=1./GreenStuff::beta;
    n+=0.5;
    return n;
}

double DMFTproc::density_mu(const arma::Cube< std::complex<double> >& G) const{
    double n=0.0;
    for (size_t j=0; j<iwnArr_l.size(); j++){ // Takes in actual mu value.
        n += ( 1.0/( G.slice(j)(0,0) + GreenStuff::mu ) - 1./iwnArr_l[j] ).real();
    }
    n*=1./GreenStuff::beta;
    n+=0.5;
    return n;
}

double DMFTproc::density_mu0(double mu0, const arma::Cube< std::complex<double> >& G0_1) const{
    double n0=0.0;
    for (size_t j=0; j<iwnArr_l.size(); j++){
        n0 += ( 1./( G0_1.slice(j)(0,0) + mu0 ) - 1./iwnArr_l[j] ).real();
    }
    n0*=1./GreenStuff::beta;
    n0+=0.5;
    return n0;
}

double DMFTproc::density_mu0(const arma::Cube< std::complex<double> >& G0_1) const{
    double n0=0.0;
    for (size_t j=0; j<iwnArr_l.size(); j++){
        n0 += ( 1./( G0_1.slice(j)(0,0) + GreenStuff::mu0 ) - 1./iwnArr_l[j] ).real();
    }
    n0*=1./GreenStuff::beta;
    n0+=0.5;
    return n0;
}

double DMFTproc::double_occupancy() const{
    double D=0.0;
    for (size_t j=static_cast<size_t>(iwnArr_l.size()/2); j<iwnArr_l.size(); j++){
        D += ( ( SelfEnergy.matsubara_w.slice(j)(0,0) - GreenStuff::U*n ) * LocalGreen.matsubara_w.slice(j)(0,0) ).real();
    }
    D*=1./(GreenStuff::beta*GreenStuff::U);
    D+=n*n;
    return D;
}


}/* End of namespace IPT2 */

/**************************************************************************************************/

void FFTtools::fft_t2w(arma::Cube<double>& data1, arma::Cube< std::complex<double> >& data2){
    std::complex<double>* inUp=new std::complex<double> [GreenStuff::N_tau];
    std::complex<double>* outUp=new std::complex<double> [GreenStuff::N_tau];
    fftw_plan pUp; //fftw_plan pDown;
    inUp[0]=0.5*data1.slice(0)(0,0);
    for(size_t j=1;j<GreenStuff::N_tau;j++){
        inUp[j]=data1.slice(j)(0.0)*exp(std::complex<double>(0.0,(double)(GreenStuff::N_tau-1)*(double)j*M_PI/(double)GreenStuff::N_tau));
    }
    pUp=fftw_plan_dft_1d(GreenStuff::N_tau, reinterpret_cast<fftw_complex*>(inUp), reinterpret_cast<fftw_complex*>(outUp), FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(pUp); //fftw_execute(pDown);
    for(size_t k=0;k<GreenStuff::N_tau;k++){
        data2.slice(k)(0,0)=(outUp[k]+0.5*data1.slice(GreenStuff::N_tau)(0,0)*exp(std::complex<double>(0.0,(double)(GreenStuff::N_tau-1)*M_PI)))*GreenStuff::beta/(double)GreenStuff::N_tau;
    }
    delete [] inUp;
    delete [] outUp;
    fftw_destroy_plan(pUp); //fftw_destroy_plan(pDown);
}

void FFTtools::fft_w2t(arma::Cube< std::complex<double> >& data1, arma::Cube<double>& data2){
    std::complex<double>* inUp=new std::complex<double> [GreenStuff::N_tau];
    std::complex<double>* outUp=new std::complex<double> [GreenStuff::N_tau];
    fftw_plan pUp; //fftw_plan pDown;
    for(size_t k=0;k<GreenStuff::N_tau;k++){
        inUp[k]=data1.slice(k)(0,0);
    }
    pUp=fftw_plan_dft_1d(GreenStuff::N_tau, reinterpret_cast<fftw_complex*>(inUp), reinterpret_cast<fftw_complex*>(outUp), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(pUp); //fftw_execute(pDown);

    for(size_t j=0;j<GreenStuff::N_tau;j++){
        data2.slice(j)(0,0)=(outUp[j]*exp(std::complex<double>(0.0,(double)(GreenStuff::N_tau-1)*(double)j*M_PI/(double)GreenStuff::N_tau))).real()/GreenStuff::beta;
    }
    data2.slice(GreenStuff::N_tau)(0,0)=0.0; // Up spin
    for(size_t k=0;k<GreenStuff::N_tau;k++){
        data2.slice(GreenStuff::N_tau)(0,0)+=(data1.slice(k)(0,0)*exp(std::complex<double>(0.0,(double)(GreenStuff::N_tau-1)*M_PI))).real()/GreenStuff::beta;
    }
    delete [] inUp;
    delete [] outUp;
    fftw_destroy_plan(pUp); //fftw_destroy_plan(pDown);
}

void FFTtools::fft_spec(GreenStuff& data1, GreenStuff& data2, arma::Cube<double>& data_dg_dtau_pos,
                                                arma::Cube<double>& data_dg_dtau_neg, Spec specialization){
    /* FFT from Matsubara frequency to imaginary time */
    arma::Cube< std::complex<double> > green_inf_v = data1.green_inf();
    const unsigned int timeGrid = 2*GreenStuff::N_tau;
    std::vector< std::complex<double> > substractG;
    for (unsigned int j=0; j<timeGrid; j++){
        substractG.push_back(data1.matsubara_w.slice(j)(0,0) - green_inf_v.slice(j)(0,0));
    }
    assert(substractG.size()==green_inf_v.n_slices && "Check FFTtools for mismatch between arrays.");
    
    double F_decal[2*timeGrid];

    double tau;
    const std::complex<double> im(0.0,1.0);
    switch (specialization){
    case plain_positive:
    
        for(size_t i=0; i<timeGrid; i++){
            F_decal[2*i]=substractG[i].real();
            F_decal[2*i+1]=substractG[i].imag();
        }
        
        gsl_fft_complex_radix2_forward(F_decal, 1, timeGrid);

        for(size_t i=0; i<timeGrid; i++){
            tau=i*GreenStuff::beta/(double)timeGrid;
            data2.matsubara_t_pos.slice(i)(0,0) = ( (1./GreenStuff::beta) * std::exp( -M_PI*i*im*(1.0/(double)timeGrid-1.0) ) * (F_decal[2*i]+im*F_decal[2*i+1]) + Hsuperior(tau, GreenStuff::mu0, GreenStuff::hyb_c, GreenStuff::beta) ).real();
        }
        // Adding value at tau=beta
        data2.matsubara_t_pos.slice(timeGrid)(0,0)=-1.0-data2.matsubara_t_pos.slice(0)(0,0);
        break;
    case plain_negative:

        for(size_t i=0; i<timeGrid; i++){
            F_decal[2*i]=substractG[i].real();
            F_decal[2*i+1]=-1.0*(substractG[i].imag()); // complex conjugate
        }
        gsl_fft_complex_radix2_forward(F_decal, 1, timeGrid);
        for(unsigned int i=0; i<timeGrid; i++){
            tau=i*GreenStuff::beta/(double)timeGrid;
            data2.matsubara_t_neg.slice(i)(0,0) = ( (1./GreenStuff::beta) * std::exp( -i*M_PI*im*(1.0/((double)timeGrid)-1.0) ) * (F_decal[2*i]+im*F_decal[2*i+1]) + Hinferior(tau, GreenStuff::mu0, GreenStuff::hyb_c, GreenStuff::beta) ).real();
        }
        // Adding value at tau=beta
        data2.matsubara_t_neg.slice(timeGrid)(0,0)=1.0-data2.matsubara_t_neg.slice(0)(0,0);
        break;
    case dG_dtau_positive:

        for(size_t i=0; i<timeGrid; i++){
            F_decal[2*i]=(-1.0*data1.iwnArr[i]*substractG[i]).real();
            F_decal[2*i+1]=(-1.0*data1.iwnArr[i]*substractG[i]).imag();
        }
        gsl_fft_complex_radix2_forward(F_decal, 1, timeGrid);
        for(size_t i=0; i<timeGrid; i++){
            tau=i*GreenStuff::beta/(double)timeGrid;
            data_dg_dtau_pos.slice(i)(0,0) = ( (1./GreenStuff::beta) * std::exp( -M_PI*i*im*(1.0/((double)timeGrid)-1.0) ) * (F_decal[2*i]+im*F_decal[2*i+1]) + Hpsuperior(tau, GreenStuff::mu0, GreenStuff::hyb_c, GreenStuff::beta) ).real();
        }
        data_dg_dtau_pos.slice(timeGrid)(0,0)= -GreenStuff::mu0 - data_dg_dtau_pos.slice(0)(0,0);
        break;
    case dG_dtau_negative:

        for(size_t i=0; i<timeGrid; i++){
            F_decal[2*i]=(-1.0*data1.iwnArr[i].imag()*substractG[i]).imag();
            F_decal[2*i+1]=(-1.0*data1.iwnArr[i].imag()*substractG[i]).real();
        }
        gsl_fft_complex_radix2_forward(F_decal, 1, timeGrid);
    
        for(size_t i=0; i<timeGrid; i++){
            tau=i*GreenStuff::beta/(double)timeGrid;
            data_dg_dtau_neg.slice(i)(0,0) = ( (1./GreenStuff::beta) * std::exp( -M_PI*i*im*(1.0/(double)timeGrid-1.0) ) * (F_decal[2*i]+im*F_decal[2*i+1]) + Hpinferior(tau, GreenStuff::mu0, GreenStuff::hyb_c, GreenStuff::beta) ).real();
        }
        data_dg_dtau_neg.slice(timeGrid)(0,0)= -GreenStuff::mu0 - data_dg_dtau_neg.slice(0)(0,0);
        break;
    }
}

void DMFTloop(IPT2::DMFTproc& sublatt1, std::ofstream& objSaveStreamGloc, std::ofstream& objSaveStreamSE, std::ofstream& objSaveStreamGW, std::vector< std::string >& vecStr, const unsigned int N_it) noexcept(false){
    /* DMFT loop */
    const Integrals integralsObj;
    unsigned int iter=1;
    double n,n0,G0_diff=0.0;
    bool converged=false;
    arma::Cube< std::complex<double> > WeissGreenTmpA(2,2,iwnArr_l.size()); // Initialization of the hybridization function.
    arma::Cube< std::complex<double> > G0_density_mu0_m(2,2,iwnArr_l.size()), G_density_mu_m(2,2,iwnArr_l.size());
    std::function<double(double)> wrapped_density_mu, wrapped_density_mu0;
    std::cout << "size of iwnArr: " << iwnArr_l.size() << std::endl;
    for (size_t i=0; i<iwnArr_l.size(); i++){
        sublatt1.Hyb.matsubara_w.slice(i)(0,0) = sublatt1.Hyb.get_hyb_c()/iwnArr_l[i];
    }
    while (iter<=N_it && !converged){ // Everything has been designed to accomodate a bipartite lattice.
        G0_diff=0.0; // resetting G0_diff
        objSaveStreamGloc.open(vecStr[0]+"_Nit_"+std::to_string(iter)+".dat", std::ios::out | std::ios::app);
        objSaveStreamSE.open(vecStr[1]+"_Nit_"+std::to_string(iter)+".dat", std::ios::out | std::ios::app);
        objSaveStreamGW.open(vecStr[2]+"_Nit_"+std::to_string(iter)+".dat", std::ios::out | std::ios::app);
        for (size_t j=0; j<iwnArr_l.size(); j++){
            sublatt1.WeissGreen.matsubara_w.slice(j)(0,0)=1.0/(iwnArr_l[j]+sublatt1.WeissGreen.get_mu0()-sublatt1.Hyb.matsubara_w.slice(j)(0,0));
        }
        sublatt1.update_impurity_self_energy(); // Spits out Sigma(iwn). To begin with, mu=U/2.0.
        for (size_t j=0; j<iwnArr_l.size(); j++){
            G0_density_mu0_m.slice(j)(0,0)=iwnArr_l[j]-sublatt1.Hyb.matsubara_w.slice(j)(0,0);
            G_density_mu_m.slice(j)(0,0)=iwnArr_l[j]-sublatt1.Hyb.matsubara_w.slice(j)(0,0)-sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0);
        }
        std::cout << "mu: " << sublatt1.SelfEnergy.get_mu() << " and mu0 " << sublatt1.SelfEnergy.get_mu0() << std::endl;
        // Compute the various chemical potentials to get n_target
        n=sublatt1.density_mu(G_density_mu_m);
        n0=sublatt1.density_mu0(G0_density_mu0_m);

        std::cout << "n0: " << n0 <<  " for mu0: " << sublatt1.WeissGreen.get_mu0() << std::endl;
        std::cout << "n: " << n << " for mu: " << sublatt1.WeissGreen.get_mu() << std::endl;

        if ( ( std::abs(n-sublatt1.n)>ROOT_FINDING_TOL ) && iter>1 ){
            try{
                wrapped_density_mu = [&](double mu){ return sublatt1.density_mu(mu,G_density_mu_m)-sublatt1.n; };
                double mu_new = integralsObj.falsePosMethod(wrapped_density_mu,-20.,20.);
                std::cout << "mu_new at iter " << iter << ": " << mu_new << std::endl;
                sublatt1.WeissGreen.update_mu(mu_new); // Updates mu from instance, even though it is a static member variable.
            }catch (const std::exception& err){
                std::cerr << err.what() << std::endl;
            }
        }
        if ( ( std::abs(n0-sublatt1.n)>ROOT_FINDING_TOL ) ){
            try{
                wrapped_density_mu0 = [&](double mu0){ return sublatt1.density_mu0(mu0,G0_density_mu0_m)-sublatt1.n; };
                double mu0_new = integralsObj.falsePosMethod(wrapped_density_mu0,-20.,20.);
                std::cout << "mu0_new at iter " << iter << ": " << mu0_new << std::endl;
                sublatt1.WeissGreen.update_mu0(mu0_new); // Updates mu from instance, even though it is a static member variable.
            }catch (const std::exception& err){
                std::cerr << err.what() << std::endl;
            }
        }
        // Determining G_loc
        for (size_t j=0; j<iwnArr_l.size(); j++){
            #if DIM == 1
            std::function<std::complex<double>(double,std::complex<double>)> G_latt = [&](double kx, std::complex<double> iwn){
                return 1.0/(iwn + sublatt1.WeissGreen.get_mu() - epsilonk(kx) - sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0));
            };
            sublatt1.LocalGreen.matsubara_w.slice(j)(0,0)=1./(2.*M_PI)*integralsObj.integrate_simps(G_latt,-M_PI,M_PI,iwnArr_l[j],0.001);
            #elif DIM == 2
            std::function<std::complex<double>(double,double,std::complex<double>)> G_latt = [&](double kx, double ky, std::complex<double> iwn){
                return 1.0/(iwn + WeissGreenA.get_mu() - epsilonk(kx,ky) - SelfEnergyA.matsubara_w.slice(j)(0,0));
            };
            LocalGreenA.matsubara_w.slice(j)(0,0)=1./(4.*M_PI*M_PI)*integralsObj.I2D(G_latt,-M_PI,M_PI,-M_PI,M_PI,iwnArr_l[j]);
            #endif
        }
        // Updating the hybridization function for next round.
        for (size_t j=0; j<iwnArr_l.size(); j++){
            sublatt1.Hyb.matsubara_w.slice(j)(0,0) = iwnArr_l[j] + sublatt1.WeissGreen.get_mu() - sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0) - 1.0/sublatt1.LocalGreen.matsubara_w.slice(j)(0,0);
        }
        if (iter>2){
            for (size_t j=0; j<iwnArr_l.size(); j++) G0_diff+=std::abs(WeissGreenTmpA.slice(j)(0,0)-sublatt1.WeissGreen.matsubara_w.slice(j)(0,0));
            std::cout << "G0_diff: " << G0_diff << std::endl;
            if (G0_diff<0.001 && std::abs(n-sublatt1.n)<ROOT_FINDING_TOL && std::abs(n0-sublatt1.n)<ROOT_FINDING_TOL) converged=true; 
        }
        if (iter>2){ // Assess the convergence process 
            for (size_t j=0; j<iwnArr_l.size(); j++){
                WeissGreenTmpA.slice(j)(0,0) = sublatt1.WeissGreen.matsubara_w.slice(j)(0,0);
            }
        }
        std::cout << "new n0: " << sublatt1.density_mu0(G0_density_mu0_m) <<  " for mu0: " << sublatt1.WeissGreen.get_mu0() << std::endl;
        std::cout << "new n: " << sublatt1.density_mu(G_density_mu_m) << " for mu: " << sublatt1.WeissGreen.get_mu() << std::endl;
        std::cout << "double occupancy: " << sublatt1.double_occupancy() << std::endl;
        saveEachIt(sublatt1,objSaveStreamGloc,objSaveStreamSE,objSaveStreamGW);
        std::cout << "iteration #" << iter << std::endl;
        iter++;
    }
    sublatt1.LocalGreen.reset_counter();
}

void saveEachIt(const IPT2::DMFTproc& sublatt1, std::ofstream& ofGloc, std::ofstream& ofSE, std::ofstream& ofGW){
    for (size_t j=0; j<sublatt1.LocalGreen.matsubara_t_pos.n_slices; j++){
        if (j==0){
            ofGloc << "/G_loc AAup tau pos" << "\t" << "G_loc AAup tau neg" << "\t" << "iwn" << "\t\t" << "G_loc AAup iwn re" << "\t" << "G_loc AAup iwn im" << "\n";
            ofSE << "/SE AAup tau pos" << "\t" << "SE AAup tau neg" << "\t" << "iwn" << "\t\t" << "SE AAup iwn re" << "\t" << "SE AAup iwn im" << "\n";
            ofGW << "/G0 AAup tau pos" << "\t" << "G0 AAup tau neg" << "\t" << "iwn" << "\t\t" << "G0 AAup iwn re" << "\t" << "G0 AAup iwn im" << "\n";

            ofGloc << sublatt1.LocalGreen.matsubara_t_pos.slice(j)(0,0) << "\t\t"; // AAup
            ofGloc << sublatt1.LocalGreen.matsubara_t_neg.slice(j)(0,0) << "\t\t"; // AAup
            ofGloc << iwnArr_l[j].imag() << "\t\t"; // AAup
            ofGloc << sublatt1.LocalGreen.matsubara_w.slice(j)(0,0).real() << "\t\t"; // AAup
            ofGloc << sublatt1.LocalGreen.matsubara_w.slice(j)(0,0).imag() << "\n"; // AAup
            //
            ofSE << sublatt1.SelfEnergy.matsubara_t_pos.slice(j)(0,0) << "\t\t"; // AAup
            ofSE << sublatt1.SelfEnergy.matsubara_t_neg.slice(j)(0,0) << "\t\t"; // AAup
            ofSE << iwnArr_l[j].imag() << "\t\t"; // AAup
            ofSE << sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0).real() << "\t\t"; // AAup
            ofSE << sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0).imag() << "\n"; // AAup
            //
            ofGW << sublatt1.WeissGreen.matsubara_t_pos.slice(j)(0,0) << "\t\t"; // AAup
            ofGW << sublatt1.WeissGreen.matsubara_t_neg.slice(j)(0,0) << "\t\t"; // AAup
            ofGW << iwnArr_l[j].imag() << "\t\t"; // AAup
            ofGW << sublatt1.WeissGreen.matsubara_w.slice(j)(0,0).real() << "\t\t"; // AAup
            ofGW << sublatt1.WeissGreen.matsubara_w.slice(j)(0,0).imag() << "\n"; // AAup
        }
        else{
            if (j<sublatt1.LocalGreen.matsubara_t_pos.n_slices-1){ // To avoid busting bounds of matsubara w array.
                ofGloc << sublatt1.LocalGreen.matsubara_t_pos.slice(j)(0,0) << "\t\t"; // AAup
                ofGloc << sublatt1.LocalGreen.matsubara_t_neg.slice(j)(0,0) << "\t\t"; // AAup
                ofGloc << iwnArr_l[j].imag() << "\t\t"; // AAup
                ofGloc << sublatt1.LocalGreen.matsubara_w.slice(j)(0,0).real() << "\t\t"; // AAup
                ofGloc << sublatt1.LocalGreen.matsubara_w.slice(j)(0,0).imag() << "\n"; // AAup
                ofSE << sublatt1.SelfEnergy.matsubara_t_pos.slice(j)(0,0) << "\t\t"; // AAup
                ofSE << sublatt1.SelfEnergy.matsubara_t_neg.slice(j)(0,0) << "\t\t"; // AAup
                ofSE << iwnArr_l[j].imag() << "\t\t"; // AAup
                ofSE << sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0).real() << "\t\t"; // AAup
                ofSE << sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0).imag() << "\n"; // AAup
                ofGW << sublatt1.WeissGreen.matsubara_t_pos.slice(j)(0,0) << "\t\t"; // AAup
                ofGW << sublatt1.WeissGreen.matsubara_t_neg.slice(j)(0,0) << "\t\t"; // AAup
                ofGW << iwnArr_l[j].imag() << "\t\t"; // AAup
                ofGW << sublatt1.WeissGreen.matsubara_w.slice(j)(0,0).real() << "\t\t"; // AAup
                ofGW << sublatt1.WeissGreen.matsubara_w.slice(j)(0,0).imag() << "\n"; // AAup
            } else{
                ofGloc << sublatt1.LocalGreen.matsubara_t_pos.slice(j)(0,0) << "\t\t"; // AAup
                ofGloc << sublatt1.LocalGreen.matsubara_t_neg.slice(j)(0,0) << "\t\t"; // AAup
                ofGloc << "\t\t" << "\t\t" << "\t\t"; // Important to be able to read off data with python programs.
                //
                ofSE << sublatt1.SelfEnergy.matsubara_t_pos.slice(j)(0,0) << "\t\t"; // AAup
                ofSE << sublatt1.SelfEnergy.matsubara_t_neg.slice(j)(0,0) << "\t\t"; // AAup
                ofSE << "\t\t" << "\t\t" << "\t\t";
                //
                ofGW << sublatt1.WeissGreen.matsubara_t_pos.slice(j)(0,0) << "\t\t"; // AAup
                ofGW << sublatt1.WeissGreen.matsubara_t_neg.slice(j)(0,0) << "\t\t"; // AAup
                ofGW << "\t\t" << "\t\t" << "\t\t";
            }
        }
    }
    ofGloc.close();
    ofSE.close();
    ofGW.close();
}

double Hsuperior(double tau, double mu, double c, double beta){
    double z1(-0.5*mu + 0.5*sqrt(mu*mu+4*c));
    double z2(-0.5*mu-0.5*sqrt(mu*mu+4*c));
    
    return -z1/(z1-z2)*1./(exp(z1*(tau-beta))+exp(z1*tau)) - z2/(z2-z1)*1./(exp(z2*(tau-beta))+exp(z2*tau));
}

double Hpsuperior(double tau, double mu, double c, double beta){
    double z1(-0.5*mu + 0.5*sqrt(mu*mu+4*c));
    double z2(-0.5*mu-0.5*sqrt(mu*mu+4*c));

    return z1*z1/(z1-z2)*1./(exp(z1*(tau-beta))+exp(z1*tau)) + z2*z2/(z2-z1)*1./(exp(z2*(tau-beta))+exp(z2*tau));
}

double Hinferior(double tau, double mu, double c, double beta){
    double z1(-0.5*mu + 0.5*sqrt(mu*mu+4*c));
    double z2(-0.5*mu-0.5*sqrt(mu*mu+4*c));
    
    return z1/(z1-z2)*1.0/(exp(z1*(beta-tau))+exp(-z1*tau)) + z2/(z2-z1)*1./(exp(z2*(beta-tau))+exp(-z2*tau));
}

double Hpinferior(double tau, double mu, double c, double beta){
    double z1(-0.5*mu + 0.5*sqrt(mu*mu+4*c));
    double z2(-0.5*mu-0.5*sqrt(mu*mu+4*c));
    
    return z1*z1/(z1-z2)*1.0/(exp(z1*(beta-tau))+exp(-z1*tau))  + z2*z2/(z2-z1)*1./(exp(z2*(beta-tau))+exp(-z2*tau));
}
