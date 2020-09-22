#include "IPT2nd3rdorderSingle2.hpp"

/**************************************************************************************************/

namespace IPT2{

double DMFTproc::n=0.0, DMFTproc::n0=0.0;
unsigned int DMFTproc::objCount=0;

DMFTproc::DMFTproc(GreenStuff& WeissGreen_,GreenStuff& Hyb_,GreenStuff& LocalGreen_, GreenStuff& SelfEnergy_,
                                                arma::Cube<double>& dg_dtau_pos, arma::Cube<double>& dg_dtau_neg,
                                                const std::vector<double>& karr_l_, const double n_t_spin_) : WeissGreen(WeissGreen_),
                                                Hyb(Hyb_), LocalGreen(LocalGreen_), SelfEnergy(SelfEnergy_),
                                                data_dg_dtau_pos(dg_dtau_pos), data_dg_dtau_neg(dg_dtau_neg), karr_l(karr_l_){
    std::cout << "DMFTproc: U: " << GreenStuff::U << "\n";
    std::cout << "DMFTproc: beta: " << GreenStuff::beta << "\n";
    std::cout << "DMFTproc: N_k: " << GreenStuff::N_k << "\n";
    std::cout << "DMFTproc: N_tau: " << GreenStuff::N_tau << std::endl;
    if (objCount<1){
        this->n=n_t_spin_;
    }
    objCount++;
}
#ifndef AFM
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
    for (int j=0; j<=2*GreenStuff::N_tau; j++){
        tau_arr.push_back(j*GreenStuff::beta/(2.0*GreenStuff::N_tau));
    }
    s.set_points(tau_arr,SelfEnergy.matsubara_t_pos);
    s.iwn_tau_spl_extrm(SelfEnergy,GreenStuff::beta,GreenStuff::N_tau); // Sets up for the new Sigma(iwn) computed with the cubic spline.
}
#else
void DMFTproc::update_parametrized_self_energy_AFM(FFTtools FFTObj, double h){
    /* update_impurity_self_energy() has to be launched before this very method */
    // tau array
    std::vector<double> tau_arr;
    for (int j=0; j<=2*GreenStuff::N_tau; j++){
        tau_arr.push_back(j*GreenStuff::beta/(2.0*GreenStuff::N_tau));
    }
    spline<double> s;
    int index;
    double left_der,right_der;
    // computing spin down Matsubara self-energy (index=0, default)
    // To compute the spline, the derivatives have to be known
    FFTObj.fft_spec_AFM(WeissGreen,WeissGreen,data_dg_dtau_pos,data_dg_dtau_neg,FFTObj.dG_dtau_positive,h);
    FFTObj.fft_spec_AFM(WeissGreen,WeissGreen,data_dg_dtau_pos,data_dg_dtau_neg,FFTObj.dG_dtau_negative,h);
    left_der=data_dg_dtau_pos.slice(0)(0,0)*WeissGreen.matsubara_t_neg.slice(0)(1,1)*WeissGreen.matsubara_t_pos.slice(0)(1,1) + WeissGreen.matsubara_t_pos.slice(0)(0,0)*data_dg_dtau_neg.slice(0)(1,1)*WeissGreen.matsubara_t_pos.slice(0)(1,1) + WeissGreen.matsubara_t_pos.slice(0)(0,0)*WeissGreen.matsubara_t_neg.slice(0)(1,1)*data_dg_dtau_pos.slice(0)(1,1);
    right_der=data_dg_dtau_pos.slice(2.0*GreenStuff::N_tau)(0,0)*WeissGreen.matsubara_t_neg.slice(2.0*GreenStuff::N_tau)(1,1)*WeissGreen.matsubara_t_pos.slice(2.0*GreenStuff::N_tau)(1,1) + WeissGreen.matsubara_t_pos.slice(2.0*GreenStuff::N_tau)(0,0)*data_dg_dtau_neg.slice(2.0*GreenStuff::N_tau)(1,1)*WeissGreen.matsubara_t_pos.slice(2.0*GreenStuff::N_tau)(1,1) + WeissGreen.matsubara_t_pos.slice(2.0*GreenStuff::N_tau)(0,0)*WeissGreen.matsubara_t_neg.slice(2.0*GreenStuff::N_tau)(1,1)*data_dg_dtau_pos.slice(2.0*GreenStuff::N_tau)(1,1);
    s.set_boundary(spline<double>::bd_type::first_deriv,left_der,spline<double>::bd_type::first_deriv,right_der);
    s.set_points(tau_arr,SelfEnergy.matsubara_t_pos);
    s.iwn_tau_spl_extrm(SelfEnergy,GreenStuff::beta,GreenStuff::N_tau); // Sets up for the new Sigma(iwn) computed with the cubic spline.
    
    // computing spin up Matsubara self-energy (index=1)
    spline<double> s2;
    index=1;
    // To compute the spline, the derivatives have to be known
    FFTObj.fft_spec_AFM(WeissGreen,WeissGreen,data_dg_dtau_pos,data_dg_dtau_neg,FFTObj.dG_dtau_positive,-1.0*h);
    FFTObj.fft_spec_AFM(WeissGreen,WeissGreen,data_dg_dtau_pos,data_dg_dtau_neg,FFTObj.dG_dtau_negative,-1.0*h);
    left_der=data_dg_dtau_pos.slice(0)(1,1)*WeissGreen.matsubara_t_neg.slice(0)(0,0)*WeissGreen.matsubara_t_pos.slice(0)(0,0) + WeissGreen.matsubara_t_pos.slice(0)(1,1)*data_dg_dtau_neg.slice(0)(0,0)*WeissGreen.matsubara_t_pos.slice(0)(0,0) + WeissGreen.matsubara_t_pos.slice(0)(1,1)*WeissGreen.matsubara_t_neg.slice(0)(0,0)*data_dg_dtau_pos.slice(0)(0,0);
    right_der=data_dg_dtau_pos.slice(2.0*GreenStuff::N_tau)(1,1)*WeissGreen.matsubara_t_neg.slice(2.0*GreenStuff::N_tau)(0,0)*WeissGreen.matsubara_t_pos.slice(2.0*GreenStuff::N_tau)(0,0) + WeissGreen.matsubara_t_pos.slice(2.0*GreenStuff::N_tau)(1,1)*data_dg_dtau_neg.slice(2.0*GreenStuff::N_tau)(0,0)*WeissGreen.matsubara_t_pos.slice(2.0*GreenStuff::N_tau)(0,0) + WeissGreen.matsubara_t_pos.slice(2.0*GreenStuff::N_tau)(1,1)*WeissGreen.matsubara_t_neg.slice(2.0*GreenStuff::N_tau)(0,0)*data_dg_dtau_pos.slice(2.0*GreenStuff::N_tau)(0,0);
    s2.set_boundary(spline<double>::bd_type::first_deriv,left_der,spline<double>::bd_type::first_deriv,right_der);
    s2.set_points(tau_arr,SelfEnergy.matsubara_t_pos,index);
    s2.iwn_tau_spl_extrm(SelfEnergy,GreenStuff::beta,GreenStuff::N_tau,index); // Sets up for the new Sigma(iwn) computed with the cubic spline.

    // std::cout << "up SE" << "\n";
    // for (size_t j=0; j<2*GreenStuff::N_tau; j++){
    //     std::cout << -1.0*SelfEnergy.matsubara_w.slice(j)(0,0) << "\n";
    // }
    // std::cout << "\n" << "down SE" << "\n";
    // for (size_t j=0; j<2*GreenStuff::N_tau; j++){
    //     std::cout << -1.0*SelfEnergy.matsubara_w.slice(j)(1,1) << "\n";
    // }
    // exit(0);
}
#endif

#ifndef AFM
void DMFTproc::update_impurity_self_energy(){ // Returns the full parametrized self-energy.
    FFTtools FFTObj;
    // parametrization of the self-energy allowing to wander off half-filling.
    double A = 2.0*n*(2.0-2.0*n)/( 2.0*n*(2.0-2.0*n) ); // IPTD becomes one because n_0 is set to be equal to n using right chemical potential mu_0.
    // Also, in the paramagnetic state for a single impurity site, n_up=n_down, hence the total electron density is 2.0*n_t_spin.
    double B = ( (1.0-n)*GreenStuff::U + this->WeissGreen.mu0 - this->WeissGreen.mu )/( n*(1.0-n)*GreenStuff::U*GreenStuff::U );
    // std::cout << "U: " << GreenStuff::U << " and beta : " << GreenStuff::beta << std::endl; 
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
#else
void DMFTproc::update_impurity_self_energy_AFM(double h){ // Returns the full parametrized self-energy.
    FFTtools FFTObj;
    // Compute the Fourier transformation to tau space for impurity self-energy calculation
    // Stores data in matsubara_t_pos and matsubara_t_neg
    FFTObj.fft_spec_AFM(WeissGreen,WeissGreen,data_dg_dtau_pos,data_dg_dtau_neg,FFTObj.plain_positive,h);
    FFTObj.fft_spec_AFM(WeissGreen,WeissGreen,data_dg_dtau_pos,data_dg_dtau_neg,FFTObj.plain_negative,h);
    for (size_t j=0; j<=2*GreenStuff::N_tau; j++){ // IPT2 self-energy diagram
        SelfEnergy.matsubara_t_pos.slice(j)(0,0)=-1.0*GreenStuff::U*GreenStuff::U*WeissGreen.matsubara_t_pos.slice(j)(0,0)*WeissGreen.matsubara_t_pos.slice(j)(1,1)*WeissGreen.matsubara_t_neg.slice(j)(1,1);
        SelfEnergy.matsubara_t_pos.slice(j)(1,1)=-1.0*GreenStuff::U*GreenStuff::U*WeissGreen.matsubara_t_pos.slice(j)(1,1)*WeissGreen.matsubara_t_pos.slice(j)(0,0)*WeissGreen.matsubara_t_neg.slice(j)(0,0);
    }
    
    update_parametrized_self_energy_AFM(FFTObj,h); // Updates Sigma(iwn) for IPT2::
    // Now dealing with the second-order Hartree term of the self-energy renormalizing the chemical potential..
    double sigma_Hartree_2nd_up=0.0, sigma_Hartree_2nd_down=0.0;
    double n0_up = -1.0*WeissGreen.matsubara_t_pos.slice(2*GreenStuff::N_tau)(0,0), n0_down = -1.0*WeissGreen.matsubara_t_pos.slice(2*GreenStuff::N_tau)(1,1);
    std::cout << "n0A_up: " << n0_up << " and n0A_down: " << n0_down << " for mu0: " << WeissGreen.get_mu0() << "\n";
    for (size_t i=0; i<=2*GreenStuff::N_tau; i++){
        sigma_Hartree_2nd_up += 1.0*GreenStuff::beta/(2.0*GreenStuff::N_tau)*WeissGreen.matsubara_t_pos.slice(i)(1,1)*WeissGreen.matsubara_t_neg.slice(i)(1,1);
        sigma_Hartree_2nd_down += 1.0*GreenStuff::beta/(2.0*GreenStuff::N_tau)*WeissGreen.matsubara_t_pos.slice(i)(0,0)*WeissGreen.matsubara_t_neg.slice(i)(0,0);
    }
    sigma_Hartree_2nd_up *= GreenStuff::U*GreenStuff::U*(n0_up-0.5); // -0.5
    sigma_Hartree_2nd_down *= GreenStuff::U*GreenStuff::U*(n0_down-0.5); // -0.5
    for (size_t i=0; i<2*GreenStuff::N_tau; i++){
        SelfEnergy.matsubara_w.slice(i)(0,0) = GreenStuff::U*(n0_down-0.5) + sigma_Hartree_2nd_up - SelfEnergy.matsubara_w.slice(i)(0,0);
        SelfEnergy.matsubara_w.slice(i)(1,1) = GreenStuff::U*(n0_up-0.5) + sigma_Hartree_2nd_down - SelfEnergy.matsubara_w.slice(i)(1,1);
    }
    // std::cout << "SE up" << "\n";
    // for (size_t i=0; i<2*GreenStuff::N_tau; i++){
    //     std::cout << SelfEnergy.matsubara_w.slice(i)(0,0) << "\n";
    // }
    // std::cout << "SE down" << "\n";
    // for (size_t i=0; i<2*GreenStuff::N_tau; i++){
    //     std::cout << SelfEnergy.matsubara_w.slice(i)(1,1) << "\n";
    // }
    // exit(0);
}
#endif

double DMFTproc::density_mu(double mu, const arma::Cube< std::complex<double> >& G) const{
    double nn=0.0;
    #ifndef AFM
    for (size_t j=0; j<iwnArr_l.size(); j++){
        nn += ( 1.0/( G.slice(j)(0,0) + mu ) - 1./iwnArr_l[j] ).real();
    }
    nn*=1./GreenStuff::beta;
    nn+=0.5;
    #else
    for (size_t j=0; j<iwnArr_l.size(); j++){
        nn += ( 1.0/( G.slice(j)(0,0) + mu ) - 1./iwnArr_l[j] ).real();
        nn += ( 1.0/( G.slice(j)(1,1) + mu ) - 1./iwnArr_l[j] ).real();
    }
    nn*=1./GreenStuff::beta;
    nn+=1.0;
    nn*=0.5; // It should give 0.5 with root finding, cause at half-filling..
    #endif
    return nn;
}

double DMFTproc::density_mu(const arma::Cube< std::complex<double> >& G) const{
    double n=0.0;
    const std::complex<double> im(0.0,1.0);
    for (size_t j=0; j<iwnArr_l.size(); j++){ // Takes in actual mu value.
        n += ( 1.0/( G.slice(j)(0,0) + this->WeissGreen.mu ) - 1./iwnArr_l[j] ).real();
    }
    n*=1./GreenStuff::beta*(std::exp(im*M_PI*(iwnArr_l.size()-1.0))).real();
    n-=0.5;
    return -1.0*n;
}

double DMFTproc::density_mu0(double mu0, const arma::Cube< std::complex<double> >& G0_1) const{
    double nn0=0.0;
    for (size_t j=0; j<iwnArr_l.size(); j++){
        nn0 += ( 1./( G0_1.slice(j)(0,0) + mu0 ) - 1./iwnArr_l[j] ).real();
    }
    nn0*=1./GreenStuff::beta;
    nn0+=0.5;
    return nn0;
}

double DMFTproc::density_mu0(const arma::Cube< std::complex<double> >& G0_1) const{
    double n0=0.0;
    const std::complex<double> im(0.0,1.0);
    for (size_t j=0; j<iwnArr_l.size(); j++){
        n0 += ( 1./( G0_1.slice(j)(0,0) + this->WeissGreen.mu0 ) - 1./iwnArr_l[j] ).real();
    }
    n0*=1./GreenStuff::beta*(std::exp(im*M_PI*(iwnArr_l.size()-1.0))).real();
    n0-=0.5;
    return -1.0*n0;
}

#ifdef AFM
tuple<double,double> DMFTproc::density_mu_AFM(const arma::Cube< std::complex<double> >& G) const{
    double n_up=0.0, n_down=0.0;
    constexpr std::complex<double> im(0.0,1.0);
    for (size_t j=0; j<iwnArr_l.size(); j++){ // Takes in actual mu value.
        n_up += ( 1.0/( G.slice(j)(0,0) + this->WeissGreen.mu ) - 1./iwnArr_l[j] ).real();
        n_down += ( 1.0/( G.slice(j)(1,1) + this->WeissGreen.mu ) - 1./iwnArr_l[j] ).real();
    }
    n_up*=1./GreenStuff::beta*(std::exp(im*M_PI*(iwnArr_l.size()-1.0))).real();
    n_down*=1./GreenStuff::beta*(std::exp(im*M_PI*(iwnArr_l.size()-1.0))).real();
    n_up-=0.5;
    n_down-=0.5;
    return tuple<double,double>(-1.0*n_up,-1.0*n_down);
}

std::tuple<double,double> DMFTproc::density_mu0_AFM(const arma::Cube< std::complex<double> >& G0_1) const{
    double n0_up=0.0,n0_down=0.0;
    constexpr std::complex<double> im(0.0,1.0);
    for (size_t j=0; j<iwnArr_l.size(); j++){
        n0_up += ( 1./( G0_1.slice(j)(0,0) + this->WeissGreen.mu0 ) - 1./iwnArr_l[j] ).real();
        n0_down += ( 1./( G0_1.slice(j)(1,1) + this->WeissGreen.mu0 ) - 1./iwnArr_l[j] ).real();
    }
    n0_up*=1./GreenStuff::beta*(std::exp(im*M_PI*(iwnArr_l.size()-1.0))).real();
    n0_down*=1./GreenStuff::beta*(std::exp(im*M_PI*(iwnArr_l.size()-1.0))).real();
    n0_up-=0.5;
    n0_down-=0.5;
    return std::make_tuple(-1.0*n0_up,-1.0*n0_down);
}
#endif

double DMFTproc::double_occupancy() const{
    double D=0.0;
    for (size_t j=0; j<iwnArr_l.size(); j++){ //static_cast<size_t>(iwnArr_l.size()/2)
        D += ( ( SelfEnergy.matsubara_w.slice(j)(0,0) - GreenStuff::U*n ) * LocalGreen.matsubara_w.slice(j)(0,0) ).real();
    }
    D*=1./(GreenStuff::beta*GreenStuff::U);
    D+=n*n;
    return D;
}

double DMFTproc::dbl_occupancy(unsigned int iter) const{
    // d=n_up*n_do-1/U*\int dtau self_energy^{M}(beta-tau)*G^{M}(tau)
    std::ofstream sigma_tau("sigma_tau_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_N_it_"+std::to_string(iter)+".dat", std::ios::app | std::ios::out);
    double D=0.0;
    std::complex<double> tail_sigma_tmp(0.0,0.0);
    const Integrals integralObj;
    FFTtools fftObj;
    std::vector<double> SigmaG_vec;
    // Removing the tails...
    for (size_t l=0; l<iwnArr_l.size(); l++){
        tail_sigma_tmp = GreenStuff::U*n + GreenStuff::U*GreenStuff::U*n*(1.0-n)/iwnArr_l[l];
        SelfEnergy.matsubara_w.slice(l)(0,0) -= tail_sigma_tmp;
        LocalGreen.matsubara_w.slice(l)(0,0) -= 1.0/iwnArr_l[l];
    }
    fftObj.fft_w2t(SelfEnergy.matsubara_w, SelfEnergy.matsubara_t_pos);
    fftObj.fft_w2t(LocalGreen.matsubara_w,LocalGreen.matsubara_t_pos);
    double tau;
    for (size_t l=0; l<=iwnArr_l.size(); l++){
        tau = GreenStuff::beta*l/(2.0*GreenStuff::N_tau);
        SelfEnergy.matsubara_t_pos.slice(l)(0,0) += - GreenStuff::U*GreenStuff::U*n*(1.0-n)/2.0; // GreenStuff::U*n
        LocalGreen.matsubara_t_pos.slice(l)(0,0) -= 0.5;
        sigma_tau << tau << "\t\t" << SelfEnergy.matsubara_t_pos.slice(l)(0,0) << "\t\t" << LocalGreen.matsubara_t_pos.slice(l)(0,0) << "\n";
        SigmaG_vec.push_back(SelfEnergy.matsubara_t_pos.slice(2*GreenStuff::N_tau-l)(0,0)*LocalGreen.matsubara_t_pos.slice(l)(0,0));
    }
    sigma_tau.close();
    // Now integrating over imaginary time
    double sigmaG = integralObj.I1D_VEC(std::move(SigmaG_vec),GreenStuff::beta/(2.0*GreenStuff::N_tau),"simpson");
    D = GreenStuff::U*n*n - sigmaG;
    D*=1./(GreenStuff::U);
    return D;
}


}/* End of namespace IPT2 */

/**************************************************************************************************/

void FFTtools::fft_w2t(arma::Cube< std::complex<double> >& data1, arma::Cube<double>& data2, int index){
    std::complex<double>* inUp=new std::complex<double> [2*GreenStuff::N_tau];
    std::complex<double>* outUp=new std::complex<double> [2*GreenStuff::N_tau];
    fftw_plan pUp; //fftw_plan pDown;
    for(size_t k=0;k<2*GreenStuff::N_tau;k++){
        inUp[k]=data1.slice(k)(index,index);
    }
    pUp=fftw_plan_dft_1d(2*GreenStuff::N_tau, reinterpret_cast<fftw_complex*>(inUp), reinterpret_cast<fftw_complex*>(outUp), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(pUp); //fftw_execute(pDown);

    for(size_t j=0;j<2*GreenStuff::N_tau;j++){
        data2.slice(j)(index,index)=(outUp[j]*std::exp(std::complex<double>(0.0,(double)(2*GreenStuff::N_tau-1)*(double)j*M_PI/((double)2*GreenStuff::N_tau)))).real()/GreenStuff::beta;
    }
    data2.slice(2*GreenStuff::N_tau)(index,index)=0.0; // Up spin
    for(size_t k=0;k<2*GreenStuff::N_tau;k++){
        data2.slice(2*GreenStuff::N_tau)(index,index)+=(data1.slice(k)(index,index)*std::exp(std::complex<double>(0.0,(double)(2*GreenStuff::N_tau-1)*M_PI))).real()/GreenStuff::beta;
    }
    delete [] inUp;
    delete [] outUp;
    fftw_destroy_plan(pUp); //fftw_destroy_plan(pDown);
}

void FFTtools::fft_w2t(const std::vector< std::complex<double> >& data1, std::vector<double>& data2, double beta, std::string dir){
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
    if (dir.compare("forward")==0){
        pUp=fftw_plan_dft_1d(N_tau-1, reinterpret_cast<fftw_complex*>(inUp), reinterpret_cast<fftw_complex*>(outUp), FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(pUp); //fftw_execute(pDown);
    } else if (dir.compare("backward")==0){
        pUp=fftw_plan_dft_1d(N_tau-1, reinterpret_cast<fftw_complex*>(inUp), reinterpret_cast<fftw_complex*>(outUp), FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(pUp); //fftw_execute(pDown);
    }

    if (dir.compare("forward")==0){
        for(size_t j=0;j<N_tau-1;j++){
            data2[j]=(outUp[j]*std::exp(im*(double)(N_tau-2)*(double)j*M_PI/((double)N_tau-1.0))).real()/beta;
        }
    } else if (dir.compare("backward")==0){
        for(size_t j=0;j<N_tau-1;j++){
            data2[j]=(outUp[j]*std::exp(-im*(double)(N_tau-2)*(double)j*M_PI/((double)N_tau-1.0))).real()/beta;
        }
    }
    data2[N_tau-1]=0.0; // Up spin
    if (dir.compare("forward")==0){
        for(size_t k=0;k<N_tau-1;k++){
            data2[N_tau-1]+=(data1[k]*std::exp(im*(double)(N_tau-2)*M_PI)).real()/beta;
        }
    } else if(dir.compare("backward")==0){
        for(size_t k=0;k<N_tau-1;k++){
            data2[N_tau-1]+=(data1[k]*std::exp(-im*(double)(N_tau-2)*M_PI)).real()/beta;
        }
    }
    delete [] inUp;
    delete [] outUp;
    fftw_destroy_plan(pUp); //fftw_destroy_plan(pDown);
}

#ifndef AFM
void FFTtools::fft_spec(GreenStuff& data1, GreenStuff& data2, arma::Cube<double>& data_dg_dtau_pos,
                                                arma::Cube<double>& data_dg_dtau_neg, Spec specialization){
    /* FFT from Matsubara frequency to imaginary time */
    arma::Cube< std::complex<double> > green_inf_v = data1.green_inf();
    const unsigned int timeGrid = 2*Data::N_tau;
    std::vector< std::complex<double> > substractG;
    for (unsigned int j=0; j<timeGrid; j++){
        substractG.push_back(data1.matsubara_w.slice(j)(0,0) - green_inf_v.slice(j)(0,0)); // Meant for the WeissGreen
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
            tau=i*Data::beta/(double)timeGrid;
            data2.matsubara_t_pos.slice(i)(0,0) = ( (1./Data::beta) * std::exp( -M_PI*i*im*(1.0/(double)timeGrid-1.0) ) * (F_decal[2*i]+im*F_decal[2*i+1]) + Hsuperior(tau, data1.mu0, Data::hyb_c, Data::beta) ).real();
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
            tau=i*Data::beta/(double)timeGrid;
            data2.matsubara_t_neg.slice(i)(0,0) = ( (1./Data::beta) * std::exp( -i*M_PI*im*(1.0/((double)timeGrid)-1.0) ) * (F_decal[2*i]+im*F_decal[2*i+1]) + Hinferior(tau, data1.mu0, Data::hyb_c, Data::beta) ).real();
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
            tau=i*Data::beta/(double)timeGrid;
            data_dg_dtau_pos.slice(i)(0,0) = ( (1./Data::beta) * std::exp( -M_PI*i*im*(1.0/((double)timeGrid)-1.0) ) * (F_decal[2*i]+im*F_decal[2*i+1]) + Hpsuperior(tau, data1.mu0, Data::hyb_c, Data::beta) ).real();
        }
        data_dg_dtau_pos.slice(timeGrid)(0,0)= -data1.mu0 - data_dg_dtau_pos.slice(0)(0,0);
        break;
    case dG_dtau_negative:

        for(size_t i=0; i<timeGrid; i++){
            F_decal[2*i]=(-1.0*data1.iwnArr[i].imag()*substractG[i]).imag();
            F_decal[2*i+1]=(-1.0*data1.iwnArr[i].imag()*substractG[i]).real();
        }
        gsl_fft_complex_radix2_forward(F_decal, 1, timeGrid);
    
        for(size_t i=0; i<timeGrid; i++){
            tau=i*Data::beta/(double)timeGrid;
            data_dg_dtau_neg.slice(i)(0,0) = ( (1./Data::beta) * std::exp( -M_PI*i*im*(1.0/(double)timeGrid-1.0) ) * (F_decal[2*i]+im*F_decal[2*i+1]) + Hpinferior(tau, data1.mu0, Data::hyb_c, Data::beta) ).real();
        }
        data_dg_dtau_neg.slice(timeGrid)(0,0)= -data1.mu0 - data_dg_dtau_neg.slice(0)(0,0);
        break;
    }
}
#else
void FFTtools::fft_spec_AFM(GreenStuff& data1, GreenStuff& data2, arma::Cube<double>& data_dg_dtau_pos,
                                                arma::Cube<double>& data_dg_dtau_neg, Spec specialization, double h){
    /* FFT from Matsubara frequency to imaginary time */
    const unsigned int timeGrid = 2*Data::N_tau;
    std::vector< std::complex<double> > substractG_up, substractG_down;
    // cubic_roots roots_up = get_cubic_roots_G_inf(1.0-n_a_up,Data::U,data1.mu,data1.mu0);
    // cubic_roots roots_down = get_cubic_roots_G_inf(n_a_up,Data::U,data1.mu,data1.mu0);

    double F_decal_up[2*timeGrid];
    double F_decal_down[2*timeGrid];
    double tau;
    constexpr std::complex<double> im(0.0,1.0);
    switch (specialization){
    case plain_positive:
        for (unsigned int j=0; j<timeGrid; j++){
            substractG_up.push_back( data1.matsubara_w.slice(j)(0,0) - 1.0/iwnArr_l[j] ); // 1.0/( iwnArr_l[j] - h + data1.mu - Data::hyb_c/iwnArr_l[j] ) ); //- get_Hyb_AFM(1.0-n_a_up,Data::U,data1.mu)/(iwnArr_l[j]*iwnArr_l[j]) ) ); // Meant for the WeissGreen
            substractG_down.push_back( data1.matsubara_w.slice(j)(1,1) - 1.0/iwnArr_l[j] ); // 1.0/( iwnArr_l[j] + h + data1.mu - Data::hyb_c/iwnArr_l[j] ) ); //- get_Hyb_AFM(n_a_up,Data::U,data1.mu)/(iwnArr_l[j]*iwnArr_l[j]) ) ); // Meant for the WeissGreen
        }
        for(size_t i=0; i<timeGrid; i++){
            F_decal_up[2*i]=substractG_up[i].real();
            F_decal_up[2*i+1]=substractG_up[i].imag();
            F_decal_down[2*i]=substractG_down[i].real();
            F_decal_down[2*i+1]=substractG_down[i].imag();
        }
        gsl_fft_complex_radix2_forward(F_decal_up, 1, timeGrid);
        gsl_fft_complex_radix2_forward(F_decal_down, 1, timeGrid);
        for(size_t i=0; i<timeGrid; i++){
            tau=i*Data::beta/(double)timeGrid;
            data2.matsubara_t_pos.slice(i)(0,0) = ( (1./Data::beta) * std::exp( -M_PI*i*im*(1.0/(double)timeGrid-1.0) ) * (F_decal_up[2*i]+im*F_decal_up[2*i+1]) - 0.5 ).real(); // + Hsuperior(tau, data1.mu-h, Data::hyb_c, Data::beta)
            data2.matsubara_t_pos.slice(i)(1,1) = ( (1./Data::beta) * std::exp( -M_PI*i*im*(1.0/(double)timeGrid-1.0) ) * (F_decal_down[2*i]+im*F_decal_down[2*i+1]) - 0.5 ).real(); // + Hsuperior(tau, data1.mu+h, Data::hyb_c, Data::beta)
        }
        // Adding value at tau=beta
        data2.matsubara_t_pos.slice(timeGrid)(0,0)=-1.0-data2.matsubara_t_pos.slice(0)(0,0);
        data2.matsubara_t_pos.slice(timeGrid)(1,1)=-1.0-data2.matsubara_t_pos.slice(0)(1,1);
        break;
    case plain_negative:
        for (unsigned int j=0; j<timeGrid; j++){
            substractG_up.push_back( data1.matsubara_w.slice(j)(0,0) - 1.0/iwnArr_l[j] );// 1.0/( iwnArr_l[j] - h + data1.mu - Data::hyb_c/iwnArr_l[j] ) ); //- get_Hyb_AFM(1.0-n_a_up,Data::U,data1.mu)/(iwnArr_l[j]*iwnArr_l[j]) ) ); // Meant for the WeissGreen
            substractG_down.push_back( data1.matsubara_w.slice(j)(1,1) - 1.0/iwnArr_l[j] ); // 1.0/( iwnArr_l[j] + h + data1.mu - Data::hyb_c/iwnArr_l[j] ) ); //- get_Hyb_AFM(n_a_up,Data::U,data1.mu)/(iwnArr_l[j]*iwnArr_l[j]) ) ); // Meant for the WeissGreen
        }
        for(size_t i=0; i<timeGrid; i++){
            F_decal_up[2*i]=substractG_up[i].real();
            F_decal_up[2*i+1]=-1.0*(substractG_up[i].imag()); // complex conjugate
            F_decal_down[2*i]=substractG_down[i].real();
            F_decal_down[2*i+1]=-1.0*(substractG_down[i].imag()); // complex conjugate
        }
        gsl_fft_complex_radix2_forward(F_decal_up, 1, timeGrid);
        gsl_fft_complex_radix2_forward(F_decal_down, 1, timeGrid);
        for(unsigned int i=0; i<timeGrid; i++){
            tau=i*Data::beta/(double)timeGrid;
            data2.matsubara_t_neg.slice(i)(0,0) = ( (1./Data::beta) * std::exp( -i*M_PI*im*(1.0/((double)timeGrid)-1.0) ) * (F_decal_up[2*i]+im*F_decal_up[2*i+1]) + 0.5 ).real(); // + Hinferior(tau, data1.mu-h, Data::hyb_c, Data::beta)
            data2.matsubara_t_neg.slice(i)(1,1) = ( (1./Data::beta) * std::exp( -i*M_PI*im*(1.0/((double)timeGrid)-1.0) ) * (F_decal_down[2*i]+im*F_decal_down[2*i+1]) + 0.5 ).real(); // + Hinferior(tau, data1.mu+h, Data::hyb_c, Data::beta)
        }
        // Adding value at tau=beta
        data2.matsubara_t_neg.slice(timeGrid)(0,0)=1.0-data2.matsubara_t_neg.slice(0)(0,0);
        data2.matsubara_t_neg.slice(timeGrid)(1,1)=1.0-data2.matsubara_t_neg.slice(0)(1,1);
        break;
    case dG_dtau_positive:
        for (unsigned int j=0; j<timeGrid; j++){
            substractG_up.push_back( data1.matsubara_w.slice(j)(0,0) - 1.0/( iwnArr_l[j] - h + data1.mu - Data::hyb_c/iwnArr_l[j] ) ); //- get_Hyb_AFM(1.0-n_a_up,Data::U,data1.mu)/(iwnArr_l[j]*iwnArr_l[j]) ) ); // Meant for the WeissGreen
            substractG_down.push_back( data1.matsubara_w.slice(j)(1,1) - 1.0/( iwnArr_l[j] - h + data1.mu - Data::hyb_c/iwnArr_l[j] ) ); //- get_Hyb_AFM(n_a_up,Data::U,data1.mu)/(iwnArr_l[j]*iwnArr_l[j]) ) ); // Meant for the WeissGreen
        }
        for(size_t i=0; i<timeGrid; i++){
            F_decal_up[2*i]=(-1.0*data1.iwnArr[i]*substractG_up[i]).real();
            F_decal_up[2*i+1]=(-1.0*data1.iwnArr[i]*substractG_up[i]).imag();
            F_decal_down[2*i]=(-1.0*data1.iwnArr[i]*substractG_down[i]).real();
            F_decal_down[2*i+1]=(-1.0*data1.iwnArr[i]*substractG_down[i]).imag();
        }
        gsl_fft_complex_radix2_forward(F_decal_up, 1, timeGrid);
        gsl_fft_complex_radix2_forward(F_decal_down, 1, timeGrid);
        for(size_t i=0; i<timeGrid; i++){
            tau=i*Data::beta/(double)timeGrid;
            data_dg_dtau_pos.slice(i)(0,0) = ( (1./Data::beta) * std::exp( -M_PI*i*im*(1.0/((double)timeGrid)-1.0) ) * (F_decal_up[2*i]+im*F_decal_up[2*i+1]) + Hpsuperior(tau, data1.mu-h, Data::hyb_c, Data::beta)).real(); // 
            data_dg_dtau_pos.slice(i)(1,1) = ( (1./Data::beta) * std::exp( -M_PI*i*im*(1.0/((double)timeGrid)-1.0) ) * (F_decal_down[2*i]+im*F_decal_down[2*i+1]) + Hpsuperior(tau, data1.mu-h, Data::hyb_c, Data::beta)).real(); //
        }
        data_dg_dtau_pos.slice(timeGrid)(0,0)= -(data1.mu-h) - data_dg_dtau_pos.slice(0)(0,0);
        data_dg_dtau_pos.slice(timeGrid)(1,1)= -(data1.mu-h) - data_dg_dtau_pos.slice(0)(1,1);
        break;
    case dG_dtau_negative:
        for (unsigned int j=0; j<timeGrid; j++){
            substractG_up.push_back( data1.matsubara_w.slice(j)(0,0) - 1.0/( iwnArr_l[j] - h + data1.mu - Data::hyb_c/iwnArr_l[j] ) ); //- get_Hyb_AFM(1.0-n_a_up,Data::U,data1.mu)/(iwnArr_l[j]*iwnArr_l[j]) ) ); // Meant for the WeissGreen
            substractG_down.push_back( data1.matsubara_w.slice(j)(1,1) - 1.0/( iwnArr_l[j] - h + data1.mu - Data::hyb_c/iwnArr_l[j] ) ); //- get_Hyb_AFM(n_a_up,Data::U,data1.mu)/(iwnArr_l[j]*iwnArr_l[j]) ) ); // Meant for the WeissGreen
        }
        for(size_t i=0; i<timeGrid; i++){
            F_decal_up[2*i]=(-1.0*data1.iwnArr[i].imag()*substractG_up[i]).imag();
            F_decal_up[2*i+1]=(-1.0*data1.iwnArr[i].imag()*substractG_up[i]).real();
            F_decal_down[2*i]=(-1.0*data1.iwnArr[i].imag()*substractG_down[i]).imag();
            F_decal_down[2*i+1]=(-1.0*data1.iwnArr[i].imag()*substractG_down[i]).real();
        }
        gsl_fft_complex_radix2_forward(F_decal_up, 1, timeGrid);
        gsl_fft_complex_radix2_forward(F_decal_down, 1, timeGrid);
        for(size_t i=0; i<timeGrid; i++){
            tau=i*Data::beta/(double)timeGrid;
            data_dg_dtau_neg.slice(i)(0,0) = ( (1./Data::beta) * std::exp( -M_PI*i*im*(1.0/(double)timeGrid-1.0) ) * (F_decal_up[2*i]+im*F_decal_up[2*i+1]) + Hpinferior(tau, data1.mu-h, Data::hyb_c, Data::beta) ).real();
            data_dg_dtau_neg.slice(i)(1,1) = ( (1./Data::beta) * std::exp( -M_PI*i*im*(1.0/(double)timeGrid-1.0) ) * (F_decal_down[2*i]+im*F_decal_down[2*i+1]) + Hpinferior(tau, data1.mu-h, Data::hyb_c, Data::beta) ).real();
        }
        data_dg_dtau_neg.slice(timeGrid)(0,0)= -(data1.mu-h) - data_dg_dtau_neg.slice(0)(0,0);
        data_dg_dtau_neg.slice(timeGrid)(1,1)= -(data1.mu-h) - data_dg_dtau_neg.slice(0)(1,1);
        break;
    }
}
#endif

#ifndef AFM
void DMFTloop(IPT2::DMFTproc& sublatt1, std::ofstream& objSaveStreamGloc, std::ofstream& objSaveStreamSE, std::ofstream& objSaveStreamGW, std::vector< std::string >& vecStr, const unsigned int N_it) noexcept(false){
    /* DMFT loop */
    const Integrals integralsObj;
    unsigned int iter=1;
    double n,n0,G0_diff;
    bool converged=false;
    arma::Cube< std::complex<double> > WeissGreenTmpA(2,2,iwnArr_l.size()); // Initialization of the hybridization function.
    arma::Cube< std::complex<double> > G0_density_mu0_m(2,2,iwnArr_l.size()), G_density_mu_m(2,2,iwnArr_l.size());
    std::function<double(double)> wrapped_density_mu, wrapped_density_mu0;
    std::cout << "size of iwnArr: " << iwnArr_l.size() << "\n";
    std::cout << "mu0: " << sublatt1.WeissGreen.get_mu0() << "\n";
    for (size_t i=0; i<iwnArr_l.size(); i++){
        sublatt1.Hyb.matsubara_w.slice(i)(0,0) = GreenStuff::get_hyb_c()/iwnArr_l[i];
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
        // Compute the various chemical potentials to get n_target
        n=sublatt1.density_mu(G_density_mu_m);
        n0=sublatt1.density_mu0(G0_density_mu0_m);

        std::cout << "n0: " << n0 <<  " for mu0: " << sublatt1.WeissGreen.get_mu0() << std::endl;
        std::cout << "n: " << n << " for mu: " << sublatt1.WeissGreen.get_mu() << std::endl;

        if ( ( std::abs(n-sublatt1.n)>ROOT_FINDING_TOL ) && iter>1 ){ // iter > 1 is important
            try{
                wrapped_density_mu = [&](double mu){ return sublatt1.density_mu(mu,G_density_mu_m)-sublatt1.n; };
                double mu_new = integralsObj.falsePosMethod(wrapped_density_mu,-20.,20.);
                std::cout << "mu_new at iter " << iter << ": " << mu_new << std::endl;
                sublatt1.WeissGreen.update_mu(mu_new); // Updates mu from instance WeissGreen.
            }catch (const std::exception& err){
                std::cerr << err.what() << "\n";
            }
        }
        if ( ( std::abs(n0-sublatt1.n)>ROOT_FINDING_TOL ) ){
            try{
                wrapped_density_mu0 = [&](double mu0){ return sublatt1.density_mu0(mu0,G0_density_mu0_m)-sublatt1.n; };
                double mu0_new = integralsObj.falsePosMethod(wrapped_density_mu0,-20.,20.);
                std::cout << "mu0_new at iter " << iter << ": " << mu0_new << std::endl;
                sublatt1.WeissGreen.update_mu0(mu0_new); // Updates mu0 from instance WeissGreen.
            }catch (const std::exception& err){
                std::cerr << err.what() << "\n";
            }
        }
        // Determining G_loc
        for (size_t j=0; j<iwnArr_l.size(); j++){
            #if DIM == 1
            std::function<std::complex<double>(double,std::complex<double>)> G_latt = [&](double kx, std::complex<double> iwn){
                return 1.0/(iwn + sublatt1.WeissGreen.get_mu() - epsilonk(kx) - sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0));
            };
            sublatt1.LocalGreen.matsubara_w.slice(j)(0,0)=1./(2.*M_PI)*integralsObj.I1D(G_latt,-M_PI,M_PI,iwnArr_l[j],"simpson",1e-5,50);
            #elif DIM == 2
            std::function<std::complex<double>(double,double)> int_2D_solver;
            double mu = sublatt1.WeissGreen.get_mu();
            int_2D_solver = [&](double kx, double ky){
                return 1.0/(iwnArr_l[j] + mu - epsilonk(kx,ky) - sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0));
            };
            sublatt1.LocalGreen.matsubara_w.slice(j)(0,0)=1./(4.*M_PI*M_PI)*integralsObj.gauss_quad_2D(int_2D_solver,0.0,2.0*M_PI,0.0,2.0*M_PI);
            #elif DIM == 3
            std::function<std::complex<double>(double,double,double)> int_3D_solver;
            // const double delta_k = 2.0*M_PI/(GreenStuff::N_k-1);
            double mu = sublatt1.WeissGreen.get_mu();
            int_3D_solver = [&](double kx, double ky, double kz){
                return 1.0/(iwnArr_l[j] + mu - epsilonk(kx,ky,kz) - sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0));
            };
            sublatt1.LocalGreen.matsubara_w.slice(j)(0,0) = 1.0/(2.0*M_PI)/(2.0*M_PI)/(2.0*M_PI)*integralsObj.gauss_quad_3D(int_3D_solver,0.0,2.0*M_PI,0.0,2.0*M_PI,0.0,2.0*M_PI);
            #endif
        }
        // Updating the hybridization function for next round.
        for (size_t j=0; j<iwnArr_l.size(); j++){
            sublatt1.Hyb.matsubara_w.slice(j)(0,0) = iwnArr_l[j] + sublatt1.WeissGreen.get_mu() - sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0) - 1.0/sublatt1.LocalGreen.matsubara_w.slice(j)(0,0);
        }
        if (iter>2){
            for (size_t j=0; j<iwnArr_l.size(); j++) G0_diff+=std::abs(WeissGreenTmpA.slice(j)(0,0)-sublatt1.WeissGreen.matsubara_w.slice(j)(0,0));
            std::cout << "G0_diff: " << G0_diff << std::endl;
            if (G0_diff<0.0005 && std::abs(n-sublatt1.n)<ROOT_FINDING_TOL && std::abs(n0-sublatt1.n)<ROOT_FINDING_TOL) converged=true; 
        }
        if (iter>2){ // Assess the convergence process 
            for (size_t j=0; j<iwnArr_l.size(); j++){
                WeissGreenTmpA.slice(j)(0,0) = sublatt1.WeissGreen.matsubara_w.slice(j)(0,0);
            }
        }
        saveEachIt(sublatt1,objSaveStreamGloc,objSaveStreamSE,objSaveStreamGW);
        std::cout << "new n0: " << sublatt1.density_mu0(G0_density_mu0_m) <<  " for mu0: " << sublatt1.WeissGreen.get_mu0() << "\n";
        std::cout << "new n: " << sublatt1.density_mu(G_density_mu_m) << " for mu: " << sublatt1.WeissGreen.get_mu() << "\n";
        std::cout << "double occupancy: " << sublatt1.double_occupancy() << "\n";
        std::cout << "iteration #" << iter << "\n";

        /******************************** testing ******************************/ 
        std::vector<double> SE_up_b(2*GreenStuff::N_tau+1,0), SE_up_f(2*GreenStuff::N_tau+1,0);
        std::ofstream outputSE_iwn_test("test_self_tau_iwn.dat",std::ios::out);
        std::vector< std::complex<double> > SE_iwn_vec( sublatt1.SelfEnergy.matsubara_w(arma::span(0,0),arma::span(0,0),arma::span::all).begin(), sublatt1.SelfEnergy.matsubara_w(arma::span(0,0),arma::span(0,0),arma::span::all).end() );
        for (size_t i=0; i<2*GreenStuff::N_tau; i++){
            outputSE_iwn_test << iwnArr_l[i].imag() << "\t\t" << SE_iwn_vec[i].real() << "\t\t" << SE_iwn_vec[i].imag() << "\n";
        }
        outputSE_iwn_test.close();
        for (size_t i=0; i<2*GreenStuff::N_tau; i++){
            SE_iwn_vec[i] -= GreenStuff::U/2.0 + GreenStuff::U*GreenStuff::U/4.0/iwnArr_l[i];
        }
        std::ofstream outputSE_tau_test("test_self_tau_m_tau.dat",std::ios::out);
        FFTtools fftObj;
        fftObj.fft_w2t(SE_iwn_vec,SE_up_b,GreenStuff::beta,"backward");
        fftObj.fft_w2t(SE_iwn_vec,SE_up_f,GreenStuff::beta,"forward");
        for (size_t j=0; j<=2*GreenStuff::N_tau; j++){
            SE_up_b[j] += GreenStuff::U/2.0 - GreenStuff::U*GreenStuff::U/4.0*0.5;
            SE_up_f[j] += GreenStuff::U/2.0 - GreenStuff::U*GreenStuff::U/4.0*0.5;
            outputSE_tau_test << GreenStuff::dtau*j << "\t\t" << SE_up_b[j] << "\t\t" << SE_up_f[j] << "\n";
        }
        outputSE_tau_test.close();
        /******************************** testing ******************************/ 
        iter++;
    }
    GreenStuff::reset_counter();
}
std::vector< std::complex<double> > DMFTloop(IPT2::DMFTproc& sublatt1, const unsigned int N_it) noexcept(false){
    /* DMFT loop */
    const Integrals integralsObj;
    unsigned int iter=1;
    double n,n0,G0_diff;
    bool converged=false;
    arma::Cube< std::complex<double> > WeissGreenTmpA(2,2,iwnArr_l.size()); // Initialization of the hybridization function.
    arma::Cube< std::complex<double> > G0_density_mu0_m(2,2,iwnArr_l.size()), G_density_mu_m(2,2,iwnArr_l.size());
    std::function<double(double)> wrapped_density_mu, wrapped_density_mu0;
    for (size_t i=0; i<iwnArr_l.size(); i++){
        sublatt1.Hyb.matsubara_w.slice(i)(0,0) = GreenStuff::get_hyb_c()/iwnArr_l[i];
    }
    while (iter<=N_it && !converged){ // Everything has been designed to accomodate a bipartite lattice.
        G0_diff=0.0; // resetting G0_diff
        for (size_t j=0; j<iwnArr_l.size(); j++){
            sublatt1.WeissGreen.matsubara_w.slice(j)(0,0)=1.0/(iwnArr_l[j]+sublatt1.WeissGreen.get_mu0()-sublatt1.Hyb.matsubara_w.slice(j)(0,0));
        }
        sublatt1.update_impurity_self_energy(); // Spits out Sigma(iwn). To begin with, mu=U/2.0.
        for (size_t j=0; j<iwnArr_l.size(); j++){
            G0_density_mu0_m.slice(j)(0,0)=iwnArr_l[j]-sublatt1.Hyb.matsubara_w.slice(j)(0,0);
            G_density_mu_m.slice(j)(0,0)=iwnArr_l[j]-sublatt1.Hyb.matsubara_w.slice(j)(0,0)-sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0);
        }
        // Compute the various chemical potentials to get n_target
        n=sublatt1.density_mu(G_density_mu_m);
        n0=sublatt1.density_mu0(G0_density_mu0_m);

        if ( ( std::abs(n-sublatt1.n)>ROOT_FINDING_TOL ) && iter>1 ){ // iter > 1 is important
            try{
                wrapped_density_mu = [&](double mu){ return sublatt1.density_mu(mu,G_density_mu_m)-sublatt1.n; };
                double mu_new = integralsObj.falsePosMethod(wrapped_density_mu,-20.,20.);
                sublatt1.WeissGreen.update_mu(mu_new); // Updates mu from instance WeissGreen.
            }catch (const std::exception& err){
                std::cerr << err.what() << "\n";
            }
        }
        if ( ( std::abs(n0-sublatt1.n)>ROOT_FINDING_TOL ) ){
            try{
                wrapped_density_mu0 = [&](double mu0){ return sublatt1.density_mu0(mu0,G0_density_mu0_m)-sublatt1.n; };
                double mu0_new = integralsObj.falsePosMethod(wrapped_density_mu0,-20.,20.);
                sublatt1.WeissGreen.update_mu0(mu0_new); // Updates mu0 from instance WeissGreen.
            }catch (const std::exception& err){
                std::cerr << err.what() << "\n";
            }
        }
        // Determining G_loc
        for (size_t j=0; j<iwnArr_l.size(); j++){
            #if DIM == 1
            std::function<std::complex<double>(double)> int_1D_solver;
            double mu = sublatt1.WeissGreen.get_mu();
            int_1D_solver = [&](double kx){
                return 1.0/(iwnArr_l[j] + mu - epsilonk(kx) - sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0));
            };
            sublatt1.LocalGreen.matsubara_w.slice(j)(0,0)=1.0/(2.0*M_PI)*integralsObj.gauss_quad_1D(int_1D_solver,0.0,2.0*M_PI);
            #elif DIM == 2
            std::function<std::complex<double>(double,double)> int_2D_solver;
            double mu = sublatt1.WeissGreen.get_mu();
            int_2D_solver = [&](double kx, double ky){
                return 1.0/(iwnArr_l[j] + mu - epsilonk(kx,ky) - sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0));
            };
            sublatt1.LocalGreen.matsubara_w.slice(j)(0,0)=1./(4.*M_PI*M_PI)*integralsObj.gauss_quad_2D(int_2D_solver,0.0,2.0*M_PI,0.0,2.0*M_PI);
            #elif DIM == 3
            std::function<std::complex<double>(double,double,double)> int_3D_solver;
            // const double delta_k = 2.0*M_PI/(GreenStuff::N_k-1);
            double mu = sublatt1.WeissGreen.get_mu();
            int_3D_solver = [&](double kx, double ky, double kz){
                return 1.0/(iwnArr_l[j] + mu - epsilonk(kx,ky,kz) - sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0));
            };
            sublatt1.LocalGreen.matsubara_w.slice(j)(0,0) = 1.0/(2.0*M_PI)/(2.0*M_PI)/(2.0*M_PI)*integralsObj.gauss_quad_3D(int_3D_solver,0.0,2.0*M_PI,0.0,2.0*M_PI,0.0,2.0*M_PI);
            #endif
        }
        // Updating the hybridization function for next round.
        for (size_t j=0; j<iwnArr_l.size(); j++){
            sublatt1.Hyb.matsubara_w.slice(j)(0,0) = iwnArr_l[j] + sublatt1.WeissGreen.get_mu() - sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0) - 1.0/sublatt1.LocalGreen.matsubara_w.slice(j)(0,0);
        }
        if (iter>2){
            for (size_t j=0; j<iwnArr_l.size(); j++) G0_diff+=std::abs(WeissGreenTmpA.slice(j)(0,0)-sublatt1.WeissGreen.matsubara_w.slice(j)(0,0));
            std::cout << "G0_diff: " << G0_diff << std::endl;
            if (G0_diff<0.0001 && std::abs(n-sublatt1.n)<ROOT_FINDING_TOL && std::abs(n0-sublatt1.n)<ROOT_FINDING_TOL) converged=true; 
        }
        if (iter>2){ // Assess the convergence process 
            for (size_t j=0; j<iwnArr_l.size(); j++){
                WeissGreenTmpA.slice(j)(0,0) = sublatt1.WeissGreen.matsubara_w.slice(j)(0,0);
            }
        }
        std::cout << "double occupancy: " << sublatt1.double_occupancy() << "\n";
        std::cout << "iteration #" << iter << "\n";

        iter++;
    }
    std::vector< std::complex<double> > SE_iwn_vec( sublatt1.SelfEnergy.matsubara_w(arma::span(0,0),arma::span(0,0),arma::span::all).begin(), sublatt1.SelfEnergy.matsubara_w(arma::span(0,0),arma::span(0,0),arma::span::all).end() );
    GreenStuff::reset_counter();
    return SE_iwn_vec;
}
#else
void DMFTloopAFM(IPT2::DMFTproc& sublatt1, std::vector<std::ofstream*> vec_sub_1_ofstream, std::vector< std::string >& vecStr, const unsigned int N_it) noexcept(false){
    /* DMFT loop */
    const Integrals integralsObj;
    unsigned int iter=1;
    double G0_diff_up, G0_diff_down;
    // Starting density values at half-filling
    double nA_up,nA_down;
    bool converged=false;
    arma::Cube< std::complex<double> > WeissGreenTmpA(2,2,iwnArr_l.size(),arma::fill::zeros);
    arma::Cube< std::complex<double> > G0_density_mu0_m_A(2,2,iwnArr_l.size()), G_density_mu_m_A(2,2,iwnArr_l.size());
    arma::Cube< std::complex<double> > LocalGreen_inverse(2,2,iwnArr_l.size());
    // To contain the inverse matrix of the LocalGreen object.
    for (size_t i=0; i<iwnArr_l.size(); i++){
        sublatt1.Hyb.matsubara_w.slice(i)(0,0) = GreenStuff::get_hyb_c()/iwnArr_l[i]; //-get_Hyb_AFM(nA_down,GreenStuff::get_U(),sublatt1.WeissGreen.get_mu())/(iwnArr_l[i]*iwnArr_l[i]); // up
        sublatt1.Hyb.matsubara_w.slice(i)(1,1) = GreenStuff::get_hyb_c()/iwnArr_l[i]; //-get_Hyb_AFM(nA_up,GreenStuff::get_U(),sublatt1.WeissGreen.get_mu())/(iwnArr_l[i]*iwnArr_l[i]); // down
    }
    double h=0.1; // Staggered magnetization
    while (iter<=N_it && !converged){ // Everything has been designed to accomodate a bipartite lattice.
        if (iter>1) 
            h=0.0;
        G0_diff_up=0.0; G0_diff_down=0.0; // resetting G0_diff
        vec_sub_1_ofstream[0]->open(vecStr[0]+"_A_Nit_"+std::to_string(iter)+".dat", std::ios::out | std::ios::app); // Gloc
        vec_sub_1_ofstream[1]->open(vecStr[1]+"_A_Nit_"+std::to_string(iter)+".dat", std::ios::out | std::ios::app); // SE
        vec_sub_1_ofstream[2]->open(vecStr[2]+"_A_Nit_"+std::to_string(iter)+".dat", std::ios::out | std::ios::app); //GW
        for (size_t j=0; j<iwnArr_l.size(); j++){
            sublatt1.WeissGreen.matsubara_w.slice(j)(0,0)=1.0/(iwnArr_l[j]-h+sublatt1.WeissGreen.get_mu()-sublatt1.Hyb.matsubara_w.slice(j)(0,0));
            sublatt1.WeissGreen.matsubara_w.slice(j)(1,1)=1.0/(iwnArr_l[j]+h+sublatt1.WeissGreen.get_mu()-sublatt1.Hyb.matsubara_w.slice(j)(1,1));
        }

        sublatt1.update_impurity_self_energy_AFM(h); // Builds Sigma(iwn). To begin with, mu=U/2.0.
        for (size_t j=0; j<iwnArr_l.size(); j++){
            G_density_mu_m_A.slice(j)(0,0)=1.0/(iwnArr_l[j]+sublatt1.WeissGreen.get_mu()-h-sublatt1.Hyb.matsubara_w.slice(j)(0,0)-sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0))-1.0/iwnArr_l[j];
            G_density_mu_m_A.slice(j)(1,1)=1.0/(iwnArr_l[j]+sublatt1.WeissGreen.get_mu()+h-sublatt1.Hyb.matsubara_w.slice(j)(1,1)-sublatt1.SelfEnergy.matsubara_w.slice(j)(1,1))-1.0/iwnArr_l[j];
        }
        FFTtools Gloc_tau_fft;
        Gloc_tau_fft.fft_w2t(G_density_mu_m_A,sublatt1.LocalGreen.matsubara_t_pos);
        Gloc_tau_fft.fft_w2t(G_density_mu_m_A,sublatt1.LocalGreen.matsubara_t_pos,1);
        for (size_t j=0; j<=2*GreenStuff::N_tau; j++) sublatt1.LocalGreen.matsubara_t_pos.slice(j)(0,0) += -0.5;
        for (size_t j=0; j<=2*GreenStuff::N_tau; j++) sublatt1.LocalGreen.matsubara_t_pos.slice(j)(1,1) += -0.5;
        // Compute the various chemical potentials to get n_target
        // auto n_up_down_A = sublatt1.density_mu_AFM(G_density_mu_m_A);
        // nA_up = std::get<0>(n_up_down_A); nA_down = std::get<1>(n_up_down_A);
        nA_up = -1.0*sublatt1.LocalGreen.matsubara_t_pos.slice(2*GreenStuff::N_tau)(0,0); nA_down = -1.0*sublatt1.LocalGreen.matsubara_t_pos.slice(2*GreenStuff::N_tau)(1,1);
        std::cout << "nA_up: " << nA_up << " and nA_down: " << nA_down << " for mu: " << sublatt1.WeissGreen.get_mu() << "\n";
        // Determining G_loc
        auto mu = sublatt1.WeissGreen.get_mu();
        for (size_t j=0; j<iwnArr_l.size(); j++){
            //std::cout << j << "\n";
            #if DIM == 1
            std::function<std::complex<double>(double)> G_latt_AA_up = [&](double kx){
                return 1.0/( iwnArr_l[j] + mu - h - sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0) - epsilonk(kx)*epsilonk(kx)/(iwnArr_l[j] + mu + h - sublatt1.SelfEnergy.matsubara_w.slice(j)(1,1)) );
            };
            sublatt1.LocalGreen.matsubara_w.slice(j)(0,0)=1./(2.0*M_PI)*integralsObj.gauss_quad_1D(G_latt_AA_up,0.0,2.0*M_PI);
            //
            std::function<std::complex<double>(double)> G_latt_AA_down = [&](double kx){
                return 1.0/( iwnArr_l[j] + mu + h - sublatt1.SelfEnergy.matsubara_w.slice(j)(1,1) - epsilonk(kx)*epsilonk(kx)/(iwnArr_l[j] + mu - h - sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0)) );
            };
            sublatt1.LocalGreen.matsubara_w.slice(j)(1,1)=1./(2.0*M_PI)*integralsObj.gauss_quad_1D(G_latt_AA_down,0.0,2.0*M_PI);
            // Off-diagonal parts AB and BA, which are equal with nearest-neighbour hopping only
            // std::function<std::complex<double>(double,std::complex<double>)> G_latt_AB = [&](double kx, std::complex<double> iwn){
            //     return epsilonk(kx)/( ( iwn + sublatt1.WeissGreen.get_mu() - h - sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0) )*( iwn + sublatt1.WeissGreen.get_mu() + h - sublatt1.SelfEnergy.matsubara_w.slice(j)(1,1) ) - epsilonk(kx)*epsilonk(kx) );
            // };
            // sublatt1.LocalGreen.matsubara_w.slice(j)(1,0)=1./(M_PI)*integralsObj.I1D(G_latt_AB,-M_PI/2.0,M_PI/2.0,iwnArr_l[j],0.00001,200);
            // sublatt1.LocalGreen.matsubara_w.slice(j)(0,1)=sublatt1.LocalGreen.matsubara_w.slice(j)(1,0);
            // LocalGreen_inverse.slice(j) = arma::inv(sublatt1.LocalGreen.matsubara_w.slice(j));
            
            #elif DIM == 2
            // std::cout << "iwn: " << iwnArr_l[j] << "\n";
            std::function<std::complex<double>(double,double)> G_latt_AA_up = [&](double kx, double ky){
                return 1.0/( iwnArr_l[j] + mu - h - sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0) - epsilonk(kx,ky)*epsilonk(kx,ky)/( iwnArr_l[j] + mu + h - sublatt1.SelfEnergy.matsubara_w.slice(j)(1,1) ) );
            };
            sublatt1.LocalGreen.matsubara_w.slice(j)(0,0)=1./(4.0*M_PI*M_PI)*integralsObj.gauss_quad_2D(G_latt_AA_up,0.0,2.0*M_PI,0.0,2.0*M_PI);
            //
            std::function<std::complex<double>(double,double)> G_latt_AA_down = [&](double kx, double ky){
                return 1.0/( iwnArr_l[j] + mu + h - sublatt1.SelfEnergy.matsubara_w.slice(j)(1,1) - epsilonk(kx,ky)*epsilonk(kx,ky)/( iwnArr_l[j] + mu - h - sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0) ) );
            };
            sublatt1.LocalGreen.matsubara_w.slice(j)(1,1)=1./(4.0*M_PI*M_PI)*integralsObj.gauss_quad_2D(G_latt_AA_down,0.0,2.0*M_PI,0.0,2.0*M_PI);
            //
            // std::function<std::complex<double>(double,double,std::complex<double>)> G_latt_AB = [&](double kx, double ky, std::complex<double> iwn){
            //     return epsilonk(kx,ky)/( ( iwn + sublatt1.WeissGreen.get_mu() - h - sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0) )*( iwn + sublatt1.WeissGreen.get_mu() + h - sublatt1.SelfEnergy.matsubara_w.slice(j)(1,1) ) - epsilonk(kx,ky)*epsilonk(kx,ky) );
            // };
            // sublatt1.LocalGreen.matsubara_w.slice(j)(1,0)=1./(M_PI*M_PI)*integralsObj.I2D(G_latt_AB,-M_PI/2.0,M_PI/2.0,-M_PI/2.0,M_PI/2.0,iwnArr_l[j],0.0001,15);
            // sublatt1.LocalGreen.matsubara_w.slice(j)(1,0) = std::complex<double>(0.0);
            // auto AA_up = 1./(M_PI*M_PI)*integralsObj.integral_functional_3d<double>(G_latt_AA_up,-M_PI,M_PI,-M_PI,M_PI,std::complex<double>(-10.0,-10.0),std::complex<double>(10.0,10.0),[](double x,double y) -> bool { 
            //     return (y>=x-M_PI && y<=x+M_PI && y>=-x-M_PI && y<=-x+M_PI);
            //     //return true;
            // });
            // sublatt1.LocalGreen.matsubara_w.slice(j)(0,1) = sublatt1.LocalGreen.matsubara_w.slice(j)(1,0);
            // LocalGreen_inverse.slice(j) = arma::inv(sublatt1.LocalGreen.matsubara_w.slice(j));
            #endif
        }
        // Updating the hybridization function for next round.
        for (size_t j=0; j<iwnArr_l.size(); j++){
            sublatt1.Hyb.matsubara_w.slice(j)(0,0) = (1.0-ALPHA)*(iwnArr_l[j] + sublatt1.WeissGreen.get_mu() - h - sublatt1.SelfEnergy.matsubara_w.slice(j)(0,0) - 1.0/sublatt1.LocalGreen.matsubara_w.slice(j)(0,0)) + ALPHA*(sublatt1.Hyb.matsubara_w.slice(j)(0,0));
            sublatt1.Hyb.matsubara_w.slice(j)(1,1) = (1.0-ALPHA)*(iwnArr_l[j] + sublatt1.WeissGreen.get_mu() + h - sublatt1.SelfEnergy.matsubara_w.slice(j)(1,1) - 1.0/sublatt1.LocalGreen.matsubara_w.slice(j)(1,1)) + ALPHA*(sublatt1.Hyb.matsubara_w.slice(j)(1,1));
        }
        if (iter>2){
            for (size_t j=0; j<iwnArr_l.size(); j++) G0_diff_up+=std::abs(WeissGreenTmpA.slice(j)(0,0)-sublatt1.WeissGreen.matsubara_w.slice(j)(0,0));
            for (size_t j=0; j<iwnArr_l.size(); j++) G0_diff_down+=std::abs(WeissGreenTmpA.slice(j)(1,1)-sublatt1.WeissGreen.matsubara_w.slice(j)(1,1));
            std::cout << "G0_diff_up: " << G0_diff_up << " and G0_diff_down: " << G0_diff_down << std::endl;
            if (G0_diff_up<0.0001 && G0_diff_down<0.0001) converged=true; 
        }
        if (iter>2){ // Assess the convergence process 
            for (size_t j=0; j<iwnArr_l.size(); j++){
                WeissGreenTmpA.slice(j)(0,0) = sublatt1.WeissGreen.matsubara_w.slice(j)(0,0);
                WeissGreenTmpA.slice(j)(1,1) = sublatt1.WeissGreen.matsubara_w.slice(j)(1,1);
            }
        }
        saveEachIt_AFM(sublatt1,*vec_sub_1_ofstream[0],*vec_sub_1_ofstream[1],*vec_sub_1_ofstream[2]);
        std::cout << "iteration #" << iter << " with h: " << h << "\n";
        iter++;
    }
    GreenStuff::reset_counter();
}
#endif

#ifndef AFM
void saveEachIt(const IPT2::DMFTproc& sublatt, std::ofstream& ofGloc, std::ofstream& ofSE, std::ofstream& ofGW) noexcept{
    for (size_t j=0; j<sublatt.LocalGreen.matsubara_w.n_slices; j++){
        if (j==0){ // The "/" is important when reading files.
            ofGloc << "/iwn" << "\t\t" << "G_loc AAup iwn re" << "\t\t" << "G_loc AAup iwn im" << "\n";
            ofSE << "/iwn" << "\t\t" << "SE AAup iwn re" << "\t\t" << "SE AAup iwn im" << "\n";
            ofGW << "/iwn" << "\t\t" << "G0 AAup iwn re" << "\t\t" << "G0 AAup iwn im" << "\n";

            ofGloc << iwnArr_l[j].imag() << "\t\t"; // AAup
            ofGloc << sublatt.LocalGreen.matsubara_w.slice(j)(0,0).real() << "\t\t"; // AAup
            ofGloc << sublatt.LocalGreen.matsubara_w.slice(j)(0,0).imag() << "\n"; // AAup
            //
            ofSE << iwnArr_l[j].imag() << "\t\t"; // AAup
            ofSE << sublatt.SelfEnergy.matsubara_w.slice(j)(0,0).real() << "\t\t"; // AAup
            ofSE << sublatt.SelfEnergy.matsubara_w.slice(j)(0,0).imag() << "\n"; // AAup
            //
            ofGW << iwnArr_l[j].imag() << "\t\t"; // AAup
            ofGW << sublatt.WeissGreen.matsubara_w.slice(j)(0,0).real() << "\t\t"; // AAup
            ofGW << sublatt.WeissGreen.matsubara_w.slice(j)(0,0).imag() << "\n"; // AAup
        }
        else{
            ofGloc << iwnArr_l[j].imag() << "\t\t"; // AAup
            ofGloc << sublatt.LocalGreen.matsubara_w.slice(j)(0,0).real() << "\t\t"; // AAup
            ofGloc << sublatt.LocalGreen.matsubara_w.slice(j)(0,0).imag() << "\n"; // AAup
            ofSE << iwnArr_l[j].imag() << "\t\t"; // AAup
            ofSE << sublatt.SelfEnergy.matsubara_w.slice(j)(0,0).real() << "\t\t"; // AAup
            ofSE << sublatt.SelfEnergy.matsubara_w.slice(j)(0,0).imag() << "\n"; // AAup
            ofGW << iwnArr_l[j].imag() << "\t\t"; // AAup
            ofGW << sublatt.WeissGreen.matsubara_w.slice(j)(0,0).real() << "\t\t"; // AAup
            ofGW << sublatt.WeissGreen.matsubara_w.slice(j)(0,0).imag() << "\n"; // AAup
        }
    }
    ofGloc.close();
    ofSE.close();
    ofGW.close();
}
#else
void saveEachIt_AFM(const IPT2::DMFTproc& sublatt, std::ofstream& ofGloc, std::ofstream& ofSE, std::ofstream& ofGloc_tau) noexcept{
    for (size_t j=0; j<sublatt.LocalGreen.matsubara_w.n_slices; j++){
        if (j==0){ // The "/" is important when reading files.
            ofGloc << "/iwn" << "\t\t" << "Re G_loc up iwn" << "\t\t" << "Im G_loc up iwn" << "\t\t" << "Re G_loc down iwn" << "\t\t" << "Im G_loc down iwn" << "\n";
            ofSE << "/iwn" << "\t\t" << "Re SE up iwn" << "\t\t" << "Im SE up iwn" << "\t\t" << "Re SE down iwn" << "\t\t" << "Im SE down iwn" << "\n";
            // ofGW << "/iwn" << "\t\t" << "Re G0 up iwn" << "\t\t" << "Im G0 up iwn" << "\t\t" << "Re G0 down iwn" << "\t\t" << "Im G0 down iwn" << "\n";
            ofGloc_tau << "/tau" << "\t\t" << "G_loc tau up" << "\t\t" << "G_loc tau down" << "\n";

            ofGloc << iwnArr_l[j].imag() << "\t\t"; // AAup
            ofGloc << sublatt.LocalGreen.matsubara_w.slice(j)(0,0).real() << "\t\t"; // up
            ofGloc << sublatt.LocalGreen.matsubara_w.slice(j)(0,0).imag() << "\t\t"; // up
            ofGloc << sublatt.LocalGreen.matsubara_w.slice(j)(1,1).real() << "\t\t"; // down
            ofGloc << sublatt.LocalGreen.matsubara_w.slice(j)(1,1).imag() << "\n"; // down
            //
            ofSE << iwnArr_l[j].imag() << "\t\t"; // AAup
            ofSE << sublatt.SelfEnergy.matsubara_w.slice(j)(0,0).real() << "\t\t"; // up
            ofSE << sublatt.SelfEnergy.matsubara_w.slice(j)(0,0).imag() << "\t\t"; // up
            ofSE << sublatt.SelfEnergy.matsubara_w.slice(j)(1,1).real() << "\t\t"; // down
            ofSE << sublatt.SelfEnergy.matsubara_w.slice(j)(1,1).imag() << "\n"; // down
            //
            // ofGW << iwnArr_l[j].imag() << "\t\t"; // AAup
            // ofGW << sublatt.WeissGreen.matsubara_w.slice(j)(0,0).real() << "\t\t"; // up
            // ofGW << sublatt.WeissGreen.matsubara_w.slice(j)(0,0).imag() << "\t\t"; // up
            // ofGW << sublatt.WeissGreen.matsubara_w.slice(j)(1,1).real() << "\t\t"; // down
            // ofGW << sublatt.WeissGreen.matsubara_w.slice(j)(1,1).imag() << "\n"; // down
            //
            ofGloc_tau << j*GreenStuff::get_beta()/sublatt.LocalGreen.matsubara_w.n_slices << "\t\t";
            ofGloc_tau << sublatt.LocalGreen.matsubara_t_pos.slice(j)(0,0) << "\t\t";
            ofGloc_tau << sublatt.LocalGreen.matsubara_t_pos.slice(j)(1,1) << "\n";
        }
        else{
            ofGloc << iwnArr_l[j].imag() << "\t\t"; // AAup
            ofGloc << sublatt.LocalGreen.matsubara_w.slice(j)(0,0).real() << "\t\t"; // up
            ofGloc << sublatt.LocalGreen.matsubara_w.slice(j)(0,0).imag() << "\t\t"; // up
            ofGloc << sublatt.LocalGreen.matsubara_w.slice(j)(1,1).real() << "\t\t"; // down
            ofGloc << sublatt.LocalGreen.matsubara_w.slice(j)(1,1).imag() << "\n"; // down
            ofSE << iwnArr_l[j].imag() << "\t\t"; // AAup
            ofSE << sublatt.SelfEnergy.matsubara_w.slice(j)(0,0).real() << "\t\t"; // up
            ofSE << sublatt.SelfEnergy.matsubara_w.slice(j)(0,0).imag() << "\t\t"; // up
            ofSE << sublatt.SelfEnergy.matsubara_w.slice(j)(1,1).real() << "\t\t"; // down
            ofSE << sublatt.SelfEnergy.matsubara_w.slice(j)(1,1).imag() << "\n"; // down
            // ofGW << iwnArr_l[j].imag() << "\t\t"; // AAup
            // ofGW << sublatt.WeissGreen.matsubara_w.slice(j)(0,0).real() << "\t\t"; // up
            // ofGW << sublatt.WeissGreen.matsubara_w.slice(j)(0,0).imag() << "\t\t"; // up
            // ofGW << sublatt.WeissGreen.matsubara_w.slice(j)(1,1).real() << "\t\t"; // down
            // ofGW << sublatt.WeissGreen.matsubara_w.slice(j)(1,1).imag() << "\n"; // down
            ofGloc_tau << j*GreenStuff::get_beta()/sublatt.LocalGreen.matsubara_w.n_slices << "\t\t";
            ofGloc_tau << sublatt.LocalGreen.matsubara_t_pos.slice(j)(0,0) << "\t\t"; // up
            ofGloc_tau << sublatt.LocalGreen.matsubara_t_pos.slice(j)(1,1) << "\n"; // down
        }
    }
    ofGloc_tau << GreenStuff::get_beta() << "\t\t";
    ofGloc_tau << sublatt.LocalGreen.matsubara_t_pos.slice(sublatt.LocalGreen.matsubara_w.n_slices)(0,0) << "\t\t"; // up
    ofGloc_tau << sublatt.LocalGreen.matsubara_t_pos.slice(sublatt.LocalGreen.matsubara_w.n_slices)(1,1) << "\n"; // down
    ofGloc.close();
    ofSE.close();
    // ofGW.close();
    ofGloc_tau.close();
}
#endif

double Hsuperior(double tau, double mu, double c, double beta){
    double z1(-0.5*mu + 0.5*sqrt(mu*mu+4*c));
    double z2(-0.5*mu - 0.5*sqrt(mu*mu+4*c));
    
    return -z1/(z1-z2)*1./(exp(z1*(tau-beta))+exp(z1*tau)) - z2/(z2-z1)*1./(exp(z2*(tau-beta))+exp(z2*tau));
}

double Hpsuperior(double tau, double mu, double c, double beta){
    double z1(-0.5*mu + 0.5*sqrt(mu*mu+4*c));
    double z2(-0.5*mu - 0.5*sqrt(mu*mu+4*c));

    return z1*z1/(z1-z2)*1./(exp(z1*(tau-beta))+exp(z1*tau)) + z2*z2/(z2-z1)*1./(exp(z2*(tau-beta))+exp(z2*tau));
}

double Hinferior(double tau, double mu, double c, double beta){
    double z1(-0.5*mu + 0.5*sqrt(mu*mu+4*c));
    double z2(-0.5*mu - 0.5*sqrt(mu*mu+4*c));
    
    return z1/(z1-z2)*1.0/(exp(z1*(beta-tau))+exp(-z1*tau)) + z2/(z2-z1)*1./(exp(z2*(beta-tau))+exp(-z2*tau));
}

double Hpinferior(double tau, double mu, double c, double beta){
    double z1(-0.5*mu + 0.5*sqrt(mu*mu+4*c));
    double z2(-0.5*mu - 0.5*sqrt(mu*mu+4*c));
    
    return z1*z1/(z1-z2)*1.0/(exp(z1*(beta-tau))+exp(-z1*tau))  + z2*z2/(z2-z1)*1./(exp(z2*(beta-tau))+exp(-z2*tau));
}

// Solving the cubic equation for the hybridisation function in the AFM case
#ifdef AFM
double get_Hyb_AFM(double n_m_sigma_sublatt, double U, double mu){
    double c_0_a=(mu-U*n_m_sigma_sublatt);
    double c_1_a=U*U*n_m_sigma_sublatt*(1.0-n_m_sigma_sublatt);
    double c_0_b=(mu-U*(1.0-n_m_sigma_sublatt)); // we used the fact that for AFM at half-filling, n_b_m = n_a_p = 1-n_a_m.
    #if DIM == 1
    double c=2.0; // This is for 1D
    #elif DIM == 2
    double c=4.0; // This is for 1D
    #elif DIM == 3
    double c=6.0;
    #endif
    return ( (c*c_0_b) + 2.0*(c*c_0_a) + 2.0*(c_0_a*c_1_a) - 2.0*(c_0_a*c_0_a*c_0_a) );
}

cubic_roots get_cubic_roots_G_inf(double n_m_sigma_sublatt, double U, double mu, double mu0){
    // The cubic equation is build here. It consists of
    // z^3 + mu*z^2 + c*z + k where c=2t^2 and k=(c*c_0_b + 2c*c_0_a + 2c_0_a*c_1_a - 2c_0_a*c_0_a*c_0_a),
    // with c_0_b=(mu-Un_b_m), c_0_a=(mu-Un_a_m), c_1_a=U^2*n_a_m*(1-n_a_m) 
    cubic_roots roots;
    // Parameters inputted in get_cubic_roots
    double a=1.0;
    double b=mu0;
    #if DIM == 1
    double c=-2.0; // This is for 1D
    #elif DIM == 2
    double c=-4.0; // This is for 1D
    #elif DIM == 3
    double c=6.0;
    #endif
    double d = get_Hyb_AFM(n_m_sigma_sublatt,U,mu);
    roots = get_cubic_roots(a,b,c,d);

    return roots;
}

double Fsuperior(double tau, double mu, double mu0, double beta, double U, double n_m_sublatt){
    cubic_roots roots = get_cubic_roots_G_inf(n_m_sublatt,U,mu,mu0);
    // A, B, and C should be real numbers, that is the discriminant in Cardano's method should be negative
    auto A = ((roots.x1*roots.x1)/((roots.x1-roots.x2)*(roots.x1-roots.x3))).real();
    auto B = ((roots.x2*roots.x2)/((roots.x2-roots.x1)*(roots.x2-roots.x3))).real();
    auto C = ((roots.x3*roots.x3)/((roots.x3-roots.x1)*(roots.x3-roots.x2))).real();

    return -A*(1.0/(exp(roots.x1.real()*(tau-beta))+exp(roots.x1.real()*tau))) - B*(1.0/(exp(roots.x2.real()*(tau-beta))+exp(roots.x2.real()*tau))) - C*(1.0/(exp(roots.x3.real()*(tau-beta))+exp(roots.x3.real()*tau)));
}

double Finferior(double tau, double mu, double mu0, double beta, double U, double n_m_sublatt){
    cubic_roots roots = get_cubic_roots_G_inf(n_m_sublatt,U,mu,mu0);
    // A, B, and C should be real numbers, that is the discriminant in Cardano's method should be negative
    auto A = ((roots.x1*roots.x1)/((roots.x1-roots.x2)*(roots.x1-roots.x3))).real();
    auto B = ((roots.x2*roots.x2)/((roots.x2-roots.x1)*(roots.x2-roots.x3))).real();
    auto C = ((roots.x3*roots.x3)/((roots.x3-roots.x1)*(roots.x3-roots.x2))).real();

    return A*(1.0/(exp(roots.x1.real()*(beta-tau))+exp(-roots.x1.real()*tau))) + B*(1.0/(exp(roots.x2.real()*(beta-tau))+exp(-roots.x2.real()*tau))) + C*(1.0/(exp(roots.x3.real()*(beta-tau))+exp(-roots.x3.real()*tau)));
}

double Fpsuperior(double tau, double mu, double mu0, double beta, double U, double n_m_sublatt){
    cubic_roots roots = get_cubic_roots_G_inf(n_m_sublatt,U,mu,mu0);
    // A, B, and C should be real numbers, that is the discriminant in Cardano's method should be negative
    auto A = ((roots.x1*roots.x1)/((roots.x1-roots.x2)*(roots.x1-roots.x3))).real();
    auto B = ((roots.x2*roots.x2)/((roots.x2-roots.x1)*(roots.x2-roots.x3))).real();
    auto C = ((roots.x3*roots.x3)/((roots.x3-roots.x1)*(roots.x3-roots.x2))).real();

    return A*roots.x1.real()*(1.0/(exp(roots.x1.real()*(tau-beta))+exp(roots.x1.real()*tau))) + B*roots.x2.real()*(1.0/(exp(roots.x2.real()*(tau-beta))+exp(roots.x2.real()*tau))) + C*roots.x3.real()*(1.0/(exp(roots.x3.real()*(tau-beta))+exp(roots.x3.real()*tau)));
}

double Fpinferior(double tau, double mu, double mu0, double beta, double U, double n_m_sublatt){
    cubic_roots roots = get_cubic_roots_G_inf(n_m_sublatt,U,mu,mu0);
    // A, B, and C should be real numbers, that is the discriminant in Cardano's method should be negative
    auto A = ((roots.x1*roots.x1)/((roots.x1-roots.x2)*(roots.x1-roots.x3))).real();
    auto B = ((roots.x2*roots.x2)/((roots.x2-roots.x1)*(roots.x2-roots.x3))).real();
    auto C = ((roots.x3*roots.x3)/((roots.x3-roots.x1)*(roots.x3-roots.x2))).real();

    return A*roots.x1.real()*(1.0/(exp(roots.x1.real()*(beta-tau))+exp(-roots.x1.real()*tau))) + B*roots.x2.real()*(1.0/(exp(roots.x2.real()*(beta-tau))+exp(-roots.x2.real()*tau))) + C*roots.x3.real()*(1.0/(exp(roots.x3.real()*(beta-tau))+exp(-roots.x3.real()*tau)));
}
#endif