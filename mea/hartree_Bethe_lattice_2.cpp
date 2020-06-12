#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <string>
#include <fftw3.h>
#include <sys/stat.h>
#include <functional>
#include <assert.h>
#include <armadillo>

#define MAXITER 150
#define MAX_ITER_INTEGRAL 40
#define MINTOL 1e-4
#define FINITEDIM 0

typedef std::tuple<std::vector<double>,std::vector<double>> FileData;

double double_occupancy(const std::vector< std::complex<double> >& SelfEnergy, const std::vector< std::complex<double> >& LocalGreen, const std::vector< std::complex<double> >& iwn, double beta, double U, double n=0.5);
FileData get_data(const std::string& strName, const unsigned int Ntau, std::string delimiter="  ") noexcept(false);
void fft_w2t(const std::vector<std::complex<double> >& data1, std::vector<double>& data2, double beta);
std::vector< std::complex<double> > partial_DFT(const std::vector< std::complex<double> >& input);
std::vector< std::complex<double> > linear_spline_Sigma_tau_to_iwn(const std::vector<double>& SE_tau, const std::vector< std::complex<double> >& iwn_array, double beta, double delta_tau);
std::vector< std::complex<double> > linear_spline_tau_to_iwn(const std::vector<double>& G_tau, const std::vector< std::complex<double> >& iwn_array, double beta, double delta_tau);
std::complex<double> I1D(std::function<std::complex<double>(double)> funct,double k0,double kf,double tol=1e-6,unsigned int maxDelta=30);
double I2D(std::function<double(double,double)> funct,double x0,double xf,double y0,double yf,const double tol,unsigned int maxVal,bool is_converged);
template<typename T, typename U> T I1D(std::vector<T>& vecfunct,std::vector<U>& delta_taus);
template<typename T, typename U> T I1D(std::vector<T>& vecfunct,std::vector<U>&& delta_taus);
double DOS(double e);
std::string exec(std::string command);
inline bool sortbysec(const std::tuple<double, double>& a,  
               const std::tuple<double, double>& b);
double Hyb_moments(double tau, double beta, double mu);

enum spin : short { up, down };

struct stat info = {0};

int main(int argc, char ** argv){
 
	double U,beta,ek,n0_up,n0_down,G_0_up_diff,G_0_down_diff,h;
    const double beta_max=70.0, beta_min=50.0, beta_step=1.0;
    const double U_max=10.0, U_min=7.0, U_step=1.0;
	const unsigned int Nomega=4096;
    const unsigned int N_e=2001; // Energy discretization
    const double e_max=2.0, alpha=0.10;
    const double de=e_max/(double)N_e;
    double mu=0.0;
    std::vector< std::complex<double> > G0_up(2*Nomega,0), G0_down(2*Nomega,0), G0_tmp_up(2*Nomega,0), G0_tmp_down(2*Nomega,0);
    std::vector< std::complex<double> > SE_up(2*Nomega,0), SE_down(2*Nomega,0); 
    std::vector< std::complex<double> > Gloc_up(2*Nomega,0), Gloc_down(2*Nomega,0);
    std::vector< std::complex<double> > Hyb_up(2*Nomega,0), Hyb_down(2*Nomega,0);
    std::vector< std::complex<double> > iwn_array;
    std::vector<double> G0_tau_up(2*Nomega+1,0), G0_tau_down(2*Nomega+1,0);
    std::vector<double> SE_tau_up(2*Nomega+1,0), SE_tau_down(2*Nomega+1,0);
    std::vector<double> Gloc_tau_up(2*Nomega+1,0), Gloc_tau_down(2*Nomega+1,0);
    arma::Cube< std::complex<double> > Gimp(2,2,2*Nomega,arma::fill::zeros);
    // std::vector<double> e_arr;
    // for (int e=1; e<2*N_e; e++){
    //     e_arr.push_back( de*(e-(double)N_e) );
    // }
    #if FINITEDIM
    std::string filename("./DOS_1D.dat");
    // Getting the number of lines in file
    std::string str = exec("cat "+filename+" | wc -l");
    double num_energy_points = std::atof(str.c_str());

    auto data_file = get_data(filename,static_cast<int>(num_energy_points));
    auto energy = std::get<0>(data_file); auto dos = std::get<1>(data_file);
    std::vector<std::tuple<double,double>> container_to_sort(energy.size());
    for (size_t i=0; i<container_to_sort.size(); i++){
        container_to_sort[i] = std::make_tuple(energy[i],dos[i]);
    }
    std::sort(container_to_sort.begin(), container_to_sort.end(), sortbysec);
    std::transform(container_to_sort.begin(),container_to_sort.end(),energy.begin(),[=](std::tuple<double,double> tup){ return std::get<0>(tup); });
    std::transform(container_to_sort.begin(),container_to_sort.end(),dos.begin(),[=](std::tuple<double,double> tup){ return std::get<1>(tup); });
    std::vector<std::complex<double>> integral_tmp_1(energy.size());
    std::vector<std::complex<double>> integral_tmp_2(energy.size());
    // std::ofstream outpdos("test_dos.dat",std::ios::out);
    // for (size_t i=0; i<energy.size(); i++){
    //     outpdos << energy[i] << "  " << dos[i] << "\n";
    // }
    // outpdos.close();
    // exit(0);
    #endif
    double tau;
    for (beta=beta_min; beta<=beta_max; beta+=beta_step){
        double delta_tau = beta/(2.0*Nomega);
        // Initializing arrays
        iwn_array.clear();
        for (signed int i=-(signed int)Nomega; i<(signed int)Nomega; i++){
            iwn_array.push_back( std::complex<double>( 0.0, (2.0*i+1.0)*M_PI/beta ) );
        }

        for (U=U_min; U<=U_max; U+=U_step) {
            std::cout << "\n\n" << "beta: " << beta << " U: " << U << "\n\n";
            bool is_converged=false;
            unsigned int iter=0;
            h=0.2;
            std::fill(G0_up.begin(),G0_up.end(),0);
            std::fill(G0_down.begin(),G0_down.end(),0);

            // Initialize the weiss Green's function with semicircular DOS. (Bethe lattice)
            #if FINITEDIM
            for (size_t i=0; i<2*Nomega; i++){
                for (size_t j=0; j<energy.size(); j++){
                    integral_tmp_1[j] = dos[j]/(iwn_array[i]+mu-h-energy[j]);
                    integral_tmp_2[j] = dos[j]/(iwn_array[i]+mu+h-energy[j]);
                }
                G0_up[i] = I1D(integral_tmp_1,energy);
                G0_down[i] = I1D(integral_tmp_2,energy);
            }
        
            std::ofstream outp("test_integral_dos.dat",std::ios::out);
            for (size_t i=0; i<2*Nomega; i++){
                outp << iwn_array[i].imag() << "  " << G0_up[i].imag() << "  " << G0_down[i].imag() << "\n";
            }
            outp.close();

            #else
            for (size_t i=0; i<2*Nomega; i++){
                for (size_t k=1; k<2*N_e; k++){
                    ek=((double)k-(double)N_e)*de;
                    G0_up[i] += de*std::sqrt(4.0-ek*ek)/(2.0*M_PI)/(iwn_array[i]+mu-h-ek);
                    G0_down[i] += de*std::sqrt(4.0-ek*ek)/(2.0*M_PI)/(iwn_array[i]+mu+h-ek);
                }
            }
            std::ofstream outp("test_integral_dos.dat",std::ios::out);
            for (size_t i=0; i<2*Nomega; i++){
                outp << iwn_array[i].imag() << "  " << G0_up[i].imag() << "  " << G0_down[i].imag() << "\n";
            }
            outp.close();
            #endif
            
            G0_tmp_up = G0_up;
            G0_tmp_down = G0_down;
            while (!is_converged && iter<MAXITER){
                std::cout << "\n" << "*************** iter " << iter << " ***************" << "\n";
                if (iter>1){
                    h=0;
                }

                // FFT iwn --> tau Weiss Green's functions
                for (size_t i=0; i<2*Nomega; i++){
                    G0_up[i] -= 1.0/iwn_array[i] + (1.0)/iwn_array[i]/iwn_array[i]/iwn_array[i]; //- 1.0/( iwn_array[i] + mu - 1.0/iwn_array[i] );
                    G0_down[i] -= 1.0/iwn_array[i] + (1.0)/iwn_array[i]/iwn_array[i]/iwn_array[i]; //- 1.0/( iwn_array[i] + mu - 1.0/iwn_array[i] );
                }
                fft_w2t(G0_up,G0_tau_up,beta);
                fft_w2t(G0_down,G0_tau_down,beta);
                for (size_t j=0; j<=2*Nomega; j++){
                    tau = j*delta_tau;
                    G0_tau_up[j] += -0.5 + 0.25*tau*(beta-tau); //+ Hyb_moments(tau,beta,mu);
                    G0_tau_down[j] += -0.5 + 0.25*tau*(beta-tau); //+ Hyb_moments(tau,beta,mu);
                }

                std::ofstream outp2("test_integral_dos_tau_G0_Niter_"+std::to_string(iter)+".dat",std::ios::out);
                for (size_t i=0; i<=2*Nomega; i++){
                    outp2 << i*delta_tau << "  " << G0_tau_up[i] << "  " << G0_tau_down[i] << "\n";
                }
                outp2.close();
                for (size_t i=0; i<2*Nomega; i++){
                    G0_up[i] += 1.0/iwn_array[i] + (1.0)/iwn_array[i]/iwn_array[i]/iwn_array[i]; //+ 1.0/( iwn_array[i] + mu - 1.0/iwn_array[i] );
                    G0_down[i] += 1.0/iwn_array[i] + (1.0)/iwn_array[i]/iwn_array[i]/iwn_array[i]; //+ 1.0/( iwn_array[i] + mu - 1.0/iwn_array[i] );
                }
                
                n0_up = -1.0*G0_tau_up[2*Nomega]; n0_down = -1.0*G0_tau_down[2*Nomega];
                std::cout << "n0_up: " << n0_up << " n0_down: " << n0_down << "\n";

                // self-energy bubble diagram
                for (size_t j=0; j<=2*Nomega; j++){
                    SE_tau_up[j] = (-1.0)*(-1.0)*U*U*G0_tau_up[j]*G0_tau_down[2*Nomega-j]*G0_tau_down[j];
                    SE_tau_down[j] = (-1.0)*(-1.0)*U*U*G0_tau_down[j]*G0_tau_up[2*Nomega-j]*G0_tau_up[j];
                }
        
                // iwn: self-energy bubble diagram
                auto SE_2nd_bubble_up = linear_spline_Sigma_tau_to_iwn(SE_tau_up,iwn_array,beta,delta_tau);
                auto SE_2nd_bubble_down = linear_spline_Sigma_tau_to_iwn(SE_tau_down,iwn_array,beta,delta_tau);
                
                // 2nd-order Hartree diagram
                double Sigma_Hartree_2nd_up = 0.0, Sigma_Hartree_2nd_down = 0.0;
                for (size_t j=0; j<=2*Nomega; j++){
                    Sigma_Hartree_2nd_up += (-1.0)*delta_tau*G0_tau_down[j]*G0_tau_down[2*Nomega-j];
                    Sigma_Hartree_2nd_down += (-1.0)*delta_tau*G0_tau_up[j]*G0_tau_up[2*Nomega-j];
                }
                Sigma_Hartree_2nd_up *= U*U*(n0_up-0.5);
                Sigma_Hartree_2nd_down *= U*U*(n0_down-0.5);
                // total self-energy
                for (size_t i=0; i<2*Nomega; i++){
                    SE_up[i] = U*(n0_down-0.5) + Sigma_Hartree_2nd_up + SE_2nd_bubble_up[i];
                    SE_down[i] = U*(n0_up-0.5) + Sigma_Hartree_2nd_down + SE_2nd_bubble_down[i];
                }
                
                // computing local density 
                for (size_t i=0; i<2*Nomega; i++){
                    Gloc_up[i] = 1.0/( 1.0/G0_up[i] - SE_up[i] ) - 1.0/iwn_array[i] - (1.0+0.25*U*U)/iwn_array[i]/iwn_array[i]/iwn_array[i];
                    Gloc_down[i] = 1.0/( 1.0/G0_down[i] - SE_down[i] ) - 1.0/iwn_array[i] - (1.0+0.25*U*U)/iwn_array[i]/iwn_array[i]/iwn_array[i];
                }
                fft_w2t(Gloc_up,Gloc_tau_up,beta);
                fft_w2t(Gloc_down,Gloc_tau_down,beta);
                
                for (size_t j=0; j<=2*Nomega; j++){
                    tau = j*delta_tau;
                    Gloc_tau_up[j] += -0.5 + 0.25*(1.0+0.25*U*U)*tau*(beta-tau);
                    Gloc_tau_down[j] += -0.5 + 0.25*(1.0+0.25*U*U)*tau*(beta-tau);
                }
                std::cout << "n_up: " << -1.0*Gloc_tau_up[2*Nomega] << " n_down: " << -1.0*Gloc_tau_down[2*Nomega] << "\n";

                for (size_t i=0; i<2*Nomega; i++){
                    Gloc_up[i] += 1.0/iwn_array[i] + (1.0+0.25*U*U)/iwn_array[i]/iwn_array[i]/iwn_array[i];
                    Gloc_down[i] += 1.0/iwn_array[i] + (1.0+0.25*U*U)/iwn_array[i]/iwn_array[i]/iwn_array[i];
                }
                // updating the local Green's function by integrating over semi-circular DOS
                for (size_t i=0; i<2*Nomega; i++){
                    // AA up
                    #if FINITEDIM
                    for (size_t j=0; j<energy.size(); j++){
                        integral_tmp_1[j] = dos[j]/( iwn_array[i] + mu - h - SE_up[i] - energy[j]*energy[j]/( iwn_array[i] + mu + h - SE_down[i] ) );
                    }
                    auto GAA_up_integrated = I1D(integral_tmp_1,energy);
                    
                    // AA down
                    for (size_t j=0; j<energy.size(); j++){
                        integral_tmp_1[j] = dos[j]/( iwn_array[i] + mu + h - SE_down[i] - energy[j]*energy[j]/( iwn_array[i] + mu - h - SE_up[i] ) );
                    }
                    auto GAA_down_integrated = I1D(integral_tmp_1,energy);

                    // AB 
                    // for (size_t j=0; j<energy.size(); j++){
                    //     integral_tmp_1[j] = energy[j]*dos[j]/( ( iwn_array[i] + mu - h - SE_up[i] )*( iwn_array[i] + mu + h - SE_down[i] ) - energy[j]*energy[j] );
                    // }
                    // auto GAB_integrated = I1D(integral_tmp_1,energy);
                    #else
                    std::function<std::complex<double>(double)> GAA_up = [&](double e){
                        return DOS(e)/( iwn_array[i] + mu - h - SE_up[i] - e*e/( iwn_array[i] + mu + h - SE_down[i] ) );
                    };
                    auto GAA_up_integrated = I1D(GAA_up,-2.0,2.0);
                    
                    // AA down
                    std::function<std::complex<double>(double)> GAA_down = [&](double e){
                        return DOS(e)/( iwn_array[i] + mu + h - SE_down[i] - e*e/( iwn_array[i] + mu - h - SE_up[i] ) );
                    };
                    auto GAA_down_integrated = I1D(GAA_down,-2.0,2.0);

                    // AB 
                    // std::function<std::complex<double>(double)> GAB = [&](double e){
                    //     return e*DOS(e)/( ( iwn_array[i] + mu - h - SE_up[i] )*( iwn_array[i] + mu + h - SE_down[i] ) - e*e );
                    // };
                    // auto GAB_integrated = I1D(GAB,-2.0,2.0);
                    #endif
                    
                    Gimp.slice(i)(up,up) = GAA_up_integrated;
                    Gimp.slice(i)(down,down) = GAA_down_integrated;
                    Gimp.slice(i)(up,down) = 0.0;
                    Gimp.slice(i)(down,up) = 0.0;
                    // inverse
                    Gimp.slice(i) = arma::inv(Gimp.slice(i));
                    
                }
                for (size_t i=0; i<2*Nomega; i++){
                    if (iter>0){
                        Hyb_up[i] = (1.0-alpha)*( iwn_array[i] + mu - h - SE_up[i] - Gimp.slice(i)(up,up) ) + alpha*( Hyb_up[i] );
                        Hyb_down[i] = (1.0-alpha)*( iwn_array[i] + mu + h - SE_down[i] - Gimp.slice(i)(down,down) ) + alpha*( Hyb_down[i] );
                    }
                    else{
                        Hyb_up[i] = ( iwn_array[i] + mu - h - SE_up[i] - Gimp.slice(i)(up,up) );
                        Hyb_down[i] = ( iwn_array[i] + mu + h - SE_down[i] - Gimp.slice(i)(down,down) );
                    }
                    G0_up[i] = 1.0/( iwn_array[i] + mu - h - Hyb_up[i] );
                    G0_down[i] = 1.0/( iwn_array[i] + mu + h - Hyb_down[i] );
                }

                // std::ofstream output("output.dat",std::ios::out);
                // for (size_t i=0; i<2*Nomega; i++){
                //     output << iwn_array[i].imag() << "  " << G0_up[i].imag() << "  " << G0_down[i].imag() << "\n";
                // }
                // output.close();
                // exit(0);

                if (iter>1){
                    G_0_up_diff = 0.0; G_0_down_diff = 0.0;
                    for (size_t l=0; l<2*Nomega; l++){
                        G_0_up_diff += std::abs(G0_tmp_up[l]-G0_up[l]);
                        G_0_down_diff += std::abs(G0_tmp_down[l]-G0_down[l]);
                    }
                    std::cout << "G0_diff up: " << G_0_up_diff << " and G0_diff down: " << G_0_down_diff << "\t";
                    if (G_0_up_diff<MINTOL && G_0_down_diff<MINTOL)
                        is_converged=true;
                }
                G0_tmp_up = G0_up;
                G0_tmp_down = G0_down;

                // Saving files at each iteration
                std::string directory_container = "Bethe_lattice_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_n_"+std::to_string(0.5)+"_Ntau_"+std::to_string(Nomega);
                std::string full_path = "./data_Bethe_test_IPT_damping_0.1_Ne_2001_more_moments/"+directory_container;
                if (stat(full_path.c_str(), &info) == -1){
                    mkdir(full_path.c_str(), 0700);
                }

                std::string filenameGloc = full_path+"/Green_loc_Bethe_lattice_AFM_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_N_tau_"+std::to_string(Nomega)+"_h_"+std::to_string(h)+"_Nit_"+std::to_string(iter)+".dat";
                std::ofstream outputGloc(filenameGloc,std::ios::out);
                for (size_t i=0; i<2*Nomega; i++){
                    if (i==0)
                        outputGloc << "iwn\t\tRe Gloc up\t\tIm Gloc up\t\tRe Gloc down\t\tIm Gloc down\n";
                    outputGloc << iwn_array[i].imag() << "\t\t" << Gloc_up[i].real() << "\t\t" << Gloc_up[i].imag() << "\t\t" << Gloc_down[i].real() << "\t\t" << Gloc_down[i].imag() << "\n";
                }
                outputGloc.close();

                std::string filenameGloctau = full_path+"/Green_loc_tau_Bethe_lattice_AFM_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_N_tau_"+std::to_string(Nomega)+"_h_"+std::to_string(h)+"_Nit_"+std::to_string(iter)+".dat";
                std::ofstream outputGloctau(filenameGloctau,std::ios::out);
                for (size_t i=0; i<=2*Nomega; i++){
                    if (i==0)
                        outputGloctau << "tau\t\tGloc up\t\tGloc down\n";
                    outputGloctau << i*delta_tau << "\t\t" << Gloc_tau_up[i] << "\t\t" << Gloc_tau_down[i] << "\n";
                }
                outputGloctau.close();
                
                std::string filenameSE = full_path+"/Self_energy_Bethe_lattice_AFM_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_N_tau_"+std::to_string(Nomega)+"_h_"+std::to_string(h)+"_Nit_"+std::to_string(iter)+".dat";
                std::ofstream outputSE(filenameSE,std::ios::out);
                for (size_t i=0; i<2*Nomega; i++){
                    if (i==0)
                        outputSE << "iwn\t\tRe SE up\t\tIm SE up\t\tRe SE down\t\tIm SE down\n";
                    outputSE << iwn_array[i].imag() << "\t\t" << SE_up[i].real() << "\t\t" << SE_up[i].imag() << "\t\t" << SE_down[i].real() << "\t\t" << SE_down[i].imag() << "\n";
                }
                outputSE.close();

                iter++;
            }
            auto DD = double_occupancy(SE_up,Gloc_up,iwn_array,beta,U);
            // std::ofstream outputD("outputD_beta_"+std::to_string(beta)+"_Ntau_"+std::to_string(Nomega)+".dat",std::ios::out | std::ios::app);
            // outputD << U << "  " << DD << "\n";
            // outputD.close();
            std::cout << "D: " << DD << "\n";
        }
    }
  return 0;
}

double Hyb_moments(double tau, double beta, double mu){
    double z1(-0.5*mu + 0.5*std::sqrt(mu*mu+4.0));
    double z2(-0.5*mu - 0.5*std::sqrt(mu*mu+4.0));

    return -1.0*z1/(z1-z2)*1.0/(std::exp(z1*tau)+std::exp((tau-beta)*z1)) - 1.0*z2/(z2-z1)*1.0/(std::exp(z2*tau)+std::exp((tau-beta)*z2));
}

double DOS(double e){
    return std::sqrt(4-e*e)/(2.0*M_PI);
}

inline bool sortbysec(const std::tuple<double, double>& a,  
               const std::tuple<double, double>& b) { 
    return (std::get<0>(a) < std::get<0>(b)); 
} 

template<typename T, typename U=void>
T I1D(std::vector<T>& vecfunct,std::vector<U>& delta_taus){
    T result(0.0);

    for (unsigned int i=1; i<vecfunct.size(); i++){
        result+=(vecfunct[i-1]+vecfunct[i])/2.0*(delta_taus[i]-delta_taus[i-1]);
    }
    
    return result;
}

template<typename T, typename U=void>
T I1D(std::vector<T>& vecfunct,std::vector<U>&& delta_taus){
    T result(0.0);

    for (unsigned int i=1; i<vecfunct.size(); i++){
        result+=(vecfunct[i-1]+vecfunct[i])/2.0*(delta_taus[i]-delta_taus[i-1]);
    }
    
    return result;
}

double double_occupancy(const std::vector< std::complex<double> >& SelfEnergy, const std::vector< std::complex<double> >& LocalGreen, const std::vector< std::complex<double> >& iwn, double beta, double U, double n){
    double D=0.0;
    for (size_t j=0; j<iwn.size(); j++){ //static_cast<size_t>(iwnArr_l.size()/2)
        D += ( ( SelfEnergy[j] - U*n ) * LocalGreen[j] ).real();
    }
    D*=1./(beta*U);
    D+=n*n;
    return D;
}

std::complex<double> I1D(std::function<std::complex<double>(double)> funct,double k0,double kf,double tol,unsigned int maxDelta){
    double dk;
    std::complex<double> result(0.0,0.0), prevResult(0.0,0.0), resultSumNk;
    unsigned int iter=0;
    bool is_converged=false;
    while(!is_converged && iter<MAX_ITER_INTEGRAL){
        resultSumNk = std::complex<double>(0.0,0.0);
        prevResult = result;
        dk=(kf-k0)/(double)maxDelta;
        for (unsigned int i=1; i<maxDelta; i++){
            resultSumNk+=funct(k0+i*dk);
        }

        result = 0.5*dk*funct(k0)+0.5*dk*funct(kf)+dk*resultSumNk;

        if (iter>0)
            is_converged = (std::abs(prevResult-result)>tol) ? false : true;

        if (is_converged)
            break;

        maxDelta*=2;
        iter++;
    }
    
    return result;
}

double I2D(std::function<double(double,double)> funct,double x0,double xf,double y0,double yf,const double tol,unsigned int maxVal,bool is_converged){
    double dx=(xf-x0)/(double)maxVal, dy=(yf-y0)/(double)maxVal;
    double resultSumNyorNx=0.0, resultSumNxNy=0.0, result, prevResult=0.0;
    unsigned int iter=1;
    while (!is_converged){
        resultSumNxNy=resultSumNyorNx=0.0;
        dx=(xf-x0)/(double)maxVal;
        dy=(yf-y0)/(double)maxVal;
        prevResult=result;
        for (unsigned int i=1; i<maxVal; i++){
            resultSumNyorNx+=funct(x0+i*dx,y0)+funct(x0+i*dx,yf)+funct(x0,y0+i*dy)+funct(xf,y0+i*dy);
        }
        for (unsigned int i=1; i<maxVal; i++){
            for (unsigned int j=1; j<maxVal; j++){
                resultSumNxNy+=funct(x0+i*dx,y0+j*dy);
            }
        }

        result=0.25*dx*dy*(funct(x0,y0)+funct(xf,y0)+funct(x0,yf)+funct(xf,yf))+0.5*dx*dy*resultSumNyorNx+dx*dy*resultSumNxNy;

        is_converged = (std::abs(prevResult-result)<tol) ? true : false;
        
        if (iter>40 || is_converged){
            break;
        }
        maxVal*=2;
        iter++;
    }

    return result;
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

FileData get_data(const std::string& strName, const unsigned int Ntau, std::string delimiter) noexcept(false){

    std::vector<double> energy(Ntau,0.0);
    std::vector<double> dos(Ntau,0.0);

    std::ifstream inputFile(strName);
    std::string increment("");
    unsigned int idx = 0;
    size_t pos=0;
    std::string token;
    if (inputFile.fail()){
        std::cerr << "Error loading the file..." << "\n";
        throw std::ios_base::failure("Check loading file procedure..");
    } 
    std::vector<double> tmp_vec;
    while (getline(inputFile,increment)){
        if (increment[0]=='/'){
            continue;
        }
        while ((pos = increment.find(delimiter)) != std::string::npos) {
            token = increment.substr(0, pos);
            increment.erase(0, pos + delimiter.length());
            tmp_vec.push_back(std::atof(token.c_str()));
        }
        tmp_vec.push_back(std::atof(increment.c_str()));
        energy[idx] = tmp_vec[0];
        dos[idx] = tmp_vec[1];
        
        increment.clear();
        tmp_vec.clear();
        idx++;
    }
    
    inputFile.close();

    FileData fileDataObj=std::make_tuple(energy,dos);
    return fileDataObj;
}

std::string exec(std::string command){
   char buffer[128];
   std::string result = "";

   // Open pipe to file
   FILE* pipe = popen(command.c_str(), "r");
   if (!pipe) {
      return "popen failed!";
   }

   // read till end of process:
   while (!feof(pipe)) {

      // use buffer to read and add to result
      if (fgets(buffer, 128, pipe) != NULL)
         result += buffer;
   }

   pclose(pipe);
   return result;
}