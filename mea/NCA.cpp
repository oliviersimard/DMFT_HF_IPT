#include <iostream>
#include <armadillo>
#include <complex>
#include <vector>
#include <fftw3.h>
#include <fstream>
#include <string>
#include <sys/stat.h>

#define NTAU 4096
#define DIM 1
#define MAXITER 150
#define MAX_ITER_INTEGRAL 20
#define MINTOL 5.0e-4

#if NTAU < 512
template<signed... Is> struct seq{};
template<signed N, signed... Is> struct gen_seq : gen_seq<N-1, N-1, Is...>{};
template<signed... Is> struct gen_seq<-NTAU, Is...> : seq<Is...>{};

struct Table {
    std::complex<double> _[2*NTAU];
};

constexpr std::complex<double> iwn_gen(signed int i, double beta){ return std::complex<double>(0.0,(2.0*i+1.0)*M_PI/beta); }

template<signed... Is>
constexpr Table iwnFunction(double beta, seq<Is...>){
  return {{ iwn_gen(Is,beta)... }};
}

constexpr Table construct_iwn(double beta){
  return iwnFunction(beta,gen_seq<NTAU>{});
}
#endif

void fft_w2t(const std::vector<std::complex<double> >& data1, std::vector<double>& data2, double beta);
std::vector< std::complex<double> > partial_DFT(const std::vector< std::complex<double> >& input);
std::vector< std::complex<double> > linear_spline_Sigma_tau_to_iwn(const std::vector<double>& SE_tau, const std::vector< std::complex<double> >& iwn_array, double beta, double delta_tau);
std::vector< std::complex<double> > linear_spline_tau_to_iwn(const std::vector<double>& G_tau, const std::vector< std::complex<double> >& iwn_array, double beta, double delta_tau);
std::complex<double> I2D(std::function<std::complex<double>(double,double,std::complex<double>)> funct,double x0,double xf,double y0,double yf,std::complex<double> iwn,std::string rule,double tol,unsigned int maxVal,bool is_converged) noexcept(false);
std::complex<double> I1D(std::function<std::complex<double>(double,std::complex<double>)> funct,double k0,double kf,std::complex<double> iwn,std::string rule="trapezoidal",double tol=1e-4,unsigned int maxDelta=40) noexcept(false);
template<typename T> inline T eps(T);
template<typename T, typename... Ts> inline T eps(T,Ts...);

enum spin : short { up, down };
struct stat info = {0};

int main(int argc, char* argv[]){

    const double beta_min=3.0, beta_step=0.2, beta_max=12.0;
    const double U_min=2.0, U_step=0.5, U_max=12.0; // For 1D, breakdown below U=4
    const double alpha=0.0;

    arma::Cube<double> Hyb(2,2,2*NTAU+1,arma::fill::zeros);
    arma::Cube<double> SE_1(2,2,2*NTAU+1,arma::fill::zeros), SE(2,2,2*NTAU+1,arma::fill::zeros);
    std::vector<double> SE_0(2*NTAU+1,0), SE_2(2*NTAU+1,0);
    arma::Cube<double> G_1(2,2,2*NTAU+1,arma::fill::zeros), dG_1(2,2,2*NTAU+1,arma::fill::zeros), G(2,2,2*NTAU+1,arma::fill::zeros); // Physical Green's function
    std::vector<double> G_0(2*NTAU+1,0), G_2(2*NTAU+1,0), dG_0(2*NTAU+1,0), dG_2(2*NTAU+1,0), G_tmp_up(2*NTAU+1), G_tmp_down(2*NTAU+1);
    arma::Cube< std::complex<double> > SE_iwn(2,2,2*NTAU,arma::fill::zeros), Gloc_iwn(2,2,2*NTAU,arma::fill::zeros);
    std::vector< std::complex<double> > G_up_iwn(2*NTAU), Hyb_up_iwn(2*NTAU), G_down_iwn(2*NTAU), Hyb_down_iwn(2*NTAU);

    for (double beta=beta_min; beta<=beta_max; beta+=beta_step){
        const double delta_tau = beta/(double)(2*NTAU);
        #if NTAU < 512
        static_assert(NTAU > 0, "NTAU must be a positive number.");
        constexpr Table table = construct_iwn(beta);
        std::vector< std::complex<double> > iwn_array(table._,table._+2*NTAU);
        #else
        std::vector< std::complex<double> > iwn_array;
        std::complex<double> tmp_iwn;
        for (signed int i=-(signed int)(NTAU); i<(signed int)(NTAU); i++){
            tmp_iwn = std::complex<double>(0.0,(2.0*i+1.0)*M_PI/beta);
            iwn_array.push_back(tmp_iwn);
        }
        #endif
        for (double U=U_min; U<=U_max; U+=U_step){

            const double E_0 = 0.0;
            const double E_1 = -U/2.0;
            const double E_2 = 0.0; // half-filling
            double h = 0.05; // Staggered magnetization
            #if DIM == 1
            const double hyb_c = 2.0;
            #elif DIM == 2
            const double hyb_c = 4.0;
            #endif

            for (int i=0; i<=2*NTAU; i++) {
                // spin splitting between spins on impurity due to h in first iterations
                G_0[i] = -std::exp(-E_0*delta_tau*i);
                G_1.slice(i)(up,up) = -std::exp(-(E_1-h)*delta_tau*i);
                G_1.slice(i)(down,down) = -std::exp(-(E_1+h)*delta_tau*i); 
                G_2[i] = -std::exp(-E_2*delta_tau*i);
                
                Hyb.slice(i)(up,up) = -0.5*hyb_c;
                Hyb.slice(i)(down,down) = -0.5*hyb_c;
            }

            // normalization of the Green's function in the Q=1 pseudo-particle subspace
            double lambda0 = log( -G_0[2*NTAU] - G_1.slice(2*NTAU)(up,up) - G_1.slice(2*NTAU)(down,down) - G_2[2*NTAU] ) / beta; // G_0 and G_2 shouldn't depend on the spin
            std::cout << "lambda0 " << lambda0 << "\n";
            for (int i=0; i<=2*NTAU; i++) {
                G_0[i] *= std::exp(-i*lambda0*delta_tau);
                G_1.slice(i) *= std::exp(-i*lambda0*delta_tau); // Multiplication is done for all the elements in the 2X2 spin matrix at each tau
                G_2[i] *= std::exp(-i*lambda0*delta_tau);
            }
    
            /******************************* Testing *********************************/
            // std::ofstream out1("G_PP_tau_first_iter_before.dat",std::ios::out);
            // for (int i=0; i<=2*NTAU; i++){
            //     out1 << i*delta_tau << "\t\t" << G_0[i] << "\t\t" << G_1.slice(i)(up,up) << "\t\t" << G_1.slice(i)(down,down) << "\t\t" << G_2[i] << "\n";
            // }
            // out1.close();
            // std::vector<double> G_up( G_1(arma::span(up,up),arma::span(up,up),arma::span::all).begin(), G_1(arma::span(up,up),arma::span(up,up),arma::span::all).end() );
            // auto G_0_iwn = linear_spline_Sigma_tau_to_iwn(G_0,iwn_array,beta,delta_tau);
            // auto G_up_iwn = linear_spline_Sigma_tau_to_iwn(G_up,iwn_array,beta,delta_tau);
            // auto G_2_iwn = linear_spline_Sigma_tau_to_iwn(G_2,iwn_array,beta,delta_tau);

            // std::ofstream outputTest2("G_PP_iwn_first_iter_before.dat",std::ios::out);
            // for (int i=0; i<2*NTAU; i++){
            //     outputTest2 << iwn_array[i].imag() << "\t\t" << G_0_iwn[i].real() << "\t\t" << G_0_iwn[i].imag() << "\t\t" << G_up_iwn[i].real() << "\t\t" << G_up_iwn[i].imag() << "\t\t" << G_2_iwn[i].real() << "\t\t" << G_2_iwn[i].imag() << "\n";
            // }
            // outputTest2.close();

            // for (size_t i=0; i<2*NTAU; i++){
            //     G_0_iwn[i] -= 1.0/iwn_array[i];
            // }
            // fft_w2t(G_0_iwn,G_0,beta);
            // for (size_t j=0; j<=2*NTAU; j++){
            //     G_0[j] += -0.5;
            // }
            // fft_w2t(G_up_iwn,G_up,beta);
            // fft_w2t(G_2_iwn,G_2,beta);
            // for (size_t i=0; i<2*NTAU; i++){
            //     G_0_iwn[i] += 1.0/iwn_array[i];
            // }

            // std::ofstream outputTest3("G_PP_tau_first_iter_after.dat",std::ios::out);
            // for (int i=0; i<2*NTAU; i++){
            //     outputTest3 << i*delta_tau << "\t\t" << G_0[i] << "\t\t" << G_up[i] << "\t\t" << G_2[i] << "\n";
            // }
            // outputTest3.close();
            // /******************************* Testing *********************************/

            double lambda = lambda0;
            unsigned int iter = 0;
            bool is_converged = false;
            double G_0_norm, G_1_up_norm, G_1_down_norm, G_2_norm;
            double rhs_0, rhs_1_up, rhs_1_down, rhs_2;
            double G_up_diff, G_down_diff;
            while (!is_converged && iter<MAXITER){
                if (iter>1){
                    h=0.0;
                }
                std::cout << "********************************** iter : " << iter << " **********************************" << "\n";
                std::cout << "U: " << U << " beta: " << beta << " h: " << h << "\n";
                // NCA self-energy update in AFM case scenario
                for (size_t i=0; i<=2*NTAU; i++) {
                    SE_0[i] = G_1.slice(i)(up,up)*Hyb.slice(2*NTAU-i)(up,up)*(-1) + G_1.slice(i)(down,down)*Hyb.slice(2*NTAU-i)(down,down)*(-1); // <0 Added twice because of spin degree of freedom
                    SE_1.slice(i)(up,up) = G_2[i]*Hyb.slice(2*NTAU-i)(down,down)*(-1) + G_0[i]*Hyb.slice(i)(up,up)*(-1);
                    SE_1.slice(i)(down,down) = G_2[i]*Hyb.slice(2*NTAU-i)(up,up)*(-1) + G_0[i]*Hyb.slice(i)(down,down)*(-1);
                    SE_2[i] = G_1.slice(i)(up,up)*Hyb.slice(i)(down,down)*(-1) + G_1.slice(i)(down,down)*Hyb.slice(i)(up,up)*(-1);
                    //
                    SE.slice(i)(up,up) = SE_0[i] + SE_1.slice(i)(up,up) + SE_2[i]; // S_tot
                    SE.slice(i)(down,down) = SE_0[i] + SE_1.slice(i)(down,down) + SE_2[i]; // S_tot
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
                for (size_t n=1; n<=2*NTAU; n++) {
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
                // std::cout << "log(...): " << -G_0[2*NTAU] - G_1.slice(2*NTAU)(up,up) - G_1.slice(2*NTAU)(down,down) - G_2[2*NTAU] << std::endl;
                double lambda_tmp = log( -G_0[2*NTAU] - G_1.slice(2*NTAU)(up,up) - G_1.slice(2*NTAU)(down,down) - G_2[2*NTAU] ) / beta;
                lambda += lambda_tmp;
                std::cout << "lambda updated " << lambda << "\n";
                for (int i=0; i<=2*NTAU; i++) {
                    G_0[i] *= std::exp(-i*lambda_tmp*delta_tau);
                    G_1.slice(i) *= std::exp(-i*lambda_tmp*delta_tau);
                    // G_1.slice(i)(down,down) *= std::exp(-i*lambda_tmp*delta_tau);
                    G_2[i] *= std::exp(-i*lambda_tmp*delta_tau);
                }
                
                for (int i=0; i<=2*NTAU; i++) {    
                    G.slice(i)(up,up) = (-1)*G_1.slice(i)(up,up)*G_0[2*NTAU-i]+G_1.slice(2*NTAU-i)(down,down)*(-1)*G_2[i];
                    G.slice(i)(down,down) = (-1)*G_1.slice(i)(down,down)*G_0[2*NTAU-i]+G_1.slice(2*NTAU-i)(up,up)*(-1)*G_2[i];
                    //std::cout << "Gnca " << i << " " << G[i] << " Hyb " << Hyb[i] << "\n"; 
                }

                // Get G and Hyb in terms of iwn...needs to transform into vectors
                std::vector<double> G_up( G(arma::span(up,up),arma::span(up,up),arma::span::all).begin(), G(arma::span(up,up),arma::span(up,up),arma::span::all).end() );
                std::vector<double> G_down( G(arma::span(down,down),arma::span(down,down),arma::span::all).begin(), G(arma::span(down,down),arma::span(down,down),arma::span::all).end() );
                std::vector<double> Hyb_up( Hyb(arma::span(up,up),arma::span(up,up),arma::span::all).begin(), Hyb(arma::span(up,up),arma::span(up,up),arma::span::all).end() );
                std::vector<double> Hyb_down( Hyb(arma::span(down,down),arma::span(down,down),arma::span::all).begin(), Hyb(arma::span(down,down),arma::span(down,down),arma::span::all).end() );
                //
                G_up_iwn = linear_spline_tau_to_iwn(G_up,iwn_array,beta,delta_tau);
                G_down_iwn = linear_spline_tau_to_iwn(G_down,iwn_array,beta,delta_tau);
                Hyb_up_iwn = linear_spline_Sigma_tau_to_iwn(Hyb_up,iwn_array,beta,delta_tau);
                Hyb_down_iwn = linear_spline_Sigma_tau_to_iwn(Hyb_down,iwn_array,beta,delta_tau);
                
                std::cout << "n_up " << -1.0*G_up[2*NTAU] << " n_down: " << -1.0*G_down[2*NTAU] << "\n"; 
                
                for (size_t i=0; i<2*NTAU; i++){
                    SE_iwn.slice(i)(up,up) = iwn_array[i] - h + U/2.0 - Hyb_up_iwn[i] - 1.0/G_up_iwn[i]; // Physical SE
                    SE_iwn.slice(i)(down,down) = iwn_array[i] + h + U/2.0 - Hyb_down_iwn[i] - 1.0/G_down_iwn[i]; // Physical SE
                    // std::cout << "SE_up: " << SE_iwn.slice(i)(up,up) << "SE_down: " << SE_iwn.slice(i)(down,down) << "\n";
                }

                // Computing G_loc with the extracted physical self-energy. DMFT procedure
                #if DIM == 1
                std::function< std::complex<double>(double,std::complex<double>) > funct_k_integration_up, funct_k_integration_down;
                #elif DIM == 2
                std::function< std::complex<double>(double,double,std::complex<double>) > funct_k_integration_up, funct_k_integration_down;
                #endif
                for (size_t i=0; i<2*NTAU; i++){
                    std::complex<double> iwn = iwn_array[i];
                    #if DIM == 1
                    // AA
                    funct_k_integration_up = [&](double kx,std::complex<double> iwn){
                        return 1.0/( iwn - h + U/2.0 - SE_iwn.slice(i)(up,up) - eps(kx)*eps(kx) / ( iwn + h + U/2.0 - SE_iwn.slice(i)(down,down) ) );
                    };
                    Gloc_iwn.slice(i)(up,up) = 1.0/(2.0*M_PI)*I1D(funct_k_integration_up,-M_PI,M_PI,iwn);
                    // BB
                    funct_k_integration_down = [&](double kx,std::complex<double> iwn){
                        return 1.0/( iwn + h + U/2.0 - SE_iwn.slice(i)(down,down) - eps(kx)*eps(kx) / ( iwn - h + U/2.0 - SE_iwn.slice(i)(up,up) ) );
                    };
                    Gloc_iwn.slice(i)(down,down) = 1.0/(2.0*M_PI)*I1D(funct_k_integration_down,-M_PI,M_PI,iwn);
                    #elif DIM == 2
                    // AA
                    funct_k_integration_up = [&](double kx,double ky,std::complex<double> iwn){
                        return 1.0/( iwn - h + U/2.0 - SE_iwn.slice(i)(up,up) - eps(kx,ky)*eps(kx,ky) / ( iwn + h + U/2.0 - SE_iwn.slice(i)(down,down) ) );
                    };
                    Gloc_iwn.slice(i)(up,up) = 1.0/(4.0*M_PI*M_PI)*I2D(funct_k_integration_up,-M_PI,M_PI,-M_PI,M_PI,iwn,"trapezoidal",1e-4,40,false);
                    // BB
                    funct_k_integration_down = [&](double kx,double ky,std::complex<double> iwn){
                        return 1.0/( iwn + h + U/2.0 - SE_iwn.slice(i)(down,down) - eps(kx,ky)*eps(kx,ky) / ( iwn - h + U/2.0 - SE_iwn.slice(i)(up,up) ) );
                    };
                    Gloc_iwn.slice(i)(down,down) = 1.0/(4.0*M_PI*M_PI)*I2D(funct_k_integration_down,-M_PI,M_PI,-M_PI,M_PI,iwn,"trapezoidal",1e-4,40,false);
                    #endif
                    Gloc_iwn.slice(i)(up,down) = 0.0;
                    Gloc_iwn.slice(i)(down,up) = Gloc_iwn.slice(i)(up,down);
                    // inversing matrices at each imaginary time
                    Gloc_iwn.slice(i) = arma::inv(Gloc_iwn.slice(i));
                }
                // Updating the hybridisation function DMFT
                for (size_t i=0; i<2*NTAU; i++){
                    Hyb_up_iwn[i] = iwn_array[i] - h + U/2.0 - SE_iwn.slice(i)(up,up) - Gloc_iwn.slice(i)(up,up) - hyb_c/iwn_array[i];
                    Hyb_down_iwn[i] = iwn_array[i] + h + U/2.0 - SE_iwn.slice(i)(down,down) - Gloc_iwn.slice(i)(down,down) - hyb_c/iwn_array[i];
                    //std::cout << Hyb_iwn[i] << "\n";
                }
                // FFT and updating at the same time
                
                fft_w2t(Hyb_up_iwn,Hyb_up,beta);
                fft_w2t(Hyb_down_iwn,Hyb_down,beta);
                // Transferring into armadillo container the dumbest way
                for (size_t i=0; i<=2*NTAU; i++){
                    Hyb.slice(i)(up,up) = (1.0-alpha)*(Hyb_up[i] - 0.5*hyb_c) + alpha*(Hyb.slice(i)(up,up));
                    Hyb.slice(i)(down,down) = (1.0-alpha)*(Hyb_down[i] - 0.5*hyb_c) + alpha*(Hyb.slice(i)(down,down));
                }
                
                // std::ofstream outHyb2("Hyb_PP_tau_first_iter.dat",std::ios::out);
                // for (int i=0; i<=2*NTAU; i++){
                //     outHyb2 << delta_tau*i << "\t\t" << Hyb.slice(i)(up,up) << "\n";
                // }
                // outHyb2.close();

                // Hyb_up = std::vector<double>( Hyb(arma::span(up,up),arma::span(up,up),arma::span::all).begin(), Hyb(arma::span(up,up),arma::span(up,up),arma::span::all).end() );
                
                // Hyb_up_iwn = linear_spline_Sigma_tau_to_iwn(Hyb_up,iwn_array,beta,delta_tau);

                // std::ofstream outHyb3("Hyb_PP_iwn_first_iter_after.dat",std::ios::out);
                // for (int i=0; i<2*NTAU; i++){
                //     outHyb3 << iwn_array[i].imag() << "\t\t" << Hyb_up_iwn[i].real() << "\t\t" << Hyb_up_iwn[i].imag() << "\n";
                // }
                // outHyb3.close();

                if (iter>0){
                    G_up_diff = 0.0; G_down_diff = 0.0;
                    for (size_t l=0; l<=2*NTAU; l++){
                        G_up_diff += std::abs(G_tmp_up[l]-G_up[l]);
                        G_down_diff += std::abs(G_tmp_down[l]-G_down[l]);
                    }
                    std::cout << "G_diff up: " << G_up_diff << " and G_diff down: " << G_down_diff << "\n";
                    if (G_up_diff<MINTOL && G_down_diff<MINTOL)
                        is_converged=true;
                }
                G_tmp_up = G_up;
                G_tmp_down = G_down;

                // Saving files at each iteration
                std::string directory_container = std::to_string(DIM)+"D_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_n_"+std::to_string(0.5)+"_Ntau_"+std::to_string(NTAU);
                std::string full_path = "./data_"+std::to_string(DIM)+"D_test_NCA_damping_0.05/"+directory_container;
                if (stat(full_path.c_str(), &info) == -1){
                    mkdir(full_path.c_str(), 0700);
                }
                // Saving data throughout iterations
                std::ofstream outSE(full_path+"/SE_"+std::to_string(DIM)+"D_NCA_AFM_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_N_tau_"+std::to_string(NTAU)+"_h_"+std::to_string(h)+"_Nit_"+std::to_string(iter)+".dat",std::ios::out);
                for (size_t i=0; i<2*NTAU; i++){
                    if (i==0){
                        outSE << "iwn" << "\t\t" << "RE SE_up" << "\t\t" << "IM SE_up" << "\t\t" << "RE SE_down" << "\t\t" << "IM SE_down" << "\n";
                    }
                    outSE << iwn_array[i].imag() << "\t\t" << SE_iwn.slice(i)(up,up).real() << "\t\t" << SE_iwn.slice(i)(up,up).imag() << "\t\t" << SE_iwn.slice(i)(down,down).real() << "\t\t" << SE_iwn.slice(i)(down,down).imag() << "\n";
                }
                outSE.close();

                std::ofstream outGtau(full_path+"/G_"+std::to_string(DIM)+"D_tau_NCA_AFM_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_N_tau_"+std::to_string(NTAU)+"_h_"+std::to_string(h)+"_Nit_"+std::to_string(iter)+".dat",std::ios::out);
                for (int i=0; i<=2*NTAU; i++){
                    if (i==0){
                        outGtau << "tau" << "\t\t" << "G_up" << "\t\t" << "G_down" << "\n";
                    }
                    outGtau << i*delta_tau << "\t\t" << G_up[i] << "\t\t" << G_down[i] << "\n";
                }
                outGtau.close();

                std::ofstream outGiwn(full_path+"/G_"+std::to_string(DIM)+"D_iwn_NCA_AFM_U_"+std::to_string(U)+"_beta_"+std::to_string(beta)+"_N_tau_"+std::to_string(NTAU)+"_h_"+std::to_string(h)+"_Nit_"+std::to_string(iter)+".dat",std::ios::out);
                for (int i=0; i<2*NTAU; i++){
                    if (i==0){
                        outGiwn << "iwn" << "\t\t" << "RE G_up" << "\t\t" << "IM G_up" << "\t\t" << "RE G_down" << "\t\t" << "IM G_down" << "\n";
                    }
                    outGiwn << iwn_array[i].imag() << "\t\t" << G_up_iwn[i].real() << "\t\t" << G_up_iwn[i].imag() << "\t\t" << G_down_iwn[i].real() << "\t\t" << G_down_iwn[i].imag() << "\n";
                }
                outGiwn.close();


                
                iter+=1;
            }
        }
    }

    return EXIT_SUCCESS;
}

// std::ofstream output("G_iwn_iter_"+std::to_string(dmftit)+".dat", std::ios::out);
//         for (int i=0; i<N; i++){
//             output << iwn_array[i].imag() << "\t\t" << G_iwn[i].real() << "\t\t" << G_iwn[i].imag() << "\n";
//         }
//         output.close();
//         std::ofstream output2("Hyb_iwn_iter_"+std::to_string(dmftit)+".dat", std::ios::out);
//         for (int i=0; i<N; i++){
//             output2 << iwn_array[i].imag() << "\t\t" << Hyb_iwn[i].real() << "\t\t" << Hyb_iwn[i].imag() << "\n";
//         }
//         output2.close();
//         std::ofstream output3("S_iwn_iter_"+std::to_string(dmftit)+".dat", std::ios::out);
//         for (int i=0; i<N; i++){
//             output3 << iwn_array[i].imag() << "\t\t" << S_iwn[i].real() << "\t\t" << S_iwn[i].imag() << "\n";
//         }
//         output3.close();

template<typename T> inline T eps(T k){
    return -2.0*std::cos(k);
}

template<typename T, typename... Ts> inline T eps(T k,Ts... ks){
    return -2.0*std::cos(k) + eps(ks...);
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

std::complex<double> I1D(std::function<std::complex<double>(double,std::complex<double>)> funct,double k0,double kf,std::complex<double> iwn,std::string rule,double tol,unsigned int maxDelta) noexcept(false){
    double dk;
    std::complex<double> result(0.0,0.0), prevResult(0.0,0.0), resultSumNk;
    unsigned int iter=0;
    bool is_converged=false;
    while(!is_converged && iter<MAX_ITER_INTEGRAL){
        resultSumNk = std::complex<double>(0.0,0.0);
        prevResult = result;
        dk=(kf-k0)/(double)maxDelta;

        if (rule.compare("trapezoidal")==0){
            for (unsigned int i=1; i<maxDelta; i++){
                resultSumNk+=funct(k0+i*dk,iwn);
            }
            result = 0.5*dk*funct(k0,iwn)+0.5*dk*funct(kf,iwn)+dk*resultSumNk;
        } else if (rule.compare("simpson")==0){
            assert(maxDelta%2==0); // The interval has to be devided into an even number of intervals for this to work in this scheme
            for (unsigned int i=1; i<maxDelta; i+=2){
                resultSumNk += funct(k0+(i-1)*dk,iwn) + 4.0*funct(k0+i*dk,iwn) + funct(k0+(i+1)*dk,iwn);
            }
            result = dk/3.0*resultSumNk;
        } else{
            throw std::invalid_argument("Unhandled rule chosen in integrals 1D. Choices are \"trapezoidal\" and \"simpson\"");
        }

        if (iter>0)
            is_converged = (std::abs(prevResult-result)>tol && iter>1) ? false : true;

        if (is_converged)
            break;

        maxDelta*=2;
        iter++;
    }
    
    return result;
}

std::complex<double> I2D(std::function<std::complex<double>(double,double,std::complex<double>)> funct,double x0,double xf,double y0,double yf,std::complex<double> iwn,std::string rule,double tol,unsigned int maxVal,bool is_converged) noexcept(false){
    double dx=(xf-x0)/(double)maxVal, dy=(yf-y0)/(double)maxVal;
    std::complex<double> result, prevResult(0.0,0.0), resultSumNyorNx, resultSumNxNy;
    unsigned int iter=0;
    while (!is_converged && iter<MAX_ITER_INTEGRAL){
        resultSumNyorNx=std::complex<double>(0.0,0.0), resultSumNxNy=std::complex<double>(0.0,0.0);
        dx=(xf-x0)/(double)maxVal;
        dy=(yf-y0)/(double)maxVal;
        prevResult=result;

        if (rule.compare("trapezoidal")==0){
            for (unsigned int i=1; i<maxVal; i++){
                resultSumNyorNx+=funct(x0+i*dx,y0,iwn)+funct(x0+i*dx,yf,iwn)+funct(x0,y0+i*dy,iwn)+funct(xf,y0+i*dy,iwn);
            }
            for (unsigned int i=1; i<maxVal; i++){
                for (unsigned int j=1; j<maxVal; j++){
                    resultSumNxNy+=funct(x0+i*dx,y0+j*dy,iwn);
                }
            }
        
            result=0.25*dx*dy*(funct(x0,y0,iwn)+funct(xf,y0,iwn)+funct(x0,yf,iwn)+funct(xf,yf,iwn))+0.5*dx*dy*resultSumNyorNx+dx*dy*resultSumNxNy;
        } else if (rule.compare("simpson")==0){
            for (unsigned int i=1; i<maxVal; i+=2){
                for (unsigned int j=1; j<maxVal; j+=2){
                    resultSumNxNy+=funct(x0+(i-1)*dx,y0+(j-1)*dy,iwn)+4.0*funct(x0+i*dx,y0+(j-1)*dy,iwn)+funct(x0+(i+1)*dx,y0+(j-1)*dy,iwn)+
                    4.0*funct(x0+(i-1)*dx,y0+j*dy,iwn)+16.0*funct(x0+i*dx,y0+j*dy,iwn)+4.0*funct(x0+(i+1)*dx,y0+j*dy,iwn)+
                    funct(x0+(i-1)*dx,y0+(j+1)*dy,iwn)+4.0*funct(x0+i*dx,y0+(j+1)*dy,iwn)+funct(x0+(i+1)*dx,y0+(j+1)*dy,iwn);
                }
            }

            result=1.0/9.0*dx*dy*resultSumNxNy;
        } else{
            throw std::invalid_argument("Unhandled rule chosen in integrals 2D. Choices are \"trapezoidal\" and \"simpson\"");
        }
        
        if (iter>0)
            is_converged = ( ( (std::abs(prevResult-result))>tol ) ) ? false : true;

        if (is_converged){
            break;
        }
        maxVal*=2;
        iter++;
    }

    return result;
}