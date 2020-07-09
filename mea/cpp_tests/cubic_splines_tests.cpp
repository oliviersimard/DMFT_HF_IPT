#include "../../src/tridiagonal.hpp"
#include <fftw3.h>
#include <fstream>

struct FileData;

std::vector<double> get_derivative(std::vector<double> x_arr, std::vector<double> y_arr);
// FileData get_data(const std::string& strName, const unsigned int& Ntau);
std::vector<double> get_iwn_to_tau(const std::vector< std::complex<double> >& F_iwn, double beta);
template<typename T> std::vector< T > generate_random_numbers(size_t arr_size, T min, T max);

struct FileData{
    std::vector<double> iwn;
    std::vector<double> re;
    std::vector<double> im;
};

int main(void){


    /* This section deals with the Fourier transformations using cubic splines for fermionic function. */
    /***************************************************************************************************/
    const unsigned int Ntau = 2*1024;
    const double beta = 40.0;
    std::string inputFilename("../Green_loc_1D_U_16.000000_beta_40.000000_n_0.500000_N_tau_1024_Nit_11.dat");
    spline<double> splObj;
    std::vector<double> beta_array;
    for (size_t j=0; j<=Ntau; j++){
        double beta_tmp = j*beta/(Ntau);
        beta_array.push_back(beta_tmp);
    }

    // Getting the data
    FileData dataFromFile = get_data(inputFilename,Ntau);

    std::vector<double> wn = dataFromFile.iwn;
    std::vector<double> re = dataFromFile.re;
    std::vector<double> im = dataFromFile.im;
    std::vector< std::complex<double> > cmplx_funct;
    for (size_t i=0; i<wn.size(); i++){
        cmplx_funct.push_back(std::complex<double>(re[i],im[i]));
    }

    std::vector< std::complex<double> > iwn(wn.size());
    std::transform(wn.begin(),wn.end(),iwn.begin(),[](double d){ return std::complex<double>(0.0,d); }); // casting into array of double for cubic spline.
    
    // Substracting the tail
    for (size_t i=0; i<iwn.size(); i++){
        cmplx_funct[i] -= 1.0/(iwn[i]);
    }

    // Computing G_tau
    std::vector<double> F_tau = get_iwn_to_tau(cmplx_funct,beta);

    // Adding back -1/2
    for (size_t j=0; j<F_tau.size(); j++){
        F_tau[j] -= 0.5;
    }

    // Taking the derivative of G_tau
    std::vector<double> der_G_tau = get_derivative(beta_array,F_tau);
    double left_der = der_G_tau[0];
    double right_der = der_G_tau.back();

    // Converting to arma Matrices
    arma::Cube<double> F_tau_arma(2,2,Ntau+1,arma::fill::ones);
    for (size_t i=0; i<F_tau.size(); i++){
        F_tau_arma.slice(i)(0,0) = F_tau[i];
    }
    
    splObj.set_boundary(spline<double>::bd_type::first_deriv,left_der,spline<double>::bd_type::first_deriv,right_der);
    splObj.set_points(beta_array,F_tau_arma);

    // Testing the spline with randomly generated beta array
    std::vector<double> rand_beta_arr = generate_random_numbers<double>(Ntau,0.0,beta);
    std::vector<double> data_spline_to_save;
    for (size_t l=0; l<rand_beta_arr.size(); l++){
        data_spline_to_save.push_back(splObj(rand_beta_arr[l]));
    }

    std::vector< std::complex<double> > cub_spl = splObj.fermionic_propagator(iwn,beta);

    // std::ofstream outputSpline("test_spline_cpp.dat", std::ios::app | std::ios::out);
    // for (size_t l=0; l<cub_spl.size(); l++){
    //     outputSpline << iwn[l].imag() << "  " << cub_spl[l].real() << "  " << cub_spl[l].imag() << "\n";
    // }
    // outputSpline.close();

    /* This section deals with the Fourier transformation (convolution) of bosonic correlation functions */
    /*****************************************************************************************************/
    spline<double> splObj_GG;

    std::vector<double> bubble_tau(F_tau.size(),0.0);
    for (size_t i=0; i<F_tau.size(); i++){
        bubble_tau[i] = (-1.0)*(-2.0)*F_tau[i]*F_tau[Ntau-i];
    }

    // Taking the derivative for boundary conditions
    std::vector<double> der_GG_tau = get_derivative(beta_array,bubble_tau);
    double left_der_GG = der_GG_tau[0];
    double right_der_GG = der_GG_tau.back();

    // Converting to arma Matrices
    arma::Cube<double> bubble_tau_arma(2,2,Ntau+1,arma::fill::zeros);
    for (size_t i=0; i<bubble_tau.size(); i++){
        bubble_tau_arma.slice(i)(0,0) = bubble_tau[i];
    }
    
    splObj_GG.set_boundary(spline<double>::bd_type::first_deriv,left_der_GG,spline<double>::bd_type::first_deriv,right_der_GG);
    splObj_GG.set_points(beta_array,bubble_tau_arma);

    // Testing the spline with randomly generated beta array
    std::vector<double> data_spline_to_save_bosons;
    for (size_t l=0; l<rand_beta_arr.size(); l++){
        data_spline_to_save_bosons.push_back(splObj_GG(rand_beta_arr[l]));
    }

    std::vector< std::complex<double> > iqn;
    for (size_t j=0; j<iwn.size(); j++){
        iqn.push_back( std::complex<double>( 0.0, (2.0*j)*M_PI/beta ) );
    }
    
    std::vector< std::complex<double> > cub_spl_GG = splObj_GG.bosonic_corr(iqn,beta);

    std::ofstream outputSplineGG("test_spline_cpp.dat", std::ios::app | std::ios::out);
    for (size_t l=0; l<cub_spl_GG.size(); l++){
        outputSplineGG << iqn[l].imag() << "  " << cub_spl_GG[l].real() << "  " << cub_spl_GG[l].imag() << "\n";
    }
    outputSplineGG.close();


}


std::vector<double> get_derivative(std::vector<double> x_arr, std::vector<double> y_arr){
    assert(x_arr.size()==y_arr.size());
    std::vector<double> der_f(x_arr.size(),0.0);
    double right_most_val = 0.0, left_most_val = 0.0;
    for (size_t i=0; i<(x_arr.size()-2); i++){
        der_f[i+1] = ( y_arr[i+2] - y_arr[i] ) / ( x_arr[i+2] - x_arr[i] );
    }
    double left_most_2nd_der = ( y_arr[2] - 2.0*y_arr[1] + y_arr[0] ) / ( (x_arr[1]-x_arr[0])*(x_arr[1]-x_arr[0]) );
    double right_most_2nd_der = ( y_arr.back() - 2.0*( *(y_arr.end()-2) ) + *(y_arr.end()-3) ) / ( (x_arr.back()-*(x_arr.end()-2))*(x_arr.back()-*(x_arr.end()-2)) );

    left_most_val = der_f[1] - left_most_2nd_der * ( x_arr[1]-x_arr[0] );
    right_most_val = der_f[x_arr.size()-2] + right_most_2nd_der * ( x_arr.back()-*(x_arr.end()-2) );

    der_f[0] = left_most_val;
    der_f[x_arr.size()-1] = right_most_val;

    return der_f;
}   


// FileData get_data(const std::string& strName, const unsigned int& Ntau){

//     std::vector<double> iwn(Ntau,0.0);
//     std::vector<double> re(Ntau,0.0);
//     std::vector<double> im(Ntau,0.0);

//     std::ifstream inputFile(strName);
//     std::string increment("");
//     unsigned int idx = 0;
//     size_t pos=0;
//     std::string token;
//     const std::string delimiter = "\t\t";
//     if (inputFile.fail()){
//         std::cerr << "Error loading the file..." << "\n";
//         throw std::ios_base::failure("Check loding file procedure..");
//     } 
//     std::vector<double> tmp_vec;
//     while (getline(inputFile,increment)){
//         if (increment[0]=='/'){
//             continue;
//         }
//         while ((pos = increment.find(delimiter)) != std::string::npos) {
//             token = increment.substr(0, pos);
//             increment.erase(0, pos + delimiter.length());
//             tmp_vec.push_back(std::atof(token.c_str()));
//         }
//         tmp_vec.push_back(std::atof(increment.c_str()));
//         iwn[idx] = tmp_vec[0];
//         re[idx] = tmp_vec[1];
//         im[idx] = tmp_vec[2];

//         increment.clear();
//         tmp_vec.clear();
//         idx++;
//     }
    
//     inputFile.close();

//     FileData fileDataObj={iwn,re,im};
//     return fileDataObj;
// }

std::vector<double> get_iwn_to_tau(const std::vector< std::complex<double> >& F_iwn, double beta){
    size_t MM = F_iwn.size();
    std::cout << "size: " << MM << std::endl;
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
    for (int i=0; i<MM; i++){
        tau_final_G[MM] += ( 1.0/beta*std::exp(-im*M_PI*(1.0-MM))*F_iwn[i] ).real();
    }
    delete[] output;
    delete[] input;

    return tau_final_G;
}

template<typename T>
std::vector< T > generate_random_numbers(size_t arr_size, T min, T max){
    srand(time(0));
    std::vector< T > rand_num_container(arr_size);
    T random_number;
    for (size_t i=0; i<arr_size; i++){
        random_number = min + (T)( ( (T)rand() ) / ( (T)RAND_MAX ) * (max - min) );
        rand_num_container[i] = random_number;
    }
    return rand_num_container;
}