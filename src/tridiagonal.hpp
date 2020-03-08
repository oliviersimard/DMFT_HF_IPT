#ifndef Tridiagonal_H_
#define Tridiagonal_H_

#include<gsl/gsl_errno.h>
#include<gsl/gsl_fft_complex.h>
#include "green_utils.hpp"

// Inspired from http://www.mymathlib.com/matrices/linearsystems/tridiagonal.html
// and https://kluge.in-chemnitz.de/opensource/spline/

template<class T>
class LUtools{
   public:
      void tridiagonal_LU_decomposition( std::vector<T>& subdiagonal, std::vector<T>& diagonal, std::vector<T>& superdiagonal ) noexcept(false);
      void tridiagonal_LU_solve( std::vector<T>& subdiagonal, std::vector<T>& diagonal, std::vector<T>& superdiagonal, 
                                       std::vector<T>& B, std::vector<T>& x ) noexcept(false);
};

// spline interpolation
template<class T>
class spline{
    public:
        enum bd_type {
            first_deriv = 1,
            second_deriv = 2
        };

    protected:
        arma::Cube< T > m_y={};            // x,y coordinates of points
        std::vector<double> m_x={};
        // interpolation parameters
        // f(x) = a*(x-x_i)^3 + b*(x-x_i)^2 + c*(x-x_i) + y_i
        std::vector< T > m_a={},m_b={},m_c={};        // spline coefficients
        T  m_b0 {0.0}, m_c0 {0.0};                     // for left extrapol
        bd_type m_left=second_deriv, m_right=second_deriv;
        T  m_left_value {0.0}, m_right_value {0.0};
        bool m_force_linear_extrapolation=false;
        //
        T _S_1_0 {0.0}, _Sp_1_0 {0.0}, _Spp_1_0 {0.0};
        T _S_N_beta {0.0}, _Sp_N_beta {0.0}, _Spp_N_beta {0.0};
        std::vector< std::complex< T > > _Sppp={};

    public:
        // set default boundary condition to be zero curvature at both ends
        // spline(): m_left(second_deriv), m_right(second_deriv),
        //     m_left_value(0.0), m_right_value(0.0),
        //     m_force_linear_extrapolation(false), m_b0(0.0), m_c0(0.0){}
        spline()=default;
        // optional, but if called it has to come be before set_points()
        void set_boundary(bd_type left, T left_value,
                      bd_type right, T right_value,
                      bool force_linear_extrapolation=false);
        void set_points(const std::vector<double>& x,
                    const arma::Cube< T >& y, bool cubic_spline=true);
        T operator() (double x) const;
        T deriv(int order, double x) const;
        void iwn_tau_spl_extrm(const GreenStuff&, const double, const unsigned int);
        std::vector< std::complex< T > > fermionic_propagator(const std::vector< std::complex< T > >& iwn_arr,double beta);
        std::vector< std::complex< T > > bosonic_corr(const std::vector< std::complex< T > >& iqn_arr,double beta);
        std::vector< std::complex< T > > bosonic_corr_single_ladder(const std::vector< std::complex< T > >& ikn_tilde_arr,double beta,size_t n_bar);
};

template<>
inline void LUtools<double>::tridiagonal_LU_decomposition( std::vector<double>& subdiagonal, std::vector<double>& diagonal, std::vector<double>& superdiagonal ) noexcept(false){
   if ( !( subdiagonal.size()==superdiagonal.size() ) || !( subdiagonal.size()==diagonal.size()-1 ) ){
      throw std::length_error("The sizes of the vectors input are wrong. Vector diagonal should have larger size by one compared to others.");
   }
   const size_t size=diagonal.size()-1;
  
   for (size_t i=0; i<size; i++){
      if (diagonal[i]==0.0){
         throw std::invalid_argument("There should be zeros along the diagonal, because this would lead to division by zero.");
      }
      else{
         subdiagonal[i] /= diagonal[i];
         diagonal[i+1] -= subdiagonal[i] * superdiagonal[i];
      }
   }
   if (diagonal[size] == 0.0){
      throw std::invalid_argument("There should be zeros along the diagonal, because this would lead to division by zero.");
   }
}

template<>
inline void LUtools< std::complex<double> >::tridiagonal_LU_decomposition( std::vector< std::complex<double> >& subdiagonal, std::vector< std::complex<double> >& diagonal, 
                                                std::vector< std::complex<double> >& superdiagonal ) noexcept(false){
   if ( !( subdiagonal.size()==superdiagonal.size() ) || !( subdiagonal.size()==diagonal.size()-1 ) ){
      throw std::length_error("The sizes of the vectors input are wrong. Vector diagonal should have larger size by one compared to others.");
   }
   const size_t size=diagonal.size()-1;
  
   for (size_t i=0; i<size; i++){
      if (diagonal[i]==std::complex<double>(0.0,0.0)){
         throw std::invalid_argument("There shouldn't be zeros along the diagonal, because this would lead to division by complex zero.");
      }
      else{
         subdiagonal[i] /= diagonal[i];
         diagonal[i+1] -= subdiagonal[i] * superdiagonal[i];
      }
   }
   if (diagonal[size] == std::complex<double>(0.0,0.0)){
      throw std::invalid_argument("There shouldn't be zeros along the diagonal, because this would lead to division by complex zero.");
   }
}

template<>
inline void LUtools<double>::tridiagonal_LU_solve( std::vector<double>& subdiagonal, std::vector<double>& diagonal, 
                              std::vector<double>& superdiagonal, std::vector<double>& B, std::vector<double>& x ) noexcept(false){
   const size_t size=diagonal.size();
   for (size_t i=0; i<diagonal.size(); i++){
      if (diagonal[i]==0.0){
         throw std::invalid_argument("Division by zero will occur: there mustn't be 0. along the diagonal.");
      }
   }
   //   Solve the linear equation Ly = B for y, where L is a lower
   //   triangular matrix.
   x[0]=B[0]; // x has been initialized. (x --> m_b, B --> rhs)
   for (size_t i=1; i<size; i++){
      x[i] = B[i] - subdiagonal[i-1] * x[i-1];
   }
   //   Solve the linear equation Ux = y, where y is the solution
   //   obtained above of Ly = B and U is an upper triangular matrix.

   x[size-1] /= diagonal[size-1];
   for (int i=size-2; i >= 0; i--){
      x[i] -= superdiagonal[i] * x[i+1];
      x[i] /= diagonal[i];
   }
}

template<>
inline void LUtools< std::complex<double> >::tridiagonal_LU_solve( std::vector< std::complex<double> >& subdiagonal, std::vector< std::complex<double> >& diagonal, 
                              std::vector< std::complex<double> >& superdiagonal, std::vector< std::complex<double> >& B, std::vector< std::complex<double> >& x ) noexcept(false){
   const size_t size=diagonal.size();
   for (unsigned int i=0; i<diagonal.size(); i++){
      if (diagonal[i]==std::complex<double>(0.0,0.0)){
         throw std::invalid_argument("Division by zero will occur: there mustn't be (0.,0.) along the diagonal.");
      }
   }
   //   Solve the linear equation Ly = B for y, where L is a lower
   //   triangular matrix.
   x[0]=B[0]; // x has been initialized.
   for (size_t i=1; i<size; i++){
      x[i] = B[i] - subdiagonal[i-1] * x[i-1];
   }
   //   Solve the linear equation Ux = y, where y is the solution
   //   obtained above of Ly = B and U is an upper triangular matrix.

   x[size-1] /= diagonal[size-1];
   for (int i=size-2; i >= 0; i--){
      x[i] -= superdiagonal[i] * x[i+1];
      x[i] /= diagonal[i];
   }
}

// spline implementation
// -----------------------
template<class T>
void spline<T>::set_boundary(spline::bd_type left, T left_value,
                          spline::bd_type right, T right_value,
                          bool force_linear_extrapolation){
    assert(m_x.size()==0);          // set_points() must not have happened yet
    m_left=left;
    m_right=right;
    m_left_value=left_value;
    m_right_value=right_value;
    m_force_linear_extrapolation=force_linear_extrapolation;
}

template<class T>
void spline<T>::set_points(const std::vector<double>& x,
                        const arma::Cube< T >& y, bool cubic_spline){
    assert(x.size()==y.n_slices);
    assert(x.size()>2);
    m_x=x;
    m_y=y;
    int n=x.size();
    LUtools< T > LUObj;
    std::vector< T > initVec_b(n,0.0), initVec_a_c(n,0.0); // <--------------------------- Tested its effects when initVec_a_c(n-1,0.0).
    m_b=initVec_b; m_a=initVec_a_c; m_c=initVec_a_c; // Must init containers holding the spline coefficients.
    // TODO: maybe sort x and y, rather than returning an error
    for(size_t i=0; i<n-1; i++) {
        assert(m_x[i]<m_x[i+1]);
    }
    if(cubic_spline==true) { // cubic spline interpolation
        // setting up the matrix and right hand side of the equation system
        // for the parameters b[]
        arma::Mat< T > A(n,n);
        std::vector< T >  rhs(n);
        for(int i=1; i<n-1; i++) {
            A(i,i-1)=1.0*(x[i]-x[i-1]);
            A(i,i)=2.0*(x[i+1]-x[i-1]);
            A(i,i+1)=1.0*(x[i+1]-x[i]);
            rhs[i]=3.0*( (y.slice(i+1)(0,0)-y.slice(i)(0,0))/(x[i+1]-x[i]) - (y.slice(i)(0,0)-y.slice(i-1)(0,0))/(x[i]-x[i-1]) );
        }
        // boundary conditions
        if(m_left == spline::second_deriv) {
            // 2*b[0] = f''
            A(0,0)=2.0;
            A(0,1)=0.0;
            rhs[0]=m_left_value;
        } else if(m_left == spline::first_deriv) {
            // c[0] = f', needs to be re-expressed in terms of b:
            // (2b[0]+b[1])(x[1]-x[0]) = 3 ((y[1]-y[0])/(x[1]-x[0]) - f')
            A(0,0)=2.0*(x[1]-x[0]);
            A(0,1)=1.0*(x[1]-x[0]);
            rhs[0]=3.0*((y.slice(1)(0,0)-y.slice(0)(0,0))/(x[1]-x[0])-m_left_value);
        } else {
            assert(false);
        }
        if(m_right == spline::second_deriv) {
            // 2*b[n-1] = f''
            A(n-1,n-1)=2.0;
            A(n-1,n-2)=0.0;
            rhs[n-1]=m_right_value;
        } else if(m_right == spline::first_deriv) {
            // c[n-1] = f', needs to be re-expressed in terms of b:
            // (b[n-2]+2b[n-1])(x[n-1]-x[n-2])
            // = 3 (f' - (y[n-1]-y[n-2])/(x[n-1]-x[n-2]))
            A(n-1,n-1)=2.0*(x[n-1]-x[n-2]);
            A(n-1,n-2)=1.0*(x[n-1]-x[n-2]);
            rhs[n-1]=3.0*(m_right_value-(y.slice(n-1)(0,0)-y.slice(n-2)(0,0))/(x[n-1]-x[n-2]));
        } else {
            assert(false);
        }
        // solve the equation system to obtain the parameters b[]. Separating first the matrix A into
        // subdiagonal, diagonal and superdiagonal parts.
        std::vector< T > subdiagonal(n-1), diagonal(n), superdiagonal(n-1);
        for (size_t i=0; i<A.n_rows-1; i++){
           subdiagonal[i]=A(i+1,i);
           superdiagonal[i]=A(i,i+1);
        }
        for (size_t i=0; i<A.n_rows; i++){
           diagonal[i]=A(i,i);
        }
        try{
            LUObj.tridiagonal_LU_decomposition(subdiagonal,diagonal,superdiagonal);
            LUObj.tridiagonal_LU_solve(subdiagonal,diagonal,superdiagonal,rhs,m_b); // Writes in m_b
        }catch (const std::exception& err){
            std::cerr << err.what() << "\n";
        }
        // calculate parameters a[] and c[] based on b[]
        for(size_t i=0; i<n-1; i++) {
            m_a[i]=1.0/3.0*(m_b[i+1]-m_b[i])/(x[i+1]-x[i]);
            m_c[i]=(y.slice(i+1)(0,0)-y.slice(i)(0,0))/(x[i+1]-x[i])
                   - 1.0/3.0*(2.0*m_b[i]+m_b[i+1])*(x[i+1]-x[i]);
        }

    } else { // linear interpolation
        for(size_t i=0; i<n-1; i++) {
            m_a[i]=0.0;
            m_b[i]=0.0;
            m_c[i]=(m_y.slice(i+1)(0,0)-m_y.slice(i)(0,0))/(m_x[i+1]-m_x[i]);
        }
    }
    // for left extrapolation coefficients
    m_b0 = (m_force_linear_extrapolation==false) ? m_b[0] : 0.0;
    m_c0 = m_c[0];
    // for the right extrapolation coefficients
    // f_{n-1}(x) = b*(x-x_{n-1})^2 + c*(x-x_{n-1}) + y_{n-1}
    double h=x[n-1]-x[n-2];
    // m_b[n-1] is determined by the boundary condition
    m_a[n-1]=0.0;
    m_c[n-1]=3.0*m_a[n-2]*h*h+2.0*m_b[n-2]*h+m_c[n-2];   // = f'_{n-2}(x_{n-1})
    if(m_force_linear_extrapolation==true)
        m_b[n-1]=0.0;
    m_c[0]=m_c0; // What the hell happened to m_c[0]? Reset to 0 for some stupid reasons...

    // for (auto el : m_c) std::cout << "m_c: " << el << std::endl;
    // for (auto el : m_b) std::cout << "m_b: " << el << std::endl;
    // for (auto el : m_a) std::cout << "m_c: " << el << std::endl; 
    
}

template<class T>
T spline<T>::operator()(double x) const{
    size_t n=m_x.size();
    // find the closest point m_x[idx] < x, idx=0 even if x<m_x[0]
    std::vector<double>::const_iterator it;
    it=std::lower_bound(m_x.begin(),m_x.end(),x);
    int idx=std::max( int(it-m_x.begin())-1, 0);

    double h=x-m_x[idx];
    T interpol;
    if(x<m_x[0]) {
        // extrapolation to the left
        interpol=(m_b0*h + m_c0)*h + m_y.slice(0)(0,0);
    } else if(x>m_x[n-1]) {
        // extrapolation to the right
        interpol=(m_b[n-1]*h + m_c[n-1])*h + m_y.slice(n-1)(0,0);
    } else {
        // interpolation
        interpol=((m_a[idx]*h + m_b[idx])*h + m_c[idx])*h + m_y.slice(idx)(0,0);
        // std::cout << "interpol: " << interpol << std::endl;
    }
    return interpol;
}

template<class T>
T spline<T>::deriv(int order, double x) const{
    assert(order>0);

    size_t n=m_x.size();
    // find the closest point m_x[idx] < x, idx=0 even if x<m_x[0]
    std::vector<double>::const_iterator it;
    it=std::lower_bound(m_x.begin(),m_x.end(),x); // Returns an iterator pointing to the first element in the range [first, last) that is not less than (i.e. greater or equal to) value, or last if no such element is found.
    int idx=std::max( int(it-m_x.begin())-1, 0 ); // minus one because f(x+h)-f(x-h)/2h minimises the approximate error.

    double h=x-m_x[idx];
    T interpol;
    if(x<m_x[0]) {
        // extrapolation to the left
        switch(order) {
        case 1:
            interpol=2.0*m_b0*h + m_c0;
            break;
        case 2:
            interpol=2.0*m_b0*h;
            break;
        default:
            interpol=0.0;
            break;
        }
    } else if(x>m_x[n-1]) {
        // extrapolation to the right
        switch(order) {
        case 1:
            interpol=2.0*m_b[n-1]*h + m_c[n-1];
            break;
        case 2:
            interpol=2.0*m_b[n-1];
            break;
        default:
            interpol=0.0;
            break;
        }
    } else {
        // interpolation
        switch(order) {
        case 1:
            interpol=(3.0*m_a[idx]*h + 2.0*m_b[idx])*h + m_c[idx];
            break;
        case 2:
            interpol=6.0*m_a[idx]*h + 2.0*m_b[idx];
            break;
        case 3:
            interpol=6.0*m_a[idx];
            break;
        default:
            interpol=0.0;
            break;
        }
    }
    return interpol;
}

template<class T>
void spline<T>::iwn_tau_spl_extrm(const GreenStuff& SelfEnergy, const double beta, const unsigned int N_tau){
    // Setting the boundary conditions. The spline f_i starts at i=1, because f_0(x) = b_0(x-x_0)^2 + c_0(x-x_0) + y_0,  x_0 <= x <= x_1.
    _S_1_0=m_y.slice(0)(0,0); // f_0(x_0) = a_0(x_0-x_0)^3 + b_0(x_0-x_0)^2 + c_0(x_0-x_0) + y_0
    _Sp_1_0=m_c0; // f'_0(x_0) = 3a_0(x_0-x_0)^2 + 2b_0(x_0-x_0)^1 + c_0 
    _Spp_1_0=2.0*m_b[0]; // f''_0(x_0) = 6a_0(x_0-x_0) + 2b_0
    // The spline f_i ends at i=N, because f_N(x) = b_N(x-x_N)^2 + c_N(x-x_N) + y_N, x >= x_N.
    std::vector<double>::const_iterator it = m_x.end(); // returns pointer after last element.
    const double h_last = *(it-1)-*(it-2); // This is kind of beta^{-}: f_{N-1}(x_N)
    _S_N_beta=m_a[2*N_tau-1]*h_last*h_last*h_last+m_b[2*N_tau-1]*h_last*h_last+m_c[2*N_tau-1]*h_last+m_y.slice(2*N_tau-1)(0,0); // In Shaheen's code it is m_y.slice(2*tau-1)...but this is not beta...
    _Sp_N_beta=3.0*m_a[2*N_tau-1]*h_last*h_last+2.0*m_b[2*N_tau-1]*h_last+m_c[2*N_tau-1];
    _Spp_N_beta=6.0*m_a[2*N_tau-1]*h_last+2.0*m_b[2*N_tau-1];
    // This will be used later for IFFT.
    for (size_t n=0; n<m_x.size()-1; n++){
        _Sppp.push_back(6.0*m_a[n]);
    }
    // Putting the _S's to use in the Fourier transformation from tau (double) to complex<double> (iwn).
    const unsigned int timeGrid = 2*N_tau;
    T F[2*timeGrid],Fp[2*timeGrid];
    const std::complex< T > im(0.0,1.0);
    for(size_t i=0; i<timeGrid; i++){
        F[2*i] = (std::exp(im*M_PI*(double)i/(double)timeGrid)*_Sppp[i]).real();
        F[2*i+1] = (std::exp(im*M_PI*(double)i/(double)timeGrid)*_Sppp[i]).imag();
    }
    
    gsl_fft_complex_radix2_backward(F, 1, timeGrid);
    
    for(size_t i=0; i<timeGrid; i++){
        if(i<timeGrid/2){ // Mirroring the data.
            Fp[2*i] = F[timeGrid+2*i];
            Fp[2*i+1] = F[timeGrid+2*i+1];
        }else{
            Fp[2*i] = F[2*i-timeGrid];
            Fp[2*i+1] = F[2*i+1-timeGrid];
        }
    }
    for(size_t i=0; i<timeGrid; i++){ // timeGrid (N in paper) factor is absorbed by IFFT.
        SelfEnergy.matsubara_w.slice(i)(0,0)=( -1.0*( -(_S_1_0+_S_N_beta)/iwnArr_l[i] + (_Sp_1_0+_Sp_N_beta)/(iwnArr_l[i]*iwnArr_l[i]) - (_Spp_1_0+_Spp_N_beta)/(iwnArr_l[i]*iwnArr_l[i]*iwnArr_l[i]) + (1.0-std::exp(1.0*iwnArr_l[i]*beta/(double)timeGrid))/(iwnArr_l[i]*iwnArr_l[i]*iwnArr_l[i]*iwnArr_l[i])*(Fp[2*i]+im*Fp[2*i+1]) ) );
    }
}

template<class T>
std::vector< std::complex< T > > spline<T>::fermionic_propagator(const std::vector< std::complex< T > >& iwn_arr,double beta){
    size_t size = iwn_arr.size();
    const std::complex< T > im(0.0,1.0);
    _S_1_0=m_y.slice(0)(0,0); // f_0(x_0) = a_0(x_0-x_0)^3 + b_0(x_0-x_0)^2 + c_0(x_0-x_0) + y_0
    _Sp_1_0=m_c0; // f'_0(x_0) = 3a_0(x_0-x_0)^2 + 2b_0(x_0-x_0)^1 + c_0 
    _Spp_1_0=2.0*m_b[0]; // f''_0(x_0) = 6a_0(x_0-x_0) + 2b_0

    std::vector<double>::const_iterator it = m_x.end(); // returns pointer after last element.
    const double h_last = *(it-1)-*(it-2);
    _S_N_beta=m_a[size-1]*h_last*h_last*h_last+m_b[size-1]*h_last*h_last+m_c[size-1]*h_last+m_y.slice(size-1)(0,0); // In Shaheen's code it is m_y.slice(2*tau-1)...but this is not beta...
    _Sp_N_beta=3.0*m_a[size-1]*h_last*h_last+2.0*m_b[size-1]*h_last+m_c[size-1];
    _Spp_N_beta=6.0*m_a[size-1]*h_last+2.0*m_b[size-1];
    
    // This will be used later for IFFT.
    for (size_t n=0; n<m_x.size()-1; n++){
        _Sppp.push_back(6.0*m_a[n]);
    }
    // Putting the _S's to use in the Fourier transformation from tau (double) to complex<double> (iwn).
    T F[2*size],Fp[2*size];
    for(size_t i=0; i<size; i++){
        F[2*i] = (std::exp(im*M_PI*(double)i/(double)size)*_Sppp[i]).real();
        F[2*i+1] = (std::exp(im*M_PI*(double)i/(double)size)*_Sppp[i]).imag();
    }
    
    gsl_fft_complex_radix2_backward(F, 1, size);
    
    for(size_t i=0; i<size; i++){
        if(i<size/2){ // Mirroring the data.
            Fp[2*i] = F[size+2*i];
            Fp[2*i+1] = F[size+2*i+1];
        }else{
            Fp[2*i] = F[2*i-size];
            Fp[2*i+1] = F[2*i+1-size];
        }
    }
    std::vector< std::complex< T > > cubic_spline(size,0.0);
    for(size_t i=0; i<size; i++){ // timeGrid (N in paper) factor is absorbed by IFFT.
        cubic_spline[i] = ( -(_S_1_0+_S_N_beta)/iwn_arr[i] + (_Sp_1_0+_Sp_N_beta)/(iwn_arr[i]*iwn_arr[i]) - (_Spp_1_0+_Spp_N_beta)/(iwn_arr[i]*iwn_arr[i]*iwn_arr[i]) + (1.0-std::exp(1.0*iwn_arr[i]*beta/(double)size))/(iwn_arr[i]*iwn_arr[i]*iwn_arr[i]*iwn_arr[i])*(Fp[2*i]+im*Fp[2*i+1]) );
    }

    return cubic_spline;
}

template<class T>
std::vector< std::complex< T > > spline<T>::bosonic_corr(const std::vector< std::complex< T > >& iqn_arr,double beta){
    size_t size = iqn_arr.size();

    _S_1_0=m_y.slice(0)(0,0); // f_0(x_0) = a_0(x_0-x_0)^3 + b_0(x_0-x_0)^2 + c_0(x_0-x_0) + y_0
    _Sp_1_0=m_c0; // f'_0(x_0) = 3a_0(x_0-x_0)^2 + 2b_0(x_0-x_0)^1 + c_0 
    _Spp_1_0=2.0*m_b[0]; // f''_0(x_0) = 6a_0(x_0-x_0) + 2b_0

    std::vector<double>::const_iterator it = m_x.end(); // returns pointer after last element.
    const double h_last = *(it-1)-*(it-2);
    _S_N_beta=m_a[size-1]*h_last*h_last*h_last+m_b[size-1]*h_last*h_last+m_c[size-1]*h_last+m_y.slice(size-1)(0,0); // In Shaheen's code it is m_y.slice(2*tau-1)...but this is not beta...
    _Sp_N_beta=3.0*m_a[size-1]*h_last*h_last+2.0*m_b[size-1]*h_last+m_c[size-1];
    _Spp_N_beta=6.0*m_a[size-1]*h_last+2.0*m_b[size-1];

    // std::cout << "S_beta: " << _S_N_beta << " S_0 " << _S_1_0 << "\n";
    // std::cout << "Sp_beta: " << _Sp_N_beta << " Sp_0 " << _Sp_1_0 << "\n";
    // std::cout << "Spp_beta: " << _Spp_N_beta << " Spp_0 " << _Spp_1_0 << "\n";

    // This will be used later for FFT.
    for (size_t n=0; n<size; n++){
        _Sppp.push_back(6.0*m_a[n]);
    }

    // FFT
    std::complex< T >* input = new std::complex< T >[size];
    std::complex< T >* output = new std::complex< T >[size];
    for (size_t i=0; i<size; i++){
        input[i] = _Sppp[i];
    }
    fftw_plan plan;
    plan = fftw_plan_dft_1d(size,reinterpret_cast<fftw_complex*>(input),reinterpret_cast<fftw_complex*>(output),FFTW_BACKWARD,FFTW_ESTIMATE);
    fftw_execute(plan);
    
    std::vector< std::complex< T > > cubic_spline(size,0.0);
    for(size_t i=0; i<size; i++){ // timeGrid (N in paper) factor is absorbed by IFFT.
        if (iqn_arr[i]!=std::complex< T >(0.0,0.0)){
            cubic_spline[i] = ( (_S_N_beta-_S_1_0)/iqn_arr[i] - (_Sp_N_beta-_Sp_1_0)/(iqn_arr[i]*iqn_arr[i]) + (_Spp_N_beta-_Spp_1_0)/(iqn_arr[i]*iqn_arr[i]*iqn_arr[i]) + (1.0-std::exp(1.0*iqn_arr[i]*beta/(double)size))/(iqn_arr[i]*iqn_arr[i]*iqn_arr[i]*iqn_arr[i])*(output[i]) );
        } else{
            for (size_t j=1; j<m_x.size(); j++){ // Dealing with the 0th bosonic Matsubara frequency...
                //std::cout << "j: " << j << " val: " << cubic_spline[i] << " ma: " << m_a[j-1] << " mb: " << m_b[j-1] << " mc: " << m_c[j-1] << " my: " << m_y[j-1] << "\n";
                cubic_spline[i] += m_a[j-1]*( (m_x[j]-m_x[j-1])*(m_x[j]-m_x[j-1])*(m_x[j]-m_x[j-1])*(m_x[j]-m_x[j-1]) )/4.0 +
                m_b[j-1]*( (m_x[j]-m_x[j-1])*(m_x[j]-m_x[j-1])*(m_x[j]-m_x[j-1]) )/3.0 + m_c[j-1]*( (m_x[j]-m_x[j-1])*(m_x[j]-m_x[j-1]) )/2.0 +
                m_y.slice(j-1)(0,0)*( (m_x[j]-m_x[j-1]) );
            }
        }
    }

    delete[] input;
    delete[] output;

    return cubic_spline;
}

template<class T>
std::vector< std::complex< T > > spline<T>::bosonic_corr_single_ladder(const std::vector< std::complex< T > >& ikn_tilde_arr,double beta,size_t n_bar){
    size_t size = ikn_tilde_arr.size();
    const double delta_tau = beta/(double)size;
    const std::complex< T > ikn_bar = ikn_tilde_arr[n_bar];

    _S_1_0=m_y.slice(0)(0,0); // f_0(x_0) = a_0(x_0-x_0)^3 + b_0(x_0-x_0)^2 + c_0(x_0-x_0) + y_0
    _Sp_1_0=m_c0; // f'_0(x_0) = 3a_0(x_0-x_0)^2 + 2b_0(x_0-x_0)^1 + c_0 
    _Spp_1_0=2.0*m_b[0]; // f''_0(x_0) = 6a_0(x_0-x_0) + 2b_0

    std::vector<double>::const_iterator it = m_x.end(); // returns pointer after last element.
    const double h_last = *(it-1)-*(it-2);
    _S_N_beta=m_a[size-1]*h_last*h_last*h_last+m_b[size-1]*h_last*h_last+m_c[size-1]*h_last+m_y.slice(size-1)(0,0); // In Shaheen's code it is m_y.slice(2*tau-1)...but this is not beta...
    _Sp_N_beta=3.0*m_a[size-1]*h_last*h_last+2.0*m_b[size-1]*h_last+m_c[size-1];
    _Spp_N_beta=6.0*m_a[size-1]*h_last+2.0*m_b[size-1];

    // std::cout << "S_beta: " << _S_N_beta << " S_0 " << _S_1_0 << "\n";
    // std::cout << "Sp_beta: " << _Sp_N_beta << " Sp_0 " << _Sp_1_0 << "\n";
    // std::cout << "Spp_beta: " << _Spp_N_beta << " Spp_0 " << _Spp_1_0 << "\n";

    // This will be used later for FFT.
    std::complex< T > n_bar_weight;
    for (size_t n=0; n<size; n++){
        n_bar_weight = std::exp(-1.0*ikn_tilde_arr[n_bar]*beta*(double)n/(double)size);
        _Sppp.push_back(6.0*n_bar_weight*m_a[n]);
    }

    // FFT
    std::complex< T >* input = new std::complex< T >[size];
    std::complex< T >* output = new std::complex< T >[size];
    for (size_t i=0; i<size; i++){
        input[i] = _Sppp[i];
    }
    fftw_plan plan;
    plan = fftw_plan_dft_1d(size,reinterpret_cast<fftw_complex*>(input),reinterpret_cast<fftw_complex*>(output),FFTW_BACKWARD,FFTW_ESTIMATE);
    fftw_execute(plan);
    
    std::vector< std::complex< T > > cubic_spline(size,0.0);
    for(size_t i=0; i<size; i++){ // timeGrid (N in paper) factor is absorbed by IFFT.
        if (std::abs(ikn_tilde_arr[i]-ikn_bar)!=0.0){
            cubic_spline[i] = ( (_S_N_beta-_S_1_0)/( ikn_tilde_arr[i]-ikn_bar ) - (_Sp_N_beta-_Sp_1_0)/( ikn_tilde_arr[i]-ikn_bar )/( ikn_tilde_arr[i]-ikn_bar ) + (_Spp_N_beta-_Spp_1_0)/( ikn_tilde_arr[i]-ikn_bar )/( ikn_tilde_arr[i]-ikn_bar )/( ikn_tilde_arr[i]-ikn_bar ) + (1.0-std::exp((ikn_tilde_arr[i]-ikn_bar)*delta_tau))*(output[i])/( ikn_tilde_arr[i]-ikn_bar )/( ikn_tilde_arr[i]-ikn_bar )/( ikn_tilde_arr[i]-ikn_bar )/( ikn_tilde_arr[i]-ikn_bar ) );
        } else{
            for (size_t j=1; j<m_x.size(); j++){ // Dealing with the 0th bosonic Matsubara frequency...
                //std::cout << "j: " << j << " val: " << cubic_spline[i] << " ma: " << m_a[j-1] << " mb: " << m_b[j-1] << " mc: " << m_c[j-1] << " my: " << m_y[j-1] << "\n";
                cubic_spline[i] += m_a[j-1]*( (m_x[j]-m_x[j-1])*(m_x[j]-m_x[j-1])*(m_x[j]-m_x[j-1])*(m_x[j]-m_x[j-1]) )/4.0 +
                m_b[j-1]*( (m_x[j]-m_x[j-1])*(m_x[j]-m_x[j-1])*(m_x[j]-m_x[j-1]) )/3.0 + m_c[j-1]*( (m_x[j]-m_x[j-1])*(m_x[j]-m_x[j-1]) )/2.0 +
                m_y.slice(j-1)(0,0)*( (m_x[j]-m_x[j-1]) );
            }
        }
    }

    delete[] input;
    delete[] output;

    return cubic_spline;
}

#endif /* end of Tridiagonal_H_ */