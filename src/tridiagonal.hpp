#include "thread_utils.hpp"

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
        arma::Cube< T > m_y;            // x,y coordinates of points
        std::vector<double> m_x;
        // interpolation parameters
        // f(x) = a*(x-x_i)^3 + b*(x-x_i)^2 + c*(x-x_i) + y_i
        std::vector< T > m_a,m_b,m_c;        // spline coefficients
        T  m_b0, m_c0;                     // for left extrapol
        bd_type m_left, m_right;
        T  m_left_value, m_right_value;
        bool    m_force_linear_extrapolation;
        //
        T _S_1_0, _Sp_1_0, _Spp_1_0;
        T _S_N_beta, _Sp_N_beta, _Spp_N_beta;
        std::vector< T > _Sppp;

    public:
        // set default boundary condition to be zero curvature at both ends
        spline(): m_left(second_deriv), m_right(second_deriv),
            m_left_value(0.0), m_right_value(0.0),
            m_force_linear_extrapolation(false){}

        // optional, but if called it has to come be before set_points()
        void set_boundary(bd_type left, T left_value,
                      bd_type right, T right_value,
                      bool force_linear_extrapolation=false);
        void set_points(const std::vector<double>& x,
                    const arma::Cube< T >& y, bool cubic_spline=true);
        T operator() (double x) const;
        T deriv(int order, double x) const;
        void iwn_tau_spl_extrm(const GreenStuff&, const double, const unsigned int);
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
         throw std::invalid_argument("There should be zeros along the diagonal, because this would lead to division by complex zero.");
      }
      else{
         subdiagonal[i] /= diagonal[i];
         diagonal[i+1] -= subdiagonal[i] * superdiagonal[i];
      }
   }
   if (diagonal[size] == std::complex<double>(0.0,0.0)){
      throw std::invalid_argument("There should be zeros along the diagonal, because this would lead to division by complex zero.");
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
    std::vector< T > initVec_b(n,0.0), initVec_a_c(n-1,0.0);
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
            std::cerr << err.what() << std::endl;
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
            m_c[i]=(m_y[i+1]-m_y[i])/(m_x[i+1]-m_x[i]);
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
    
}

template<class T>
T spline<T>::operator() (double x) const{
    size_t n=m_x.size();
    // find the closest point m_x[idx] < x, idx=0 even if x<m_x[0]
    std::vector<double>::const_iterator it;
    it=std::lower_bound(m_x.begin(),m_x.end(),x);
    int idx=std::max( int(it-m_x.begin())-1, 0);

    double h=x-m_x[idx];
    T interpol;
    if(x<m_x[0]) {
        // extrapolation to the left
        interpol=(m_b0*h + m_c0)*h + m_y[0];
    } else if(x>m_x[n-1]) {
        // extrapolation to the right
        interpol=(m_b[n-1]*h + m_c[n-1])*h + m_y[n-1];
    } else {
        // interpolation
        interpol=((m_a[idx]*h + m_b[idx])*h + m_c[idx])*h + m_y[idx];
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
    // Setting the boundary conditions. The spline f_i starts at i=1, because f_0(x) = b_1(x-x_1) + c_1(x-x_1) + y_1, x <= x_1.
    _S_1_0=m_y[0]; // f_1(x_0) = a_1(x_0-x_1)^3 + b_1(x_0-x_1)^2 + c_1(x_0-x_1) + y_1 = y_0
    _Sp_1_0=m_c0; // f'_1(x_0) = 3a_1(x_0-x_1)^2 + 2b_1(x_0-x_1)^1 + c_1 = c_0
    _Spp_1_0=2.0*m_b[0]; // f''_1(x_0) = 6a_1(x_0-x_1) + 2b_1 = b_0
    // The spline f_i ends at i=N, because f_N(x) = b_N(x-x_N) + c_N(x-x_N) + y_N, x >= x_1.
    std::vector<double>::const_iterator it = m_x.end(); // returns pointer after last element.
    const double h_last = *(it-1)-*(it-2);
    _S_N_beta=m_a.back()*h_last*h_last*h_last+m_b[2*N_tau-1]*h_last*h_last+m_c.back()*h_last+m_y.slice(2*N_tau-1)(0,0); // In Shaheen's code it is m_y.slice(2*tau-1)...but this is not beta...
    _Sp_N_beta=3.0*m_a.back()*h_last*h_last+2.0*m_b[2*N_tau-1]*h_last+m_c.back();
    _Spp_N_beta=6.0*m_a.back()*h_last+2.0*m_b[2*N_tau-1];
    // This will be used later for IFFT.
    for (size_t n=0; n<m_x.size()-1; n++){
        _Sppp.push_back(6.0*m_a[n]);
    }
    // Putting the _S's to use in the Fourier transformation from tau (double) to complex<double> (iwn).
    const unsigned int timeGrid = 2*N_tau;
    double F[2*timeGrid],Fp[2*timeGrid];
    const std::complex<double> im(0.0,1.0);
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
            Fp[2*i]= F[2*i-timeGrid];
            Fp[2*i+1] = F[2*i+1-timeGrid];
        }
    }
    for(size_t i=0; i<timeGrid; i++){ // timeGrid (N in paper) factor is absorbed by IFFT.
        SelfEnergy.matsubara_w.slice(i)(0,0)=( -1.0*( -(_S_1_0+_S_N_beta)/iwnArr_l[i] + (_Sp_1_0+_Sp_N_beta)/(iwnArr_l[i]*iwnArr_l[i]) - (_Spp_1_0+_Spp_N_beta)/(iwnArr_l[i]*iwnArr_l[i]*iwnArr_l[i]) + (1.0-std::exp(1.0*iwnArr_l[i]*beta/(double)timeGrid))/(iwnArr_l[i]*iwnArr_l[i]*iwnArr_l[i]*iwnArr_l[i])*(Fp[2*i]+im*Fp[2*i+1]) ) );
    }
}   


// int main(int argc, char** argv){

//    // Solving Ax = b
//    double B[NN] = { 2. , 0.5 , 7. }, x[NN];

//    double mat[][NN] = { { 2., -1., -2. },
//                     { -4., 6., 3. },
//                     { -4., -2., 8. } };

//    double subdiagonal[2], diagonal[NN], superdiagonal[2];

//    // Building main parts of the routine.
//    for (int k=0; k<2; k++){
//       subdiagonal[k]=mat[k+1][k];
//       superdiagonal[k]=mat[k][k+1];
//    }
//    for (int k=0; k<NN; k++){
//       diagonal[k]=mat[k][k];
//    }

//    for (int i=0; i<NN; i++){
//       for (int j=0; j<(int)(sizeof(mat)/sizeof(mat[0])); j++){
//          std::cout << mat[i][j] << " ";
//       }
//       std::cout << "\n";
//    }

//    int err = Tridiagonal_LU_Decomposition(subdiagonal, diagonal, superdiagonal, NN);
//    if (err < 0) printf(" Matrix A failed the LU decomposition\n");
//    else{
//       err = Tridiagonal_LU_Solve(subdiagonal,diagonal,superdiagonal,B,x,NN);
//    }

//    // Testing the spline methods
//    const std::complex<double> im = std::complex<double>(0.0,1.0);
//    std::vector< std::complex<double> > Y(5);
//    std::vector<double> X(5);
//    X[0]=0.1; X[1]=0.4; X[2]=1.2; X[3]=1.8; X[4]=2.0; /// <------------------ test passed
//    Y[0]=0.1+0.1*im; Y[1]=0.7-0.2*im; Y[2]=0.6+0.0*im; Y[3]=1.1-0.4*im; Y[4]=0.9;

//    spline< std::complex<double> > s;
//    s.set_points(X,Y);
//    std::cout << s(1.5) << std::endl;
//    s.iwn_tau_spl_extrm();

