#ifndef Integral_utils_H_
#define Integral_utils_H_

#include <armadillo>
#include <complex>
#include <functional>
#include <exception>
#include <assert.h>
#include <type_traits>

// #define INTEGRAL
#define MAX_ITER_INTEGRAL 20
#define MAX_ITER_ROOT 100000
#define ROOT_FINDING_TOL 0.0001
// for Gaussian quadratures
#define ABS_TOL 1e-2
#define MAX_DEPTH 6

template<class T>
struct GaussEl{
    T jj{0};
    T szsz{0};
    GaussEl()=default;
    GaussEl(T a, T b) : jj(a), szsz(b){};
    GaussEl operator+(const GaussEl& a) const noexcept;
    GaussEl operator+(const T& a) const noexcept;
    GaussEl operator*(const T& a) const noexcept;
};

template<class T>
GaussEl<T> GaussEl<T>::operator+(const T& a) const noexcept{
    return GaussEl<T>(this->jj+a,this->szsz+a);
}
template<class T>
GaussEl<T> GaussEl<T>::operator*(const T& a) const noexcept{
    return GaussEl<T>(this->jj*a,this->szsz*a);
}
template<class T>
GaussEl<T> GaussEl<T>::operator+(const GaussEl<T>& a) const noexcept{
    return GaussEl<T>( this->jj+a.jj, this->szsz+a.szsz );
}
template<class T>
GaussEl<T> operator*(const GaussEl<T>& a, const T& num) noexcept{
    return GaussEl<T>(a.jj*num, a.szsz*num);
}
template<class T>
GaussEl<T> operator*(const GaussEl<T>& a, const double& num) noexcept{
    return GaussEl<T>(a.jj*num, a.szsz*num);
}
template<class T>
GaussEl<T> operator*(const double& num,const GaussEl<T>& a) noexcept{
    return operator*(a, num);
}
// overloading operators to handle the vectors in gauss integral methods
template<typename T, typename U>
std::vector<T> operator*(std::vector<T> src, const U& number) noexcept{
    for (size_t i=0; i<src.size(); i++){
        src[i] *= number;
    }
    return src;
}
template<typename T, typename U>
std::vector<T> operator*(const U& number, std::vector<T> src) noexcept{
    return operator*(src,number);
}
template<typename T, typename U>
std::vector<T> operator+(std::vector<T> src, const U& number) noexcept{
    for (size_t i=0; i<src.size(); i++){
        src[i] += number;
    }
    return src;
}
template<typename T>
std::vector<T> operator+(std::vector<T> src, std::vector<T> other) noexcept{
    for (size_t i=0; i<src.size(); i++){
        src[i] += other[i];
    }
    return src;
}
template<typename T, typename U>
std::vector<T> operator+(const U& number, std::vector<T> src) noexcept{
    return operator+(src,number);
}

struct cubic_roots{ // Hosting the roots of cubic equation
    std::complex<double> x1;
    std::complex<double> x2;
    std::complex<double> x3;
};

std::complex<double> cbrt(std::complex<double> num);
cubic_roots get_cubic_roots(double a, double b, double c, double d);

class Integrals{
    public:
        double coarse_app(std::function< double(double) >,double,double) const;
        std::complex<double> coarse_app(std::function< std::complex<double>(double,std::complex<double>) >,double,double,std::complex<double>) const;
        double trap_app(std::function< double(double) >,double,double) const;
        double simps_app(std::function< double(double) >,double,double) const;
        std::complex<double> simps_app(std::function< std::complex<double>(double,std::complex<double>) >,double,double,std::complex<double>) const;


        double trap(std::function< double(double) >,double,double,double,unsigned int) const;
        double simps(std::function< double(double) >,double,double,double,unsigned int) const;
        std::complex<double> simps(std::function< std::complex<double>(double,std::complex<double>) >,double,double,std::complex<double>,double,unsigned int) const;

        double integrate_trap(std::function< double(double) >,double,double,double) const;
        double integrate_simps(std::function< double(double) >,double,double,double) const;
        std::complex<double> integrate_simps(std::function< std::complex<double>(double,std::complex<double>) >,double,double,std::complex<double>,double) const;
        
        // D. Keffer, ChE 505 ,University of Tennessee, September, 1999
        //double I2DRec(std::function<double(double,double)>,double,double,double,double,unsigned int,double tol=0.001,unsigned int maxVal=10,bool is_converged=false,double prevResult=0.0) const;
        double I2D(std::function<double(double,double)>,double,double,double,double,std::string rule="trapezoidal",double tol=1e-4,unsigned int maxVal=40,bool is_converged=false) const noexcept(false);
        std::complex<double> I2D(std::function<std::complex<double>(double,double,std::complex<double>)>,double,double,double,double,std::complex<double>,std::string rule="trapezoidal",double tol=1e-4,unsigned int maxVal=40,bool is_converged=false) const noexcept(false);
        std::complex<double> I1D(std::function<std::complex<double>(double,std::complex<double>)> funct,double k0,double kf,std::complex<double> iwn,std::string rule="trapezoidal",double tol=1e-5,unsigned int maxDelta=100) const noexcept(false);
        double I1D(std::function<double(double)> funct,double k0,double kf,std::string rule="trapezoidal",double tol=1e-5,unsigned int maxDelta=100) const noexcept(false);
        // MonteCarlo integral
        template<typename T, typename U> double integral_functional_3d(std::function<U(T,T)> f, T xmin, T xmax, T ymin, T ymax, U zmin, U zmax, bool(*y_cond)(T x,T y)) const;
        // template<typename T, typename U> inline typename std::enable_if< std::is_same< U,std::complex<double> >::value >::type integral_functional_3d(std::function<std::complex<double>(T,T)> f, T xmin, T xmax, T ymin, T ymax, std::complex<double> zmin, std::complex<double> zmax, bool(*y_cond)(T x,T y)) const;
        template<typename T> std::complex<double> integral_functional_3d(std::function<std::complex<double>(T,T)> f, T xmin, T xmax, T ymin, T ymax, std::complex<double> zmin, std::complex<double> zmax, bool(*y_cond)(T x,T y)) const;
        template<typename T, typename U> U I2D_constrained(std::function<U(T,T)> funct,std::function<bool(T,T)> cond,T x0,T xf,T y0,T yf,const double tol=1e-2,unsigned int maxVal=20,bool is_converged=false) const;
    
        // double I1D(std::vector<double>& vecfunct,double delta_tau) const;
        template<typename T> T I1D_VEC(std::vector< T >& vecfunct,double delta,std::string rule="trapezoidal") const noexcept(false);
        template<typename T> T I1D_VEC(std::vector< T >&& vecfunct,double delta,std::string rule="trapezoidal") const noexcept(false);
        // Gaussian Quadratures
        template<typename T, typename U>
        inline U gauss_quad_1D(std::function<U(T)> func, T x1, T x2, int depth = 1) const noexcept;

        template<typename T, typename U, typename std::enable_if< std::is_same< U, std::complex<T> >::value >::type* = nullptr >
        inline U gauss_quad_2D(std::function<U(T,T)> func, T x1, T x2, T y1, T y2, int depth=1) const noexcept{
            // sqrt(3) = 1.7320508075688772
            T w1 = static_cast<T>(1.0), w2 = static_cast<T>(1.0);
            constexpr T xi1 = (T)(-1.0/1.7320508075688772), xi2 = (T)(1.0/1.7320508075688772);
            constexpr T eta1 = (T)(-1.0/1.7320508075688772), eta2 = (T)(1.0/1.7320508075688772);
            T hx = (x2-x1)/static_cast<T>(2.0), hy = (y2-y1)/static_cast<T>(2.0);
            T dx = (x2+x1)/static_cast<T>(2.0), dy = (y2+y1)/static_cast<T>(2.0);
            // U abst{0};

            U layer = hx*hy*w1*w1*func(hx*xi1+dx,hy*eta1+dy) + hx*hy*w1*w2*func(hx*xi1+dx,hy*eta2+dy) + 
                hx*hy*w2*w1*func(hx*xi2+dx,hy*eta1+dy) + hx*hy*w2*w2*func(hx*xi2+dx,hy*eta2+dy);
            
            // if (depth > 1){
            //     abst = std::abs(layer-prev_result)/4.0;
            // }
            // if ( (depth > 1) && (std::abs(abst) < std::abs(abs_tol)) ){
            //     return layer;
            if (depth==MAX_DEPTH){
                return layer;
            } else{
                depth += 1;
                // layer /= static_cast<T>(4.0);
                return gauss_quad_2D(func,x1,x1+hx,y1,y1+hy,depth) + gauss_quad_2D(func,x1,x1+hx,y1+hy,y2,depth) +
                    gauss_quad_2D(func,x1+hx,x2,y1,y1+hy,depth) + gauss_quad_2D(func,x1+hx,x2,y1+hy,y2,depth);
            }
        }

        template<typename T, typename U, typename std::enable_if< std::is_same< U, GaussEl<std::complex<T>> >::value >::type* = nullptr>
        inline U gauss_quad_2D(std::function<U(T,T)> func, T x1, T x2, T y1, T y2, int depth=1) const noexcept{
            // sqrt(3) = 1.7320508075688772
            T w1 = static_cast<T>(1.0), w2 = static_cast<T>(1.0);
            constexpr T xi1 = (T)(-1.0/1.7320508075688772), xi2 = (T)(1.0/1.7320508075688772);
            constexpr T eta1 = (T)(-1.0/1.7320508075688772), eta2 = (T)(1.0/1.7320508075688772);
            T hx = (x2-x1)/static_cast<T>(2.0), hy = (y2-y1)/static_cast<T>(2.0);
            T dx = (x2+x1)/static_cast<T>(2.0), dy = (y2+y1)/static_cast<T>(2.0);

            U down_left = func(hx*xi1+dx,hy*eta1+dy);
            U up_left = func(hx*xi1+dx,hy*eta2+dy);
            U down_right = func(hx*xi2+dx,hy*eta1+dy);
            U up_right = func(hx*xi2+dx,hy*eta2+dy);

            U layer = hx*hy*w1*w1*down_left + hx*hy*w1*w2*up_left + 
                hx*hy*w2*w1*down_right + hx*hy*w2*w2*up_right;
            
            if (depth==3){
                return layer;
            } else{
                depth += 1;
                // layer /= static_cast<T>(4.0);
                return gauss_quad_2D(func,x1,x1+hx,y1,y1+hy,depth) + gauss_quad_2D(func,x1,x1+hx,y1+hy,y2,depth) +
                    gauss_quad_2D(func,x1+hx,x2,y1,y1+hy,depth) + gauss_quad_2D(func,x1+hx,x2,y1+hy,y2,depth);
            }
        }
        
        template<typename T, typename U>
        inline U gauss_quad_3D( std::function<U(T,T,T)>, T x1, T x2, T y1, T y2, T z1, T z2, int depth = 1) const noexcept;
        template<typename T, typename U>
        inline void gauss_quad_2D_generate_max_depth_vals( U (*func)(T,T), T x1, T x2, T y1, T y2, std::vector<U>& container, int depth=1) const noexcept;
        // False position method to find roots
        double falsePosMethod(std::function<double(double)>,double,double,const double tol=ROOT_FINDING_TOL) const noexcept(false);
    
    private:
        static const unsigned int _maxLevel;
        static const unsigned int _minLevel;

};

template<typename T, typename U>
double Integrals::integral_functional_3d(std::function<U(T,T)> f, T xmin, T xmax, T ymin, T ymax, U zmin, U zmax, bool(*y_cond)(T x,T y)) const{
    int count;
    int total(0), inBox(0);
    for (count=0; count < 1000000; count++){
        T u1 = (T)rand()/(T)RAND_MAX;
        T xcoord = ((xmax - xmin)*u1) + xmin;

        T u2 = (T)rand()/(T)RAND_MAX;
        T ycoord = ((ymax - ymin)*u2) + ymin;

        U u3 = (U)rand()/(U)RAND_MAX;
        U zcoord = ((zmax - zmin)*u3) + zmin;

        if (y_cond(xcoord,ycoord)){
            total++;
            auto val = f(xcoord,ycoord);
            // std::cout << "bool: " << ((val < zcoord)) << " val: " << val << " z: " << zcoord << "\n";
            if ( (val > zcoord) && (zcoord > (U)0.0) ){
                inBox++;
                // std::cout << "inBox ++: " << inBox << "\n";
            } else if( (val < zcoord) && (zcoord < (U)0.0) ){
                inBox--;
                // std::cout << "inBox --: " << inBox << "\n";
            }
        }
    }
    std::cout << "total: " << total << " inBox: " << inBox << std::endl;
    double density = inBox/(double)total;
    std::cout << "density: " << density << " and volume " << (xmax - xmin)*(ymax - ymin)*(zmax - zmin) << "\n";
    return density*(double)(xmax - xmin)*(ymax - ymin)*(zmax - zmin);
}

template<typename T> 
inline std::complex<double> Integrals::integral_functional_3d(std::function<std::complex<double>(T,T)> f, 
                    T xmin, T xmax, T ymin, T ymax, std::complex<double> zmin, std::complex<double> zmax, bool(*y_cond)(T x,T y)) const{
    int count;
    int total(0), inBoxR(0), inBoxI(0);
    // srand((unsigned) time(NULL));
    for (count=0; count < 1000000; count++){
        T u1 = (T)rand()/(T)RAND_MAX;
        T xcoord = ((xmax - xmin)*u1) + xmin;

        T u2 = (T)rand()/(T)RAND_MAX;
        T ycoord = ((ymax - ymin)*u2) + ymin;

        double u3R = (double)rand()/(double)RAND_MAX;
        double u3I = (double)rand()/(double)RAND_MAX;
        std::complex<double> zcoord( (zmax - zmin).real()*u3R + zmin.real(), (zmax - zmin).imag()*u3I + zmin.imag() );
        if (y_cond(xcoord,ycoord)){
            total++;
            std::complex<double> val = f(xcoord,ycoord);
            //std::cout << " zcoord: " << zcoord << " val: " << val << " xcoord: " << xcoord << " ycoord: " << ycoord << "\n";
            // std::cout << "bool: " << ((val < zcoord)) << " val: " << val << " z: " << zcoord << "\n";
            if ( (val.real() > zcoord.real()) && (zcoord.real() > 0.0) ){
                inBoxR++;
                //std::cout << "inBoxR ++: " << inBoxR << "\n";
            }
            else if ( (val.real() < zcoord.real()) && (zcoord.real() < 0.0) ){
                inBoxR--;
                //std::cout << "inBoxR --: " << inBoxR << "\n";
            }
            if ( (val.imag() > zcoord.imag()) && (zcoord.imag() > 0.0) ){
                inBoxI++;
                //std::cout << "inBoxI ++: " << inBoxI << "\n";
            }
            else if ( (val.imag() < zcoord.imag()) && (zcoord.imag() < 0.0) ){
                inBoxI--;
                //std::cout << "inBoxI --: " << inBoxI << "\n";
            }
        }
    }
    //std::cout << "total: " << total << " inBoxR: " << inBoxR << " inBoxI: " << inBoxI << std::endl;
    double densityR = inBoxR/(double)total;
    double densityI = inBoxI/(double)total;
    //std::cout << "densityR: " << densityR << " densityI: " << densityI << " and volume " << (xmax - xmin)*(ymax - ymin)*(zmax - zmin) << "\n";
    std::complex<double> result = std::complex<double>(densityR*(xmax - xmin)*(ymax - ymin)*(zmax - zmin).real(),densityI*(xmax - xmin)*(ymax - ymin)*(zmax - zmin).imag());
    std::cout << "result: " << result << "\n";
    return result;
}

template<typename T, typename U>
U Integrals::I2D_constrained(std::function<U(T,T)> funct,std::function<bool(T,T)> cond,T x0,T xf,T y0,T yf,const double tol,unsigned int maxVal,bool is_converged) const{
    T dx=(xf-x0)/(T)maxVal, dy=(yf-y0)/(T)maxVal;
    U resultSumNyorNx(0.0), resultSumNxNy(0.0), result, prevResult(0.0);
    unsigned int iter=1;
    while (!is_converged){
        resultSumNxNy=resultSumNyorNx=U(0.0);
        dx=(xf-x0)/(T)maxVal;
        dy=(yf-y0)/(T)maxVal;
        prevResult=result;
        for (unsigned int i=1; i<maxVal; i++){
            if (cond(x0+i*dx,y0)){
                resultSumNyorNx+=funct(x0+i*dx,y0);
            }
            if (cond(x0+i*dx,yf)){
                resultSumNyorNx+=funct(x0+i*dx,yf);
            }
            if (cond(x0,y0+i*dy)){
                resultSumNyorNx+=funct(x0,y0+i*dy);
            }
            if (cond(xf,y0+i*dy)){
                resultSumNyorNx+=funct(xf,y0+i*dy);
            }
        }
        for (unsigned int i=1; i<maxVal; i++){
            for (unsigned int j=1; j<maxVal; j++){
                if (cond(x0+i*dx,y0+j*dy)){
                    resultSumNxNy+=funct(x0+i*dx,y0+j*dy);
                }
            }
        }

        result=0.5*dx*dy*resultSumNyorNx+dx*dy*resultSumNxNy;
        if (cond(x0,y0)){
            result+=0.25*dx*dy*funct(x0,y0);
        }
        if (cond(xf,y0)){
            result+=0.25*dx*dy*funct(xf,y0);
        }
        if (cond(x0,yf)){
            result+=0.25*dx*dy*funct(x0,yf);
        }
        if (cond(xf,yf)){
            result+=0.25*dx*dy*funct(xf,yf);
        }

        is_converged = (std::abs(prevResult-result)<tol && iter>=2) ? true : false;
        
        if (iter>MAX_ITER_INTEGRAL || is_converged){
            break;
        }
        maxVal*=2;
        iter++;
    }

    return result;
}

template<typename T>
T Integrals::I1D_VEC(std::vector< T >& vecfunct,double delta,std::string rule) const noexcept(false){
    T result, resultSumNk{0.0};

    if (rule.compare("trapezoidal")==0){
        for (unsigned int i=1; i<vecfunct.size()-1; i++){
            resultSumNk+=vecfunct[i];
        }
        result = 0.5*delta*vecfunct[0]+0.5*delta*vecfunct.back()+delta*resultSumNk;
    } else if (rule.compare("simpson")==0){
        assert(vecfunct.size()%2==1); // simpson's rule work when the size of the array is odd...
        for (unsigned int i=1; i<vecfunct.size()-1; i+=2){
            resultSumNk += vecfunct[i-1] + 4.0*vecfunct[i] + vecfunct[i+1];
        }
        result = delta/3.0*resultSumNk;   
    } else{
        throw std::invalid_argument("Unhandeled rule chosen in integrals. Choices are \"trapezoidal\" and \"simpson\"");
    }
    
    return result;
}

template<typename T>
T Integrals::I1D_VEC(std::vector< T >&& vecfunct,double delta,std::string rule) const noexcept(false){
    T result, resultSumNk{0.0};

    if (rule.compare("trapezoidal")==0){
        for (unsigned int i=1; i<vecfunct.size()-1; i++){
            resultSumNk+=vecfunct[i];
        }
        result = 0.5*delta*vecfunct[0]+0.5*delta*vecfunct.back()+delta*resultSumNk;
    } else if (rule.compare("simpson")==0){
        assert(vecfunct.size()%2==1); // simpson's rule work when the size of the array is odd...
        for (unsigned int i=1; i<vecfunct.size()-1; i+=2){
            resultSumNk += vecfunct[i-1] + 4.0*vecfunct[i] + vecfunct[i+1];
        }
        result = delta/3.0*resultSumNk;   
    } else{
        throw std::invalid_argument("Unhandeled rule chosen in integrals. Choices are \"trapezoidal\" and \"simpson\"");
    }
    
    return result;
}

template<typename T, typename U>
inline U Integrals::gauss_quad_1D(std::function<U(T)> func, T x1, T x2, int depth) const noexcept{
    T w1 = static_cast<T>(1.0), w2 = static_cast<T>(1.0);
    T xi1 = (T)(-1.0/1.7320508075688772), xi2 = (T)(1.0/1.7320508075688772);
    T hx = (x2-x1)/static_cast<T>(2.0);
    T dx = (x2+x1)/static_cast<T>(2.0);

    U layer = hx*w1*func(hx*xi1+dx) + hx*w2*func(hx*xi2+dx);
    
    if (depth==MAX_DEPTH){
        return layer;
    } else{
        depth += 1;
        return gauss_quad_1D(func,x1,x1+hx,depth) + gauss_quad_1D(func,x1+hx,x2,depth);
    }
}

template<typename T, typename U>
inline U Integrals::gauss_quad_3D( std::function<U(T,T,T)> func, T x1, T x2, T y1, T y2, T z1, T z2, int depth) const noexcept{
    T w1 = static_cast<T>(1.0), w2 = static_cast<T>(1.0);
    T xi1 = -(T)(1.0/1.7320508075688772), xi2 = (T)(1.0/1.7320508075688772);
    T eta1 = -(T)(1.0/1.7320508075688772), eta2 = (T)(1.0/1.7320508075688772);
    T gamma1 = (T)(1.0/1.7320508075688772), gamma2 = -(T)(1.0/1.7320508075688772);
    T hx = (x2-x1)/static_cast<T>(2.0), hy = (y2-y1)/static_cast<T>(2.0), hz = (z2-z1)/static_cast<T>(2.0);
    T dx = (x2+x1)/static_cast<T>(2.0), dy = (y2+y1)/static_cast<T>(2.0), dz = (z2+z1)/static_cast<T>(2.0);

    U layer = hz*hx*hy*w1*w1*w1*func(hx*xi1+dx,hy*eta1+dy,hz*gamma1+dz) + hz*hx*hy*w1*w2*w1*func(hx*xi1+dx,hy*eta2+dy,hz*gamma1+dz) + 
        hz*hx*hy*w2*w1*w1*func(hx*xi2+dx,hy*eta1+dy,hz*gamma1+dz) + hz*hx*hy*w2*w2*w1*func(hx*xi2+dx,hy*eta2+dy,hz*gamma1+dz) + 
        hz*hx*hy*w1*w1*w2*func(hx*xi1+dx,hy*eta1+dy,hz*gamma2+dz) + hz*hx*hy*w1*w2*w2*func(hx*xi1+dx,hy*eta2+dy,hz*gamma2+dz) + 
        hz*hx*hy*w2*w1*w2*func(hx*xi2+dx,hy*eta1+dy,hz*gamma2+dz) + hz*hx*hy*w2*w2*w2*func(hx*xi2+dx,hy*eta2+dy,hz*gamma2+dz);
    
    if (depth==MAX_DEPTH){
        return layer;
    } else{
        depth += 1;
        return gauss_quad_3D(func,x1,x1+hx,y1,y1+hy,z1,z1+hz,depth) + gauss_quad_3D(func,x1,x1+hx,y1+hy,y2,z1,z1+hz,depth) +
            gauss_quad_3D(func,x1+hx,x2,y1,y1+hy,z1,z1+hz,depth) + gauss_quad_3D(func,x1+hx,x2,y1+hy,y2,z1,z1+hz,depth) + 
            gauss_quad_3D(func,x1,x1+hx,y1,y1+hy,z1+hz,z2,depth) + gauss_quad_3D(func,x1,x1+hx,y1+hy,y2,z1+hz,z2,depth) +
            gauss_quad_3D(func,x1+hx,x2,y1,y1+hy,z1+hz,z2,depth) + gauss_quad_3D(func,x1+hx,x2,y1+hy,y2,z1+hz,z2,depth);
    }
}


template<typename T, typename U>
inline void Integrals::gauss_quad_2D_generate_max_depth_vals( U (*func)(T,T), T x1, T x2, T y1, T y2, std::vector<U>& container, int depth) const noexcept{
    // sqrt(3) = 1.7320508075688772
    constexpr T xi1 = (T)(-1.0/1.7320508075688772), xi2 = (T)(1.0/1.7320508075688772);
    constexpr T eta1 = (T)(-1.0/1.7320508075688772), eta2 = (T)(1.0/1.7320508075688772);
    T hx = (x2-x1)/static_cast<T>(2.0), hy = (y2-y1)/static_cast<T>(2.0);
    T dx = (x2+x1)/static_cast<T>(2.0), dy = (y2+y1)/static_cast<T>(2.0);

    U layer1 = func(hx*xi1+dx,hy*eta1+dy);
    U layer2 = func(hx*xi1+dx,hy*eta2+dy);
    U layer3 = func(hx*xi2+dx,hy*eta1+dy);
    U layer4 = func(hx*xi2+dx,hy*eta2+dy);
    if (depth==MAX_DEPTH){
        container.push_back(layer1);
        container.push_back(layer2);
        container.push_back(layer3);
        container.push_back(layer4);
    }

    if (depth==MAX_DEPTH){
        return;
    } else{
        depth += 1;
        gauss_quad_2D_generate_max_depth_vals(func,x1,x1+hx,y1,y1+hy,container,depth); // lower left corner
        gauss_quad_2D_generate_max_depth_vals(func,x1,x1+hx,y1+hy,y2,container,depth); // upper left corner
        gauss_quad_2D_generate_max_depth_vals(func,x1+hx,x2,y1,y1+hy,container,depth); // lower right corner
        gauss_quad_2D_generate_max_depth_vals(func,x1+hx,x2,y1+hy,y2,container,depth); // upper right corner
    }
}

#endif /* Integral_utils_H_ */
