#ifndef Integral_utils_H_
#define Integral_utils_H_

#include <armadillo>
#include <complex>
#include <functional>
#include <exception>

// #define INTEGRAL
#define MAX_ITER_INTEGRAL 20
#define MAX_ITER_ROOT 100000
#define ROOT_FINDING_TOL 0.0001

/* Template structure to call functions in classes. */
template<typename T, typename C, typename Q>
struct functorStruct{
    //using matCplx = arma::Mat< std::complex<double> >;
    using funct_init_t = arma::Mat< std::complex<double> > (C::*)(Q model, T kk, T qq, int n, int l);
    using funct_con_t = arma::Mat< std::complex<double> > (C::*)(Q model, T kk, T qq, int n, int l, arma::Mat< std::complex<double> > SE);

    functorStruct(funct_init_t initFunct, funct_con_t conFunct);
    arma::Mat< std::complex<double> > callInitFunct(C& obj, Q model, T kk, T qq, int n, int l);
    arma::Mat< std::complex<double> > callConFunct(C& obj, Q model, T kk, T qq, int n, int l, arma::Mat< std::complex<double> > SE);

    private:
        funct_init_t _initFunct;
        funct_con_t _conFunct;
};

template<typename T, typename C, typename Q>
functorStruct<T,C,Q>::functorStruct(funct_init_t initFunct, funct_con_t conFunct) : _initFunct(initFunct), _conFunct(conFunct){};

template<typename T, typename C, typename Q>
arma::Mat< std::complex<double> > functorStruct<T,C,Q>::callInitFunct(C& obj, Q model, T kk, T qq, int n, int l){
    return (obj.*_initFunct)(model, kk, qq, n, l);
}

template<typename T, typename C, typename Q>
arma::Mat< std::complex<double> > functorStruct<T,C,Q>::callConFunct(C& obj, Q model, T kk, T qq, int n, int l, arma::Mat< std::complex<double> > SE){
    return (obj.*_conFunct)(model, kk, qq, n, l, SE);
}

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
        double I2D(std::function<double(double,double)>,double,double,double,double,const double tol=0.0001,unsigned int maxVal=10,bool is_converged=false) const;
        std::complex<double> I2D(std::function<std::complex<double>(double,double,std::complex<double>)>,double,double,double,double,std::complex<double>,const double tol=0.00001,unsigned int maxVal=10,bool is_converged=false) const;
        std::complex<double> I1D(std::function<std::complex<double>(double,std::complex<double>)> funct,double k0,double kf,std::complex<double> iwn,double tol=0.00001,unsigned int maxDelta=10) const;
        double I1D(std::function<double(double)> funct,double k0,double kf,double tol=0.00001,unsigned int maxDelta=10) const;
        double I1D(std::vector<double>& vecfunct,double delta_tau) const;
        std::complex<double> I1D_CPLX(std::vector< std::complex<double> >& vecfunct,double delta) const;

        // False position method to find roots
        double falsePosMethod(std::function<double(double)>,double,double,const double tol=ROOT_FINDING_TOL) const noexcept(false);
    
    private:
        static const unsigned int _maxLevel;
        static const unsigned int _minLevel;

};

#endif /* Integral_utils_H_ */