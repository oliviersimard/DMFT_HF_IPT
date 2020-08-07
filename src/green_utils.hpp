#ifndef Green_Utils_H_
#define Green_Utils_H_

#include <vector>
#include <complex>
#include <exception>
#include <iostream>
#include <assert.h> // For assert()
extern "C" {
#include<fftw3.h>
}
//#include <type_traits> // for use of std::is_same in static_assert
#include "wrapper_utils.hpp"

#define DIM 1 // 1, 2 or 3
#define SPINDEG 2 // Should be 2, unless bipartite lattice used
#define VERBOSE 0
#define MULT_N_TAU 2 // Has to be an even number
#define AFM

// Some declarations
namespace IPT2{ class DMFTproc; };
namespace IPT2{ template<class T> class SplineInline; };
namespace HF{ class FunctorBuildGk; };
namespace ThreadFunctor{ class ThreadWrapper; };
namespace ThreadFunctor{ enum solver_prototype : short; };
namespace IPT2{ enum spline_type : short; };
//class FFTtools;
template<class T> class Susceptibility;
namespace ThreadFunctor{ struct mpistruct; };
typedef ThreadFunctor::mpistruct mpistruct_t;

extern std::vector<double> vecK;
extern std::vector< std::complex<double> > iwnArr_l;
extern std::vector< std::complex<double> > iqnArr_l;
//extern unsigned int iter;
extern const arma::Mat< std::complex<double> > ZEROS_;
extern arma::Mat< std::complex<double> > statMat;

extern double epsilonk(double kx, double ky, double kz) noexcept;
extern double epsilonk(double,double) noexcept;
extern double epsilonk(double) noexcept;
double epsilonk2D(double kx, double ky);

namespace ThreadFunctor{ struct gamma_tensor_content; };
template<typename T> inline void calculateSusceptibilities(T&,const std::string&,const std::string&,const bool&,const bool&);
template<typename T> inline void calculateSusceptibilitiesParallel(T,std::string,std::string,bool,bool,double,ThreadFunctor::solver_prototype);
void DMFTloop(IPT2::DMFTproc&, std::ofstream&, std::ofstream&, std::ofstream&, std::vector< std::string >&, const unsigned int) noexcept(false);
void DMFTloopAFM(IPT2::DMFTproc& sublatt1, std::vector<std::ofstream*> vec_sub_1_ofstream, std::vector< std::string >& vecStr, const unsigned int N_it) noexcept(false);
std::ostream& operator<<(std::ostream&, const HF::FunctorBuildGk&);
struct Data{
    friend class FFTtools;
    friend class IPT2::DMFTproc;
    friend class ThreadFunctor::ThreadWrapper;
    template<class T>
    friend class Susceptibility;
    template<typename T>
    friend void calculateSusceptibilities(T&,const IPT2::SplineInline< std::complex<double> >&,const std::string&,const std::string&,const bool&,const bool&);
    template<typename T> friend void calculateSusceptibilitiesParallel(IPT2::SplineInline< std::complex<double> >,std::string,std::string,bool,bool,ThreadFunctor::solver_prototype);
    protected:
        static double beta;
        static double hyb_c;
        static double U;
        static double dtau;
        static unsigned int N_tau;
        static unsigned int N_k;
        static unsigned int objectCount; // they are treated similarly to global variables, and get initialized when the program starts
        //
        double mu, mu0{0.0};
    explicit Data(const unsigned int, const unsigned int, const double, const double, const double);
    Data()=default;
    Data(const Data& obj);
    Data& operator=(const Data& obj);
};

inline Data::Data(const unsigned int N_tau_, const unsigned int N_k_, const double beta_, const double U_, const double hyb_c_){
    if (objectCount<1){ // Maybe try to implement it as a singleton.
        N_k=N_k_;
        hyb_c=hyb_c_;
        N_tau=N_tau_;
        beta=beta_;
        U=U_;
        dtau=beta/N_tau;
    }
    objectCount++;
    #ifdef AFM
    this->mu=0.0; // starting at half-filling, where the Ising spin is 1/2, so the chemical potential is 0
    #else
    this->mu=U/2.0; // starting at half-filling
    #endif
}

inline Data::Data(const Data& obj){}

struct GreenStuff final : Data{ // Non-subclassable
    friend class IPT2::DMFTproc; // Important to access the private static variables from this struct.
    friend void ::DMFTloop(IPT2::DMFTproc& sublatt1, std::ofstream& objSaveStreamGloc, std::ofstream& objSaveStreamSE, std::ofstream& objSaveStreamGW, std::vector< std::string >& vecStr,const unsigned int N_it) noexcept(false);
    friend void ::DMFTloopAFM(IPT2::DMFTproc& sublatt1, std::vector<std::ofstream*> vec_sub_1_ofstream, std::vector< std::string >& vecStr, const unsigned int N_it) noexcept(false);
    // Member variables
    arma::Cube<double>& matsubara_t_pos; // Ctor: n_rows, n_cols, n_slices
    arma::Cube<double>& matsubara_t_neg;
    arma::Cube< std::complex<double> >& matsubara_w;
    std::vector< std::complex<double> > iwnArr;
    // Member functions
    GreenStuff();
    explicit GreenStuff(const unsigned int, const unsigned int, const double, const double,
                                        const double, std::vector< std::complex<double> >,
                                        arma::Cube<double>&, arma::Cube<double>&, arma::Cube< std::complex<double> >&) noexcept(false);
    GreenStuff(const GreenStuff& obj)=delete; // Copy constructor
    GreenStuff& operator=(const GreenStuff& obj)=delete; // Copy assignment
    arma::Cube< std::complex<double> > green_inf() const;
    static void reset_counter(){ objectCount=0; } // Important to update parameters in loop.
    double get_mu() { return this->mu; }
    double get_mu0() { return this->mu0; }
    static double get_beta() { return beta; }
    static double get_hyb_c() { return hyb_c; }
    static double get_U() { return U; }
    void update_mu(double mu){ this->mu=mu; }
    void update_mu0(double mu0){ this->mu0=mu0; }
    
    private:
    /* The static variables are useful whenever default constructor called. */
        static arma::Cube<double> dblVec_pos, dblVec_neg;
        static arma::Cube< std::complex<double> > cplxVec;
};

inline GreenStuff::GreenStuff() : Data(), matsubara_t_pos(dblVec_pos), matsubara_t_neg(dblVec_neg), matsubara_w(cplxVec){
    //this->iwnArr=std::vector< std::complex<double> >(1+1,0.0);
    std::cerr << "Constructor GreenStuff()\n";
}

namespace HF{

    class FunctorBuildGk{
        friend std::ostream& ::operator<<(std::ostream& os, const HF::FunctorBuildGk& obj);
        friend class ::ThreadFunctor::ThreadWrapper;
        template<class T>
        friend class ::Susceptibility;
        template<typename T>
        friend void ::calculateSusceptibilities(T&,const std::string&,const std::string&,const bool&,const bool&);
        template<typename T>
        friend void ::calculateSusceptibilitiesParallel(T,std::string,std::string,bool,bool,double,ThreadFunctor::solver_prototype);
        public:
            explicit FunctorBuildGk(double,double,double,double,std::vector<double>&,unsigned int,unsigned int,std::vector< std::complex<double> >&);
            FunctorBuildGk& operator=(const FunctorBuildGk& obj);
            FunctorBuildGk()=default;
            #if DIM == 1
            arma::Mat< std::complex<double> > operator()(int j, double kx) const{
                return buildGkAA_1D(j,kx);
            } // This inline function is called in other translation units!
            void update_ndo_1D();
            arma::Mat< std::complex<double> > operator()(std::complex<double> w, double kx) const{
                return buildGkAA_1D_w(w,kx);
            } // This inline functions are called in other translation units!
            #elif DIM == 2
            arma::Mat< std::complex<double> > operator()(int j, double kx, double ky) const{
                return buildGkAA_2D(j,kx,ky);
            };
            void update_ndo_2D();
            arma::Mat< std::complex<double> > operator()(std::complex<double> w, double kx, double ky) const{
                return buildGkAA_2D_w(w,kx,ky);
            };
            #endif
            std::complex<double> w(int,double) const;
            std::complex<double> q(int) const;
            arma::Mat< std::complex<double> >& swap(arma::Mat< std::complex<double> >&) const;
            #if DIM == 1
            arma::Mat< std::complex<double> > buildGkAA_1D(int,double) const;
            arma::Mat< std::complex<double> > buildGkAA_1D_w(std::complex<double>,double) const;
            #elif DIM == 2
            arma::Mat< std::complex<double> > buildGkAA_2D(int,double,double) const;
            arma::Mat< std::complex<double> > buildGkAA_2D_w(std::complex<double>,double,double) const;
            #endif
            
            double get_double_occupancy_AA() const;
            double get_ndo() const{
                return this->_ndo;
            } // Called in main.

        private:
            double _mu {0.0}, _u {0.0}, _ndo {0.0}, _beta {0.0};
            unsigned int _Nit {0}, _Nk {0};
            std::vector<double> _kArr_l={};
            std::complex<double>* _Gup_k=nullptr;
            size_t _size {0};
            std::vector< std::complex<double> > _precomp_wn={}, _precomp_qn={};
    };

    inline std::complex<double> FunctorBuildGk::w(int n, double mu) const{
        return std::complex<double>(0.0,1.0)*(2.0*(double)n+1.0)*M_PI/_beta + mu;
    }

    inline std::complex<double> FunctorBuildGk::q(int n) const{
        return std::complex<double>(0.0,1.0)*(2.0*(double)n)*M_PI/_beta;
    }

    struct K_1D{
        explicit K_1D(double qx, std::complex<double> iwn) : _qx(qx), _iwn(iwn){};
        K_1D()=default;
        ~K_1D()=default;
        K_1D operator+(const K_1D& rhs) const;
        K_1D operator-(const K_1D& rhs) const;

        double _qx {0.0};
        std::complex<double> _iwn={};
    };

    struct K_2D : K_1D{
        explicit K_2D(double qx, double qy, std::complex<double> iwn) : K_1D(qx,iwn){
            this->_qy = qy;
        }
        K_2D()=default;
        ~K_2D()=default;
        K_2D operator+(const K_2D& rhs) const;
        K_2D operator-(const K_2D& rhs) const;
    
        double _qx {0.0}, _qy {0.0};
        std::complex<double> _iwn={};
    };

} /* end of namespace HF */


#endif /* Green_Utils_H_ */
