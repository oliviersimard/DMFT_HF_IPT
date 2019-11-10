#ifndef Green_Utils_H_
#define Green_Utils_H_

#include <vector>
#include <complex>
#include <exception>
#include <armadillo>
#include <iostream>

#define DIM 1

namespace IPT2{ class DMFTproc; };
class FFTtools;

extern std::vector<double> vecK;
extern std::vector< std::complex<double> > iwnArr_l;
extern unsigned int iter;

struct Data{
    friend class FFTtools;
    friend class IPT2::DMFTproc;
    protected:
        static double beta;
        static double hyb_c;
        static double U;
        static double dtau;
        static double mu, mu0;
        static unsigned int N_tau;
        static unsigned int N_k;
        static unsigned int objectCount; // they are treated similarly to global variables, and get initialized when the program starts
        //
    explicit Data(const unsigned int, const unsigned int, const double, const double, const double);
    Data()=default;
    Data(const Data& obj);
    Data& operator=(const Data& obj);
};

struct GreenStuff final : Data{ // Non-subclassable
    friend class IPT2::DMFTproc; // Important to access the private static variables from this struct.
    friend void get_tau_obj(GreenStuff&,GreenStuff&); // Spits out tau-defined object.
    // Member variables
    arma::Cube<double>& matsubara_t_pos; // Ctor: n_rows, n_cols, n_slices
    arma::Cube<double>& matsubara_t_neg;
    arma::Cube< std::complex<double> >& matsubara_w;
    std::vector< std::complex<double> > iwnArr;
    // Member functions
    GreenStuff();
    explicit GreenStuff(const unsigned int N_tau, const unsigned int N_k, const double beta, const double U,
                                        const double hyb_c, std::vector< std::complex<double> >,
                                        arma::Cube<double>&, arma::Cube<double>&, arma::Cube< std::complex<double> >&) noexcept(false);
    GreenStuff(const GreenStuff& obj)=delete; // Copy constructor
    GreenStuff& operator=(const GreenStuff& obj)=delete; // Copy assignment
    arma::Cube< std::complex<double> > green_inf() const;
    void reset_counter(){
        this->objectCount=0; // Important to update parameters in loop.
    }
    double get_mu() const{
        return this->mu;
    }
    double get_mu0() const{
        return this->mu0;
    }
    double get_hyb_c() const{
        return this->hyb_c;
    }
    void update_mu(double mu){
        this->mu=mu;
    }
    void update_mu0(double mu0){
        this->mu0=mu0;
    }
    
    private:
    /* The static variables are useful whenever default constructor called. */
        static arma::Cube<double> dblVec_pos, dblVec_neg;
        static arma::Cube< std::complex<double> > cplxVec;
};


#endif /* Green_Utils_H_ */