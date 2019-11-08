#ifndef IPT2Single_H_
#define IPT2Single_H_

#include<functional>
extern "C" {
#include<fftw3.h>
}
#include<cmath> // double abs(double)
#include<stdexcept>
//#include<algorithm> // copy() and assign()
#include<armadillo>
#include<gsl/gsl_errno.h>
#include<gsl/gsl_fft_complex.h>
#include "tridiagonal.hpp"
#include "file_utils.hpp"

#define TOL 0.000000001

namespace IPT2{ class DMFTproc; };
struct GreenStuff;

// Prototypes
//bool file_exists(const std::string&);
//std::vector<std::string> glob(const std::string& pattern) noexcept(false);
void saveEachIt(const IPT2::DMFTproc&, std::ofstream&, std::ofstream&, std::ofstream&);
//void check_file_content(const std::vector< std::string >&, std::string) noexcept(false);
void get_tau_obj(GreenStuff&, GreenStuff&);
double Hsuperior(double tau, double mu, double c, double beta);
double Hpsuperior(double tau, double mu, double c, double beta);
double Hinferior(double tau, double mu, double c, double beta);
double Hpinferior(double tau, double mu, double c, double beta);


extern double epsilonk(double,double);
extern double epsilonk(double);


class FFTtools{

    public:
        enum Spec { plain_positive, plain_negative, dG_dtau_positive, dG_dtau_negative };

        void fft_t2w(arma::Cube<double>& data1, arma::Cube< std::complex<double> >& data2);
        void fft_w2t(arma::Cube< std::complex<double> >& data1, arma::Cube<double>& data2);
        void fft_spec(GreenStuff& data1, GreenStuff& data2, arma::Cube<double>&, arma::Cube<double>&, Spec);
        
};

namespace IPT2{

class DMFTproc{
    friend void ::saveEachIt(const IPT2::DMFTproc& sublatt1, std::ofstream& ofGloc, std::ofstream& ofSE, std::ofstream& ofGW);
    public:
        DMFTproc(GreenStuff&,GreenStuff&,GreenStuff&,GreenStuff&, arma::Cube<double>&, arma::Cube<double>&,
                                                        const std::vector<double>&, const double);
        DMFTproc(const DMFTproc&)=delete;
        DMFTproc& operator=(const DMFTproc&)=delete;
        void update_impurity_self_energy();
        void update_parametrized_self_energy(FFTtools);
        double density_mu(double, const arma::Cube< std::complex<double> >&) const; // used for false position method
        double density_mu(const arma::Cube< std::complex<double> >&) const;
        double density_mu0(double, const arma::Cube< std::complex<double> >&) const; // used for false position method
        double density_mu0(const arma::Cube< std::complex<double> >&) const;
        double double_occupancy() const;
    private:
        GreenStuff& WeissGreen;
        GreenStuff& Hyb;
        GreenStuff& LocalGreen;
        GreenStuff& SelfEnergy;
        arma::Cube<double>& data_dg_dtau_pos;
        arma::Cube<double>& data_dg_dtau_neg;
        const std::vector<double>& karr_l;
        //
        static double n, n0;
        static unsigned int objCount;
};

}

#endif /* end of IPT2Single_H_ */