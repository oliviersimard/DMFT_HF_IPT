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
#include "tridiagonal.hpp"
#include "file_utils.hpp"
#include "integral_utils.hpp"

#define TOL 0.000000001

namespace IPT2{ class DMFTproc; };
struct GreenStuff;

// Prototypes
void saveEachIt(const IPT2::DMFTproc&, std::ofstream&, std::ofstream&, std::ofstream&);
void DMFTloop(IPT2::DMFTproc&, std::ofstream&, std::ofstream&, std::ofstream&, std::vector< std::string >&, const unsigned int) noexcept(false);
void get_tau_obj(GreenStuff&, GreenStuff&);
double Hsuperior(double tau, double mu, double c, double beta);
double Hpsuperior(double tau, double mu, double c, double beta);
double Hinferior(double tau, double mu, double c, double beta);
double Hpinferior(double tau, double mu, double c, double beta);


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
    friend void ::DMFTloop(IPT2::DMFTproc& sublatt1, std::ofstream& objSaveStreamGloc, std::ofstream& objSaveStreamSE, std::ofstream& objSaveStreamGW, std::vector< std::string >& vecStr,const unsigned int N_it) noexcept(false);
    template<class T>
    friend class ::Susceptibility;
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

template<class T>
class SplineInline{
    public:
        void loadFileSpline(const std::string&) noexcept(false);
        T calculateSpline(double) const;
        SplineInline(const size_t,const std::vector<double>&);
    private:
        const size_t N_tau_size;
        std::vector<double> iwn;
        std::vector<double> iwn_re;
        std::vector<double> iwn_im;
        static spline<T> spl;
};

template<class T> spline<T> SplineInline<T>::spl;

}

template<typename T>
IPT2::SplineInline<T>::SplineInline(const size_t sizeArr,const std::vector<double>& initVec) : N_tau_size(sizeArr),
                                            iwn(initVec),iwn_re(initVec),iwn_im(initVec){}

template<>
inline void IPT2::SplineInline< std::complex<double> >::loadFileSpline(const std::string& filename) noexcept(false){
    std::ifstream infile;   
    std::string firstline("");
    std::string patternFile = filename+"*";
    std::vector< std::string > vecStr;
    try{
        vecStr=glob(patternFile);
    }catch (const std::exception& err){
        std::cerr << err.what() << "\n";
    }
    int largestNum=-1; // Numbers after N_it are always positive.
    for (auto str : vecStr){ // Getting the largest iteration number to eventually load for interpolation.
        int tempNum=extractIntegerLastWords(str);
        if (tempNum>largestNum)
            largestNum=tempNum;
    }
    unsigned int num = 0; // num must start at 0
    arma::Cube< std::complex<double> > inputFunct(2,2,2*N_tau_size);
    std::string finalFile=filename+"_Nit_"+std::to_string(largestNum)+".dat";
    infile.open(finalFile);// file containing numbers in 3 columns
    std::cout << "The file from which the spline is done: "+finalFile << "\n";
    if(infile.fail()){ // checks to see if file opended 
        std::cout << "error" << std::endl; 
        throw std::ios_base::failure("File not Found in SplineInline!!"); // no point continuing if the file didn't open...
    }
    while(!infile.eof()){ // reads file to end of *file*, not line
        if (num==0 && firstline==""){ 
        getline(infile,firstline);
        //std::cout << firstline << std::endl;
        if (firstline[0]!='/'){
            std::cerr << "Gotta remove first line of: "+finalFile << "\n";
            throw std::invalid_argument("The files loaded should have the marker \"/\" in front of the lines commented.");
        }
        }else{
	        infile >> iwn[num]; // Spoiled by the last two arrays
            infile >> iwn_re[num];
            infile >> iwn_im[num];

            ++num;
        }
    }
    infile.close();
    for (size_t i=0; i<2*N_tau_size; i++){
        inputFunct.slice(i)(0,0)=std::complex<double>(iwn_re[i],iwn_im[i]);
        //std::cout << inputFunct.slice(i)(0,0) << std::endl;
        //std::cout << iwn[i] << std::endl;
    }
    spl.set_points(iwn,inputFunct);
}

template<>
inline std::complex<double> IPT2::SplineInline< std::complex<double> >::calculateSpline(double reIwn) const{
    return spl(reIwn);
}

#endif /* end of IPT2Single_H_ */