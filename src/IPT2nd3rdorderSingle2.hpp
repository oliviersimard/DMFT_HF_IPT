#ifndef IPT2Single_H_
#define IPT2Single_H_

#include<functional>
#include<cmath> // double abs(double)
#include<stdexcept>
//#include<algorithm> // copy() and assign()
#include<armadillo>
//#include<limits> // numeric_limits
#include<iomanip> // set_precision
#include "tridiagonal.hpp"
#include "file_utils.hpp"
#include "integral_utils.hpp"

#define TOL 0.000000001

//namespace IPT2{ class DMFTproc; };
struct GreenStuff;

// Prototypes
void saveEachIt(const IPT2::DMFTproc&, std::ofstream&, std::ofstream&, std::ofstream&);
void DMFTloop(IPT2::DMFTproc&, std::ofstream&, std::ofstream&, std::ofstream&, std::vector< std::string >&, const unsigned int) noexcept(false);
double Hsuperior(double tau, double mu, double c, double beta);
double Hpsuperior(double tau, double mu, double c, double beta);
double Hinferior(double tau, double mu, double c, double beta);
double Hpinferior(double tau, double mu, double c, double beta);


class FFTtools{

    public:
        enum Spec { plain_positive, plain_negative, dG_dtau_positive, dG_dtau_negative };

        void fft_w2t(arma::Cube< std::complex<double> >& data1, arma::Cube<double>& data2);
        void fft_spec(GreenStuff& data1, GreenStuff& data2, arma::Cube<double>&, arma::Cube<double>&, Spec);
        
};

namespace IPT2{
enum spline_type : short { linear, cubic };

class DMFTproc{
    friend void ::saveEachIt(const IPT2::DMFTproc& sublatt1, std::ofstream& ofGloc, std::ofstream& ofSE, std::ofstream& ofGW);
    friend void ::DMFTloop(IPT2::DMFTproc& sublatt1, std::ofstream& objSaveStreamGloc, std::ofstream& objSaveStreamSE, std::ofstream& objSaveStreamGW, std::vector< std::string >& vecStr,const unsigned int N_it) noexcept(false);
    template<class T>
    friend class ::Susceptibility;
    public:
        explicit DMFTproc(GreenStuff&,GreenStuff&,GreenStuff&,GreenStuff&, arma::Cube<double>&, arma::Cube<double>&,
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
        double dbl_occupancy(unsigned int iter) const;
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
    friend class ::ThreadFunctor::ThreadWrapper;
    public:
        void loadFileSpline(const std::string&, const spline_type&) noexcept(false);
        T calculateSpline(double) const;
        std::vector< T > get_loaded_data_interpolated() const{
            return _iwn_cplx;
        }
        //SplineInline(const size_t,const std::vector<double>&);
        SplineInline(const size_t,std::vector<double>,std::vector<double>,std::vector< std::complex<double> >,std::vector< std::complex<double> >);
        SplineInline()=default;
        SplineInline<T>& operator=(const SplineInline<T>& obj);
    private:
        const size_t _N_tau_size {0};
        std::vector<double> _iwn={}, _iwn_re={}, _iwn_im={}, _k_array={};
        std::vector< T > _iwn_array={}, _iqn_array={}, _iwn_cplx={};
        spline_type _spline_choice=cubic;
        static spline<T> _spl;
        T compute_linear_spline(double reIwn, std::vector< T > y) const;
};

template<class T> spline<T> SplineInline<T>::_spl=spline<T>();

}

// template<typename T>
// IPT2::SplineInline<T>::SplineInline(const size_t sizeArr,const std::vector<double>& initVec) : N_tau_size(sizeArr),
//                                             iwn(initVec),iwn_re(initVec),iwn_im(initVec){}

template<typename T>
IPT2::SplineInline<T>::SplineInline(const size_t sizeArr,std::vector<double> initVec,std::vector<double> k_arr,std::vector< std::complex<double> > iwn_arr,std::vector< std::complex<double> > iqn_arr) : _N_tau_size(sizeArr),
                                                                _iwn(initVec),_iwn_re(initVec),_iwn_im(initVec),_k_array(k_arr),_iwn_array(iwn_arr),_iqn_array(iqn_arr){

                                                                }

template<typename T>
IPT2::SplineInline<T>& IPT2::SplineInline<T>::operator=(const IPT2::SplineInline<T>& obj){
    if (this!=&obj){
        *(const_cast<size_t*>(&this->_N_tau_size)) = obj._N_tau_size;
        this->_iwn=obj._iwn;
        this->_iwn_re=obj._iwn_re;
        this->_iwn_im=obj._iwn_im;
        this->_iwn_cplx=obj._iwn_cplx;
        this->_k_array=obj._k_array;
        this->_iwn_array=obj._iwn_array;
        this->_iqn_array=obj._iqn_array;
        this->_spline_choice=obj._spline_choice;
    }
    return *this;
}

template<typename T>
T IPT2::SplineInline<T>::compute_linear_spline(double reIwn, std::vector< T > y) const {
    assert(_iwn.size()==y.size());
    assert(_iwn.size()>2);
    size_t n = _iwn.size();
    std::vector<double>::const_iterator it;
    it=std::lower_bound(_iwn.begin(),_iwn.end(),reIwn);
    if (VERBOSE>0)
        std::cout << std::setprecision(10) << "it: " << *it << " and reIwn: " << reIwn << " and it+1: " << *(it+1) << std::endl; //" and _iwn[idx]: " << _iwn[idx] << std::endl;
    int idx=std::max( int( it - _iwn.begin() ) - 1, 0);
    // Have to correct for the boundaries. Loaded iwn data doesn't have the same precision as the values iwn produced in main, so the 
    // function lower_bound returns "it" has the iwn.size()/2 value: with the -1 in the definition of "idx", it means one jumps across the discontinuity..
    if (reIwn>0.0){ // Taking care of the boundary conditions (rounding errors especially)
        if ( idx == (static_cast<int>(n/2)-1) ){
            idx++;
        } else if ( idx == (static_cast<int>(n)-1) ){
            idx--;
        }
    } else if (reIwn<0.0){
        if ( idx == (static_cast<int>(n/2)-1) ){
            idx--;
        }
    }
    // std::cout << std::setprecision(10) << "idx: " << idx << " and iwn[idx]: " << _iwn[idx] << " reIwn: " << reIwn << " iwn[idx+1] " << _iwn[idx+1] << std::endl;
    double h = reIwn-_iwn[idx];
    T interpol;
    // interpol = y_i + (y_{i+1}-y_i) / (x_{i+1}-x_i) * (x - x_i). h and x_i would be purely imaginary, so no need to transform back to imaginary data, because it cancels out due to division.
    interpol = y[idx] + ( (y[idx+1]-y[idx]) / (_iwn[idx+1]-_iwn[idx]) ) * h;

    return interpol;
}

template<>
inline void IPT2::SplineInline< std::complex<double> >::loadFileSpline(const std::string& filename, const IPT2::spline_type& spl_t) noexcept(false){
    std::ifstream infile;   
    std::string firstline("");
    std::string patternFile = filename+"*";
    std::vector< std::string > vecStr;
    try{
        vecStr=glob(patternFile);
    }catch (const std::runtime_error& err){
        std::cerr << err.what() << "\n";
        exit(1);
    }
    
    int largestNum=1; // Numbers after N_it are always positive.
    for (auto str : vecStr){ // Getting the largest iteration number to eventually load for interpolation.
        int tempNum=extractIntegerLastWords(str);
        if (tempNum>largestNum)
            largestNum=tempNum;
    }
    unsigned int num = 0; // num must start at 0
    arma::Cube< std::complex<double> > inputFunct(2,2,2*_N_tau_size);
    std::cout << "Size of loaded self-energy array: " << 2*_N_tau_size << "\n";
    std::string finalFile=filename+"_Nit_"+std::to_string(largestNum)+".dat";
    infile.open(finalFile);// file containing numbers in 3 columns
    std::cout << "The file from which the spline is done: "+finalFile << "\n";
    if(infile.fail()){ // checks to see if file opended 
        std::cerr << "error" << "\n"; 
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
	        infile >> _iwn[num]; // Spoiled by the last two arrays
            infile >> _iwn_re[num];
            infile >> _iwn_im[num];

            ++num;
        }
    }
    infile.close();
    for (size_t i=0; i<2*_N_tau_size; i++){
        inputFunct.slice(i)(0,0)=std::complex<double>(_iwn_re[i],_iwn_im[i]);
        _iwn_cplx.push_back(std::complex<double>(_iwn_re[i],_iwn_im[i])); // For the linear spline..
    }
    // Only if cubic spline
    _spline_choice=spl_t;
    if (spl_t==IPT2::spline_type::cubic)
        _spl.set_points(_iwn,inputFunct);
}

template<>
inline std::complex<double> IPT2::SplineInline< std::complex<double> >::calculateSpline(double reIwn) const{
    switch (_spline_choice){
    case IPT2::spline_type::linear:
        return compute_linear_spline(reIwn,_iwn_cplx);
        break;
    default:
        return _spl(reIwn);
        break;
    }
}

#endif /* end of IPT2Single_H_ */