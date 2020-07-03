#ifndef IPT2Single_H_
#define IPT2Single_H_

#include<functional>
#include<cmath> // double abs(double)
#include<stdexcept>
#include<algorithm> // copy() and assign()
//#include<limits> // numeric_limits
#include<iomanip> // set_precision
#include<tuple>
#include "tridiagonal.hpp"
#include "file_utils.hpp"
#include "integral_utils.hpp"

#define TOL 0.000000001
#define ALPHA 0.05 // Parameter to mix the Hybridisation function between the different iterations (current and previous ones)

//namespace IPT2{ class DMFTproc; };
struct GreenStuff;
namespace IPT2{ template< class T > class OneLadder; template< class T > class InfiniteLadders; };

// Prototypes
void saveEachIt(const IPT2::DMFTproc&, std::ofstream&, std::ofstream&, std::ofstream&) noexcept;
void saveEachIt_AFM(const IPT2::DMFTproc& sublatt, std::ofstream& ofGloc, std::ofstream& ofSE, std::ofstream& ofGloc_tau) noexcept;
double Hsuperior(double tau, double mu, double c, double beta);
double Hpsuperior(double tau, double mu, double c, double beta);
double Hinferior(double tau, double mu, double c, double beta);
double Hpinferior(double tau, double mu, double c, double beta);
#ifdef AFM
cubic_roots get_cubic_roots_G_inf(double n_m_sigma_sublatt, double U, double mu, double mu0);
double get_Hyb_AFM(double n_m_sigma_sublatt, double U, double mu);
double Fsuperior(double tau, double mu, double mu0, double beta, double U, double n_m_sublatt);
double Finferior(double tau, double mu, double mu0, double beta, double U, double n_m_sublatt);
double Fpsuperior(double tau, double mu, double mu0, double beta, double U, double n_m_sublatt);
double Fpinferior(double tau, double mu, double mu0, double beta, double U, double n_m_sublatt);
#endif


class FFTtools{

    public:
        enum Spec { plain_positive, plain_negative, dG_dtau_positive, dG_dtau_negative };

        void fft_w2t(arma::Cube< std::complex<double> >& data1, arma::Cube<double>& data2, int index=0);
        void fft_w2t(const std::vector< std::complex<double> >& data1, std::vector<double>& data2, double beta, std::string dir="forward");
        void fft_spec(GreenStuff& data1, GreenStuff& data2, arma::Cube<double>&, arma::Cube<double>&, Spec);
        void fft_spec_AFM(GreenStuff& data1, GreenStuff& data2, arma::Cube<double>& data_dg_dtau_pos,arma::Cube<double>& data_dg_dtau_neg, Spec specialization, double n_a_up);
        
};

namespace IPT2{
enum spline_type : short { linear, cubic };

class DMFTproc{
    friend void ::saveEachIt(const IPT2::DMFTproc& sublatt1, std::ofstream& ofGloc, std::ofstream& ofSE, std::ofstream& ofGW) noexcept;
    friend void ::saveEachIt_AFM(const IPT2::DMFTproc& sublatt1, std::ofstream& ofGloc, std::ofstream& ofSE, std::ofstream& ofGloc_tau) noexcept;
    friend void ::DMFTloop(IPT2::DMFTproc& sublatt1, std::ofstream& objSaveStreamGloc, std::ofstream& objSaveStreamSE, std::ofstream& objSaveStreamGW, std::vector< std::string >& vecStr,const unsigned int N_it) noexcept(false);
    friend void ::DMFTloopAFM(IPT2::DMFTproc& sublatt1, std::vector<std::ofstream*> vec_sub_1_ofstream, std::vector< std::string >& vecStr, const unsigned int N_it) noexcept(false);
    template<class T>
    friend class ::Susceptibility;
    public:
        explicit DMFTproc(GreenStuff&,GreenStuff&,GreenStuff&,GreenStuff&, arma::Cube<double>&, arma::Cube<double>&,
                                                        const std::vector<double>&, const double);
        DMFTproc(const DMFTproc&)=delete;
        DMFTproc& operator=(const DMFTproc&)=delete;
        void update_impurity_self_energy();
        void update_impurity_self_energy_AFM(double h);
        void update_parametrized_self_energy(FFTtools);
        void update_parametrized_self_energy_AFM(FFTtools,double h);
        double density_mu(double, const arma::Cube< std::complex<double> >&) const; // used for false position method
        double density_mu(const arma::Cube< std::complex<double> >&) const;
        #ifdef AFM
        #ifndef SUS
        tuple<double,double> density_mu_AFM(const arma::Cube< std::complex<double> >& G) const;
        #endif
        #endif
        double density_mu0(double, const arma::Cube< std::complex<double> >&) const; // used for false position method
        double density_mu0(const arma::Cube< std::complex<double> >&) const;
        std::tuple<double,double> density_mu0_AFM(const arma::Cube< std::complex<double> >& G0_1) const;
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
    friend class OneLadder< T >;
    friend class InfiniteLadders< T >;
    public:
        void loadFileSpline(const std::string&, const spline_type&) noexcept(false);
        T calculateSpline(double) const;
        std::vector< T > get_loaded_data_interpolated() const{
            return _iwn_cplx;
        }
        //SplineInline(const size_t,const std::vector<double>&);
        SplineInline(const size_t,std::vector<double>,std::vector<double>,std::vector< T >);
        SplineInline()=default;
        SplineInline<T>& operator=(const SplineInline<T>& obj);
    private:
        const size_t _N_tau_size {0};
        std::vector<double> _iwn={}, _iwn_re={}, _iwn_im={}, _k_array={};
        std::vector< T > _iwn_array={}, _iwn_cplx={};
        spline_type _spline_choice=cubic;
        static spline< T > _spl;
        T compute_linear_spline(double reIwn) const;
};

template<class T> spline<T> SplineInline<T>::_spl=spline<T>();

}

// template<typename T>
// IPT2::SplineInline<T>::SplineInline(const size_t sizeArr,const std::vector<double>& initVec) : N_tau_size(sizeArr),
//                                             iwn(initVec),iwn_re(initVec),iwn_im(initVec){}

template<typename T>
IPT2::SplineInline< T >::SplineInline(const size_t sizeArr,std::vector<double> initVec,std::vector<double> k_arr,std::vector< T > iwn_arr) : _N_tau_size(sizeArr),
                                                                _iwn(initVec),_iwn_re(initVec),_iwn_im(initVec),_k_array(k_arr),_iwn_array(iwn_arr){

                                                                }

template<typename T>
IPT2::SplineInline< T >& IPT2::SplineInline< T >::operator=(const IPT2::SplineInline< T >& obj){
    if (this!=&obj){
        *(const_cast<size_t*>(&this->_N_tau_size)) = obj._N_tau_size;
        this->_iwn=obj._iwn;
        this->_iwn_re=obj._iwn_re;
        this->_iwn_im=obj._iwn_im;
        this->_iwn_cplx=obj._iwn_cplx;
        this->_k_array=obj._k_array;
        this->_iwn_array=obj._iwn_array;
        this->_spline_choice=obj._spline_choice;
    }
    return *this;
}

template<typename T>
T IPT2::SplineInline<T>::compute_linear_spline(double reIwn) const {
    #ifdef DEBUG
    assert(_iwn.size()==_iwn_cplx.size());
    assert(_iwn.size()>2);
    #endif
    size_t n = _iwn.size();
    std::vector<double>::const_iterator it;
    it=std::lower_bound(_iwn.begin(),_iwn.end(),reIwn);
    #ifdef DEBUG
    if (VERBOSE>0)
        std::cout << std::setprecision(10) << "it: " << *it << " and reIwn: " << reIwn << " and it+1: " << *(it+1) << std::endl; //" and _iwn[idx]: " << _iwn[idx] << std::endl;
    #endif
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
    // interpol = y_i + (y_{i+1}-y_i) / (x_{i+1}-x_i) * (x - x_i). h and x_i would be purely imaginary, so no need to transform back to imaginary data, because it cancels out due to division.
    return _iwn_cplx[idx] + ( (_iwn_cplx[idx+1]-_iwn_cplx[idx]) / (_iwn[idx+1]-_iwn[idx]) ) * h;
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
    arma::Cube< std::complex<double> > inputFunct(2,2,2*_N_tau_size);
    std::cout << "Size of loaded self-energy array: " << 2*_N_tau_size << "\n";
    std::string finalFile=filename+"_Nit_"+std::to_string(largestNum)+".dat";
    std::cout << "The file from which the spline is done: "+finalFile << "\n";

    FileData dataFromFile;
    dataFromFile = get_data(finalFile,2*_N_tau_size);
    _iwn = static_cast<std::vector<double>&&>(dataFromFile.iwn);
    _iwn_re = static_cast<std::vector<double>&&>(dataFromFile.re);
    _iwn_im = static_cast<std::vector<double>&&>(dataFromFile.im);
    
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
        return compute_linear_spline(reIwn);
        break;
    default:
        return _spl(reIwn);
        break;
    }
}

#endif /* end of IPT2Single_H_ */