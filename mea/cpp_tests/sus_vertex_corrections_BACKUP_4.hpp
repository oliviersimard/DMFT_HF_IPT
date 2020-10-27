#include "../../src/IPT2nd3rdorderSingle2.hpp"
#include <ctime>
// #include <mpi.h>
/*
Info HDF5:
https://support.hdfgroup.org/HDF5/doc/cpplus_RM/compound_8cpp-example.html
https://www.hdfgroup.org/downloads/hdf5/source-code/
*/

#define SEND_DATA_TAG 2000
#define SEND_NUM_TO_SLAVES 2001
#define SEND_NUM_START_TO_SLAVES 4001
#define RETURN_DATA_TAG 3000
#define RETURN_NUM_RECV_TO_ROOT 3001
#define RETURN_TAGS_TO_ROOT 3002
#define SHIFT_TO_DIFFERENTIATE_TAGS 1000000
#define RENORMALIZING_FACTOR 1.335  // U=2 --> 1.335, U=3 --> 1.400
#define RENORMALIZING_FACTOR_IL 1.1 // U=2 --> 1.1, U=3,1 --> 1.0

//static bool slaves_can_write_in_file = false; // This prevents that the slave processes
static int root_process = 0;
extern const int SE_multiple_matsubara_Ntau;

#define INFINITE

typedef struct{
    size_t k_tilde;
    size_t k_bar;
    std::complex<double> cplx_data_jj;
    std::complex<double> cplx_data_szsz;
} MPIData;

typedef struct{
    size_t n_iqn;
    size_t n_ikn;
    size_t n_k;
    std::complex<double> cplx_denom_corr;
} MPIDataCorr;

typedef struct{
    size_t n_iqpn;
    size_t n_qp;
    std::complex<double> cplx_val;
} MPIDataLadder;

template<class T>
struct MPIDataReceive{
    T* data_struct;
    size_t size;
};

inline std::tuple<int,int> inverse_Cantor_pairing(int number);

// Adding methods to IPT2 namespace for this particular translation unit
namespace IPT2{
    std::vector<double> get_iwn_to_tau(const std::vector< std::complex<double> >& F_iwn, double beta, std::string obj="Green") noexcept;
    template<typename T> std::vector< T > generate_random_numbers(size_t arr_size, T min, T max) noexcept;
    inline double velocity(double k) noexcept;
    void set_vector_processes(std::vector<MPIData>*,unsigned int N_q) noexcept;
    void create_mpi_data_struct(MPI_Datatype& custom_type);
    void create_mpi_data_struct_corr(MPI_Datatype& custom_type);
    void create_mpi_data_struct_cplx(MPI_Datatype& custom_type);

    template< class T >
    class OneLadder{
        /* Class to compute single ladder vertex corrections.
        */
        public:
            OneLadder& operator=(const OneLadder&) = delete;
            OneLadder(const OneLadder&) = delete;
            #if DIM == 1
            std::vector< MPIData > operator()(size_t n_k_bar, size_t n_k_tilde, bool is_single_ladder_precomputed=false, void* arma_ptr=nullptr, double qq=0.0) const noexcept(false);
            #elif DIM == 2
            std::vector< MPIData > operator()(size_t n_k_tilde_x, size_t n_k_tilde_y, bool is_single_ladder_precomputed=false, void* arma_ptr=nullptr, double qqx=0.0, double qqy=0.0) const noexcept(false);
            #elif DIM == 3
            std::vector< MPIData > operator()(size_t n_k_tilde_x, size_t n_k_tilde_y, bool is_single_ladder_precomputed, void* arma_ptr, double qqx=0.0, double qqy=0.0, double qqz=0.0) const noexcept(false);
            #endif
            OneLadder()=default;
            explicit OneLadder(const SplineInline< T >& splInlineobj, const std::vector< T >& SE, const std::vector< T >& iqn, const std::vector<double>& k_arr, const std::vector< T >& iqn_tilde, const std::vector< T >& iqn_big_array,
                        double mu, double U, double beta) : _splInlineobj(splInlineobj), _SE(SE), _iqn(iqn), _k_t_b(k_arr), _iqn_tilde(iqn_tilde), _iqn_big_array(iqn_big_array){
                this->_mu = mu;
                this->_U = U;
                this->_beta = beta;
            };
            static std::vector<int> tag_vec;
            //static std::vector< T** > mat_sl_vec; 

        protected:
            const SplineInline< T >& _splInlineobj;
            const std::vector< T >& _SE;
            const std::vector< T >& _iqn;
            const std::vector<double>& _k_t_b;
            const std::vector< T >& _iqn_tilde;
            const std::vector< T >& _iqn_big_array;
            double _mu {0.0}, _U {0.0}, _beta {0.0};
            #if DIM == 1
            inline T getGreen(double k, T iwn) const noexcept;
            T Gamma(double k_bar, double k_tilde, size_t n_ikn_bar, size_t n_ikn_tilde) const noexcept(false);
            #elif DIM == 2
            inline T getGreen(double kx, double ky, T iwn) const noexcept;
            T Gamma(double k_tilde_x, double k_tilde_y, double k_bar_x, double k_bar_y, size_t n_ikn_bar, size_t n_ikn_tilde, const Integrals& integralsObj) const noexcept(false);
            #elif DIM == 3
            inline T getGreen(double kx, double ky, double kz, T iwn) const noexcept;
            T Gamma(double k_tilde_x, double k_tilde_y, double k_tilde_z, double k_bar_x, double k_bar_y, double k_bar_z, double qqx, double qqy, double qqz, size_t n_ikn_bar, size_t n_ikn_tilde, size_t n_iqn, const Integrals& integralsObj) const noexcept(false);
            #endif

    };
    // contains tags to dicriminate matrices contained in mat_sl_vec (ordered arrangement)
    template< class T > std::vector<int> OneLadder< T >::tag_vec = {};
    //template< class T > std::vector< T** > OneLadder< T >::mat_sl_vec = {};

    template< class T >
    class InfiniteLadders : public OneLadder< T > {
        /*  Class to compute infinite ladder vertex correction diagram.
        */
        public:
            InfiniteLadders& operator=(const InfiniteLadders&) = delete;
            InfiniteLadders(const InfiniteLadders&) = delete;
            explicit InfiniteLadders(const SplineInline< T >& splInlineobj, const std::vector< T >& SE, const std::vector< T >& iqn, const std::vector<double>& k_arr, const std::vector< T >& iqn_tilde, const std::vector< T >& iqn_big_array, double mu, double U, double beta) : OneLadder< T >(splInlineobj,SE,iqn,k_arr,iqn_tilde,iqn_big_array,mu,U,beta){
                std::cout << "InfiniteLadder U: " << OneLadder< T >::_U << " and InfiniteLadder beta: " << OneLadder< T >::_beta << std::endl;
            }
            std::vector< MPIData > operator()(size_t n_k_bar, size_t n_k_tilde, double qq=0.0) const noexcept(false);
            T Gamma_merged_corr(size_t n_ikn_bar, double k_bar, size_t n_iqn, double qq) const noexcept(false);
            T ladder(size_t n_iqpn, double k_qp, int lower_bound, std::vector< T >& iqn_arr) const noexcept;
            T determine_lambda_even(size_t n_ikn_bar, double k_bar, size_t n_iqn, double qq) const noexcept;
            static std::string _FILE_NAME;
            // static arma::Cube< T > _denom_corr;
            static arma::Mat< T > _ladder;
            static arma::Mat< T > _ladder_larger;
            static arma::Cube< T > _lambda_even;

        private:
            using OneLadder< T >::getGreen;
            using OneLadder< T >::Gamma;
            T Gamma_correction_denominator(double k_bar, double kpp, double qq, size_t n_ikn_bar, size_t n_ikppn) const noexcept(false);
            T chi_corr(size_t n_ikn_tilde, size_t n_k_tilde, size_t n_ikn_bar, size_t n_k_bar, size_t n_iqpn, size_t n_qp, size_t n_iqn, double qq) const noexcept;
            T ladder_2(T iwn, double k, size_t n_iwn) const noexcept;
            T ladder_merged_with_chi_corr(size_t n_ikn_tilde, size_t n_k_tilde, size_t n_ikn_bar, size_t n_k_bar, size_t n_iqn, double qq) const noexcept;
            
    };

    template< class T > std::string InfiniteLadders< T >::_FILE_NAME = std::string("");
    template< class T > arma::Mat< T > InfiniteLadders< T >::_ladder = arma::Mat< T >();
    template< class T > arma::Mat< T > InfiniteLadders< T >::_ladder_larger = arma::Mat< T >();
    template< class T > arma::Cube< T > InfiniteLadders< T >::_lambda_even = arma::Cube<T>();
    // template< class T > arma::Cube< T > InfiniteLadders< T >::_denom_corr = arma::Cube< T >();

}

#if DIM == 1
template< class T >
inline T IPT2::OneLadder< T >::getGreen(double k, T iwn) const noexcept{
    /*  This method computes the dressed Green's function of the one-band model. 
        
        Parameters:
            k (double): object defined in fermionic Matsubara frequencies to translate to imaginary time.
            iwn (T): fermionic Matsubara frequency.
        
        Returns:
            (T): dressed Green's function.
    */
    return 1.0 / ( iwn + _mu - epsilonk(k) - _splInlineobj.calculateSpline( iwn.imag() ) );
}
#elif DIM == 2
template< class T >
inline T IPT2::OneLadder< T >::getGreen(double kx, double ky, T iwn) const noexcept{
    /*  This method computes the dressed Green's function of the one-band model. 
        
        Parameters:
            k (double): object defined in fermionic Matsubara frequencies to translate to imaginary time.
            iwn (T): fermionic Matsubara frequency.
        
        Returns:
            (T): dressed Green's function.
    */
    return 1.0 / ( iwn + _mu - epsilonk(kx,ky) - _splInlineobj.calculateSpline( iwn.imag() ) );
}
#endif

#if DIM == 1

template< class T >
T IPT2::OneLadder< T >::Gamma(double k_bar, double k_tilde, size_t n_ikn_bar, size_t n_ikn_tilde) const noexcept(false){
    /*  This method computes the current-vertex correction for the single ladder diagram. It uses the dressed Green's
    functions computed in the paramagnetic state. 
        
        Parameters:
            k_bar (double): right doublon momentum (Feynman diagram picture).
            k_tilde (double): left doublon momentum (Feynman diagram picture).
            ikn_bar (T): fermionic Matsubara frequency for the right doublon (Feynman diagram picture).
            ikn_tilde (T): fermionic Matsubara frequency for the left doublon (Feynman diagram picture).
        
        Returns:
            lower_val (T): the current-vertex function for the given doublon parameter set. Eventually, the elements are gathered
            in a matrix (Fermionic Matsubara frequencies) before being squeezed in between the four outer Green's functions; this is done
            in operator() public member function.
    */
    T lower_val{0.0};
    const Integrals intObj;
    const size_t Ntau = _splInlineobj._iwn_array.size(); // corresponds to half the size of the SE array
    std::function<T(double)> int_k_1D;
    T ikn_tilde = _splInlineobj._iwn_array[n_ikn_tilde], ikn_bar = _splInlineobj._iwn_array[n_ikn_bar], iqn_tilde;
    for (size_t j=0; j<_iqn_tilde.size(); j++){
        iqn_tilde = _iqn_tilde[j];
        int_k_1D = [&](double k){
            return ( 1.0 / ( ikn_tilde-iqn_tilde + _mu - epsilonk(k_tilde-k) - _SE[(SE_multiple_matsubara_Ntau/2*Ntau-1)+n_ikn_tilde-j] ) 
            )*( 1.0 / ( ikn_bar-iqn_tilde + _mu - epsilonk(k_bar-k) - _SE[(SE_multiple_matsubara_Ntau/2*Ntau-1)+n_ikn_bar-j] ) 
            );
        };
        lower_val += 1.0/(2.0*M_PI)*intObj.gauss_quad_1D(int_k_1D,0.0,2.0*M_PI);
    }
    
    lower_val *= _U/_beta/RENORMALIZING_FACTOR;
    lower_val += 1.0;
    lower_val = _U/lower_val/RENORMALIZING_FACTOR;

    return lower_val;
}

template< class T >
std::vector< MPIData > IPT2::OneLadder< T >::operator()(size_t n_k_bar, size_t n_k_tilde, bool is_single_ladder_precomputed, void* arma_ptr, double qq) const noexcept(false){
    /*  This method computes the susceptibility given the current-vertex correction for the single ladder diagram. It does so for a set
    of momenta (k_bar,ktilde). It uses the dressed Green's functions computed in the paramagnetic state. 
        
        Parameters:
            n_k_bar (size_t): right doublon momentum index (Feynman diagram picture).
            n_k_tilde (size_t): left doublon momentum index (Feynman diagram picture).
            qq (double): momentum injected in the system. Defaults to 0, because interested in the thermodynamic limit (infinite volume).
        
        Returns:
            GG_iqn (std::vector< MPIData >): vector whose elements are MPIData structures containing footprint of n_k_bar, n_k_tilde and 
            the susceptibility for each bosonic Matsubara frequency. Therefore, the length of the vector is same as that of the incoming
            bosonic Matsubara frequencies.
    */
    std::vector< MPIData > GG_iqn;
    // Computing Gamma
    const size_t NI = _splInlineobj._iwn_array.size();
    T jj_resp_iqn{0.0}, szsz_resp_iqn{0.0};
    arma::Mat< T > Gamma_n_tilde_n_bar(NI,NI);
    for (size_t n_tilde=0; n_tilde<NI; n_tilde++){
        for (size_t n_bar=0; n_bar<NI; n_bar++){
            Gamma_n_tilde_n_bar(n_tilde,n_bar) = Gamma(_k_t_b[n_k_bar],_k_t_b[n_k_tilde],n_bar,n_tilde);
        }
    }
    if (is_single_ladder_precomputed){
        // Come up with UNIQUE way to attribute the tags to different Gamma matrices using Cantor pairing function
        int tag = (int)( ((n_k_bar+n_k_tilde)*(n_k_bar+n_k_tilde+1))/2 ) + (int)n_k_tilde;
        std::cout << "tag: " << tag << std::endl;
        tag_vec.push_back(tag);
        *( static_cast< arma::Mat< T >* >(arma_ptr) ) = Gamma_n_tilde_n_bar;
    }

    arma::Mat< T > GG_n_tilde_n_bar(NI,NI);
    const size_t starting_point_SE = ((double)SE_multiple_matsubara_Ntau/2.0-1.0/2.0)*(int)NI;
    for (size_t n_em=0; n_em<_iqn.size(); n_em++){
        for (size_t n_bar=0; n_bar<NI; n_bar++){
            for (size_t n_tilde=0; n_tilde<NI; n_tilde++){
                GG_n_tilde_n_bar(n_tilde,n_bar) = ( 1.0/( _splInlineobj._iwn_array[n_tilde] + _mu - epsilonk(_k_t_b[n_k_tilde]) - _SE[starting_point_SE+n_tilde] )
                )*( 1.0/( _splInlineobj._iwn_array[n_tilde]-_iqn[n_em] + _mu - epsilonk(_k_t_b[n_k_tilde]-qq) - _SE[starting_point_SE+n_tilde-n_em] )
                )*Gamma_n_tilde_n_bar.at(n_tilde,n_bar
                )*( 1.0/( _splInlineobj._iwn_array[n_bar] + _mu - epsilonk(_k_t_b[n_k_bar]) - _SE[starting_point_SE+n_bar] )
                )*( 1.0/( _splInlineobj._iwn_array[n_bar]-_iqn[n_em] + _mu - epsilonk(_k_t_b[n_k_bar]-qq) - _SE[starting_point_SE+n_bar-n_em] )
                );
            }
        }
        
        // summing over the internal ikn_tilde and ikn_bar
        jj_resp_iqn = -1.0*velocity(_k_t_b[n_k_tilde])*velocity(_k_t_b[n_k_bar])*(2.0/_beta/_beta)*arma::accu(GG_n_tilde_n_bar);
        szsz_resp_iqn = (2.0/_beta/_beta)*arma::accu(GG_n_tilde_n_bar);
        // ADDED A FACTOR OF 2 FOR THE SPIN BELOW
        MPIData mpi_data_tmp { n_k_tilde, n_k_bar, jj_resp_iqn, szsz_resp_iqn }; // summing over the internal ikn_tilde and ikn_bar
        GG_iqn.push_back(static_cast<MPIData&&>(mpi_data_tmp));
    }
    
    return GG_iqn;
}

template< class T >
T IPT2::InfiniteLadders< T >::ladder(size_t n_iqpn, double qp, int lower_bound, std::vector< T >& iqn_arr) const noexcept{
    /*  This method computes a single ladder diagram (box in the notes). It uses the dressed Green's
    functions computed in the paramagnetic state. 
        
        Parameters:
            n_iqpn (size_t): bosonic Matsubara frequency (Feynman diagram picture).
            qp (double): momentum (Feynman diagram picture).
        
        Returns:
            lower_val (T): single-ladder value.
    */
    T lower_val{0.0};
    const Integrals intObj;
    const size_t NI = OneLadder< T >::_splInlineobj._iwn_array.size(); // corresponds to half the size of the SE array
    const size_t starting_point_SE = ((double)SE_multiple_matsubara_Ntau/2.0-1.0/2.0)*(int)NI;
    const size_t q_starting_point_SE = ((double)SE_multiple_matsubara_Ntau/2.0-1.0/2.0)*(int)NI + lower_bound;
    std::function<T(double)> int_k_1D;
    T iqpn = iqn_arr[n_iqpn], ikpn;
    for (size_t n_ikpn=0; n_ikpn<NI; n_ikpn++){
        ikpn = OneLadder< T >::_splInlineobj._iwn_array[n_ikpn];
        int_k_1D = [&](double k){
            return ( 1.0 / ( ikpn + OneLadder< T >::_mu - epsilonk(k) - OneLadder< T >::_SE[starting_point_SE+n_ikpn] ) 
            )*( 1.0 / ( ikpn-iqpn + OneLadder< T >::_mu - epsilonk(k-qp) - OneLadder< T >::_SE[q_starting_point_SE+n_ikpn-n_iqpn] ) 
            );
        };
        lower_val += 1.0/(2.0*M_PI)*intObj.gauss_quad_1D(int_k_1D,0.0,2.0*M_PI);
    }
    
    lower_val *= OneLadder< T >::_U/OneLadder< T >::_beta;
    lower_val += 1.0;
    lower_val = OneLadder< T >::_U/lower_val;

    return lower_val;
};

template< class T >
T IPT2::InfiniteLadders< T >::ladder_2(T iwn, double k, size_t n_iwn) const noexcept{
    /*  This method computes a single ladder diagram (box in the notes). It uses the dressed Green's
    functions computed in the paramagnetic state. 
        
        Parameters:
            n_iqpn (size_t): bosonic Matsubara frequency (Feynman diagram picture).
            qp (double): momentum (Feynman diagram picture).
        
        Returns:
            lower_val (T): single-ladder value.
    */
    T lower_val{0.0};
    const Integrals intObj;
    const size_t NI = OneLadder< T >::_splInlineobj._iwn_array.size(); // corresponds to half the size of the SE array
    const size_t starting_point_SE = ((double)SE_multiple_matsubara_Ntau/2.0+1.0)*(int)NI - 2;
    // const size_t other_starting_point_SE = ((double)SE_multiple_matsubara_Ntau/2.0-1.0)*(int)NI + 1;
    std::function<T(double)> int_k_1D;
    T ikpn;
    for (size_t n_ikpn=0; n_ikpn<NI; n_ikpn++){
        ikpn = OneLadder< T >::_splInlineobj._iwn_array[n_ikpn];
        int_k_1D = [&](double kk){
            return ( 1.0 / ( ikpn + OneLadder< T >::_mu - epsilonk(kk) - OneLadder< T >::_SE[starting_point_SE+n_ikpn] ) 
            )*( 1.0 / ( ikpn+iwn + OneLadder< T >::_mu - epsilonk(kk-k) - OneLadder< T >::_SE[starting_point_SE+n_ikpn+n_iwn] ) 
            );
        };
        lower_val += 1.0/(2.0*M_PI)*intObj.gauss_quad_1D(int_k_1D,0.0,2.0*M_PI);
    }
    
    lower_val *= OneLadder< T >::_U/OneLadder< T >::_beta;
    lower_val += 1.0;
    lower_val = OneLadder< T >::_U/lower_val;

    return lower_val;
};

template< class T > 
T IPT2::InfiniteLadders< T >::determine_lambda_even(size_t n_ikn_bar, double k_bar, size_t n_iqn, double qq) const noexcept{
    /*  This method computes the Lambda function making up the even correction to the infinite ladder-of-ladders. 
        
        Parameters:
            k_bar (double): right doublon momentum (Feynman diagram picture).
            k_tilde (double): left doublon momentum (Feynman diagram picture).
        
        Returns:
            lower_val (T): the current-vertex function for the given doublon parameter set. Eventually, the elements are gathered
            in a matrix (Fermionic Matsubara frequencies) before being squeezed in between the four outer Green's functions; this is done
            in operator() public member function.
    */
    T val{0.0};
    const Integrals intObj;
    const double delta = 2.0*M_PI/(double)(OneLadder< T >::_k_t_b.size()-1);
    const size_t NI = OneLadder< T >::_splInlineobj._iwn_array.size(); // corresponds to half the size of the SE array
    std::vector< std::complex<double> > int_k_1D(OneLadder< T >::_k_t_b.size());
    T ikn_bar = OneLadder< T >::_splInlineobj._iwn_array[n_ikn_bar], iqn_bar, iqqn = OneLadder< T >::_iqn[n_iqn];
    const int ladder_shift = ((int)OneLadder< T >::_iqn_big_array.size()/2-(int)NI/2)+1;
    double q_bar;
    for (size_t n_iqn_bar=0; n_iqn_bar<OneLadder< T >::_iqn_tilde.size(); n_iqn_bar++){
        iqn_bar = OneLadder< T >::_iqn_tilde[n_iqn_bar];
        for (size_t n_q_bar=0; n_q_bar<OneLadder< T >::_k_t_b.size(); n_q_bar++){
            q_bar = OneLadder< T >::_k_t_b[n_q_bar];
            int_k_1D[n_q_bar] = ( 1.0 / ( ikn_bar-iqn_bar + OneLadder< T >::_mu - epsilonk(k_bar-q_bar) - OneLadder< T >::_SE[(SE_multiple_matsubara_Ntau/2*NI-1)+n_ikn_bar-n_iqn_bar] ) 
            )*( 1.0 / ( ikn_bar-iqn_bar-iqqn + OneLadder< T >::_mu - epsilonk(k_bar-q_bar-qq) - OneLadder< T >::_SE[(SE_multiple_matsubara_Ntau/2*NI-1)+n_ikn_bar-n_iqn_bar-n_iqn] ) 
            )*_ladder_larger(n_iqn_bar+ladder_shift,n_q_bar);
        }
        val += 1.0/(2.0*M_PI)*intObj.I1D_VEC(int_k_1D,delta,"simpson"); // Does it necessitate the normalization????
    }
    
    return val/OneLadder< T >::_beta;
};

template< class T >
T IPT2::InfiniteLadders< T >::chi_corr(size_t n_ikn_tilde, size_t n_k_tilde, size_t n_ikn_bar, size_t n_k_bar, size_t n_iqpn, size_t n_qp, size_t n_iqn, double qq) const noexcept{
    T val{0.0};
    const Integrals intObj;
    const int size_k_arr = static_cast<int>(OneLadder< T >::_k_t_b.size());
    auto k_resizing = [size_k_arr](int n_k_val) -> int {if (n_k_val>=0) return n_k_val%(size_k_arr-1); else return (size_k_arr-1)+n_k_val%(size_k_arr-1);};
    double delta = 2.0*M_PI/(double)(size_k_arr-1);
    const size_t NI = OneLadder< T >::_splInlineobj._iwn_array.size();
    std::vector< T > int_k_1D(size_k_arr);
    T ikn_tilde = OneLadder< T >::_splInlineobj._iwn_array[n_ikn_tilde], iqppn; // ikn_bar = OneLadder< T >::_splInlineobj._iwn_array[n_ikn_bar];
    T iqpn = OneLadder< T >::_iqn_tilde[n_iqpn], iqqn = OneLadder< T >::_iqn[n_iqn];
    double qpp, qp=OneLadder< T >::_k_t_b[n_qp], k_tilde=OneLadder< T >::_k_t_b[n_k_tilde];//, k_bar=OneLadder< T >::_k_t_b[n_k_bar];
    const int ladder_shift_simple = ((int)OneLadder< T >::_iqn_big_array.size()/2-(int)NI/2)+1;
    const int ladder_shift_complex = ((int)OneLadder< T >::_iqn_big_array.size()/2-(int)NI)+2;
    for (size_t n_iqppn=0; n_iqppn<OneLadder< T >::_iqn_tilde.size(); n_iqppn++){
        iqppn = OneLadder< T >::_iqn_tilde[n_iqppn];
        for (size_t n_qpp=0; n_qpp<OneLadder< T >::_k_t_b.size(); n_qpp++){
            //std::cout << (int)ladder_shift_complex+(int)n_ikn_tilde+(int)n_iqn+(int)n_iqpn+(int)n_iqppn+(int)n_ikn_bar << " " << k_resizing((int)n_k_tilde-(int)n_qp-(int)n_qpp-(int)n_k_bar) << std::endl;
            qpp = OneLadder< T >::_k_t_b[n_qpp];
            int_k_1D[n_qpp] = ( 1.0 / ( ikn_tilde+iqqn-iqpn + OneLadder< T >::_mu - epsilonk(k_tilde+qq-qp) - OneLadder< T >::_SE[(SE_multiple_matsubara_Ntau/2*NI-1)+n_ikn_tilde+n_iqn-n_iqpn] ) 
            )*( 1.0 / ( ikn_tilde-iqpn + OneLadder< T >::_mu - epsilonk(k_tilde-qp) - OneLadder< T >::_SE[(SE_multiple_matsubara_Ntau/2*NI-1)+n_ikn_tilde-n_iqpn] ) 
            )*_ladder_larger(n_iqppn+ladder_shift_simple,n_qpp)*( 1.0 / ( ikn_tilde+iqqn-iqpn-iqppn + OneLadder< T >::_mu - epsilonk(k_tilde+qq-qp-qpp) - OneLadder< T >::_SE[(SE_multiple_matsubara_Ntau/2*NI+NI/2-2)+n_ikn_tilde+n_iqn-n_iqpn-n_iqppn] ) 
            )*( 1.0 / ( ikn_tilde-iqpn-iqppn + OneLadder< T >::_mu - epsilonk(k_tilde-qp-qpp) - OneLadder< T >::_SE[(SE_multiple_matsubara_Ntau/2*NI+NI/2-2)+n_ikn_tilde-n_iqpn-n_iqppn] ) 
            )*_ladder_larger(ladder_shift_complex-n_ikn_tilde-n_iqn+n_iqpn+n_iqppn+n_ikn_bar,k_resizing((int)n_k_tilde-(int)n_qp-(int)n_qpp-(int)n_k_bar));
        }
        val += 1.0/(2.0*M_PI)*intObj.I1D_VEC(int_k_1D,delta,"simpson");
    }
    val *= -1.0/OneLadder< T >::_beta;
    val += 1.0;

    return 1.0/val;
}

template<class T>
T IPT2::InfiniteLadders< T >::ladder_merged_with_chi_corr(size_t n_ikn_tilde, size_t n_k_tilde, size_t n_ikn_bar, size_t n_k_bar, size_t n_iqn, double qq) const noexcept{
    T val{0.0};
    Integrals intObj;
    std::vector< T > int_k_1D(OneLadder< T >::_k_t_b.size());
    double delta = 2.0*M_PI/(double)(OneLadder< T >::_k_t_b.size()-1);
    const size_t NI = OneLadder< T >::_splInlineobj._iwn_array.size();
    const int ladder_shift = ((int)OneLadder< T >::_iqn_big_array.size()/2-(int)NI/2)+1;
    
    for (size_t n_iqpn=0; n_iqpn<OneLadder< T >::_iqn_tilde.size(); n_iqpn++){
        for (size_t n_qp=0; n_qp<OneLadder< T >::_k_t_b.size(); n_qp++){
            int_k_1D[n_qp] = _ladder_larger(n_iqpn+ladder_shift,n_qp)*chi_corr(n_ikn_tilde,n_k_tilde,n_ikn_bar,n_k_bar,n_iqpn,n_qp,n_iqn,qq);
        }
        val += 1.0/(2.0*M_PI)*intObj.I1D_VEC(int_k_1D,delta,"simpson");
    }
    
    val *= 1.0/OneLadder< T >::_beta;

    return val;
};

template< class T >
T IPT2::InfiniteLadders< T >::Gamma_correction_denominator(double k_bar, double kpp, double qq, size_t n_ikn_bar, size_t n_ikppn) const noexcept(false){
    /*  This method computes the (q,iqn)-independent part of the correction to the current-vertex for the single ladder diagram. 
    It corresponds to the denominator of the correction term. This correction to the single ladder diagrams depends solely on the 
    following incoming tuple (k_bar,ikn_bar). It uses the dressed Green's functions computed in the paramagnetic state. 
        
        Parameters:
            k_bar (double): left doublon momentum (Feynman diagram picture).
            k_pp (double): momentum local to the full correction term accountable for the infinite ladder summation (Feynman diagram picture).
            ikn_bar (T): left doublon fermionic Matsubara frequency (Feynman diagram picture).
            ikppn (T): fermionic Matsubara frequency local to the full correction term (pairing up with k_pp).
        
        Returns:
            (T): denominator of the correction term accountable to the infinite ladder summation for the current-vertex corrections.
            This value is independent of the tuple (q,iqn).
    */
    T denom_val{0.0};
    const Integrals intObj;
    // const double delta = 2.0*M_PI/(double)(OneLadder< T >::_splInlineobj._k_array.size()-1);
    const size_t NI = OneLadder< T >::_splInlineobj._iwn_array.size();
    const size_t starting_point_SE = ((double)SE_multiple_matsubara_Ntau/2.0-1.0/2.0)*(int)NI;
    std::function<T(double)> int_k_1D;
    T ikpppn, ikn_bar = OneLadder< T >::_splInlineobj._iwn_array[n_ikn_bar], ikppn = OneLadder< T >::_splInlineobj._iwn_array[n_ikppn];
    for (size_t n_ppp=0; n_ppp<OneLadder<T>::_splInlineobj._iwn_array.size(); n_ppp++){
        ikpppn = OneLadder< T >::_splInlineobj._iwn_array[n_ppp];
        int_k_1D = [&](double k_ppp){
            return ( 1.0 / ( ikpppn+ikppn-ikn_bar + OneLadder< T >::_mu - epsilonk(k_ppp+kpp-k_bar) - OneLadder< T >::_SE[starting_point_SE+n_ppp+n_ikppn-n_ikn_bar] ) 
                )*( 1.0 / ( ikpppn + OneLadder< T >::_mu - epsilonk(k_ppp) - OneLadder< T >::_SE[starting_point_SE+n_ppp] ) 
                );
        };
        denom_val += intObj.gauss_quad_1D(int_k_1D,0.0,2.0*M_PI);
    }
    denom_val *= OneLadder< T >::_U/OneLadder< T >::_beta/(2.0*M_PI)/RENORMALIZING_FACTOR_IL;
    denom_val += 1.0;

    return 1.0/denom_val;
}

template< class T >
T IPT2::InfiniteLadders< T >::Gamma_merged_corr(size_t n_ikn_bar, double k_bar, size_t n_iqn, double qq) const noexcept(false){
    /*  This method finishes off the computation of the correction term to the single ladder by looping over the local
    momentum k_pp and the local fermionic Matsubara frequency ikn_bar and including the numerator, which depends on (iqn,qq).
        
        Parameters:
            denom_corr (arma::Mat< T >&): matrix containing the correction term to the single ladder as a function of the local 4-vector
            (k_pp,ikppn).
            iqn (T): injected bosonic Matsubara frequency.
            qq (double): momentum injected in the system. Defaults to 0, because interested in the thermodynamic limit (infinite volume).
        
        Returns:
            tot_corr (T): total contribution comming from the infinite summation of the ladder diagrams as a function of iqn.
    */
    // This function method is an implicit function of k_bar and ikn_bar
    T tot_corr{0.0};
    const Integrals intObj;
    // const double delta = 2.0*M_PI/(double)(OneLadder< T >::_splInlineobj._k_array.size()-1);
    const size_t NI = OneLadder< T >::_splInlineobj._iwn_array.size();
    const size_t starting_point_SE = ((double)SE_multiple_matsubara_Ntau/2.0-1.0/2.0)*(int)NI;
    std::function<T(double)> int_k_1D;
    T iqn = OneLadder< T >::_iqn[n_iqn], ikppn;
    for (size_t n_pp=0; n_pp<NI; n_pp++){
        ikppn = OneLadder< T >::_splInlineobj._iwn_array[n_pp];
        int_k_1D = [&](double k_pp){
            return ( 1.0/( ikppn + OneLadder< T >::_mu - epsilonk(k_pp) - OneLadder< T >::_SE[starting_point_SE+n_pp] )
                )*Gamma_correction_denominator(k_bar,k_pp,qq,n_ikn_bar,n_pp
                )*( 1.0/( ikppn-iqn + OneLadder< T >::_mu - epsilonk(k_pp-qq) - OneLadder< T >::_SE[starting_point_SE+n_pp-n_iqn] )
                );
        };
        tot_corr += intObj.gauss_quad_1D(int_k_1D,0.0,2.0*M_PI);
    }
    tot_corr *= OneLadder< T >::_U/OneLadder< T >::_beta/(2.0*M_PI)/RENORMALIZING_FACTOR_IL;
    
    return tot_corr; 
}

template< class T >
std::vector< MPIData > IPT2::InfiniteLadders< T >::operator()(size_t n_k_bar, size_t n_k_tilde, double qq) const noexcept(false){
    /*  This method computes the susceptibility given the current-vertex correction for the infinite ladder diagram. It does so for a set
    of momenta (k_bar,ktilde). It uses the dressed Green's functions computed in the paramagnetic state. 
        
        Parameters:
            n_k_bar (size_t): right doublon momentum index (Feynman diagram picture).
            n_k_tilde (size_t): left doublon momentum index (Feynman diagram picture).
            qq (double): momentum injected in the system. Defaults to 0, because interested in the thermodynamic limit (infinite volume).
            is_single_ladder_precomputed (bool): boolean used to choose whether the single ladder part is loaded from previous matching 
            calculations (true), or if it is computed from the beginning (false).
        
        Returns:
            GG_iqn (std::vector< MPIData >): vector whose elements are MPIData structures containing footprint of n_k_bar, n_k_tilde and 
            the susceptibility for each bosonic Matsubara frequency. Therefore, the length of the vector is same as that of the incoming
            bosonic Matsubara frequencies.
    */
    std::vector< MPIData > GG_iqn;
    // Computing Gamma
    // Here should have the choice the load the precomputed single ladder denominator or not to save time...
    const size_t NI = OneLadder< T >::_splInlineobj._iwn_array.size();
    const size_t starting_point_SE = ((double)SE_multiple_matsubara_Ntau/2.0-1.0/2.0)*(int)NI;

    // Single ladder and its corrections
    arma::Mat< T > GG_n_tilde_n_bar_even(NI,NI), GG_n_tilde_n_bar_odd(NI,NI);
    T jj_resp_iqn{0.0}, szsz_resp_iqn{0.0};
    clock_t begin, end;
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    for (size_t n_em=0; n_em<OneLadder< T >::_iqn.size(); n_em++){
        begin = clock();
        //std::cout << "world_rank " << world_rank << " issued " << n_em << std::endl;
        for (size_t n_bar=0; n_bar<NI; n_bar++){
            for (size_t n_tilde=0; n_tilde<NI; n_tilde++){
                GG_n_tilde_n_bar_odd(n_tilde,n_bar) = ( 1.0/( OneLadder< T >::_splInlineobj._iwn_array[n_tilde] + OneLadder< T >::_mu - epsilonk(OneLadder< T >::_k_t_b[n_k_tilde]) - OneLadder< T >::_SE[starting_point_SE+n_tilde] ) 
                    )*( 1.0/( OneLadder< T >::_splInlineobj._iwn_array[n_tilde]+OneLadder< T >::_iqn[n_em] + OneLadder< T >::_mu - epsilonk(OneLadder< T >::_k_t_b[n_k_tilde]+qq) - OneLadder< T >::_SE[starting_point_SE+n_tilde+n_em] ) 
                    )*ladder_merged_with_chi_corr(n_tilde,n_k_tilde,n_bar,n_k_bar,n_em,qq
                    )*( 1.0/( OneLadder< T >::_splInlineobj._iwn_array[n_bar] + OneLadder< T >::_mu - epsilonk(OneLadder< T >::_k_t_b[n_k_bar]) - OneLadder< T >::_SE[starting_point_SE+n_bar] ) 
                    )*( 1.0/( OneLadder< T >::_splInlineobj._iwn_array[n_bar]-OneLadder< T >::_iqn[n_em] + OneLadder< T >::_mu - epsilonk(OneLadder< T >::_k_t_b[n_k_bar]-qq) - OneLadder< T >::_SE[starting_point_SE+n_bar-n_em] ) );
                
                GG_n_tilde_n_bar_even(n_tilde,n_bar) = GG_n_tilde_n_bar_odd(n_tilde,n_bar)*_lambda_even(n_em,n_bar,n_k_bar);
            }
        }
        
        // summing over the internal ikn_tilde and ikn_bar
        jj_resp_iqn = -2.0*velocity(OneLadder< T >::_k_t_b[n_k_tilde])*velocity(OneLadder< T >::_k_t_b[n_k_bar])*(1.0/OneLadder< T >::_beta/OneLadder< T >::_beta)*(arma::accu(GG_n_tilde_n_bar_even)+arma::accu(GG_n_tilde_n_bar_odd));
        szsz_resp_iqn = (2.0/OneLadder< T >::_beta/OneLadder< T >::_beta)*(arma::accu(GG_n_tilde_n_bar_odd)-arma::accu(GG_n_tilde_n_bar_even));
        MPIData mpi_data_tmp { n_k_tilde, n_k_bar, jj_resp_iqn, szsz_resp_iqn };
        GG_iqn.push_back(static_cast<MPIData&&>(mpi_data_tmp));
       
        end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "n_em: " << n_em << " done in " << elapsed_secs << " secs.." << "\n";
    }

    return GG_iqn;
}

#elif DIM == 2

template< class T >
T IPT2::OneLadder< T >::Gamma(double k_tilde_x, double k_tilde_y, double k_bar_x, double k_bar_y, size_t n_ikn_bar, size_t n_ikn_tilde, const Integrals& integralsObj) const noexcept(false){
    /*  This method computes the current-vertex correction for the single ladder diagram. It uses the dressed Green's
    functions computed in the paramagnetic state. 
    
        Parameters:
            k_tilde_m_k_bar_x (double): x component of difference in doublon momentum (Feynman diagram picture).
            k_tilde_m_k_bar_y (double): y component of difference in doublon momentum (Feynman diagram picture).
            ikn_bar (T): fermionic Matsubara frequency for the right doublon (Feynman diagram picture).
            ikn_tilde (T): fermionic Matsubara frequency for the left doublon (Feynman diagram picture).
        
        Returns:
            lower_val (T): the current-vertex function for the given doublon parameter set. Eventually, the elements are gathered
            in a matrix (Fermionic Matsubara frequencies) before being squeezed in between the four outer Green's functions; this is done
            in operator() public member function.
    */
    T lower_val{0.0};
    std::function<T(double,double)> inner_2D_int;
    const size_t Ntau = _splInlineobj._iwn_array.size();
    T ikn_tilde = _splInlineobj._iwn_array[n_ikn_tilde], ikn_bar = _splInlineobj._iwn_array[n_ikn_bar];
    for (size_t j=0; j<_iqn_tilde.size(); j++){
        inner_2D_int = [&](double kx, double ky){
            return ( 1.0/(  ikn_tilde-_iqn_tilde[j] + _mu - epsilonk(k_tilde_x-kx,k_tilde_y-ky) - _SE[(2*Ntau-1)+n_ikn_tilde-j] ) 
                )*( 1.0/( ikn_bar-_iqn_tilde[j] + _mu - epsilonk(k_bar_x-kx,k_bar_y-ky) - _SE[(2*Ntau-1)+n_ikn_bar-j] ) 
                );
        };
        lower_val += integralsObj.gauss_quad_2D<double,T>(inner_2D_int,0.0,2.0*M_PI,0.0,2.0*M_PI);
    }
    // Summing over the bosonic Matsubara frequencies
    lower_val *= _U/_beta/(2.0*M_PI)/(2.0*M_PI);
    lower_val += 1.0;
    lower_val = _U/lower_val;

    return lower_val;
}

template< class T >
std::vector< MPIData > IPT2::OneLadder< T >::operator()(size_t n_k_tilde_x, size_t n_k_tilde_y, bool is_single_ladder_precomputed, void* arma_ptr, double qqx, double qqy) const noexcept(false){
    /*  This method computes the susceptibility given the current-vertex correction for the single ladder diagram. It does so for a set
    of momenta (k_bar,ktilde). It uses the dressed Green's functions computed in the paramagnetic state. 
        
        Parameters:
            n_k_bar (size_t): right doublon momentum index (Feynman diagram picture).
            n_k_tilde (size_t): left doublon momentum index (Feynman diagram picture).
            qq (double): momentum injected in the system. Defaults to 0, because interested in the thermodynamic limit (infinite volume).
        
        Returns:
            GG_iqn (std::vector< MPIData >): vector whose elements are MPIData structures containing footprint of n_k_bar, n_k_tilde and 
            the susceptibility for each bosonic Matsubara frequency. Therefore, the length of the vector is same as that of the incoming
            bosonic Matsubara frequencies.
    */
    std::vector< MPIData > GG_iqn;
    // Computing Gamma
    const size_t NI = _splInlineobj._iwn_array.size();
    const double delta = 2.0*M_PI/(double)(_k_t_b.size()-1);
    const Integrals integralsObj;
    arma::Cube< T >** Gamma_n_tilde_n_bar = new arma::Cube< T >* [_k_t_b.size()];
    for (size_t i=0; i<_k_t_b.size(); i++){
        Gamma_n_tilde_n_bar[i] = new arma::Cube< T >(NI,NI,_k_t_b.size()); // n_tilde, n_bar, kbar_x
    }
    T jj_resp_iqn{0}, szsz_resp_iqn{0}, jj_tmp{0}, szsz_tmp{0};
    clock_t begin, end;
    for (size_t n_k_bar_y=0; n_k_bar_y<_k_t_b.size(); n_k_bar_y++){
        for (size_t n_k_bar_x=0; n_k_bar_x<_k_t_b.size(); n_k_bar_x++){
            begin = clock();
            std::cout << "(" << n_k_bar_x << "," << n_k_bar_y << ")" << std::endl;
            for (size_t n_tilde=0; n_tilde<NI; n_tilde++){
                for (size_t n_bar=0; n_bar<NI; n_bar++){
                    Gamma_n_tilde_n_bar[n_k_bar_y]->at(n_tilde,n_bar,n_k_bar_x) = Gamma(_k_t_b[n_k_tilde_x],_k_t_b[n_k_tilde_y],_k_t_b[n_k_bar_x],_k_t_b[n_k_bar_y],n_bar,n_tilde,integralsObj);
                }
            }
            end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "(" << n_k_bar_x << "," << n_k_bar_y << ")" << " done in " << elapsed_secs << " secs.." << "\n";
        }
    }
    // if (is_single_ladder_precomputed){
    //     // Using Cantor pairing function
    //     int tag = (int)( ((n_k_tilde_x+n_k_tilde_y)*(n_k_tilde_x+n_k_tilde_y+1))/2 ) + (int)n_k_tilde_y;
    //     std::cout << "tag: " << tag << std::endl;
    //     tag_vec.push_back(tag);
    //     *( static_cast< arma::Mat< T >* >(arma_ptr) ) = Gamma_n_tilde_n_bar;
    // }
    std::vector< T > k_integral_tmp_x_jj(_k_t_b.size()), k_integral_tmp_x_szsz(_k_t_b.size());
    std::vector< T > k_integral_tmp_y_jj(_k_t_b.size()), k_integral_tmp_y_szsz(_k_t_b.size());
    for (size_t n_em=0; n_em<_iqn.size(); n_em++){
        begin = clock();
        std::cout << "n_em: " << n_em << std::endl;
        // numerator
        for (size_t n_bar=0; n_bar<NI; n_bar++){
            for (size_t n_tilde=0; n_tilde<NI; n_tilde++){
                for (size_t n_k_bar_y=0; n_k_bar_y<_k_t_b.size(); n_k_bar_y++){
                    auto k_bar_y = _k_t_b[n_k_bar_y];
                    for (size_t n_k_bar_x=0; n_k_bar_x<_k_t_b.size(); n_k_bar_x++){
                        auto k_bar_x = _k_t_b[n_k_bar_x];
                        // jj (nearest-neighbour only)
                        jj_tmp = -1.0*velocity(_k_t_b[n_k_tilde_x])*velocity(k_bar_x)*
                            ( 1.0/( _splInlineobj._iwn_array[n_tilde] + _mu - epsilonk(_k_t_b[n_k_tilde_x],_k_t_b[n_k_tilde_y]) - _SE[3*NI/2+n_tilde] ) 
                            )*( 1.0/( _splInlineobj._iwn_array[n_tilde]-_iqn[n_em] + _mu - epsilonk(_k_t_b[n_k_tilde_x]-qqx,_k_t_b[n_k_tilde_y]-qqy) - _SE[3*NI/2+n_tilde-n_em] )
                            )*Gamma_n_tilde_n_bar[n_k_bar_y]->at(n_tilde,n_bar,n_k_bar_x
                            )*( 1.0/( _splInlineobj._iwn_array[n_bar] + _mu - epsilonk(k_bar_x,k_bar_y) - _SE[3*NI/2+n_bar] )
                            )*( 1.0/( _splInlineobj._iwn_array[n_bar]-_iqn[n_em] + _mu - epsilonk(k_bar_x-qqx,k_bar_y-qqy) - _SE[3*NI/2+n_bar-n_em] ) 
                            );
                        k_integral_tmp_x_jj[n_k_bar_x] = jj_tmp;
                        // jj (nearest-neighbour only)
                        szsz_tmp = ( 1.0/( _splInlineobj._iwn_array[n_tilde] + _mu - epsilonk(_k_t_b[n_k_tilde_x],_k_t_b[n_k_tilde_y]) - _SE[3*NI/2+n_tilde] ) 
                            )*( 1.0/( _splInlineobj._iwn_array[n_tilde]-_iqn[n_em] + _mu - epsilonk(_k_t_b[n_k_tilde_x]-qqx,_k_t_b[n_k_tilde_y]-qqy) - _SE[3*NI/2+n_tilde-n_em] )
                            )*Gamma_n_tilde_n_bar[n_k_bar_y]->at(n_tilde,n_bar,n_k_bar_x
                            )*( 1.0/( _splInlineobj._iwn_array[n_bar] + _mu - epsilonk(k_bar_x,k_bar_y) - _SE[3*NI/2+n_bar] )
                            )*( 1.0/( _splInlineobj._iwn_array[n_bar]-_iqn[n_em] + _mu - epsilonk(k_bar_x-qqx,k_bar_y-qqy) - _SE[3*NI/2+n_bar-n_em] ) 
                            );
                        k_integral_tmp_x_szsz[n_k_bar_x] = szsz_tmp;
                    }
                    k_integral_tmp_y_jj[n_k_bar_y] = integralsObj.I1D_VEC(k_integral_tmp_x_jj,delta);
                    k_integral_tmp_y_szsz[n_k_bar_y] = integralsObj.I1D_VEC(k_integral_tmp_x_szsz,delta);
                }
                jj_resp_iqn += integralsObj.I1D_VEC(k_integral_tmp_y_jj,delta);
                szsz_resp_iqn += integralsObj.I1D_VEC(k_integral_tmp_y_szsz,delta); // k_tilde_x and k_tilde_y are left for the python script
            }
        }
        end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "loop n_em: " << n_em << " done in " << elapsed_secs << " secs.." << "\n";
        // one 2 is for spin and the other for iqn, since only positive iqn is considered
        jj_resp_iqn *= (2.0/_beta/_beta/(2.0*M_PI)/(2.0*M_PI)); //*arma::accu(GG_n_tilde_n_bar_jj);
        szsz_resp_iqn *= (2.0/_beta/_beta/(2.0*M_PI)/(2.0*M_PI)); //*arma::accu(GG_n_tilde_n_bar_szsz);
        MPIData mpi_data_tmp { n_k_tilde_x, n_k_tilde_y, jj_resp_iqn, szsz_resp_iqn }; // summing over the internal ikn_tilde and ikn_bar
        GG_iqn.push_back(static_cast<MPIData&&>(mpi_data_tmp));
    }
    std::cout << "After the loop.." << std::endl;
    // cleaning up
    for (size_t i=0; i<_k_t_b.size(); i++){
        delete Gamma_n_tilde_n_bar[i];
    }
    delete [] Gamma_n_tilde_n_bar;
    return GG_iqn;
}

#elif DIM == 3

template< class T >
T IPT2::OneLadder< T >::Gamma(double k_tilde_x, double k_tilde_y, double k_tilde_z, double k_bar_x, double k_bar_y, double k_bar_z, double qqx, double qqy, double qqz, size_t n_ikn_bar, size_t n_ikn_tilde, size_t n_iqn, const Integrals& integralsObj) const noexcept(false){
    /*  This method computes the current-vertex correction for the single ladder diagram. It uses the dressed Green's
    functions computed in the paramagnetic state. 
    
        Parameters:
            k_tilde_m_k_bar_x (double): x component of difference in doublon momentum (Feynman diagram picture).
            k_tilde_m_k_bar_y (double): y component of difference in doublon momentum (Feynman diagram picture).
            ikn_bar (T): fermionic Matsubara frequency for the right doublon (Feynman diagram picture).
            ikn_tilde (T): fermionic Matsubara frequency for the left doublon (Feynman diagram picture).
        
        Returns:
            lower_val (T): the current-vertex function for the given doublon parameter set. Eventually, the elements are gathered
            in a matrix (Fermionic Matsubara frequencies) before being squeezed in between the four outer Green's functions; this is done
            in operator() public member function.
    */
    T lower_val{0.0};
    std::function<T(double,double,double)> inner_3D_int;
    #ifndef DEBUG
    const size_t Ntau = 2*_splInlineobj._iwn_array.size();
    #endif
    T ikn_tilde = _splInlineobj._iwn_array[n_ikn_tilde], ikn_bar = _splInlineobj._iwn_array[n_ikn_bar];
    #ifdef DEBUG
    T iqn = _iqn[n_iqn];
    #endif
    for (size_t j=0; j<_iqn_tilde.size(); j++){
        #ifndef DEBUG
        inner_3D_int = [&](double kx, double ky, double kz){
            return ( 1.0/(  ikn_tilde-_iqn_tilde[j] + _mu - epsilonk(k_tilde_x-kx,k_tilde_y-ky,k_tilde_z-kz) - _SE[(Ntau-1)+n_ikn_tilde-j] ) 
                )*( 1.0/( ikn_bar-_iqn_tilde[j]+_iqn[n_iqn] + _mu - epsilonk(k_bar_x-kx+qqx,k_bar_y-ky+qqy,k_bar_z-kz+qqz) - _SE[(Ntau-1)+n_ikn_bar-j+n_iqn] ) 
                );
        };
        // to sum over Matsubara frequencies
        lower_val += integralsObj.gauss_quad_3D(inner_3D_int,0.0,2.0*M_PI,0.0,2.0*M_PI,0.0,2.0*M_PI);
        #else
        inner_3D_int = [&](double kx, double ky, double kz){
            return getGreen(k_tilde_x-kx,k_tilde_y-ky,k_tilde_z-kz,ikn_tilde-_iqn_tilde[j])*getGreen(k_bar_x-kx+qqx,k_bar_y-ky+qqy,k_bar_z-kz+qqz,ikn_bar-_iqn_tilde[j]+iqn);
        };
        // to sum over Matsubara frequencies
        lower_val += integralsObj.gauss_quad_3D(inner_3D_int,0.0,2.0*M_PI,0.0,2.0*M_PI,0.0,2.0*M_PI);
        #endif
    }
    // Summing over the bosonic Matsubara frequencies
    lower_val *= _U/_beta/(2.0*M_PI)/(2.0*M_PI)/(2.0*M_PI);
    lower_val += 1.0;
    lower_val = _U/lower_val;

    return lower_val;
}

template< class T >
std::vector< MPIData > IPT2::OneLadder< T >::operator()(size_t n_k_tilde_x, size_t n_k_tilde_y, bool is_single_ladder_precomputed, void* arma_ptr, double qqx, double qqy, double qqz) const noexcept(false){
    /*  This method computes the susceptibility given the current-vertex correction for the single ladder diagram. It does so for a set
    of momenta (k_bar,ktilde). It uses the dressed Green's functions computed in the paramagnetic state. 
        
        Parameters:
            n_k_bar (size_t): right doublon momentum index (Feynman diagram picture).
            n_k_tilde (size_t): left doublon momentum index (Feynman diagram picture).
            qq (double): momentum injected in the system. Defaults to 0, because interested in the thermodynamic limit (infinite volume).
        
        Returns:
            GG_iqn (std::vector< MPIData >): vector whose elements are MPIData structures containing footprint of n_k_bar, n_k_tilde and 
            the susceptibility for each bosonic Matsubara frequency. Therefore, the length of the vector is same as that of the incoming
            bosonic Matsubara frequencies.
    */
    std::vector< MPIData > GG_iqn;
    // Computing Gamma
    const size_t NI = _splInlineobj._iwn_array.size();
    const Integrals integralsObj;
    arma::Cube< T > Gamma_n_tilde_n_bar(NI,NI,_iqn.size()); // Doesn't depend on iq_n
    // arma::Mat< T > GG_n_tilde_n_bar_jj(NI,NI), GG_n_tilde_n_bar_szsz(NI,NI);
    T jj_resp_iqn{0}, szsz_resp_iqn{0}, jj_tmp{0}, szsz_tmp{0};

    std::function<T(double,double,double)> inner_3D_int_jj, inner_3D_int_szsz;
    std::function<T(double)> inner_1D_int_jj, inner_1D_int_szsz;
    clock_t begin, end;
    for (size_t n_em=0; n_em<_iqn.size(); n_em++){
        // numerator
        for (size_t n_bar=0; n_bar<NI; n_bar++){
            begin = clock();
            std::cout << "n_bar: " << n_bar << std::endl;
            for (size_t n_tilde=0; n_tilde<NI; n_tilde++){
                std::cout << "n_tilde" << n_tilde << std::endl;
                #ifndef DEBUG
                inner_3D_int_jj = [&](double k_bar_x, double k_bar_y, double k_bar_z){
                    inner_1D_int_jj = [&](double k_tilde_z){
                        // denominator
                        Gamma_n_tilde_n_bar.at(n_tilde,n_bar,n_em) = Gamma(_k_t_b[n_k_tilde_x],_k_t_b[n_k_tilde_y],k_tilde_z,k_bar_x,k_bar_y,k_bar_z,qqx,qqy,qqz,n_bar,n_tilde,n_em,integralsObj);
                        // jj (nearest-neighbour only)
                        jj_tmp = -1.0*velocity(_k_t_b[n_k_tilde_x])*velocity(k_bar_x)*
                            ( 1.0/( _splInlineobj._iwn_array[n_tilde] + _mu - epsilonk(_k_t_b[n_k_tilde_x],_k_t_b[n_k_tilde_y],k_tilde_z) - _SE[3*NI/2+n_tilde] ) 
                            )*( 1.0/( _splInlineobj._iwn_array[n_tilde]-_iqn[n_em] + _mu - epsilonk(_k_t_b[n_k_tilde_x]-qqx,_k_t_b[n_k_tilde_y]-qqy,k_tilde_z-qqz) - _SE[3*NI/2+n_tilde-n_em] )
                            )*Gamma_n_tilde_n_bar.at(n_tilde,n_bar,n_em
                            )*( 1.0/( _splInlineobj._iwn_array[n_bar] + _mu - epsilonk(k_bar_x,k_bar_y,k_bar_z) - _SE[3*NI/2+n_bar] )
                            )*( 1.0/( _splInlineobj._iwn_array[n_bar]+_iqn[n_em] + _mu - epsilonk(k_bar_x+qqx,k_bar_y+qqy,k_bar_z+qqz) - _SE[3*NI/2+n_bar+n_em] ) 
                            );
                        return jj_tmp;
                    };
                    T res_1D = 1.0/(2.0*M_PI)*integralsObj.gauss_quad_1D(inner_1D_int_jj,0.0,2.0*M_PI);
                    return res_1D;
                };
                inner_3D_int_szsz = [&](double k_bar_x, double k_bar_y, double k_bar_z){
                    inner_1D_int_szsz = [&](double k_tilde_z){
                        // denominator
                        Gamma_n_tilde_n_bar.at(n_tilde,n_bar,n_em) = Gamma(_k_t_b[n_k_tilde_x],_k_t_b[n_k_tilde_y],k_tilde_z,k_bar_x,k_bar_y,k_bar_z,qqx,qqy,qqz,n_bar,n_tilde,n_em,integralsObj);
                        // jj (nearest-neighbour only)
                        szsz_tmp = ( 1.0/( _splInlineobj._iwn_array[n_tilde] + _mu - epsilonk(_k_t_b[n_k_tilde_x],_k_t_b[n_k_tilde_y],k_tilde_z) - _SE[3*NI/2+n_tilde] ) 
                            )*( 1.0/( _splInlineobj._iwn_array[n_tilde]-_iqn[n_em] + _mu - epsilonk(_k_t_b[n_k_tilde_x]-qqx,_k_t_b[n_k_tilde_y]-qqy,k_tilde_z-qqz) - _SE[3*NI/2+n_tilde-n_em] )
                            )*Gamma_n_tilde_n_bar.at(n_tilde,n_bar,n_em
                            )*( 1.0/( _splInlineobj._iwn_array[n_bar] + _mu - epsilonk(k_bar_x,k_bar_y,k_bar_z) - _SE[3*NI/2+n_bar] )
                            )*( 1.0/( _splInlineobj._iwn_array[n_bar]+_iqn[n_em] + _mu - epsilonk(k_bar_x+qqx,k_bar_y+qqy,k_bar_z+qqz) - _SE[3*NI/2+n_bar+n_em] ) 
                            );
                        return szsz_tmp;
                    };
                    T res_1D = 1.0/(2.0*M_PI)*integralsObj.gauss_quad_1D(inner_1D_int_szsz,0.0,2.0*M_PI);
                    return res_1D;
                };
                #else
                inner_2D_int_jj = [&](double k_bar_x, double k_bar_y){
                    // denominator
                    Gamma_n_tilde_n_bar.at(n_tilde,n_bar,n_em) = Gamma(_k_t_b[n_k_tilde_x],_k_t_b[n_k_tilde_y],k_bar_x,k_bar_y,qqx,qqy,n_bar,n_tilde,n_em,integralsObj);
                    // jj
                    jj_tmp = -1.0*velocity(_k_t_b[n_k_tilde_x])*velocity(k_bar_x
                        )*getGreen(_k_t_b[n_k_tilde_x],_k_t_b[n_k_tilde_y],_splInlineobj._iwn_array[n_tilde]
                        )*getGreen(_k_t_b[n_k_tilde_x]-qqx,_k_t_b[n_k_tilde_y]-qqy,_splInlineobj._iwn_array[n_tilde]-_iqn[n_em]
                        )*Gamma_n_tilde_n_bar.at(n_tilde,n_bar,n_em
                        )*getGreen(k_bar_x,k_bar_y,_splInlineobj._iwn_array[n_bar]
                        )*getGreen(k_bar_x+qqx,k_bar_y+qqy,_splInlineobj._iwn_array[n_bar]+_iqn[n_em]);
                
                    return jj_tmp;
                };
                inner_2D_int_szsz = [&](double k_bar_x, double k_bar_y){
                    // denominator
                    Gamma_n_tilde_n_bar.at(n_tilde,n_bar,n_em) = Gamma(_k_t_b[n_k_tilde_x],_k_t_b[n_k_tilde_y],k_bar_x,k_bar_y,qqx,qqy,n_bar,n_tilde,n_em,integralsObj);
                    // szsz
                    szsz_tmp = getGreen(_k_t_b[n_k_tilde_x],_k_t_b[n_k_tilde_y],_splInlineobj._iwn_array[n_tilde]
                        )*getGreen(_k_t_b[n_k_tilde_x]-qqx,_k_t_b[n_k_tilde_y]-qqy,_splInlineobj._iwn_array[n_tilde]-_iqn[n_em]
                        )*Gamma_n_tilde_n_bar.at(n_tilde,n_bar,n_em
                        )*getGreen(k_bar_x,k_bar_y,_splInlineobj._iwn_array[n_bar]
                        )*getGreen(k_bar_x+qqx,k_bar_y+qqy,_splInlineobj._iwn_array[n_bar]+_iqn[n_em]);
                    return szsz_tmp;
                };
                #endif
                if (is_single_ladder_precomputed){
                    // Come up with UNIQUE way to attribute the tags to different Gamma matrices using Cantor pairing function
                    int tag = (int)( ((n_k_tilde_x+n_k_tilde_y)*(n_k_tilde_x+n_k_tilde_y+1))/2 ) + (int)n_k_tilde_y;
                    std::cout << "tag: " << tag << std::endl;
                    tag_vec.push_back(tag);
                    *( static_cast< arma::Cube< T >* >(arma_ptr) ) = Gamma_n_tilde_n_bar;
                }
                // auto integral_res = integralsObj.gauss_quad_2D<double,GaussEl<T>>(inner_2D_int,0.0,2.0*M_PI,0.0,2.0*M_PI);
                jj_resp_iqn += integralsObj.gauss_quad_3D(inner_3D_int_jj,0.0,2.0*M_PI,0.0,2.0*M_PI,0.0,2.0*M_PI);
                szsz_resp_iqn += integralsObj.gauss_quad_3D(inner_3D_int_szsz,0.0,2.0*M_PI,0.0,2.0*M_PI,0.0,2.0*M_PI);
            }
            end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "loop n_em: " << n_em << " done in " << elapsed_secs << " secs.." << "\n";
        }
        jj_resp_iqn *= (2.0/_beta/_beta/(2.0*M_PI)/(2.0*M_PI)/(2.0*M_PI)); //*arma::accu(GG_n_tilde_n_bar_jj);
        szsz_resp_iqn *= (2.0/_beta/_beta/(2.0*M_PI)/(2.0*M_PI)/(2.0*M_PI)); //*arma::accu(GG_n_tilde_n_bar_szsz);
        MPIData mpi_data_tmp { n_k_tilde_x, n_k_tilde_y, jj_resp_iqn, szsz_resp_iqn }; // summing over the internal ikn_tilde and ikn_bar
        GG_iqn.push_back(static_cast<MPIData&&>(mpi_data_tmp));
    }
    std::cout << "After the loop.." << std::endl;
    
    return GG_iqn;
}

#endif

namespace IPT2{

    inline double velocity(double k) noexcept{
        /* This method computes the current vertex in the case of a 1D nearest-neighbour dispersion relation.

        Parameters:
            k (double): k-point.

        Returns:
            (double): current vertex.
        */
        return 2.0*std::sin(k);
    }

    std::vector<double> get_iwn_to_tau(const std::vector< std::complex<double> >& F_iwn, double beta, std::string obj) noexcept{
        /*  This method computes the imaginary-time object related to "F_iwn". Prior to calling in this method, one needs
        to subtract the leading moments of the object, be it the self-energy or the Green's function. 
            
            Parameters:
                F_iwn (const std::vector< std::complex<double> >&): Object defined in fermionic Matsubara frequencies to translate to imaginary time.
                beta (double): Inverse temperature.
                obj (std::string): string specifying the nature of the object to be transformed. Defaults to "Green". Anything else chosen,
                i.e "Derivative", means that the last component to the imaginary-time object (tau=beta) is left for later. This is particularly important for the derivative.
            
            Returns:
                tau_final_G (std::vector<double>): vector containing the imaginary-time definition of the inputted object "F_iwn".
        */

        size_t MM = F_iwn.size();
        //std::cout << "size: " << MM << std::endl;
        const std::complex<double> im(0.0,1.0);
        std::vector<double> tau_final_G(MM+1,0.0);
        std::complex<double>* output = new std::complex<double>[MM];
        std::complex<double>* input = new std::complex<double>[MM];
        for (size_t j=0; j<MM; j++){
            input[j] = F_iwn[j];
        }
        // FFT
        fftw_plan plan;
        plan=fftw_plan_dft_1d(MM, reinterpret_cast<fftw_complex*>(input), reinterpret_cast<fftw_complex*>(output), FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan);
        for (int i=0; i<MM; i++){
            tau_final_G[i] = ( ( 1.0/beta*std::exp( -im*(double)i*M_PI*(1.0/MM - 1.0) ) )*output[i] ).real();
        }
        if (obj.compare("Green")==0){
            for (int i=0; i<MM; i++){
                tau_final_G[MM] += ( 1.0/beta*std::exp(-im*M_PI*(1.0-MM))*F_iwn[i] ).real();
            }
        }
        
        delete[] output;
        delete[] input;

        return tau_final_G;
    }

    template<typename T>
    std::vector< T > generate_random_numbers(size_t arr_size, T min, T max) noexcept{
        /*  This template (T) method generates an array of random numbers. 
            
            Parameters:
                arr_size (size_t): size of the vector to be generated randomly.
                min (T): minimum value of the randomly generated vector.
                max (T): maximum value of the randomly generated vector.
            
            Returns:
                rand_num_container (std::vector< T >): vector containing the numbers generated randomly.
        */

        srand(time(0));
        std::vector< T > rand_num_container(arr_size);
        T random_number;
        for (size_t i=0; i<arr_size; i++){
            random_number = min + (T)( ( (T)rand() ) / ( (T)RAND_MAX ) * (max - min) );
            rand_num_container[i] = random_number;
        }
        return rand_num_container;
    }

    void set_vector_processes(std::vector<MPIData>* vec_to_processes, unsigned int N_q) noexcept{
        /* This function sets up the vector of MPIData structures to be dispatched to slave processes by
        root process.

            Parameters:
                vec_to_processes (std::vector<MPIData>*): pointer to vector of MPIData instances.
                N_q (unsigned int): number of k-points (1D) of the doublon momentum to be considered (k_bar or k_tilde).
            
            Returns:
                (void): fills up vec_to_processes with the MPIData instances.
        */
        for (size_t k_t=0; k_t<N_q; k_t++){
            for (size_t k_b=0; k_b<N_q; k_b++){
                MPIData mpi_data{ k_t, k_b, std::complex<double>(0.0,0.0), std::complex<double>(0.0,0.0) };
                vec_to_processes->push_back(static_cast<MPIData&&>(mpi_data));
            }
        }
    }

    void create_mpi_data_struct(MPI_Datatype& custom_type){
        /* This function build a new MPI datatype by reference to deal with the struct MPIData that is used to send across the different 
        processes.

            Parameters:
                custom_type (MPI_Datatype&): MPI datatype to be created based upon MPIData struct.
            
            Returns:
                (void): committed MPI datatype.
        */
        int lengths[4]={ 1, 1, 1, 1 };
        MPI_Aint offsets[4]={ offsetof(MPIData,k_tilde), offsetof(MPIData,k_bar), offsetof(MPIData,cplx_data_jj), offsetof(MPIData,cplx_data_szsz) };
        MPI_Datatype types[4]={ MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_CXX_DOUBLE_COMPLEX, MPI_CXX_DOUBLE_COMPLEX }, tmp_type;
        MPI_Type_create_struct(4,lengths,offsets,types,&tmp_type);
        // Proper padding
        MPI_Type_create_resized(tmp_type, 0, sizeof(MPIData), &custom_type);
        MPI_Type_commit(&custom_type);
    }

    void create_mpi_data_struct_cplx(MPI_Datatype& custom_type){
        /* This function build a new MPI datatype by reference to deal with the struct cplx_t that is used to send across the different 
        processes.

            Parameters:
                custom_type (MPI_Datatype&): MPI datatype to be created based upon cplx_t struct.
            
            Returns:
                (void): committed MPI datatype.
        */
        int lengths[2]={ 1, 1 };
        MPI_Aint offsets[2]={ offsetof(cplx_t,re), offsetof(cplx_t,im) };
        MPI_Datatype types[2]={ MPI_DOUBLE, MPI_DOUBLE }, tmp_type;
        MPI_Type_create_struct(2,lengths,offsets,types,&tmp_type);
        // Proper padding
        MPI_Type_create_resized(tmp_type, 0, sizeof(cplx_t), &custom_type);
        MPI_Type_commit(&custom_type);
    }

    void create_mpi_data_struct_corr(MPI_Datatype& custom_type){
        /* This function build a new MPI datatype by reference to deal with the struct MPIDataCorr that is used to send across the different 
        processes.

            Parameters:
                custom_type (MPI_Datatype&): MPI datatype to be created based upon MPIDataCorr struct.
            
            Returns:
                (void): committed MPI datatype.
        */
        int lengths[4]={ 1, 1, 1, 1 };
        MPI_Aint offsets[4]={ offsetof(MPIDataCorr,n_iqn), offsetof(MPIDataCorr,n_ikn), offsetof(MPIDataCorr,n_k), offsetof(MPIDataCorr,cplx_denom_corr) };
        MPI_Datatype types[4]={ MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_CXX_DOUBLE_COMPLEX }, tmp_type;
        MPI_Type_create_struct(4,lengths,offsets,types,&tmp_type);
        // Proper padding
        MPI_Type_create_resized(tmp_type, 0, sizeof(MPIDataCorr), &custom_type);
        MPI_Type_commit(&custom_type);
    }

    void create_mpi_data_struct_ladder(MPI_Datatype& custom_type){
        /* This function build a new MPI datatype by reference to deal with the struct MPIDataCorr that is used to send across the different 
        processes.

            Parameters:
                custom_type (MPI_Datatype&): MPI datatype to be created based upon MPIDataCorr struct.
            
            Returns:
                (void): committed MPI datatype.
        */
        int lengths[3]={ 1, 1, 1 };
        MPI_Aint offsets[3]={ offsetof(MPIDataLadder,n_iqpn), offsetof(MPIDataLadder,n_qp), offsetof(MPIDataLadder,cplx_val) };
        MPI_Datatype types[3]={ MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_CXX_DOUBLE_COMPLEX }, tmp_type;
        MPI_Type_create_struct(3,lengths,offsets,types,&tmp_type);
        // Proper padding
        MPI_Type_create_resized(tmp_type, 0, sizeof(MPIDataLadder), &custom_type);
        MPI_Type_commit(&custom_type);
    }
}

inline std::tuple<int,int> inverse_Cantor_pairing(int number){
    // see https://en.wikipedia.org/wiki/Pairing_function for notation and explanation
    int w = (int)std::floor( ( std::sqrt( 8.0*number + 1.0 ) - 1.0 ) / 2.0 );
    int t = static_cast<int>( (w)*(w+1)/2 );
    int n_k_tilde = number - t;
    int n_k_bar = w - n_k_tilde;

    return std::make_tuple( n_k_bar, n_k_tilde );
}

// some functions used to send off and receive data from unidimensional k_array (or any arrays) across processes

void MPI_send_k_array(int& world_size, int& ierr, int& num_elem_remaining, int& num_elem_to_send, const int& num_elem_per_proc_precomp, int& shift, int& start, int& end, const unsigned int& N_k, double* k_t_b_array_data) noexcept(false){
    for (int an_id=1; an_id<world_size; an_id++){ // This loop is skipped if world_size=1
        if (num_elem_remaining<=1){
            start = an_id*num_elem_per_proc_precomp + 1 + ( (num_elem_remaining != 0) ? shift-1 : 0 );
            if((N_k - start) < num_elem_per_proc_precomp){ // Taking care of the case where remaining data is 0.
                end = N_k - 1;
            } else{
                end = (an_id + 1)*num_elem_per_proc_precomp + ( (num_elem_remaining != 0) ? shift-1 : 0 );
            }
        } else{
            if (an_id==1){
                start = an_id*num_elem_per_proc_precomp + 1;
            } else{
                start = end+1;
            }
            end = start+num_elem_per_proc_precomp;
            num_elem_remaining--;
        }
        std::cout << "num_elem_remaining: " << num_elem_remaining << " for an_id: " << an_id << " start: " << start << " end: " << end << std::endl;
        num_elem_to_send = end - start + 1;
        std::cout << "num elem to send for id " << an_id << " is " << num_elem_to_send << "\n";
        ierr = MPI_Send( &num_elem_to_send, 1 , MPI_INT, an_id, SEND_NUM_TO_SLAVES, MPI_COMM_WORLD );
        ierr = MPI_Send( &start, 1 , MPI_INT, an_id, SEND_NUM_START_TO_SLAVES, MPI_COMM_WORLD );
        ierr = MPI_Send( (void*)(k_t_b_array_data+start), num_elem_to_send, MPI_DOUBLE,
                an_id, SEND_DATA_TAG, MPI_COMM_WORLD );
    }
}

template<typename T>
void MPI_recv_k_array_from_slaves(int& world_size, int& ierr, int& recv_root_num_elem, std::vector<T>& local_container, std::vector<T>& container_bcast, MPI_Status& status, MPI_Datatype& MPI_DataCorr_struct_t) noexcept(false){
    MPIDataReceive<T> mpi_datacorr_receive;
    for (int an_id=1; an_id<world_size; an_id++){ // This loop is skipped if world_size=1
        ierr = MPI_Recv( &recv_root_num_elem, 1, MPI_INT, 
            an_id, RETURN_NUM_RECV_TO_ROOT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "recv_root_num_elem: " << recv_root_num_elem << " for id " << an_id << std::endl;
        mpi_datacorr_receive.size = (size_t)recv_root_num_elem;
        mpi_datacorr_receive.data_struct = (T*)malloc(mpi_datacorr_receive.size*sizeof(T));
        ierr = MPI_Recv((void*)mpi_datacorr_receive.data_struct,mpi_datacorr_receive.size,MPI_DataCorr_struct_t,an_id,an_id+SHIFT_TO_DIFFERENTIATE_TAGS,MPI_COMM_WORLD,&status);
        for (size_t el=0; el<mpi_datacorr_receive.size; el++){
            local_container.push_back( mpi_datacorr_receive.data_struct[el] );
        }
        free(mpi_datacorr_receive.data_struct);
    }
    // unpacking denom_corr into denom_corr_tensor
    // Broadcasting the data from the root process for the correction to the denominator
    assert(container_bcast.size()==local_container.size());
    std::cout << "BCAST Before " << local_container.size() << " " << container_bcast.size() << std::endl;
    container_bcast = std::move(local_container);
}