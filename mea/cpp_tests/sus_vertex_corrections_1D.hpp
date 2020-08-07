#include "../../src/IPT2nd3rdorderSingle2.hpp"
#include <ctime>
#include <mpi.h>
/*
Info HDF5:
https://support.hdfgroup.org/HDF5/doc/cpplus_RM/compound_8cpp-example.html
https://www.hdfgroup.org/downloads/hdf5/source-code/
*/

#define SEND_DATA_TAG 2000
#define SEND_NUM_TO_SLAVES 2001
#define RETURN_DATA_TAG 3000
#define RETURN_NUM_RECV_TO_ROOT 3001
#define RETURN_TAGS_TO_ROOT 3002
#define SHIFT_TO_DIFFERENTIATE_TAGS 1000000

//static bool slaves_can_write_in_file = false; // This prevents that the slave processes
static int root_process = 0;

//#define INFINITE

typedef struct{
    size_t k_tilde;
    size_t k_bar;
    std::complex<double> cplx_data;
} MPIData;

struct MPIDataReceive{
    MPIData* data_struct;
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
    void create_mpi_data_struct_cplx(MPI_Datatype& custom_type);

    template< class T >
    class OneLadder{
        /* Class to compute single ladder vertex corrections.
        */
        public:
            OneLadder& operator=(const OneLadder&) = delete;
            OneLadder(const OneLadder&) = delete;
            std::vector< MPIData > operator()(size_t n_k_bar, size_t n_k_tilde, bool is_jj, bool is_single_ladder_precomputed=false, void* arma_ptr=nullptr, double qq=0.0) const noexcept(false);
            OneLadder()=default;
            explicit OneLadder(const SplineInline< T >& splInlineobj, const std::vector< T >& iqn, const std::vector<double>& k_arr, const std::vector< T >& iqn_tilde, double mu, double U, double beta) : _splInlineobj(splInlineobj), _iqn(iqn), _k_t_b(k_arr), _iqn_tilde(iqn_tilde){
                this->_mu = mu;
                this->_U = U;
                this->_beta = beta;
            };
            static std::vector<int> tag_vec;
            //static std::vector< T** > mat_sl_vec; 

        protected:
            const SplineInline< T >& _splInlineobj;
            const std::vector< T >& _iqn;
            const std::vector<double>& _k_t_b;
            double _mu {0.0}, _U {0.0}, _beta {0.0};
            inline T getGreen(double k, T iwn) const noexcept;
            T Gamma(double k_bar, double k_tilde, T ikn_bar, T ikn_tilde) const noexcept(false);

        private:
            const std::vector< T >& _iqn_tilde;

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
            explicit InfiniteLadders(const SplineInline< T >& splInlineobj, const std::vector< T >& iqn, const std::vector<double>& k_arr, const std::vector< T >& iqn_tilde, double mu, double U, double beta) : OneLadder< T >(splInlineobj,iqn,k_arr,iqn_tilde,mu,U,beta){
                std::cout << "InfiniteLadder U: " << OneLadder< T >::_U << " and InfiniteLadder beta: " << OneLadder< T >::_beta << std::endl;
            }
            std::vector< MPIData > operator()(size_t n_k_bar, size_t n_k_tilde, bool is_jj, bool is_simple_ladder_precomputed=false, double qq=0.0) const noexcept(false);
            static std::string _FILE_NAME;

        private:
            using OneLadder< T >::getGreen;
            using OneLadder< T >::Gamma;
            T Gamma_correction_denominator(double k_bar, double kpp, T ikn_bar, T ikppn) const noexcept(false);
            arma::Mat< T > Gamma_to_merge_corr(double k_bar, T ikn_bar, double qq) const noexcept(false);
            T Gamma_merged_corr(arma::Mat< T >& denom_corr, T iqn, double qq) const noexcept(false);
            
    };

    template< class T > std::string InfiniteLadders< T >::_FILE_NAME = std::string("");

}

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

template< class T >
T IPT2::OneLadder< T >::Gamma(double k_bar, double k_tilde, T ikn_bar, T ikn_tilde) const noexcept(false){
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
    const double delta = 2.0*M_PI/(double)(_splInlineobj._k_array.size()-1);
    std::vector< T > tmp_container_integral_k(_splInlineobj._k_array.size());
    std::vector< T > tmp_container_sum_iqn(_iqn_tilde.size());
    for (size_t j=0; j<_iqn_tilde.size(); j++){
        for (size_t l=0; l<_splInlineobj._k_array.size(); l++){
            tmp_container_integral_k[l] = getGreen(k_bar-_splInlineobj._k_array[l],ikn_bar-_iqn_tilde[j])*getGreen(k_tilde-_splInlineobj._k_array[l],ikn_tilde-_iqn_tilde[j]);
        }
        tmp_container_sum_iqn[j] = 1.0/(2.0*M_PI)*intObj.I1D_VEC(tmp_container_integral_k,delta,"simpson");
    }
    std::for_each(tmp_container_sum_iqn.begin(),tmp_container_sum_iqn.end(),[&](T n){ return lower_val+=n; });
    
    lower_val *= _U/_beta;
    lower_val += 1.0;
    lower_val = _U/lower_val;

    return lower_val;
}

template< class T >
std::vector< MPIData > IPT2::OneLadder< T >::operator()(size_t n_k_bar, size_t n_k_tilde, bool is_jj, bool is_single_ladder_precomputed, void* arma_ptr, double qq) const noexcept(false){
    /*  This method computes the susceptibility given the current-vertex correction for the single ladder diagram. It does so for a set
    of momenta (k_bar,ktilde). It uses the dressed Green's functions computed in the paramagnetic state. 
        
        Parameters:
            n_k_bar (size_t): right doublon momentum index (Feynman diagram picture).
            n_k_tilde (size_t): left doublon momentum index (Feynman diagram picture).
            is_jj (bool): boolean parameters that selects the nature of the susceptibility to be computed: spin-spin (false) or current-current (true).
            qq (double): momentum injected in the system. Defaults to 0, because interested in the thermodynamic limit (infinite volume).
        
        Returns:
            GG_iqn (std::vector< MPIData >): vector whose elements are MPIData structures containing footprint of n_k_bar, n_k_tilde and 
            the susceptibility for each bosonic Matsubara frequency. Therefore, the length of the vector is same as that of the incoming
            bosonic Matsubara frequencies.
    */
    std::vector< MPIData > GG_iqn;
    // Computing Gamma
    const size_t NI = _splInlineobj._iwn_array.size();
    arma::Mat< T > Gamma_n_bar_n_tilde(NI,NI); // Doesn't depend on iq_n
    for (size_t n_bar=0; n_bar<NI; n_bar++){
        // clock_t begin = clock();
        for (size_t n_tilde=0; n_tilde<NI; n_tilde++){
            Gamma_n_bar_n_tilde(n_bar,n_tilde) = Gamma(_k_t_b[n_k_bar],_k_t_b[n_k_tilde],_splInlineobj._iwn_array[n_bar],_splInlineobj._iwn_array[n_tilde]);
        }
        // clock_t end = clock();
        // double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        // std::cout << "outer loop n_bar: " << n_bar << " done in " << elapsed_secs << " secs.." << "\n";
    }
    if (is_single_ladder_precomputed){
        // Come up with UNIQUE way to attribute the tags to different Gamma matrices using Cantor pairing function
        int tag = (int)( ((n_k_bar+n_k_tilde)*(n_k_bar+n_k_tilde+1))/2 ) + (int)n_k_tilde;
        std::cout << "tag: " << tag << std::endl;
        tag_vec.push_back(tag);
        *( static_cast< arma::Mat< T >* >(arma_ptr) ) = Gamma_n_bar_n_tilde;
        // for (size_t i=0; i<_splInlineobj._iwn_array.size(); i++){
        //     std::cout << "el after: " << sl_vec[0][3*_splInlineobj._iwn_array.size()+i] << std::endl;
        // }
    }

    arma::Mat< T > GG_n_bar_n_tilde(NI,NI);
    for (size_t em=0; em<_iqn.size(); em++){
        for (size_t n_bar=0; n_bar<NI; n_bar++){
            for (size_t n_tilde=0; n_tilde<NI; n_tilde++){
                // This part remains to be done....
                GG_n_bar_n_tilde(n_bar,n_tilde) = getGreen(_k_t_b[n_k_tilde],_splInlineobj._iwn_array[n_tilde])*getGreen(_k_t_b[n_k_tilde]+qq,_splInlineobj._iwn_array[n_tilde]+_iqn[em])*Gamma_n_bar_n_tilde(n_bar,n_tilde)*getGreen(_k_t_b[n_k_bar],_splInlineobj._iwn_array[n_bar])*getGreen(_k_t_b[n_k_bar]+qq,_splInlineobj._iwn_array[n_bar]+_iqn[em]);
            }
        }
        if (is_jj){
            MPIData mpi_data_tmp { n_k_tilde, n_k_bar, -1.0*velocity(_k_t_b[n_k_tilde])*velocity(_k_t_b[n_k_bar])*(2.0/_beta/_beta)*arma::accu(GG_n_bar_n_tilde) }; // summing over the internal ikn_tilde and ikn_bar
            GG_iqn.push_back(static_cast<MPIData&&>(mpi_data_tmp));
        } else{ // ADDED A FACTOR OF 2 FOR THE SPIN BELOW
            MPIData mpi_data_tmp { n_k_tilde, n_k_bar, (2.0/_beta/_beta)*arma::accu(GG_n_bar_n_tilde) }; // summing over the internal ikn_tilde and ikn_bar
            GG_iqn.push_back(static_cast<MPIData&&>(mpi_data_tmp));
        }
    }
    
    return GG_iqn;
}

template< class T >
T IPT2::InfiniteLadders< T >::Gamma_correction_denominator(double k_bar, double kpp, T ikn_bar, T ikppn) const noexcept(false){
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
    const double delta = 2.0*M_PI/(double)(OneLadder< T >::_splInlineobj._k_array.size()-1);
    std::vector< T > tmp_container_integral_k(OneLadder< T >::_splInlineobj._k_array.size());
    std::vector< T > tmp_container_sum_iwn(OneLadder< T >::_splInlineobj._iwn_array.size());
    const size_t NI = OneLadder< T >::_splInlineobj._iwn_array.size();
    for (size_t n_ppp=0; n_ppp<NI; n_ppp++){
        for (size_t n_k_ppp=0; n_k_ppp<OneLadder< T >::_splInlineobj._k_array.size(); n_k_ppp++){
            tmp_container_integral_k[n_k_ppp] = getGreen(OneLadder< T >::_splInlineobj._k_array[n_k_ppp]+kpp-k_bar,OneLadder< T >::_splInlineobj._iwn_array[n_ppp]+ikppn-ikn_bar)*getGreen(OneLadder< T >::_splInlineobj._k_array[n_k_ppp],OneLadder< T >::_splInlineobj._iwn_array[n_ppp]);
        }
        tmp_container_sum_iwn[n_ppp] = intObj.I1D_VEC(tmp_container_integral_k,delta,"simpson");
    }
    std::for_each(tmp_container_sum_iwn.begin(),tmp_container_sum_iwn.end(),[&](T n){ return denom_val+=n; });
    denom_val *= OneLadder< T >::_U/OneLadder< T >::_beta/(2.0*M_PI);
    denom_val += 1.0;

    return 1.0/denom_val;
}

template< class T >
arma::Mat< T > IPT2::InfiniteLadders< T >::Gamma_to_merge_corr(double k_bar, T ikn_bar, double qq) const noexcept(false){
    /*  This method computes the matrix containing the correction term to the single ladder as a function of the local
    momentum k_pp and the local fermionic Matsubara frequency ikn_bar. This method mainly produces data container fit to be used
    to sum over the local overall variables (k_pp,ikppn) defining the correction to the single ladder contribution.
        
        Parameters:
            k_bar (double): left doublon momentum (Feynman diagram picture).
            ikn_bar (T): left doublon fermionic Matsubara frequency (Feynman diagram picture).
            qq (double): momentum injected in the system. Defaults to 0, because interested in the thermodynamic limit (infinite volume).
        
        Returns:
            deniom_corr (arma::Mat<T>): Matrix containing the correction term to the single ladder as a function of the local 4-vector
            (k_pp,ikppn).
    */
    // compute the denominator of correction term in bunch (doesn't depend on iqn)
    arma::Mat< T > denom_corr(OneLadder< T >::_splInlineobj._k_array.size(),OneLadder< T >::_splInlineobj._iwn_array.size());
    for (size_t n_k_pp=0; n_k_pp<OneLadder< T >::_splInlineobj._k_array.size(); n_k_pp++){
        clock_t begin = clock();
        for (size_t n_pp=0; n_pp<OneLadder< T >::_splInlineobj._iwn_array.size(); n_pp++){
            denom_corr(n_k_pp,n_pp) = Gamma_correction_denominator(k_bar,OneLadder< T >::_splInlineobj._k_array[n_k_pp],ikn_bar,OneLadder< T >::_splInlineobj._iwn_array[n_pp]);
        }
        clock_t end = clock();
        double elapsed_secs = double(end-begin) / CLOCKS_PER_SEC;
        std::cout << "infinite ladder loop n_k_pp: " << n_k_pp << elapsed_secs << "secs.." << "\n";
    }

    return denom_corr;
}

template< class T >
T IPT2::InfiniteLadders< T >::Gamma_merged_corr(arma::Mat< T >& denom_corr, T iqn, double qq) const noexcept(false){
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
    const double delta = 2.0*M_PI/(double)(OneLadder< T >::_splInlineobj._k_array.size()-1);
    std::vector< T > tmp_container_integral_k(OneLadder< T >::_splInlineobj._k_array.size());
    std::vector< T > tmp_container_sum_iwn(OneLadder< T >::_splInlineobj._iwn_array.size());
    const size_t NI = OneLadder< T >::_splInlineobj._iwn_array.size();
    for (size_t n_pp=0; n_pp<NI; n_pp++){
        for (size_t n_k_pp=0; n_k_pp<OneLadder< T >::_splInlineobj._k_array.size(); n_k_pp++){
            tmp_container_integral_k[n_k_pp] = getGreen(OneLadder< T >::_splInlineobj._k_array[n_k_pp],OneLadder< T >::_splInlineobj._iwn_array[n_pp])*denom_corr(n_k_pp,n_pp)*getGreen(OneLadder< T >::_splInlineobj._k_array[n_k_pp]+qq,OneLadder< T >::_splInlineobj._iwn_array[n_pp]+iqn);
        }
        tmp_container_sum_iwn[n_pp] = 1.0/(2.0*M_PI)*intObj.I1D_VEC(tmp_container_integral_k,delta,"simpson");
    }

    std::for_each(tmp_container_sum_iwn.begin(),tmp_container_sum_iwn.end(),[&](T n){ return tot_corr+=n; });
    tot_corr *= OneLadder< T >::_U/OneLadder< T >::_beta/(2.0*M_PI);
    
    return tot_corr; 
}

template< class T >
std::vector< MPIData > IPT2::InfiniteLadders< T >::operator()(size_t n_k_bar, size_t n_k_tilde, bool is_jj, bool is_single_ladder_precomputed, double qq) const noexcept(false){
    /*  This method computes the susceptibility given the current-vertex correction for the infinite ladder diagram. It does so for a set
    of momenta (k_bar,ktilde). It uses the dressed Green's functions computed in the paramagnetic state. 
        
        Parameters:
            n_k_bar (size_t): right doublon momentum index (Feynman diagram picture).
            n_k_tilde (size_t): left doublon momentum index (Feynman diagram picture).
            is_jj (bool): boolean parameters that selects the nature of the susceptibility to be computed: spin-spin (false) or current-current (true).
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
    arma::Mat< T > Gamma_n_bar_n_tilde(NI,NI); // Doesn't depend on iq_n
    if (is_single_ladder_precomputed){
        // Should load the data saved previously saved in the simpler simgle ladder calculation..
        const H5std_string DATASET_NAME_OPEN("kbar_"+std::to_string(OneLadder<T>::_k_t_b[n_k_bar])+"ktilde_"+std::to_string(OneLadder<T>::_k_t_b[n_k_tilde]));
        std::cout << "_FILE_NAME: " << _FILE_NAME << std::endl;
        H5::H5File* file_open = new H5::H5File(_FILE_NAME,H5F_ACC_RDONLY);

        if ( std::is_same< T,std::complex<double> >::value ){
            try{
                Gamma_n_bar_n_tilde = readFromHDF5File(file_open,DATASET_NAME_OPEN);
            } catch(std::runtime_error& err){
                std::cerr << err.what() << "\n";
                exit(0);
            }
        }
        
        delete file_open;
    } else{
        for (size_t n_bar=0; n_bar<NI; n_bar++){
            clock_t begin = clock();
            for (size_t n_tilde=0; n_tilde<NI; n_tilde++){
                Gamma_n_bar_n_tilde(n_bar,n_tilde) = OneLadder< T >::_U/Gamma(OneLadder< T >::_k_t_b[n_k_bar],OneLadder< T >::_k_t_b[n_k_tilde],OneLadder< T >::_splInlineobj._iwn_array[n_bar],OneLadder< T >::_splInlineobj._iwn_array[n_tilde]);
            }
            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "infinite ladder loop n_bar: " << n_bar << " done in " << elapsed_secs << " secs.." << "\n";
        }
    }

    // ikn_bar-diagonal summation over the correction term degrees of freedom
    arma::Cube< T > denom_corr(OneLadder< T >::_splInlineobj._k_array.size(),NI,NI);  // layout as follows: kpp, ikppn, ikn_bar
    for (size_t n_bar=0; n_bar<NI; n_bar++){
        clock_t begin = clock();
        denom_corr.slice(n_bar) = Gamma_to_merge_corr(OneLadder< T >::_k_t_b[n_k_bar],OneLadder< T >::_splInlineobj._iwn_array[n_bar],qq);
        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "infinite ladder loop n_bar: " << n_bar << " done in " << elapsed_secs << " secs.." << "\n";
    }
    
    arma::Mat< T > tmp_ikn_bar_corr;
    for (size_t em=0; em<OneLadder< T >::_iqn.size(); em++){
        // Now, for each ikn_bar value, one has to complete the summation over kpp and ikppn by calling Gamma_merged_corr
        std::vector< T > ikn_bar_corr(NI);
        for (size_t n_bar=0; n_bar<NI; n_bar++){
            tmp_ikn_bar_corr = denom_corr.slice(n_bar);
            ikn_bar_corr[n_bar] = Gamma_merged_corr(tmp_ikn_bar_corr,OneLadder< T >::_iqn[em],qq);
        }
        // Now considering both the single ladder and its corrections
        arma::Mat< T > GG_n_bar_n_tilde(NI,NI);
        for (size_t n_bar=0; n_bar<NI; n_bar++){
            for (size_t n_tilde=0; n_tilde<NI; n_tilde++){
                GG_n_bar_n_tilde(n_bar,n_tilde) = getGreen(OneLadder< T >::_k_t_b[n_k_tilde],OneLadder< T >::_splInlineobj._iwn_array[n_tilde])*getGreen(OneLadder< T >::_k_t_b[n_k_tilde]+qq,OneLadder< T >::_splInlineobj._iwn_array[n_tilde]+OneLadder< T >::_iqn[em]) * ( 
                    1.0 / ( Gamma_n_bar_n_tilde(n_bar,n_tilde) - ikn_bar_corr[n_bar] ) 
                    ) * getGreen(OneLadder< T >::_k_t_b[n_k_bar],OneLadder< T >::_splInlineobj._iwn_array[n_bar])*getGreen(OneLadder< T >::_k_t_b[n_k_bar]+qq,OneLadder< T >::_splInlineobj._iwn_array[n_bar]+OneLadder< T >::_iqn[em]);
            }
        }
        if (is_jj){
            MPIData mpi_data_tmp { n_k_tilde, n_k_bar, -1.0*velocity(OneLadder< T >::_k_t_b[n_k_tilde])*velocity(OneLadder< T >::_k_t_b[n_k_bar])*(OneLadder< T >::_U/OneLadder< T >::_beta/OneLadder< T >::_beta)*arma::accu(GG_n_bar_n_tilde) }; // summing over the internal ikn_tilde and ikn_bar
            GG_iqn.push_back(static_cast<MPIData&&>(mpi_data_tmp));
        } else{
            MPIData mpi_data_tmp { n_k_tilde, n_k_bar, (OneLadder< T >::_U/OneLadder< T >::_beta/OneLadder< T >::_beta)*arma::accu(GG_n_bar_n_tilde) }; // summing over the internal ikn_tilde and ikn_bar
            GG_iqn.push_back(static_cast<MPIData&&>(mpi_data_tmp));
        }
    }

    return GG_iqn;
}

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
                MPIData mpi_data{ k_t, k_b, std::complex<double>(0.0,0.0) };
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
        int lengths[3]={ 1, 1, 1 };
        MPI_Aint offsets[3]={ offsetof(MPIData,k_tilde), offsetof(MPIData,k_bar), offsetof(MPIData,cplx_data) };
        MPI_Datatype types[3]={ MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_CXX_DOUBLE_COMPLEX }, tmp_type;
        MPI_Type_create_struct(3,lengths,offsets,types,&tmp_type);
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
}

inline std::tuple<int,int> inverse_Cantor_pairing(int number){
    // see https://en.wikipedia.org/wiki/Pairing_function for notation and explanation
    int w = (int)std::floor( ( std::sqrt( 8.0*number + 1.0 ) - 1.0 ) / 2.0 );
    int t = static_cast<int>( (w)*(w+1)/2 );
    int n_k_tilde = number - t;
    int n_k_bar = w - n_k_tilde;

    return std::make_tuple( n_k_bar, n_k_tilde );
}