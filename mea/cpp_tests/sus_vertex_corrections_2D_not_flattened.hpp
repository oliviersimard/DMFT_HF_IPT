#include "../../src/IPT2nd3rdorderSingle2.hpp"
#include <ctime>
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
#define SHIFT_TO_DIFFERENTIATE_TAGS 1000000 // To differentiate from the tags using Cantor function

#define TRIGTABLES // forces use of precomputed sine and cosine tables

// procomputed sine and cosine tables
static std::vector<double> sine_table;
static std::vector<double> cosine_table;
static std::vector<double> sine_table_finer;
static std::vector<double> cosine_table_finer;

//static bool slaves_can_write_in_file = false; // This prevents that the slave processes
static int root_process = 0;

typedef struct{
    #ifndef TRIGTABLES
    size_t k_tilde_x;
    size_t k_tilde_y;
    #else
    int k_tilde_x;
    int k_tilde_y;
    #endif
    std::complex<double> cplx_data_jj;
    std::complex<double> cplx_data_szsz;
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
    #ifndef TRIGTABLES
    inline double velocity(double k) noexcept;
    #else
    inline double velocity(int k) noexcept;
    #endif
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
            #ifndef TRIGTABLES
            std::vector< MPIData > operator()(size_t n_k_tilde_x, size_t n_k_tilde_y, double qqx=0.0, double qqy=0.0) const noexcept(false);
            #else
            std::vector< MPIData > operator()(int n_k_tilde_x, int n_k_tilde_y, int qqx=0, int qqy=0) const noexcept(false);
            #endif
            OneLadder()=default;
            #ifdef TRIGTABLES
            explicit OneLadder(const TrigP<double,int>& trigObj,const TrigP<double,int>& trigObj_finer,
                            const SplineInline< T >& splInlineobj, const std::vector< T >& iqn, const std::vector<double>& k_arr, 
                            const std::vector< T >& iqn_tilde, double mu, double U, 
                            double beta) : _splInlineobj(splInlineobj), _iqn(iqn), _k_t_b(k_arr), _trigObj(trigObj), _trigObj_finer(trigObj_finer), _iqn_tilde(iqn_tilde){
                this->_mu = mu;
                this->_U = U;
                this->_beta = beta;
            };
            #else
            explicit OneLadder(const SplineInline< T >& splInlineobj, const std::vector< T >& iqn, const std::vector<double>& k_arr, 
                            const std::vector< T >& iqn_tilde, double mu, double U, 
                            double beta) : _splInlineobj(splInlineobj), _iqn(iqn), _k_t_b(k_arr), _iqn_tilde(iqn_tilde){
                this->_mu = mu;
                this->_U = U;
                this->_beta = beta;
            };
            #endif

        protected:
            const SplineInline< T >& _splInlineobj;
            const std::vector< T >& _iqn;
            const std::vector<double>& _k_t_b;
            double _mu {0.0}, _U {0.0}, _beta {0.0};
            #ifndef TRIGTABLES
            inline T getGreen(double kx, double ky, T iwn) const noexcept;
            T Gamma(double k_tilde_x, double k_tilde_y, double k_bar_x, double k_bar_y, double qqx, double qqy, T ikn_bar, T ikn_tilde, T iqn) const noexcept(false);
            #else
            inline T getGreen(int kx, int ky, T iwn) const noexcept;
            inline T getGreen_finer(int k_tilde_x, int kx, int k_tilde_y, int ky, T iwn) const noexcept;
            T Gamma(int k_tilde_x, int k_tilde_y, int k_bar_x, int k_bar_y, int qqx, int qqy, T ikn_bar, T ikn_tilde, T iqn) const noexcept(false);
            #endif

        private:
            #ifdef TRIGTABLES
            const TrigP<double,int>& _trigObj;
            const TrigP<double,int>& _trigObj_finer;
            #endif
            const std::vector< T >& _iqn_tilde;

    };

}

#ifndef TRIGTABLES
template< class T >
inline T IPT2::OneLadder< T >::getGreen(double kx, double ky, T iwn) const noexcept{
    /*  This method computes the dressed Green's function of the one-band model. 
        
        Parameters:
            kx (double): x component of the momentum.
            ky (double): y component of the momentum.
            iwn (T): fermionic Matsubara frequency.
        
        Returns:
            (T): dressed 2D Green's function.
    */

    return 1.0 / ( iwn + _mu - epsilonk(kx,ky) - _splInlineobj.calculateSpline( iwn.imag() ) );
}
#else
template< class T >
inline T IPT2::OneLadder< T >::getGreen(int kx, int ky, T iwn) const noexcept{
    /*  This method computes the dressed Green's function of the one-band model. 
        
        Parameters:
            kx (double): x component of the momentum.
            ky (double): y component of the momentum.
            iwn (T): fermionic Matsubara frequency.
        
        Returns:
            (T): dressed 2D Green's function.
    */
    return 1.0 / ( iwn + _mu + 2.0*(_trigObj.COS(kx)+_trigObj.COS(ky)) - _splInlineobj.calculateSpline( iwn.imag() ) );
}
template< class T >
inline T IPT2::OneLadder< T >::getGreen_finer(int k_tilde_x, int kx, int k_tilde_y, int ky, T iwn) const noexcept{
    /*  This method computes the dressed Green's function of the one-band model. 
        
        Parameters:
            kx (double): x component of the momentum.
            ky (double): y component of the momentum.
            iwn (T): fermionic Matsubara frequency.
        
        Returns:
            (T): dressed 2D Green's function.
    */
    return 1.0 / ( iwn + _mu + 2.0*(_trigObj_finer.COS(k_tilde_x,kx)+_trigObj_finer.COS(k_tilde_y,ky)) - _splInlineobj.calculateSpline( iwn.imag() ) );
}
#endif

#ifndef TRIGTABLES
template< class T >
T IPT2::OneLadder< T >::Gamma(double k_tilde_x, double k_tilde_y, double k_bar_x, double k_bar_y, double qqx, double qqy, T ikn_bar, T ikn_tilde, T iqn) const noexcept(false){
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
    const Integrals integralsObj;
    const double delta = 2.0*M_PI/(double)(_splInlineobj._k_array.size()-1);
    std::vector< T > tmp_k_integral_inner(_splInlineobj._k_array.size()), tmp_k_integral_outer(_splInlineobj._k_array.size());
    std::vector< T > tmp_iqn_integral(_iqn_tilde.size());
    T integrated_ky;
    for (size_t j=0; j<_iqn_tilde.size(); j++){
        for (size_t kx=0; kx<_splInlineobj._k_array.size(); kx++){
            for (size_t ky=0; ky<_splInlineobj._k_array.size(); ky++){
                tmp_k_integral_inner[ky] = getGreen(k_tilde_x-_splInlineobj._k_array[kx],k_tilde_y-_splInlineobj._k_array[ky],ikn_tilde-_iqn_tilde[j])*getGreen(k_bar_x-_splInlineobj._k_array[kx]+qqx,k_bar_y-_splInlineobj._k_array[ky]+qqy,ikn_bar-_iqn_tilde[j]+iqn);
            }
            integrated_ky = integralsObj.I1D_VEC(tmp_k_integral_inner,delta,"simpson");
            tmp_k_integral_outer[kx] = integrated_ky;
        }
        // to sum over Matsubara frequencies
        tmp_iqn_integral[j] = integralsObj.I1D_VEC(tmp_k_integral_outer,delta,"simpson");
    }
    // Summing over the bosonic Matsubara frequencies
    std::for_each(tmp_iqn_integral.begin(),tmp_iqn_integral.end(),[&](T n){ return lower_val+=n; });
    lower_val *= _U/_beta/(2.0*M_PI)/(2.0*M_PI);
    lower_val += 1.0;
    lower_val = _U/lower_val;

    return lower_val;
}

template< class T >
std::vector< MPIData > IPT2::OneLadder< T >::operator()(size_t n_k_tilde_x, size_t n_k_tilde_y, double qqx, double qqy) const noexcept(false){
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
    const double delta = 2.0*M_PI/(_k_t_b.size()-1);
    arma::Cube< T > Gamma_n_tilde_n_bar(NI,NI,_iqn.size()); // Doesn't depend on iq_n
    arma::Mat< T > GG_n_tilde_n_bar_jj(NI,NI), GG_n_tilde_n_bar_szsz(NI,NI);
    // containers to integrate over k_bar_i's
    std::vector< T > tmp_k_integral_inner_jj(_k_t_b.size()), tmp_k_integral_outer_jj(_k_t_b.size());
    std::vector< T > tmp_k_integral_inner_szsz(_k_t_b.size()), tmp_k_integral_outer_szsz(_k_t_b.size());
    T jj_resp_iqn{0.0}, szsz_resp_iqn{0.0};
    for (size_t n_em=0; n_em<_iqn.size(); n_em++){
        clock_t begin = clock();
        // numerator
        for (size_t n_bar=0; n_bar<NI; n_bar++){
            std::cout << "n_bar: " << n_bar << std::endl;
            for (size_t n_tilde=0; n_tilde<NI; n_tilde++){
                for (size_t n_k_bar_x=0; n_k_bar_x<_k_t_b.size(); n_k_bar_x++){
                    for (size_t n_k_bar_y=0; n_k_bar_y<_k_t_b.size(); n_k_bar_y++){
                        // denominator
                        Gamma_n_tilde_n_bar.at(n_tilde,n_bar,n_em) = Gamma(_k_t_b[n_k_tilde_x],_k_t_b[n_k_tilde_y],_k_t_b[n_k_bar_x],_k_t_b[n_k_bar_y],qqx,qqy,_splInlineobj._iwn_array[n_bar],_splInlineobj._iwn_array[n_tilde],_iqn[n_em]);
                        // jj for nearest-neighbour hoppings
                        tmp_k_integral_inner_jj[n_k_bar_y] = -1.0*velocity(_k_t_b[n_k_tilde_x])*velocity(_k_t_b[n_k_bar_x]
                        )*getGreen(_k_t_b[n_k_tilde_x],_k_t_b[n_k_tilde_y],_splInlineobj._iwn_array[n_tilde]
                        )*getGreen(_k_t_b[n_k_tilde_x]-qqx,_k_t_b[n_k_tilde_y]-qqy,_splInlineobj._iwn_array[n_tilde]-_iqn[n_em]
                        )*Gamma_n_tilde_n_bar.at(n_tilde,n_bar,n_em
                        )*getGreen(_k_t_b[n_k_bar_x],_k_t_b[n_k_bar_y],_splInlineobj._iwn_array[n_bar]
                        )*getGreen(_k_t_b[n_k_bar_x]+qqx,_k_t_b[n_k_bar_y]+qqy,_splInlineobj._iwn_array[n_bar]+_iqn[n_em]);

                        // szsz
                        tmp_k_integral_inner_szsz[n_k_bar_y] = getGreen(_k_t_b[n_k_tilde_x],_k_t_b[n_k_tilde_y],_splInlineobj._iwn_array[n_tilde]
                        )*getGreen(_k_t_b[n_k_tilde_x]-qqx,_k_t_b[n_k_tilde_y]-qqy,_splInlineobj._iwn_array[n_tilde]-_iqn[n_em]
                        )*Gamma_n_tilde_n_bar.at(n_tilde,n_bar,n_em
                        )*getGreen(_k_t_b[n_k_bar_x],_k_t_b[n_k_bar_y],_splInlineobj._iwn_array[n_bar]
                        )*getGreen(_k_t_b[n_k_bar_x]+qqx,_k_t_b[n_k_bar_y]+qqy,_splInlineobj._iwn_array[n_bar]+_iqn[n_em]);

                    }
                    tmp_k_integral_outer_jj[n_k_bar_x] = 1.0/(2.0*M_PI)*integralsObj.I1D_VEC(tmp_k_integral_inner_jj,delta,"simpson");
                    tmp_k_integral_outer_szsz[n_k_bar_x] = 1.0/(2.0*M_PI)*integralsObj.I1D_VEC(tmp_k_integral_inner_szsz,delta,"simpson");
                }
                GG_n_tilde_n_bar_jj.at(n_tilde,n_bar) = 1.0/(2.0*M_PI)*integralsObj.I1D_VEC(tmp_k_integral_outer_jj,delta,"simpson");
                GG_n_tilde_n_bar_szsz.at(n_tilde,n_bar) = 1.0/(2.0*M_PI)*integralsObj.I1D_VEC(tmp_k_integral_outer_szsz,delta,"simpson");
            }
        }
        jj_resp_iqn = (1.0/_beta/_beta)*arma::accu(GG_n_tilde_n_bar_jj);
        szsz_resp_iqn = (1.0/_beta/_beta)*arma::accu(GG_n_tilde_n_bar_szsz);
        MPIData mpi_data_tmp { n_k_tilde_x, n_k_tilde_y, jj_resp_iqn, szsz_resp_iqn }; // summing over the internal ikn_tilde and ikn_bar
        GG_iqn.push_back(static_cast<MPIData&&>(mpi_data_tmp));
        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "loop n_em: " << n_em << " done in " << elapsed_secs << " secs.." << "\n";
    }
    std::cout << "After the loop.." << std::endl;
    
    return GG_iqn;
}
#else
template< class T >
T IPT2::OneLadder< T >::Gamma(int k_tilde_x, int k_tilde_y, int k_bar_x, int k_bar_y, int qqx, int qqy, T ikn_bar, T ikn_tilde, T iqn) const noexcept(false){
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
    const Integrals integralsObj;
    const double delta = 2.0*M_PI/(double)(_splInlineobj._k_array.size()-1);
    std::vector< T > tmp_k_integral_inner(_splInlineobj._k_array.size()), tmp_k_integral_outer(_splInlineobj._k_array.size());
    std::vector< T > tmp_iqn_integral(_iqn_tilde.size());
    T integrated_ky;
    for (size_t j=0; j<_iqn_tilde.size(); j++){
        for (int kx=0; kx<static_cast<int>(_splInlineobj._k_array.size()); kx++){
            for (int ky=0; ky<static_cast<int>(_splInlineobj._k_array.size()); ky++){
                tmp_k_integral_inner[ky] = getGreen_finer(k_tilde_x,-kx,k_tilde_y,-ky,ikn_tilde-_iqn_tilde[j])*getGreen_finer(k_bar_x,-kx+qqx,k_bar_y,-ky+qqy,ikn_bar-_iqn_tilde[j]+iqn); // qqx and qqy are 0
            }
            integrated_ky = integralsObj.I1D_VEC(tmp_k_integral_inner,delta,"simpson");
            tmp_k_integral_outer[kx] = integrated_ky;
        }
        // to sum over Matsubara frequencies
        tmp_iqn_integral[j] = integralsObj.I1D_VEC(tmp_k_integral_outer,delta,"simpson");
    }
    // Summing over the bosonic Matsubara frequencies
    std::for_each(tmp_iqn_integral.begin(),tmp_iqn_integral.end(),[&](T n){ return lower_val+=n; });
    lower_val *= _U/_beta/(2.0*M_PI)/(2.0*M_PI);
    lower_val += 1.0;
    lower_val = _U/lower_val;

    return lower_val;
}

template< class T >
std::vector< MPIData > IPT2::OneLadder< T >::operator()(int n_k_tilde_x, int n_k_tilde_y, int qqx, int qqy) const noexcept(false){
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
    const double delta = 2.0*M_PI/(double)(_k_t_b.size()-1);
    arma::Cube< T > Gamma_n_tilde_n_bar(NI,NI,_iqn.size()); // Doesn't depend on iq_n
    arma::Mat< T > GG_n_tilde_n_bar_jj(NI,NI), GG_n_tilde_n_bar_szsz(NI,NI);
    // containers to integrate over k_bar_i's
    std::vector< T > tmp_k_integral_inner_jj(_k_t_b.size()), tmp_k_integral_outer_jj(_k_t_b.size());
    std::vector< T > tmp_k_integral_inner_szsz(_k_t_b.size()), tmp_k_integral_outer_szsz(_k_t_b.size());
    T jj_resp_iqn{0.0}, szsz_resp_iqn{0.0};
    for (size_t n_em=0; n_em<_iqn.size(); n_em++){
        clock_t begin = clock();
        // numerator
        for (size_t n_bar=0; n_bar<NI; n_bar++){
            std::cout << "n_bar: " << n_bar << std::endl;
            for (size_t n_tilde=0; n_tilde<NI; n_tilde++){
                for (int n_k_bar_x=0; n_k_bar_x<static_cast<int>(_k_t_b.size()); n_k_bar_x++){
                    for (int n_k_bar_y=0; n_k_bar_y<static_cast<int>(_k_t_b.size()); n_k_bar_y++){
                        // denominator
                        Gamma_n_tilde_n_bar.at(n_tilde,n_bar,n_em) = Gamma(n_k_tilde_x,n_k_tilde_y,n_k_bar_x,n_k_bar_y,qqx,qqy,_splInlineobj._iwn_array[n_bar],_splInlineobj._iwn_array[n_tilde],_iqn[n_em]);
                        // jj for nearest-neighbour hoppings
                        tmp_k_integral_inner_jj[n_k_bar_y] = -1.0*velocity(n_k_tilde_x)*velocity(n_k_bar_x
                        )*getGreen(n_k_tilde_x,n_k_tilde_y,_splInlineobj._iwn_array[n_tilde]
                        )*getGreen(n_k_tilde_x-qqx,n_k_tilde_y-qqy,_splInlineobj._iwn_array[n_tilde]-_iqn[n_em]
                        )*Gamma_n_tilde_n_bar.at(n_tilde,n_bar,n_em
                        )*getGreen(n_k_bar_x,n_k_bar_y,_splInlineobj._iwn_array[n_bar]
                        )*getGreen(n_k_bar_x+qqx,n_k_bar_y+qqy,_splInlineobj._iwn_array[n_bar]+_iqn[n_em]);

                        // szsz
                        tmp_k_integral_inner_szsz[n_k_bar_y] = getGreen(n_k_tilde_x,n_k_tilde_y,_splInlineobj._iwn_array[n_tilde]
                        )*getGreen(n_k_tilde_x-qqx,n_k_tilde_y-qqy,_splInlineobj._iwn_array[n_tilde]-_iqn[n_em]
                        )*Gamma_n_tilde_n_bar.at(n_tilde,n_bar,n_em
                        )*getGreen(n_k_bar_x,n_k_bar_y,_splInlineobj._iwn_array[n_bar]
                        )*getGreen(n_k_bar_x+qqx,n_k_bar_y+qqy,_splInlineobj._iwn_array[n_bar]+_iqn[n_em]);

                    }
                    tmp_k_integral_outer_jj[n_k_bar_x] = 1.0/(2.0*M_PI)*integralsObj.I1D_VEC(tmp_k_integral_inner_jj,delta,"simpson");
                    tmp_k_integral_outer_szsz[n_k_bar_x] = 1.0/(2.0*M_PI)*integralsObj.I1D_VEC(tmp_k_integral_inner_szsz,delta,"simpson");
                }
                GG_n_tilde_n_bar_jj.at(n_tilde,n_bar) = 1.0/(2.0*M_PI)*integralsObj.I1D_VEC(tmp_k_integral_outer_jj,delta,"simpson");
                GG_n_tilde_n_bar_szsz.at(n_tilde,n_bar) = 1.0/(2.0*M_PI)*integralsObj.I1D_VEC(tmp_k_integral_outer_szsz,delta,"simpson");
            }
        }
        jj_resp_iqn = (2.0/_beta/_beta)*arma::accu(GG_n_tilde_n_bar_jj);
        szsz_resp_iqn = (2.0/_beta/_beta)*arma::accu(GG_n_tilde_n_bar_szsz);
        MPIData mpi_data_tmp { n_k_tilde_x, n_k_tilde_y, jj_resp_iqn, szsz_resp_iqn }; // summing over the internal ikn_tilde and ikn_bar
        GG_iqn.push_back(static_cast<MPIData&&>(mpi_data_tmp));
        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "loop n_em: " << n_em << " done in " << elapsed_secs << " secs.." << "\n";
    }
    std::cout << "After the loop.." << std::endl;
    
    return GG_iqn;
}
#endif



namespace IPT2{

    #ifndef TRIGTABLES
    inline double velocity(double k) noexcept{
        /* This method computes the current vertex in the case of a 1D nearest-neighbour dispersion relation.

        Parameters:
            k (double): k-point.

        Returns:
            (double): current vertex.
        */
        return 2.0*std::sin(k);
    }
    #else
    inline double velocity(int k) noexcept{
        /* This method computes the current vertex in the case of a 1D nearest-neighbour dispersion relation.

        Parameters:
            k (double): k-point.

        Returns:
            (double): current vertex.
        */
        return 2.0*sine_table[k];
    }
    #endif

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
        #ifndef TRIGTABLES
        for (size_t k_t=0; k_t<N_q; k_t++){
            for (size_t k_b=0; k_b<N_q; k_b++){
                MPIData mpi_data{ k_t, k_b, std::complex<double>(0.0,0.0), std::complex<double>(0.0,0.0) };
                vec_to_processes->push_back(static_cast<MPIData&&>(mpi_data));
            }
        }
        #else   
        for (int k_t=0; k_t<static_cast<int>(N_q); k_t++){
            for (int k_b=0; k_b<static_cast<int>(N_q); k_b++){
                MPIData mpi_data{ k_t, k_b, std::complex<double>(0.0,0.0), std::complex<double>(0.0,0.0) };
                vec_to_processes->push_back(static_cast<MPIData&&>(mpi_data));
            }
        }
        #endif
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
        MPI_Aint offsets[4]={ offsetof(MPIData,k_tilde_x), offsetof(MPIData,k_tilde_y), offsetof(MPIData,cplx_data_jj), offsetof(MPIData,cplx_data_szsz) };
        #ifndef TRIGTABLES
        MPI_Datatype types[4]={ MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_CXX_DOUBLE_COMPLEX, MPI_CXX_DOUBLE_COMPLEX }, tmp_type;
        #else
        MPI_Datatype types[4]={ MPI_INT, MPI_INT, MPI_CXX_DOUBLE_COMPLEX, MPI_CXX_DOUBLE_COMPLEX }, tmp_type;
        #endif
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
}

inline std::tuple<int,int> inverse_Cantor_pairing(int number){
    // see https://en.wikipedia.org/wiki/Pairing_function for notation and explanation
    int w = (int)std::floor( ( std::sqrt( 8.0*number + 1.0 ) - 1.0 ) / 2.0 );
    int t = static_cast<int>( (w)*(w+1)/2 );
    int n_k_tilde = number - t;
    int n_k_bar = w - n_k_tilde;

    return std::make_tuple( n_k_bar, n_k_tilde );
}