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
#define SHIFT_TO_DIFFERENTIATE_TAGS 1000000 // To differentiate from the tags using Cantor function

//static bool slaves_can_write_in_file = false; // This prevents that the slave processes
static int root_process = 0;

//#define INFINITE

typedef struct{
    size_t k_tilde_m_k_bar_x;
    size_t k_tilde_m_k_bar_y;
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
            std::vector< MPIData > operator()(size_t n_k_tilde_m_k_bar_x, size_t n_k_tilde_m_k_bar_y, bool is_jj, bool is_single_ladder_precomputed=false, void* arma_ptr=nullptr, double qqx=0.0, double qqy=0.0) const noexcept(false);
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
            inline T getGreen(double kx, double ky, T iwn) const noexcept;
            T Gamma(double k_tilde_m_k_bar_x, double k_tilde_m_k_bar_y, T ikn_bar, T ikn_tilde) const noexcept(false);

        private:
            const std::vector< T >& _iqn_tilde;

    };
    // contains tags to dicriminate matrices contained in mat_sl_vec (ordered arrangement)
    template< class T > std::vector<int> OneLadder< T >::tag_vec = {};
    //template< class T > std::vector< T** > OneLadder< T >::mat_sl_vec = {};

}

template< class T >
class ArmaMPI{
    /* This specific class sends T-typed Armadillo matrices across MPI processes */
    friend class IPT2::OneLadder< T >;
    public:
        explicit ArmaMPI(size_t n_rows, size_t n_cols);
        ~ArmaMPI();
        void send_Arma_mat_MPI(arma::Mat< T >& mat,int dest, int tag) const noexcept;
        arma::Mat< T > recv_Arma_mat_MPI(int tag, int src) const noexcept;
        void fill_TArr(arma::Mat< T >& mat) const noexcept;

    private:
        T** _TArr;
        T* _Tmemptr;
        size_t _n_rows;
        size_t _n_cols;
        //MPI_Datatype _cplx_custom_t;
};

template< class T >
ArmaMPI< T >::ArmaMPI(size_t n_rows, size_t n_cols) : _n_rows(n_rows),_n_cols(n_cols){
    T* data = new T[_n_rows*_n_cols];
    _TArr = new T*[_n_rows];
    for (size_t i=0; i<_n_rows; i++){
        _TArr[i] = &data[i*_n_cols];
    }
    // constructing the MPI datatype, even if not needed..
    //IPT2::create_mpi_data_struct_cplx(_cplx_custom_t);
    _Tmemptr = new T[_n_rows*_n_cols];
}

template< class T >
ArmaMPI< T >::~ArmaMPI(){
    delete[] _TArr[0]; // Effectively deletes data*
    delete[] _TArr;
    delete[] _Tmemptr;
    //MPI_Type_free(&_cplx_custom_t);
}

template< class T >
void ArmaMPI< T >::send_Arma_mat_MPI(arma::Mat< T >& mat,int dest,int tag) const noexcept{
    for (size_t i=0; i<_n_rows; i++){
        for (size_t j=0; j<_n_cols; j++){
            _TArr[i][j] = mat(i,j);
        }
    }
    // Contiguous memory
    MPI_Send(&(_TArr[0][0]),_n_cols*_n_rows*sizeof(T),MPI_BYTE,dest,tag,MPI_COMM_WORLD);
}

template<>
inline void ArmaMPI< std::complex<double> >::send_Arma_mat_MPI(arma::Mat< std::complex<double> >& mat,int dest,int tag) const noexcept{
    for (size_t i=0; i<_n_rows; i++){
        for (size_t j=0; j<_n_cols; j++){
            _TArr[i][j] = mat(i,j);
        }
    }
    MPI_Send(&(_TArr[0][0]),_n_cols*_n_rows,MPI_CXX_DOUBLE_COMPLEX,dest,tag,MPI_COMM_WORLD);
}

template< class T >
arma::Mat< T > ArmaMPI< T >::recv_Arma_mat_MPI(int tag, int src) const noexcept{
    arma::Mat< T > returned_mat(_n_rows,_n_cols);
    MPI_Recv(&(_TArr[0][0]),_n_cols*_n_rows*sizeof(T),MPI_BYTE,src,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    
    for (size_t i=0; i<_n_rows; i++){
        for (size_t j=0; j<_n_cols; j++){
            returned_mat(i,j) = _TArr[i][j];
        }
    }

    return returned_mat;
}

template<>
inline arma::Mat< std::complex<double> > ArmaMPI< std::complex<double> >::recv_Arma_mat_MPI(int tag, int src) const noexcept{
    arma::Mat< std::complex<double> > returned_mat(_n_rows,_n_cols);
    MPI_Recv(_Tmemptr,_n_cols*_n_rows,MPI_CXX_DOUBLE_COMPLEX,src,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    
    for (size_t i=0; i<_n_rows; i++){
        for (size_t j=0; j<_n_cols; j++){
            returned_mat(i,j) = _Tmemptr[i*_n_cols+j];
        }
    }

    return returned_mat;
}

template< class T >
inline void ArmaMPI< T >::fill_TArr(arma::Mat< T >& mat) const noexcept{
    for (size_t i=0; i<_n_rows; i++){
        for (size_t j=0; j<_n_cols; j++){
            _TArr[i][j] = mat(i,j);
        }
    }
}

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

template< class T >
T IPT2::OneLadder< T >::Gamma(double k_tilde_m_k_bar_x, double k_tilde_m_k_bar_y, T ikn_bar, T ikn_tilde) const noexcept(false){
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
    for (size_t j=0; j<_iqn_tilde.size(); j++){
        for (size_t kx=0; kx<_splInlineobj._k_array.size(); kx++){
            for (size_t ky=0; ky<_splInlineobj._k_array.size(); ky++){
                tmp_k_integral_inner[ky] = getGreen(_splInlineobj._k_array[kx]+k_tilde_m_k_bar_x,_splInlineobj._k_array[ky]+k_tilde_m_k_bar_y,ikn_tilde-_iqn_tilde[j])*getGreen(_splInlineobj._k_array[kx],_splInlineobj._k_array[ky],ikn_bar-_iqn_tilde[j]);
            }
            T integrated_ky = integralsObj.I1D_VEC(tmp_k_integral_inner,delta,"simpson");
            tmp_k_integral_outer[kx] = integrated_ky;
        }
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
std::vector< MPIData > IPT2::OneLadder< T >::operator()(size_t n_k_tilde_m_k_bar_x, size_t n_k_tilde_m_k_bar_y, bool is_jj, bool is_single_ladder_precomputed, void* arma_ptr, double qqx, double qqy) const noexcept(false){
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
    const Integrals integralsObj;
    const double delta = 2.0*M_PI/(_k_t_b.size()-1);
    arma::Mat< T > Gamma_n_bar_n_tilde(NI,NI); // Doesn't depend on iq_n
    
    for (size_t n_bar=0; n_bar<NI; n_bar++){
        clock_t begin = clock();
        for (size_t n_tilde=0; n_tilde<NI; n_tilde++){
            Gamma_n_bar_n_tilde(n_bar,n_tilde) = Gamma(_k_t_b[n_k_tilde_m_k_bar_x],_k_t_b[n_k_tilde_m_k_bar_y],_splInlineobj._iwn_array[n_bar],_splInlineobj._iwn_array[n_tilde]);
        }
        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "outer loop n_bar: " << n_bar << " done in " << elapsed_secs << " secs.." << "\n";
    }
    if (is_single_ladder_precomputed){
        // Come up with UNIQUE way to attribute the tags to different Gamma matrices using Cantor pairing function
        int tag = (int)( ((n_k_tilde_m_k_bar_x+n_k_tilde_m_k_bar_y)*(n_k_tilde_m_k_bar_x+n_k_tilde_m_k_bar_y+1))/2 ) + (int)n_k_tilde_m_k_bar_y;
        std::cout << "tag: " << tag << std::endl;
        tag_vec.push_back(tag);
        *( static_cast< arma::Mat< T >* >(arma_ptr) ) = Gamma_n_bar_n_tilde;
        // for (size_t i=0; i<_splInlineobj._iwn_array.size(); i++){
        //     std::cout << "el after: " << sl_vec[0][3*_splInlineobj._iwn_array.size()+i] << std::endl;
        // }
    }

    arma::Mat< T > GG_n_bar_n_tilde(NI,NI);
    std::vector< T > tmp_k_integral_inner(_k_t_b.size()), tmp_k_integral_outer(_k_t_b.size());
    for (size_t em=0; em<_iqn.size(); em++){
        for (size_t n_bar=0; n_bar<NI; n_bar++){
            for (size_t n_tilde=0; n_tilde<NI; n_tilde++){
                if (is_jj){
                    for (size_t n_k_bar_x=0; n_k_bar_x<_k_t_b.size(); n_k_bar_x++){
                        for (size_t n_k_bar_y=0; n_k_bar_y<_k_t_b.size(); n_k_bar_y++){
                            // This part remains to be done....
                            tmp_k_integral_inner[n_k_bar_y] = getGreen(_k_t_b[n_k_tilde_m_k_bar_x]+_k_t_b[n_k_bar_x],_k_t_b[n_k_tilde_m_k_bar_y]+_k_t_b[n_k_bar_y],_splInlineobj._iwn_array[n_tilde]
                            )*getGreen(_k_t_b[n_k_tilde_m_k_bar_x]+_k_t_b[n_k_bar_x]+qqx,_k_t_b[n_k_tilde_m_k_bar_y]+_k_t_b[n_k_bar_y]+qqy,_splInlineobj._iwn_array[n_tilde]+_iqn[em]
                            )*Gamma_n_bar_n_tilde(n_bar,n_tilde
                            )*getGreen(_k_t_b[n_k_tilde_m_k_bar_x],_k_t_b[n_k_tilde_m_k_bar_y],_splInlineobj._iwn_array[n_bar]
                            )*getGreen(_k_t_b[n_k_tilde_m_k_bar_x]+qqx,_k_t_b[n_k_tilde_m_k_bar_y]+qqy,_splInlineobj._iwn_array[n_bar]+_iqn[em]);
                        }
                        tmp_k_integral_outer[n_k_bar_x] = -1.0/(2.0*M_PI)*velocity(_k_t_b[n_k_tilde_m_k_bar_x]+_k_t_b[n_k_bar_x])*velocity(_k_t_b[n_k_bar_x])*integralsObj.I1D_VEC(tmp_k_integral_inner,delta,"simpson");
                    }
                    GG_n_bar_n_tilde(n_bar,n_tilde) = 1.0/(2.0*M_PI)*integralsObj.I1D_VEC(tmp_k_integral_outer,delta,"simpson");
                } else{
                    for (size_t n_k_bar_x=0; n_k_bar_x<_k_t_b.size(); n_k_bar_x++){
                        for (size_t n_k_bar_y=0; n_k_bar_y<_k_t_b.size(); n_k_bar_y++){
                            // This part remains to be done....
                            tmp_k_integral_inner[n_k_bar_y] = getGreen(_k_t_b[n_k_tilde_m_k_bar_x]+_k_t_b[n_k_bar_x],_k_t_b[n_k_tilde_m_k_bar_y]+_k_t_b[n_k_bar_y],_splInlineobj._iwn_array[n_tilde]
                            )*getGreen(_k_t_b[n_k_tilde_m_k_bar_x]+_k_t_b[n_k_bar_x]+qqx,_k_t_b[n_k_tilde_m_k_bar_y]+_k_t_b[n_k_bar_y]+qqy,_splInlineobj._iwn_array[n_tilde]+_iqn[em]
                            )*Gamma_n_bar_n_tilde(n_bar,n_tilde
                            )*getGreen(_k_t_b[n_k_tilde_m_k_bar_x],_k_t_b[n_k_tilde_m_k_bar_y],_splInlineobj._iwn_array[n_bar]
                            )*getGreen(_k_t_b[n_k_tilde_m_k_bar_x]+qqx,_k_t_b[n_k_tilde_m_k_bar_y]+qqy,_splInlineobj._iwn_array[n_bar]+_iqn[em]);
                        }
                        tmp_k_integral_outer[n_k_bar_x] = 1.0/(2.0*M_PI)*integralsObj.I1D_VEC(tmp_k_integral_inner,delta,"simpson");
                    }
                    GG_n_bar_n_tilde(n_bar,n_tilde) = 1.0/(2.0*M_PI)*integralsObj.I1D_VEC(tmp_k_integral_outer,delta,"simpson");
                }
            }
        }
        MPIData mpi_data_tmp { n_k_tilde_m_k_bar_y, n_k_tilde_m_k_bar_x, (1.0/_beta/_beta)*arma::accu(GG_n_bar_n_tilde) }; // summing over the internal ikn_tilde and ikn_bar
        GG_iqn.push_back(static_cast<MPIData&&>(mpi_data_tmp));
    }
    std::cout << "After the loop.." << std::endl;
    
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
        MPI_Aint offsets[3]={ offsetof(MPIData,k_tilde_m_k_bar_x), offsetof(MPIData,k_tilde_m_k_bar_y), offsetof(MPIData,cplx_data) };
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