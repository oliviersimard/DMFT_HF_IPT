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

typedef struct{
    size_t k_tilde;
    size_t k_bar;
    std::complex<double> cplx_data;
} MPIData;

struct MPIDataReceive{
    MPIData* data_struct;
    size_t size;
};

// Adding methods to IPT2 namespace for this particular translation unit
namespace IPT2{
    std::vector<double> get_iwn_to_tau(const std::vector< std::complex<double> >& F_iwn, double beta, std::string obj="Green") noexcept;
    template<typename T> std::vector< T > generate_random_numbers(size_t arr_size, T min, T max) noexcept;
    inline double velocity(double k) noexcept;
    void set_vector_processes(std::vector<MPIData>*,unsigned int N_q) noexcept;
    void create_mpi_data_struct(MPI_Datatype& custom_type);

    template< class T >
    class OneLadder{
        public:
            OneLadder& operator=(const OneLadder&) = delete;
            OneLadder(const OneLadder&) = delete;
            std::vector< MPIData > operator()(size_t n_k_bar, size_t n_k_tilde, double qq=0.0) const noexcept;
            OneLadder()=default;
            explicit OneLadder(const SplineInline< T >& splInlineobj, const std::vector< T >& iqn_tilde, const std::vector< T >& iqn, const std::vector<double>& k_arr, double mu, double U, double beta) : _splInlineobj(splInlineobj), _iqn_tilde(iqn_tilde), _iqn(iqn), _k_t_b(k_arr){
                this->_mu = mu;
                this->_U = U;
                this->_beta = beta;
            };

            inline T getGreen(double k, T iwn) const noexcept;
            T Gamma(double k_bar, double k_tilde, T ikn_bar, T ikn_tilde) const noexcept;
        
        private:
            // using SplineInline< T >::_k_array; // if inheriting public SplineInline< T >
            // using SplineInline< T >::_iwn_array;
            const SplineInline< T >& _splInlineobj;
            const std::vector< T >& _iqn_tilde;
            const std::vector< T >& _iqn;
            const std::vector<double>& _k_t_b;
            double _mu {0.0}, _U {0.0}, _beta {0.0};

    };

    template< class T >
    class InfiniteLadders : OneLadder< T > {
        public:
            explicit InfiniteLadders(const SplineInline< T >& splInlineobj, const std::vector< T >& iqn_tilde, const std::vector< T >& iqn, double mu, double U, double beta) : OneLadder< T >(splInlineobj,iqn_tilde,iqn,mu,U,beta){};

        private:
            using OneLadder< T >::getGreen;
            using OneLadder< T >::Gamma;
    };

}

template< class T >
inline T IPT2::OneLadder< T >::getGreen(double k, T iwn) const noexcept{
    return 1.0 / ( iwn + _mu - epsilonk(k) - _splInlineobj.calculateSpline( iwn.imag() ) );
}

template< class T >
T IPT2::OneLadder< T >::Gamma(double k_bar, double k_tilde, T ikn_bar, T ikn_tilde) const noexcept{
    T lower_val(0.0,0.0);
    for (size_t l=0; l<_splInlineobj._k_array.size(); l++){
        if ( (l==0) || (l==(_splInlineobj._k_array.size()-1)) ){
            for (size_t j=0; j<_iqn_tilde.size(); j++){
                lower_val += 0.5*getGreen(k_bar-_splInlineobj._k_array[l],ikn_bar-_iqn_tilde[j])*getGreen(k_tilde-_splInlineobj._k_array[l],ikn_tilde-_iqn_tilde[j]);
            }
        } else{
            for (size_t j=0; j<_iqn_tilde.size(); j++){
                lower_val += getGreen(k_bar-_splInlineobj._k_array[l],ikn_bar-_iqn_tilde[j])*getGreen(k_tilde-_splInlineobj._k_array[l],ikn_tilde-_iqn_tilde[j]);
            }
        }
    }
    lower_val*=_U/_beta/_splInlineobj._k_array.size();
    lower_val+=1.0;
    lower_val = _U/lower_val;

    return lower_val;
}

template< class T >
std::vector< MPIData > IPT2::OneLadder< T >::operator()(size_t n_k_bar, size_t n_k_tilde, double qq) const noexcept{
    std::vector< MPIData > cubic_spline_GG_iqn;
    // Computing Gamma
    arma::Mat< T > Gamma_n_bar_n_tilde(_splInlineobj._iwn_array.size(),_splInlineobj._iwn_array.size()); // Doesn't depend on iq_n
    for (size_t n_bar=0; n_bar<_splInlineobj._iwn_array.size(); n_bar++){
        clock_t begin = clock();
        for (size_t n_tilde=0; n_tilde<_splInlineobj._iwn_array.size(); n_tilde++){
            Gamma_n_bar_n_tilde(n_bar,n_tilde) = Gamma(_k_t_b[n_k_bar],_k_t_b[n_k_tilde],_splInlineobj._iwn_array[n_bar],_splInlineobj._iwn_array[n_tilde]);
        }
        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "outer loop n_bar: " << n_bar << " done in " << elapsed_secs << " secs.." << "\n";
    }

    arma::Mat< T > GG_n_bar_n_tilde(_splInlineobj._iwn_array.size(),_splInlineobj._iwn_array.size());
    for (size_t em=0; em<_iqn.size(); em++){
        for (size_t n_bar=0; n_bar<_splInlineobj._iwn_array.size(); n_bar++){
            for (size_t n_tilde=0; n_tilde<_splInlineobj._iwn_array.size(); n_tilde++){
                // This part remains to be done....
                GG_n_bar_n_tilde(n_bar,n_tilde) = getGreen(_k_t_b[n_k_tilde],_splInlineobj._iwn_array[n_tilde])*getGreen(_k_t_b[n_k_tilde]+qq,_splInlineobj._iwn_array[n_tilde]+_iqn[em])*Gamma_n_bar_n_tilde(n_bar,n_tilde)*getGreen(_k_t_b[n_k_bar],_splInlineobj._iwn_array[n_bar])*getGreen(_k_t_b[n_k_bar]+qq,_splInlineobj._iwn_array[n_bar]+_iqn[em]);
            }
        }
        MPIData mpi_data_tmp { n_k_tilde, n_k_bar, (1.0/_beta/_beta)*arma::accu(GG_n_bar_n_tilde) }; // summing over the internal ikn_tilde and ikn_bar
        cubic_spline_GG_iqn.push_back(static_cast<MPIData&&>(mpi_data_tmp));
    }
    std::cout << "After the loop.." << std::endl;
    
    return cubic_spline_GG_iqn;
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
        for (size_t k_t=0; k_t<N_q; k_t++){
            for (size_t k_b=0; k_b<N_q; k_b++){
                MPIData mpi_data{ k_t, k_b, std::complex<double>(0.0,0.0) };
                vec_to_processes->push_back(static_cast<MPIData&&>(mpi_data));
            }
        }
    }

    void create_mpi_data_struct(MPI_Datatype& custom_type){
        int lengths[3]={ 1, 1, 1 };
        MPI_Aint offsets[3]={ offsetof(MPIData,k_tilde), offsetof(MPIData,k_bar), offsetof(MPIData,cplx_data) };
        MPI_Datatype types[3]={ MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_CXX_DOUBLE_COMPLEX }, tmp_type;
        MPI_Type_create_struct(3,lengths,offsets,types,&tmp_type);
        // Proper padding
        MPI_Type_create_resized(tmp_type, 0, sizeof(MPIData), &custom_type);
        MPI_Type_commit(&custom_type);
    }
}