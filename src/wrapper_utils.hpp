#ifndef Wrapper_utils_H_
#define Wrapper_utils_H_

// #define ARMA_ALLOW_FAKE_GCC
// #define ARMA_NO_DEBUG // to disable bound checks

#define NCA
// #define DEBUG
// #define SUS // Enables only methods relevant when calculating the susceptibility

#include <iostream>
#include <complex>
#include <armadillo>
#include <mpi.h>

namespace IPT2{ template<class T> class OneLadder; };
#ifndef SUS
/* Template structure to call functions in classes. */
template<typename T, typename C, typename Q>
struct functorStruct{
    //using matCplx = arma::Mat< std::complex<double> >;
    using funct_init_t = arma::Mat< std::complex<double> > (C::*)(Q model, T kk, T qq, int n, int l);
    using funct_con_t = arma::Mat< std::complex<double> > (C::*)(Q model, T kk, T qq, int n, int l, arma::Mat< std::complex<double> > SE);

    functorStruct(funct_init_t initFunct, funct_con_t conFunct);
    arma::Mat< std::complex<double> > callInitFunct(C& obj, Q model, T kk, T qq, int n, int l);
    arma::Mat< std::complex<double> > callConFunct(C& obj, Q model, T kk, T qq, int n, int l, arma::Mat< std::complex<double> > SE);

    private:
        funct_init_t _initFunct;
        funct_con_t _conFunct;
};

template<typename T, typename C, typename Q>
functorStruct<T,C,Q>::functorStruct(funct_init_t initFunct, funct_con_t conFunct) : _initFunct(initFunct), _conFunct(conFunct){};

template<typename T, typename C, typename Q>
arma::Mat< std::complex<double> > functorStruct<T,C,Q>::callInitFunct(C& obj, Q model, T kk, T qq, int n, int l){
    return (obj.*_initFunct)(model, kk, qq, n, l);
}

template<typename T, typename C, typename Q>
arma::Mat< std::complex<double> > functorStruct<T,C,Q>::callConFunct(C& obj, Q model, T kk, T qq, int n, int l, arma::Mat< std::complex<double> > SE){
    return (obj.*_conFunct)(model, kk, qq, n, l, SE);
}

template<typename> struct holder;

template<typename R, typename... Ts> struct holder<R(Ts...)>{
    R (*my_ptr)(Ts...);
    //
    holder(R(*funct)(Ts...)) : my_ptr(funct) {};
    R operator()(Ts... args){
        return my_ptr(args...);
    }
    holder& operator=( R (*f) (Ts...) ){
        my_ptr = f;
        return *this;
    }
};

template<typename Fn, Fn fn, typename... Args>
typename std::result_of<Fn(Args...)>::type
wrapper(Args&&... args) {
    return fn(std::forward<Args>(args)...);
}
#define WRAPPER(FUNC) wrapper<decltype(&FUNC), &FUNC>

template <class... Ts> struct tuple {};

template <class T, class... Ts>
struct tuple<T, Ts...> : tuple<Ts...> {
  tuple(T t, Ts... ts) : tuple<Ts...>(ts...), tail(t) {}

  T tail;
};

template <std::size_t, class> struct elem_type_holder;

template <class T, class... Ts>
struct elem_type_holder<0, tuple<T, Ts...>> {
  typedef T type;
};

template <std::size_t k, class T, class... Ts>
struct elem_type_holder<k, tuple<T, Ts...>> {
  typedef typename elem_type_holder<k - 1, tuple<Ts...>>::type type;
};

template <std::size_t k, class... Ts>
typename std::enable_if<k == 0, typename elem_type_holder<0, tuple<Ts...>>::type&>::type get(tuple<Ts...>& t) {
  return t.tail;
}

template <std::size_t k, class T, class... Ts>
typename std::enable_if<k != 0, typename elem_type_holder<k, tuple<T, Ts...>>::type&>::type get(tuple<T, Ts...>& t) {
  tuple<Ts...>& base = t;
  return get<k - 1>(base);
}

#endif

template<class T>
class auto_ptr {
    // Custom pointer-wrapper to ensure allocated memory inside objects is destroyed, even if the code crashes or throws in its course.
    public:
        auto_ptr(T* ptr = nullptr) : _m_ptr(ptr){};
        ~auto_ptr(){
            delete _m_ptr;
        }
        // auto_ptr(const auto_ptr&) = delete;
        auto_ptr(const auto_ptr&);
        // auto_ptr& operator=(const auto_ptr&)=delete;
        auto_ptr& operator=(const auto_ptr&);
        // move ctor
        auto_ptr(auto_ptr&&) noexcept;
        // move assignment
        auto_ptr& operator=(auto_ptr&&) noexcept;
        T& operator*() const noexcept;
        T* operator->() const noexcept;
    private:
        T* _m_ptr;
};

template<class T>
inline auto_ptr<T>::auto_ptr(const auto_ptr<T>& src){
    this->_m_ptr = new T;
    *this->_m_ptr = *src._m_ptr;
}

template<class T>
inline auto_ptr<T>& auto_ptr<T>::operator=(const auto_ptr<T>& src){
    if (this==&src){
        return *this;
    } else{
        delete this->_m_ptr;

        this->_m_ptr = new T;
        *(this->_m_ptr) = *(src._m_ptr);

        return *this;
    }
}

template<class T>
inline auto_ptr<T>::auto_ptr(auto_ptr<T>&& src) noexcept{
    this->_m_ptr=src._m_ptr;
    // avoid that destructor destroys data pointed to
    src._m_ptr=nullptr;
}

template<class T>
inline auto_ptr<T>& auto_ptr<T>::operator=(auto_ptr<T>&& src) noexcept{
    if (this == &src){
        return *this;
    } else{
        delete this->_m_ptr;
        this->_m_ptr=src._m_ptr;
        src._m_ptr=nullptr;

        return *this;
    }
}

template<class T>
inline T& auto_ptr<T>::operator*() const noexcept{
    return *_m_ptr;
}

template<class T>
inline T* auto_ptr<T>::operator->() const noexcept{
    return _m_ptr;
}

template< class T >
class ArmaMPI{
    /* This specific class sends T-typed Armadillo matrices across MPI processes */
    friend class IPT2::OneLadder< T >;
    public:
        explicit ArmaMPI(size_t n_rows, size_t n_cols, size_t n_slices=0);
        ~ArmaMPI();
        void send_Arma_mat_MPI(arma::Mat< T >& mat,int dest, int tag) const noexcept;
        arma::Mat< T > recv_Arma_mat_MPI(int tag, int src) const noexcept;
        arma::Cube< T > recv_Arma_cube_MPI(int tag, int src) const noexcept;
        void fill_TArr(arma::Mat< T >& mat) const noexcept;

    private:
        T** _TArr;
        T* _Tmemptr;
        size_t _n_rows{0};
        size_t _n_cols{0};
        size_t _n_slices{0};
        //MPI_Datatype _cplx_custom_t;
};

template< class T >
ArmaMPI< T >::ArmaMPI(size_t n_rows, size_t n_cols, size_t n_slices) : _n_rows(n_rows),_n_cols(n_cols),_n_slices(n_slices){
    if (_n_slices == 0){
        T* data = new T[_n_rows*_n_cols];
        _TArr = new T*[_n_rows];
        for (size_t i=0; i<_n_rows; i++){
            _TArr[i] = &data[i*_n_cols];
        }
        // constructing the MPI datatype, even if not needed..
        //IPT2::create_mpi_data_struct_cplx(_cplx_custom_t);
        _Tmemptr = new T[_n_rows*_n_cols];
        
    } else if (_n_slices>0){
        // constructing the MPI datatype, even if not needed..
        //IPT2::create_mpi_data_struct_cplx(_cplx_custom_t);
        _Tmemptr = new T[_n_rows*_n_cols*_n_slices];
    }
}

template< class T >
ArmaMPI< T >::~ArmaMPI(){
    if (_n_slices==0){
        delete[] _TArr[0]; // Effectively deletes data*
        delete[] _TArr;
    }
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
    
    for (size_t j=0; j<_n_cols; j++){
        for (size_t i=0; i<_n_rows; i++){
            returned_mat(i,j) = _Tmemptr[j*_n_rows+i]; // column-major storage
        }
    }

    return returned_mat;
}

template<>
inline arma::Cube< std::complex<double> > ArmaMPI< std::complex<double> >::recv_Arma_cube_MPI(int tag, int src) const noexcept{
    assert(_n_slices>0);
    arma::Cube< std::complex<double> > returned_cube(_n_rows,_n_cols,_n_slices);
    MPI_Recv(_Tmemptr,_n_cols*_n_rows*_n_slices,MPI_CXX_DOUBLE_COMPLEX,src,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    
    for (size_t k=0; k<_n_slices; k++){ // column-major storage, matrix by matrix
        for (size_t j=0; j<_n_cols; j++){
            for (size_t i=0; i<_n_rows; i++){
                returned_cube(i,j,k) = _Tmemptr[i + j*_n_rows + k*(_n_rows*_n_cols)];
            }
        }
    }

    return returned_cube;
}

template< class T >
inline void ArmaMPI< T >::fill_TArr(arma::Mat< T >& mat) const noexcept{
    for (size_t i=0; i<_n_rows; i++){
        for (size_t j=0; j<_n_cols; j++){
            _TArr[i][j] = mat(i,j);
        }
    }
}

template<typename U, typename T, typename ...Ts>
struct TrigP{
    // Custom class to compute trigonometric functions using a predetermined array instead of always calling std::cos or std::sin.
    // If for example we want to compute cos(k_array[l]-k_array[j]), one cannot simply compute _cos_prec[l-j]. One should rather use this method
    // in the following form: trigObj.COS(l,-j).
    U COS(T first) const noexcept{
        return _cosine_p[first];
    }
    U COS(T first, T second) const noexcept{
        // cos(x+y) = cos(x)*cos(y) - sin(x)*sin(y)
        if ( first < static_cast<T>(0) )
            return _cosine_p[-first]*_cosine_p[second] + _sine_p[-first]*_sine_p[second];
        else if ( second < static_cast<T>(0) )
            return _cosine_p[first]*_cosine_p[-second] + _sine_p[first]*_sine_p[-second];
        else
            return _cosine_p[first]*_cosine_p[second] - _sine_p[first]*_sine_p[second];
    }
    
    // U COS(T first, T second, Ts ...others) const noexcept{
    //     // cos(x+y+z+...) = cos(x)*cos(y+z+...) - sin(x)*sin(y+z+...)
    //     if (sizeof...(others)>0){
    //         if ( first < static_cast<T>(0) )
    //             return _cosine_p[first]*COS(second, others ...)+_sine_p[first]*SIN(second, others ...);
    //         else
    //             return _cosine_p[first]*COS(second, others ...)-_sine_p[first]*SIN(second, others ...);
    //     } else{
    //         return COS(first, second);
    //     }
    // }

    U SIN(T first) const noexcept{
        return _sine_p[first];
    }
    U SIN(T first, T second) const noexcept{
        // sin(x+y) = sin(x)*cos(y) + cos(x)*sin(y)
        if ( first < static_cast<T>(0) )
            return -_cosine_p[second]*_sine_p[first] + _cosine_p[first]*_sine_p[second];
        else if ( second < static_cast<T>(0) )
            return _cosine_p[second]*_sine_p[first] - _cosine_p[first]*_sine_p[second];
        else if ( second < static_cast<T>(0) && first < static_cast<T>(0) )
            return -_cosine_p[second]*_sine_p[first] - _cosine_p[first]*_sine_p[second];
        else
            return _cosine_p[second]*_sine_p[first] + _cosine_p[first]*_sine_p[second];
    }
    // U SIN(T first, T second, Ts ...others) const noexcept{
    //     // sin(x+y+z+...) = sin(x)*cos(y+z+...) + cos(x)*sin(y+z+...)
    //     if (sizeof...(others)>0){
    //         return _sine_p[first]*COS(second, others ...)+_cosine_p[first]*SIN(second, others ...);
    //     } else{
    //         return SIN(first, second);
    //     }
    // }
    TrigP(std::vector< U >& sine_p, std::vector< U >& cosine_p) : _sine_p(sine_p), _cosine_p(cosine_p){};
    private:
        std::vector< U >& _sine_p;
        std::vector< U >& _cosine_p;
};

#endif /* Wrapper_utils_H_ */