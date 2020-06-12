#ifndef Wrapper_utils_H_
#define Wrapper_utils_H_

#include <iostream>
#include <complex>

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

struct A{
    std::complex<double> _a {0};
    double _b {0};
    A(){ std::cout << "default A ctor" << "\n"; };
    A(std::complex<double> a, double b) : _a(a), _b(b){ std::cout << "A ctor" << "\n"; };
    ~A(){ std::cout << "A dtor" << "\n"; };
};

struct B : A{
    int _k {0};
    int _q {1};
    B(){ std::cout << "default B ctor" << "\n"; };
    B(int k, int q, std::complex<double> a, double b) : A(a,b), _k(k), _q(q) { std::cout << "B ctor" << "\n"; };
    ~B(){ std::cout << "B dtor" << "\n"; };
};

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


// int main(void){
//     int&& fuck = 6;
//     auto&& obj_B = generate_B();
//     std::cout << "_a: " << obj_B->_a << "\n";
//     std::cout << "_q: " << obj_B->_q << "\n";
//     auto lol = (*obj_B)._a;
//     std::cout << "fuck " << fuck << "\n";

//     return 0;
// }

#endif /* Wrapper_utils_H_ */