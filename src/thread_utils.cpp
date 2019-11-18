#include "thread_utils.hpp"

using namespace ThreadFunctor;

arma::Mat< std::complex<double> > matGamma; // Matrices used in case parallel.
arma::Mat< std::complex<double> > matWeigths;
arma::Mat< std::complex<double> > matTotSus;
arma::Mat< std::complex<double> > matCorr;
arma::Mat< std::complex<double> > matMidLev;


ThreadWrapper::ThreadWrapper(HF::FunctorBuildGk Gk,HF::K_1D q,double ndo_converged) : _q(q){
    this->_ndo_converged=ndo_converged;
    this->_Gk=Gk;
}

ThreadWrapper::ThreadWrapper(HF::FunctorBuildGk Gk,HF::K_2D q,double ndo_converged) : _q(q){
    this->_ndo_converged=ndo_converged;
    this->_Gk=Gk;
}

ThreadWrapper::ThreadWrapper(HF::K_1D q,IPT2::SplineInline< std::complex<double> > splInline){
    this->_q=q;
    this->_splInline=splInline;
}

void ThreadWrapper::operator()(size_t ktilde, size_t kbar, double beta, bool is_jj){
    std::complex<double> tmp_val_kt_kb(0.0,0.0);
    std::complex<double> tmp_val_weights(0.0,0.0);
    std::complex<double> tmp_val_tot_sus(0.0,0.0);
    for (size_t wtilde=0; wtilde<_Gk._size; wtilde++){
        for (size_t wbar=0; wbar<_Gk._size; wbar++){
            std::complex<double> tmp_val_kt_kb_tmp = gamma_oneD_spsp(_Gk._kArr_l[ktilde],_Gk._precomp_wn[wtilde],_Gk._kArr_l[kbar],_Gk._precomp_wn[wbar]);
            std::complex<double> tmp_val_weights_tmp = buildGK1D(
                                    _Gk._precomp_wn[wtilde],_Gk._kArr_l[ktilde]
                                    )[0]*buildGK1D(
                                    _Gk._precomp_wn[wtilde]-_q._iwn,_Gk._kArr_l[ktilde]-_q._qx
                                    )[0]*buildGK1D(
                                    _Gk._precomp_wn[wbar]-_q._iwn,_Gk._kArr_l[kbar]-_q._qx
                                    )[1]*buildGK1D(
                                    _Gk._precomp_wn[wbar],_Gk._kArr_l[kbar]
                                    )[1]; 
            tmp_val_weights += tmp_val_weights_tmp;
            tmp_val_kt_kb += tmp_val_kt_kb_tmp;
            tmp_val_tot_sus += tmp_val_kt_kb_tmp*tmp_val_weights_tmp;
            if ((wtilde==0) && (wbar==0)){
                std::cout << "Thread id: " << std::this_thread::get_id() << std::endl;
                std::cout << "ktilde: " << _Gk._kArr_l[ktilde] << std::endl;
                std::cout << "kbar: " << _Gk._kArr_l[kbar] << std::endl;
                std::cout << "gamma_oneD_spsp: " << tmp_val_kt_kb << std::endl;
                std::cout << "weights: " << tmp_val_weights << std::endl;
            }
        } 
    }
    // lock_guard<mutex> guard(mutx); 
    matGamma(kbar,ktilde) = tmp_val_kt_kb; // These matrices are static variables.
    matWeigths(kbar,ktilde) = tmp_val_weights;
    if (!is_jj){
        matTotSus(kbar,ktilde) = 1.0/(_Gk._beta)/(_Gk._beta)*tmp_val_tot_sus; // This gives the total susceptibility resolved in k-space. Summation performed on beta only.
    }
    else if (is_jj){
        matTotSus(kbar,ktilde) = -1.0/(_Gk._beta)/(_Gk._beta)*(-2.0*std::sin(_Gk._kArr_l[ktilde]))*tmp_val_tot_sus*(-2.0*std::sin(_Gk._kArr_l[kbar]));
    }
    //cout << "Gamma for " << "ktilde " << ktilde << " and kbar " << kbar << ": " << matGamma(kbar,ktilde) << "\n";
    //cout << "Weigths for " << "ktilde " << ktilde << " and kbar " << kbar << ": " << matWeigths(kbar,ktilde) << "\n";
}

void ThreadWrapper::operator()(size_t ktilde, size_t kbar, bool is_jj){
    std::complex<double> tmp_val_kt_kb(0.0,0.0);
    std::complex<double> tmp_val_weights(0.0,0.0);
    std::complex<double> tmp_val_tot_sus(0.0,0.0);
    for (size_t wtilde=static_cast<size_t>(_splInline.iwn_array.size()/2); wtilde<_splInline.iwn_array.size(); wtilde++){
        for (size_t wbar=static_cast<size_t>(_splInline.iwn_array.size()/2); wbar<_splInline.iwn_array.size(); wbar++){
            std::complex<double> tmp_val_kt_kb_tmp = gamma_oneD_spsp_IPT(_splInline.k_array[ktilde],_splInline.iwn_array[wtilde],_splInline.k_array[kbar],_splInline.iwn_array[wbar]);
            std::complex<double> tmp_val_weights_tmp = buildGK1D_IPT(
                                    _splInline.iwn_array[wtilde],_splInline.k_array[ktilde]
                                    )[0]*buildGK1D_IPT(
                                    _splInline.iwn_array[wtilde]-_q._iwn,_splInline.k_array[ktilde]-_q._qx
                                    )[0]*buildGK1D_IPT(
                                    _splInline.iwn_array[wbar]-_q._iwn,_splInline.k_array[kbar]-_q._qx
                                    )[1]*buildGK1D_IPT(
                                    _splInline.iwn_array[wbar],_splInline.k_array[kbar]
                                    )[1]; 
            tmp_val_weights += tmp_val_weights_tmp;
            tmp_val_kt_kb += tmp_val_kt_kb_tmp;
            tmp_val_tot_sus += tmp_val_kt_kb_tmp*tmp_val_weights_tmp;
            if ((wtilde==static_cast<size_t>(_splInline.iwn_array.size()/2)) && (wbar==static_cast<size_t>(_splInline.iwn_array.size()/2))){
                std::cout << "Thread id: " << std::this_thread::get_id() << std::endl;
                std::cout << "ktilde: " << _splInline.k_array[ktilde] << std::endl;
                std::cout << "kbar: " << _splInline.k_array[kbar] << std::endl;
                std::cout << "gamma_oneD_spsp: " << tmp_val_kt_kb << std::endl;
                std::cout << "weights: " << tmp_val_weights << std::endl;
            }
        } 
    }
    // lock_guard<mutex> guard(mutx); 
    matGamma(kbar,ktilde) = tmp_val_kt_kb; // These matrices are static variables.
    matWeigths(kbar,ktilde) = tmp_val_weights;
    if (!is_jj){
        matTotSus(kbar,ktilde) = 1.0/(GreenStuff::beta)/(GreenStuff::beta)*tmp_val_tot_sus; // This gives the total susceptibility resolved in k-space. Summation performed on beta only.
    }
    else if (is_jj){
        matTotSus(kbar,ktilde) = -1.0/(GreenStuff::beta)/(GreenStuff::beta)*(-2.0*std::sin(_splInline.k_array[ktilde]))*tmp_val_tot_sus*(-2.0*std::sin(_splInline.k_array[kbar]));
    }
    //cout << "Gamma for " << "ktilde " << ktilde << " and kbar " << kbar << ": " << matGamma(kbar,ktilde) << "\n";
    //cout << "Weigths for " << "ktilde " << ktilde << " and kbar " << kbar << ": " << matWeigths(kbar,ktilde) << "\n";
}


void ThreadWrapper::operator()(size_t kbarx_m_tildex, size_t kbary_m_tildey){ // no beta to overload operator() function.
/* In 2D, calculating the weights implies computing whilst dismissing the translational invariance of the vertex function (Gamma). */
    std::complex<double> tmp_val_kt_kb(0.0,0.0);
    for (int wtilde=0; wtilde<_Gk._size; wtilde++){
        for (int wbar=0; wbar<_Gk._size; wbar++){
            tmp_val_kt_kb += gamma_twoD_spsp(_Gk._kArr_l[kbarx_m_tildex],_Gk._kArr_l[kbary_m_tildey],_Gk._precomp_wn[wtilde],_Gk._precomp_wn[wbar]);
            if ((wtilde==0) && (wbar==0)){
                std::cout << "Thread id: " << std::this_thread::get_id() << std::endl;
                std::cout << "kbarx_m_tildex: " << _Gk._kArr_l[kbarx_m_tildex] << std::endl;
                std::cout << "kbary_m_tildey: " << _Gk._kArr_l[kbary_m_tildey] << std::endl;
                std::cout << "gamma_oneD_spsp: " << tmp_val_kt_kb << std::endl;
            }
        } 
    }
    // lock_guard<mutex> guard(mutx);
    matGamma(kbary_m_tildey,kbarx_m_tildex) = tmp_val_kt_kb; // These matrices are static variables.
}

std::complex<double> ThreadWrapper::gamma_oneD_spsp(double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar){
    // Watch out: do not use _Gk() because operator() involves static matrix in FunctorbuildGk and 
    // therefore generates data race when more then one thread used.
    std::complex<double> lower_level=0.0;
    for (int wttilde=0; wttilde<_Gk._size; wttilde++){
        for (size_t qttilde=0; qttilde<_Gk._kArr_l.size(); qttilde++){
            lower_level += buildGK1D(wtilde-_Gk._precomp_qn[wttilde],ktilde-_Gk._kArr_l[qttilde])[0]*buildGK1D(wbar-_Gk._precomp_qn[wttilde],kbar-_Gk._kArr_l[qttilde])[1];
            //lower_level += _Gk(wtilde-_Gk._precomp_qn[wttilde],ktilde-_Gk._kArr_l[qttilde])(0,0) * _Gk(wbar-_Gk._precomp_qn[wttilde],kbar-_Gk._kArr_l[qttilde])(1,1);
        }
    }
    
    lower_level *= SPINDEG*_Gk._u/(_Gk._beta*_Gk._Nk); /// Removed minus sign
    lower_level += 1.0;
    //cout << "gamma_oneD_spsp: " << _Gk._u/lower_level << endl;
    return _Gk._u/lower_level;
}

std::complex<double> ThreadWrapper::gamma_oneD_spsp_IPT(double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar){
    std::complex<double> lower_level=0.0;
    for (int wttilde=0; wttilde<_splInline.iqn_array.size(); wttilde++){
        for (size_t qttilde=0; qttilde<_splInline.k_array.size(); qttilde++){
            lower_level += buildGK1D_IPT(wtilde-_splInline.iqn_array[wttilde],ktilde-_splInline.k_array[qttilde])[0]*buildGK1D_IPT(wbar-_splInline.iqn_array[wttilde],kbar-_splInline.k_array[qttilde])[1];
            //lower_level += _Gk(wtilde-_Gk._precomp_qn[wttilde],ktilde-_Gk._kArr_l[qttilde])(0,0) * _Gk(wbar-_Gk._precomp_qn[wttilde],kbar-_Gk._kArr_l[qttilde])(1,1);
        }
    }
    
    lower_level *= SPINDEG*GreenStuff::U/(GreenStuff::beta*GreenStuff::N_k); /// Removed minus sign
    lower_level += 1.0;
    //cout << "gamma_oneD_spsp: " << _Gk._u/lower_level << endl;
    return GreenStuff::U/lower_level;
}

std::complex<double> ThreadWrapper::gamma_twoD_spsp(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar){
    std::complex<double> lower_level(0.0,0.0);
    for (int wttilde=0; wttilde<_Gk._size; wttilde++){
        for (size_t qttildey=0; qttildey<_Gk._kArr_l.size(); qttildey++){
            for (size_t qttildex=0; qttildex<_Gk._kArr_l.size(); qttildex++){ // the change of variable only applies to k-space, due to periodicity modulo 2pi.
                lower_level += buildGK2D(-(wtilde-_Gk._precomp_qn[wttilde]),-_Gk._kArr_l[qttildex],-_Gk._kArr_l[qttildey])[0]*buildGK2D(-(wbar-_Gk._precomp_qn[wttilde]),-(_Gk._kArr_l[qttildex]+kbarx_m_tildex),-(_Gk._kArr_l[qttildey]+kbary_m_tildey))[1];
            }
        }
    }
    lower_level *= SPINDEG*_Gk._u/(_Gk._beta*_Gk._Nk*_Gk._Nk); //factor 2 for the spin and minus sign added
    lower_level += 1.0;

    return _Gk._u/lower_level;
}

inline std::vector< std::complex<double> > ThreadWrapper::buildGK1D(std::complex<double> ik, double k){
    std::vector< std::complex<double> > GK = { 1.0/( ik + _Gk._mu - epsilonk(k) - _Gk._u*_ndo_converged ), 1.0/( ik + _Gk._mu - epsilonk(k) - _Gk._u*(1.0-_ndo_converged) ) }; // UP, DOWN
    return GK;
}

inline std::vector< std::complex<double> > ThreadWrapper::buildGK1D_IPT(std::complex<double> ik, double k){
    std::vector< std::complex<double> > GK = { 1.0/( ik + GreenStuff::mu - epsilonk(k) - _splInline.calculateSpline(ik.imag()) ), 1.0/( ik + GreenStuff::mu - epsilonk(k) - _splInline.calculateSpline(ik.imag()) ) }; // UP, DOWN
    return GK;
}

inline std::vector< std::complex<double> > ThreadWrapper::buildGK2D(std::complex<double> ik, double kx, double ky){
    std::vector< std::complex<double> > GK = { 1.0/( ik + _Gk._mu - epsilonk(kx,ky) - _Gk._u*_ndo_converged ), 1.0/( ik + _Gk._mu - epsilonk(kx,ky) - _Gk._u*(1.0-_ndo_converged) ) }; // UP, DOWN
    return GK;
}

void ThreadWrapper::join_all(std::vector<std::thread>& grp){
    for (auto& thread : grp){
        if (thread.joinable()){
            thread.join();
        }
    }
}