#include "thread_utils.hpp"

using namespace ThreadFunctor;

arma::Mat< std::complex<double> > matGamma; // Matrices used in case parallel.
arma::Mat< std::complex<double> > matWeigths;
arma::Mat< std::complex<double> > matTotSus;
arma::Mat< std::complex<double> > matCorr;
arma::Mat< std::complex<double> > matMidLev;
int root_process=0;
std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecGammaSlaves = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >();
std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecWeightsSlaves = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >();
std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecTotSusSlaves = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >();
std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecCorrSlaves = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >();
std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecMidLevSlaves = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >();

#if DIM == 1
void ThreadWrapper::operator()(size_t ktilde, size_t kbar, bool is_jj, solver_prototype sp) const{
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    std::complex<double> tmp_val_kt_kb(0.0,0.0);
    std::complex<double> tmp_val_weights(0.0,0.0);
    std::complex<double> tmp_val_tot_sus(0.0,0.0);
    std::complex<double> val_jj;
    std::complex<double> tmp_val_kt_kb_tmp, tmp_val_weights_tmp;
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    switch(sp){
    case solver_prototype::HF_prot:
        for (size_t wtilde=0; wtilde<_Gk._size; wtilde++){
            for (size_t wbar=0; wbar<_Gk._size; wbar++){
                tmp_val_kt_kb_tmp = gamma_oneD_spsp(_Gk._kArr_l[ktilde],_Gk._precomp_wn[wtilde],_Gk._kArr_l[kbar],_Gk._precomp_wn[wbar]);
                tmp_val_weights_tmp = buildGK1D(
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
                    std::cout << "Process id: " << world_rank << "\n";
                    std::cout << "ktilde: " << _Gk._kArr_l[ktilde] << "\n";
                    std::cout << "kbar: " << _Gk._kArr_l[kbar] << "\n";
                    std::cout << "gamma_oneD_spsp: " << tmp_val_kt_kb << "\n";
                    std::cout << "weights: " << tmp_val_weights << std::endl;
                }
            } 
        }
        // lock_guard<mutex> guard(mutx); 
        matGamma(kbar,ktilde) = tmp_val_kt_kb*1.0/(_Gk._beta)/(_Gk._beta); // These matrices are static variables.
        matWeigths(kbar,ktilde) = tmp_val_weights*1.0/(_Gk._beta)/(_Gk._beta);
        if (world_rank != root_process){ // Saving into vector of tuples.
            vecGammaSlaves->push_back( std::make_tuple( kbar, ktilde, tmp_val_kt_kb*1.0/(_Gk._beta)/(_Gk._beta) ) );
            vecWeightsSlaves->push_back( std::make_tuple( kbar, ktilde, tmp_val_weights*1.0/(_Gk._beta)/(_Gk._beta) ) );
        }
        if (!is_jj){
            matTotSus(kbar,ktilde) = 1.0/(_Gk._beta)/(_Gk._beta)*tmp_val_tot_sus; // This gives the total susceptibility resolved in k-space. Summation performed on beta only.
            if (world_rank != root_process)
                vecTotSusSlaves->push_back( std::make_tuple( kbar, ktilde, 1.0/(_Gk._beta)/(_Gk._beta)*tmp_val_tot_sus ) );
        }
        else if (is_jj){
            val_jj = -1.0/(_Gk._beta)/(_Gk._beta)*(-2.0*std::sin(_Gk._kArr_l[ktilde]))*tmp_val_tot_sus*(-2.0*std::sin(_Gk._kArr_l[kbar]));
            matTotSus(kbar,ktilde) = val_jj;
            if (world_rank != root_process)
                vecTotSusSlaves->push_back( std::make_tuple( kbar, ktilde, val_jj ) );
        }
        break;
    case solver_prototype::IPT2_prot:    
        for (size_t wtilde=static_cast<size_t>(_splInline.iwn_array.size()/2); wtilde<_splInline.iwn_array.size(); wtilde++){
            for (size_t wbar=static_cast<size_t>(_splInline.iwn_array.size()/2); wbar<_splInline.iwn_array.size(); wbar++){
                tmp_val_kt_kb_tmp = gamma_oneD_spsp_IPT(_splInline.k_array[ktilde],_splInline.iwn_array[wtilde],_splInline.k_array[kbar],_splInline.iwn_array[wbar]);
                tmp_val_weights_tmp = buildGK1D_IPT(
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
                    std::cout << "Process id: " << world_rank << "\n";
                    std::cout << "ktilde: " << _splInline.k_array[ktilde] << "\n";
                    std::cout << "kbar: " << _splInline.k_array[kbar] << "\n";
                    std::cout << "gamma_oneD_spsp: " << tmp_val_kt_kb << "\n";
                    std::cout << "weights: " << tmp_val_weights << std::endl;
                }
            } 
        }
        // lock_guard<mutex> guard(mutx); 
        matGamma(kbar,ktilde) = tmp_val_kt_kb*1.0/(GreenStuff::beta)/(GreenStuff::beta); // These matrices are static variables.
        matWeigths(kbar,ktilde) = tmp_val_weights*1.0/(GreenStuff::beta)/(GreenStuff::beta);
        if (world_rank != root_process){
            vecGammaSlaves->push_back( std::make_tuple( kbar, ktilde, tmp_val_kt_kb*1.0/(GreenStuff::beta)/(GreenStuff::beta) ) );
            vecWeightsSlaves->push_back( std::make_tuple( kbar, ktilde, tmp_val_weights*1.0/(GreenStuff::beta)/(GreenStuff::beta) ) );
        }
        if (!is_jj){
            matTotSus(kbar,ktilde) = 1.0/(GreenStuff::beta)/(GreenStuff::beta)*tmp_val_tot_sus; // This gives the total susceptibility resolved in k-space. Summation performed on beta only.
            if (world_rank != root_process)
                vecTotSusSlaves->push_back( std::make_tuple( kbar, ktilde, 1.0/(GreenStuff::beta)/(GreenStuff::beta)*tmp_val_tot_sus ) );
        }
        else if (is_jj){
            val_jj = -1.0/(GreenStuff::beta)/(GreenStuff::beta)*(-2.0*std::sin(_splInline.k_array[ktilde]))*tmp_val_tot_sus*(-2.0*std::sin(_splInline.k_array[kbar]));
            matTotSus(kbar,ktilde) = val_jj;
            if (world_rank != root_process)
                vecTotSusSlaves->push_back( std::make_tuple( kbar, ktilde, val_jj ) );
        }
        break;
    }
    //cout << "Gamma for " << "ktilde " << ktilde << " and kbar " << kbar << ": " << matGamma(kbar,ktilde) << "\n";
    //cout << "Weigths for " << "ktilde " << ktilde << " and kbar " << kbar << ": " << matWeigths(kbar,ktilde) << "\n";
}

std::complex<double> ThreadWrapper::gamma_oneD_spsp(double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar) const{
    // Watch out: do not use _Gk() because operator() involves static matrix in FunctorbuildGk and 
    // therefore generates data race when more then one thread used.
    std::complex<double> lower_level=0.0;
    for (size_t wttilde=0; wttilde<_Gk._size; wttilde++){
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

std::complex<double> ThreadWrapper::gamma_oneD_spsp_IPT(double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar) const{
    std::complex<double> lower_level=0.0;
    for (size_t wttilde=0; wttilde<_splInline.iqn_array.size(); wttilde++){
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

#elif DIM == 2
void ThreadWrapper::operator()(solver_prototype sp, size_t kbarx_m_tildex, size_t kbary_m_tildey, bool is_jj) const{
/* In 2D, calculating the weights implies computing whilst dismissing the translational invariance of the vertex function (Gamma). */
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    std::complex<double> tmp_val_kt_kb(0.0,0.0), tmp_val_weights(0.0,0.0), tmp_val_tot_sus(0.0,0.0);
    std::complex<double> val_jj(0.0,0.0);
    double kbarx;
    switch(sp){
    case solver_prototype::HF_prot:
        // cout << beta << " " << _Gk._kArr_l[ktilde] << " " << _q._iwn << " " << _q._qx << " " << _Gk._kArr_l[kbar] << endl;
        for (size_t wtilde=0; wtilde<_Gk._size; wtilde++){
            for (size_t wbar=0; wbar<_Gk._size; wbar++){
                tmp_val_kt_kb += gamma_twoD_spsp(_Gk._kArr_l[kbarx_m_tildex],_Gk._kArr_l[kbary_m_tildey],_Gk._precomp_wn[wtilde],_Gk._precomp_wn[wbar]);
                tmp_val_weights += getWeightsHF(kbarx_m_tildex,kbary_m_tildey,wtilde,wbar);
                if ((wtilde==0) && (wbar==0)){
                    std::cout << "Process id: " << world_rank << "\n";
                    std::cout << "kbarx_m_tildex: " << _Gk._kArr_l[kbarx_m_tildex] << "\n";
                    std::cout << "kbary_m_tildey: " << _Gk._kArr_l[kbary_m_tildey] << "\n";
                    std::cout << "gamma_twoD_spsp: " << tmp_val_kt_kb << "\n";
                    std::cout << "weights: " << tmp_val_weights << std::endl;
                }
                tmp_val_tot_sus += tmp_val_kt_kb*tmp_val_weights;
            } 
        }
        // lock_guard<mutex> guard(mutx);
        matWeigths(kbary_m_tildey,kbarx_m_tildex) = tmp_val_weights*(1.0/_Gk._beta)*(1.0/_Gk._beta);
        matGamma(kbary_m_tildey,kbarx_m_tildex) = tmp_val_kt_kb*(1.0/_Gk._beta)*(1.0/_Gk._beta);
        if (world_rank != root_process){ // Saving into vector of tuples..
            vecGammaSlaves->push_back( std::make_tuple( kbary_m_tildey,kbarx_m_tildex, tmp_val_kt_kb*(1.0/_Gk._beta)*(1.0/_Gk._beta) ) );
            vecWeightsSlaves->push_back( std::make_tuple( kbary_m_tildey,kbarx_m_tildex, tmp_val_weights*(1.0/_Gk._beta)*(1.0/_Gk._beta) ) );
        }
        if (!is_jj){
            matTotSus(kbary_m_tildey,kbarx_m_tildex) = (1.0/_Gk._beta)*(1.0/_Gk._beta)*tmp_val_tot_sus; // These matrices are static variables.
            if (world_rank != root_process)
                vecTotSusSlaves->push_back( std::make_tuple( kbary_m_tildey,kbarx_m_tildex,(1.0/_Gk._beta)*(1.0/_Gk._beta)*tmp_val_tot_sus ) );
        }
        else if (is_jj){ // kbarx could also be kbary
            for (size_t ktildex=0; ktildex<_Gk._kArr_l.size(); ktildex++){
                kbarx = (_Gk._kArr_l[ktildex]-_Gk._kArr_l[kbarx_m_tildex]); // It brakets within [-2pi,2pi].
                val_jj += -1.0*(-2.0*std::sin(_Gk._kArr_l[ktildex]))*tmp_val_tot_sus*(-2.0*std::sin(kbarx));
            }
            matTotSus(kbary_m_tildey,kbarx_m_tildex)=val_jj*(1.0/_Gk._beta)*(1.0/_Gk._beta)*(1.0/_Gk._Nk);
            if (world_rank != root_process)
                vecTotSusSlaves->push_back( std::make_tuple( kbary_m_tildey,kbarx_m_tildex,val_jj ) );
        }
        break;
    case solver_prototype::IPT2_prot:
        for (size_t wtilde=static_cast<size_t>(_splInline.iwn_array.size()/2); wtilde<_splInline.iwn_array.size(); wtilde++){
            for (size_t wbar=static_cast<size_t>(_splInline.iwn_array.size()/2); wbar<_splInline.iwn_array.size(); wbar++){
                tmp_val_kt_kb += gamma_twoD_spsp_IPT(_splInline.k_array[kbarx_m_tildex],_splInline.k_array[kbary_m_tildey],_splInline.iwn_array[wtilde],_splInline.iwn_array[wbar]);
                tmp_val_weights += getWeightsIPT(kbarx_m_tildex,kbary_m_tildey,wtilde,wbar);
                if ((wtilde==0) && (wbar==0)){
                    std::cout << "Process id: " << world_rank << "\n";
                    std::cout << "kbarx_m_tildex: " << _splInline.k_array[kbarx_m_tildex] << "\n";
                    std::cout << "kbary_m_tildey: " << _splInline.k_array[kbary_m_tildey] << "\n";
                    std::cout << "gamma_twoD_spsp: " << tmp_val_kt_kb << "\n";
                    std::cout << "weights: " << tmp_val_weights << std::endl;
                }
                tmp_val_tot_sus+=tmp_val_kt_kb*tmp_val_weights;
            } 
        }
        // lock_guard<mutex> guard(mutx);
        matGamma(kbary_m_tildey,kbarx_m_tildex) = tmp_val_kt_kb*(1.0/GreenStuff::beta)*(1.0/GreenStuff::beta);
        matWeigths(kbary_m_tildey,kbarx_m_tildex) = tmp_val_weights*(1.0/GreenStuff::beta)*(1.0/GreenStuff::beta);
        if (world_rank != root_process){
            vecGammaSlaves->push_back( std::make_tuple( kbary_m_tildey,kbarx_m_tildex,tmp_val_kt_kb*(1.0/GreenStuff::beta)*(1.0/GreenStuff::beta) ) );
            vecWeightsSlaves->push_back( std::make_tuple( kbary_m_tildey,kbarx_m_tildex,tmp_val_weights*(1.0/GreenStuff::beta)*(1.0/GreenStuff::beta) ) );
        }
        if (!is_jj){
            matTotSus(kbary_m_tildey,kbarx_m_tildex) = (1.0/GreenStuff::beta)*(1.0/GreenStuff::beta)*tmp_val_tot_sus; // These matrices are static variables.
            if (world_rank != root_process)
                vecTotSusSlaves->push_back( std::make_tuple( kbary_m_tildey,kbarx_m_tildex,(1.0/GreenStuff::beta)*(1.0/GreenStuff::beta)*tmp_val_tot_sus ) );
        }
        else if (is_jj){
            for (size_t ktildex=0; ktildex<_splInline.k_array.size(); ktildex++){
                kbarx = (_splInline.k_array[ktildex]-_splInline.k_array[kbarx_m_tildex]); // It brakets within [-2pi,2pi].
                val_jj += -1.0*(-2.0*std::sin(_splInline.k_array[ktildex]))*tmp_val_tot_sus*(-2.0*std::sin(kbarx));
            }
            matTotSus(kbary_m_tildey,kbarx_m_tildex) = val_jj*(1.0/GreenStuff::N_k)*(1.0/GreenStuff::beta)*(1.0/GreenStuff::beta);
            if (world_rank != root_process)
                vecTotSusSlaves->push_back( std::make_tuple( kbary_m_tildey,kbarx_m_tildex,val_jj ) );
        }
        break;
    }

}

std::complex<double> ThreadWrapper::gamma_twoD_spsp(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar) const{
    std::complex<double> lower_level(0.0,0.0);
    for (size_t wttilde=0; wttilde<_Gk._size; wttilde++){
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

std::complex<double> ThreadWrapper::gamma_twoD_spsp_IPT(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar) const{
    std::complex<double> lower_level(0.0,0.0);
    for (size_t wttilde=0; wttilde<_splInline.iqn_array.size(); wttilde++){
        for (size_t qttildey=0; qttildey<_splInline.k_array.size(); qttildey++){
            for (size_t qttildex=0; qttildex<_splInline.k_array.size(); qttildex++){ // the change of variable only applies to k-space, due to periodicity modulo 2pi.
                lower_level += buildGK2D_IPT((wtilde-_splInline.iqn_array[wttilde]),_splInline.k_array[qttildex],_splInline.k_array[qttildey])[0]*buildGK2D_IPT((wbar-_splInline.iqn_array[wttilde]),(_splInline.k_array[qttildex]+kbarx_m_tildex),(_splInline.k_array[qttildey]+kbary_m_tildey))[1];
            }
        }
    }
    lower_level *= SPINDEG*GreenStuff::U/(GreenStuff::beta*GreenStuff::N_k*GreenStuff::N_k); //factor 2 for the spin and minus sign added
    lower_level += 1.0;
    return GreenStuff::U/lower_level;
}

std::complex<double> ThreadWrapper::gamma_twoD_spsp_full_lower(double kpx,double kpy,double kbarx,double kbary,std::complex<double> iknp,std::complex<double> wbar) const{
    std::complex<double> lower_level(0.0,0.0);
    for (size_t iknpp=0; iknpp<_Gk._size; iknpp++){
        for (size_t kppx=0; kppx<_Gk._kArr_l.size(); kppx++){
            for (size_t kppy=0; kppy<_Gk._kArr_l.size(); kppy++){
                lower_level += buildGK2D(_Gk._precomp_wn[iknpp]+iknp-wbar,_Gk._kArr_l[kppx]+kpx-kbarx,_Gk._kArr_l[kppy]+kpy-kbary)[0]*buildGK2D(_Gk._precomp_wn[iknpp],_Gk._kArr_l[kppx],_Gk._kArr_l[kppy])[1];
            }
        }
    }
    lower_level *= SPINDEG*_Gk._u/(_Gk._beta*(_Gk._Nk)*(_Gk._Nk)); /// No minus sign at ground level. Factor 2 for spin.
    lower_level += 1.0;
    return _Gk._u/lower_level; // Means that we have to multiply the middle level of this component by the two missing Green's functions.
}

std::complex<double> ThreadWrapper::gamma_twoD_spsp_full_lower_IPT(double kpx,double kpy,double kbarx,double kbary,std::complex<double> iknp,std::complex<double> wbar) const{
    std::complex<double> lower_level(0.0,0.0);
    for (size_t iknpp=static_cast<size_t>(_splInline.iwn_array.size()/2); iknpp<_splInline.iwn_array.size(); iknpp++){
        for (size_t kppx=0; kppx<_splInline.k_array.size(); kppx++){
            for (size_t kppy=0; kppy<_splInline.k_array.size(); kppy++){
                lower_level += buildGK2D_IPT(_splInline.iwn_array[iknpp]+iknp-wbar,_splInline.k_array[kppx]+kpx-kbarx,_splInline.k_array[kppy]+kpy-kbary)[0]*buildGK2D_IPT(_splInline.iwn_array[iknpp],_splInline.k_array[kppx],_splInline.k_array[kppy])[1];
            }
        }
    }
    lower_level *= SPINDEG*GreenStuff::U/(GreenStuff::beta*(GreenStuff::N_k)*(GreenStuff::N_k)); /// No minus sign at ground level. Factor 2 for spin.
    lower_level += 1.0;
    return GreenStuff::U/lower_level; // Means that we have to multiply the middle level of this component by the two missing Green's functions.
}


std::complex<double> ThreadWrapper::getWeightsHF(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar) const{
    std::complex<double> tmp_val_weigths(0.0,0.0);
    double kbary, kbarx;
    for (size_t ktildey=0; ktildey<_Gk._kArr_l.size(); ktildey++){
        kbary = _Gk._kArr_l[ktildey] - kbary_m_tildey;
        for (size_t ktildex=0; ktildex<_Gk._kArr_l.size(); ktildex++){
            kbarx = _Gk._kArr_l[ktildex] - kbarx_m_tildex;
            tmp_val_weigths += _Gk(wtilde,_Gk._kArr_l[ktildex],_Gk._kArr_l[ktildey])(0,0)*_Gk(wtilde+_qq._iwn,_Gk._kArr_l[ktildex]+_qq._qx,_Gk._kArr_l[ktildey]+_qq._qy
                )(0,0)*_Gk(wbar+_qq._iwn,kbarx+_qq._qx,kbary+_qq._qy)(1,1)*_Gk(wbar,kbarx,kbary)(1,1);
        }      
    }
    tmp_val_weigths*=1.0/_Gk._kArr_l.size()/_Gk._kArr_l.size();
    return tmp_val_weigths;
}

std::complex<double> ThreadWrapper::getWeightsIPT(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar) const{
    std::complex<double> tmp_val_weigths(0.0,0.0);
    double kbary, kbarx;
    for (size_t ktildey=0; ktildey<_splInline.k_array.size(); ktildey++){
        kbary = _splInline.k_array[ktildey] - kbary_m_tildey;
        for (size_t ktildex=0; ktildex<_splInline.k_array.size(); ktildex++){
            kbarx = _splInline.k_array[ktildex] - kbarx_m_tildex;
            tmp_val_weigths += buildGK2D_IPT(wtilde,ktildex,ktildey)[0] * buildGK2D_IPT(wtilde+_qq._iwn,ktildex+_qq._qx,ktildey+_qq._qy
                                )[0] * buildGK2D_IPT(wbar+_qq._iwn,kbarx+_qq._qx,kbary+_qq._qy)[1] * buildGK2D_IPT(wbar,kbarx,kbary)[1];
        }      
    }
    tmp_val_weigths*=1.0/_splInline.k_array.size()/_splInline.k_array.size();
    return tmp_val_weigths;
}

#endif /* DIM */


void ThreadWrapper::join_all(std::vector<std::thread>& grp) const{
    for (auto& thread : grp){
        if (thread.joinable()){
            thread.join();
        }
    }
}

void get_vector_mpi(size_t totSize,bool is_jj,solver_prototype sp,std::vector<mpistruct_t>* vec_root_process){
    size_t idx=0;
    for (size_t lkt=0; lkt<vecK.size(); lkt++){
        for (size_t lkb=0; lkb<vecK.size(); lkb++){
            mpistruct_t MPIObj;
            MPIObj._lkt=lkt;
            MPIObj._lkb=lkb;
            MPIObj._is_jj=is_jj;
            MPIObj._sp=sp;
            vec_root_process->at(idx) = MPIObj;
            idx++;
        }
    }       
}