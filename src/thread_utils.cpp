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
void ThreadWrapper::operator()(size_t ktilde, size_t kbar, bool is_jj, bool is_full, solver_prototype sp) const{
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    std::complex<double> tmp_val_kt_kb(0.0,0.0),tmp_val_weights(0.0,0.0),tmp_val_tot_sus(0.0,0.0),
    tmp_val_mid_lev(0.0,0.0),tmp_val_corr(0.0,0.0);
    std::complex<double> val_jj;
    std::complex<double> tmp_val_kt_kb_tmp, tmp_val_weights_tmp;
    std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > tupOfValsFull;
    std::tuple< std::complex<double>,std::complex<double> > tupOfVals;
    switch(sp){
    case solver_prototype::HF_prot:
        for (size_t wtilde=0; wtilde<_Gk._size; wtilde++){
            for (size_t wbar=0; wbar<_Gk._size; wbar++){
                if (!is_full){
                    tupOfVals = gamma_oneD_spsp(_Gk._kArr_l[ktilde],_Gk._precomp_wn[wtilde],_Gk._kArr_l[kbar],_Gk._precomp_wn[wbar]);
                    tmp_val_kt_kb_tmp = std::get<0>(tupOfVals);
                    tmp_val_mid_lev += std::get<1>(tupOfVals);
                }
                else{
                    tupOfValsFull = gamma_oneD_spsp_full_middle_plotting(_Gk._kArr_l[ktilde],_Gk._kArr_l[kbar],_Gk._precomp_wn[wbar],_Gk._precomp_wn[wtilde]);
                    tmp_val_kt_kb_tmp = std::get<0>(tupOfValsFull);
                    tmp_val_corr += std::get<1>(tupOfValsFull);
                    tmp_val_mid_lev += std::get<2>(tupOfValsFull);
                }
                tmp_val_weights_tmp = _Gk(_Gk._precomp_wn[wtilde],_Gk._kArr_l[ktilde]
                                    )(0,0)*_Gk(_Gk._precomp_wn[wtilde]-_q._iwn,_Gk._kArr_l[ktilde]-_q._qx
                                    )(0,0)*_Gk(_Gk._precomp_wn[wbar]-_q._iwn,_Gk._kArr_l[kbar]-_q._qx
                                    )(1,1)*_Gk(_Gk._precomp_wn[wbar],_Gk._kArr_l[kbar]
                                    )(1,1); 
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
        matGamma(kbar,ktilde) = tmp_val_kt_kb*1.0/(_Gk._beta)/(_Gk._beta); // These matrices are extern variables.
        matWeigths(kbar,ktilde) = tmp_val_weights*1.0/(_Gk._beta)/(_Gk._beta);
        matMidLev(kbar,ktilde) = tmp_val_mid_lev*1.0/(_Gk._beta)/(_Gk._beta);
        if (is_full)
            matCorr(kbar,ktilde) = tmp_val_corr*1.0/(_Gk._beta)/(_Gk._beta);
        if (world_rank != root_process){ // Saving into vector of tuples.
            vecGammaSlaves->push_back( std::make_tuple( kbar, ktilde,tmp_val_kt_kb*1.0/(_Gk._beta)/(_Gk._beta) ) );
            vecWeightsSlaves->push_back( std::make_tuple( kbar, ktilde,tmp_val_weights*1.0/(_Gk._beta)/(_Gk._beta) ) );
            vecMidLevSlaves->push_back( std::make_tuple( kbar,ktilde,tmp_val_mid_lev*1.0/(_Gk._beta)/(_Gk._beta) ) );
            if (is_full)
                vecCorrSlaves->push_back( std::make_tuple( kbar,ktilde,tmp_val_corr*1.0/(_Gk._beta)/(_Gk._beta) ) );
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
                if (!is_full){
                    tupOfVals = gamma_oneD_spsp_IPT(_splInline.k_array[ktilde],_splInline.iwn_array[wtilde],_splInline.k_array[kbar],_splInline.iwn_array[wbar]);
                    tmp_val_kt_kb_tmp = std::get<0>(tupOfVals);
                    tmp_val_mid_lev += std::get<1>(tupOfVals);
                }
                else{
                    tupOfValsFull = gamma_oneD_spsp_full_middle_plotting_IPT(_splInline.k_array[ktilde],_splInline.k_array[kbar],_splInline.iwn_array[wbar],_splInline.iwn_array[wtilde]);
                    tmp_val_kt_kb_tmp = std::get<0>(tupOfValsFull);
                    tmp_val_corr += std::get<1>(tupOfValsFull);
                    tmp_val_mid_lev += std::get<2>(tupOfValsFull);
                }
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
        matGamma(kbar,ktilde) = tmp_val_kt_kb*1.0/(GreenStuff::beta)/(GreenStuff::beta); // These matrices are extern variables.
        matWeigths(kbar,ktilde) = tmp_val_weights*1.0/(GreenStuff::beta)/(GreenStuff::beta);
        matMidLev(kbar,ktilde) = tmp_val_mid_lev*1.0/(GreenStuff::beta)/(GreenStuff::beta);
        if (is_full)
            matCorr(kbar,ktilde) = tmp_val_corr*1.0/(GreenStuff::beta)/(GreenStuff::beta);
        if (world_rank != root_process){
            vecGammaSlaves->push_back( std::make_tuple( kbar,ktilde,tmp_val_kt_kb*1.0/(GreenStuff::beta)/(GreenStuff::beta) ) );
            vecWeightsSlaves->push_back( std::make_tuple( kbar,ktilde,tmp_val_weights*1.0/(GreenStuff::beta)/(GreenStuff::beta) ) );
            vecMidLevSlaves->push_back( std::make_tuple( kbar,ktilde,tmp_val_mid_lev*1.0/(GreenStuff::beta)/(GreenStuff::beta) ) );
            if (is_full)
                vecCorrSlaves->push_back( std::make_tuple( kbar,ktilde,tmp_val_corr*1.0/(GreenStuff::beta)/(GreenStuff::beta) ) );
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
}

std::tuple< std::complex<double>,std::complex<double> > ThreadWrapper::gamma_oneD_spsp(double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar) const{
    std::complex<double> lower_level(0.0,0.0), bubble(0.0,0.0);
    for (size_t wttilde=0; wttilde<_Gk._size; wttilde++){
        for (size_t qttilde=0; qttilde<_Gk._kArr_l.size(); qttilde++){
            //lower_level += buildGK1D(wtilde-_Gk._precomp_qn[wttilde],ktilde-_Gk._kArr_l[qttilde])[0]*buildGK1D(wbar-_Gk._precomp_qn[wttilde],kbar-_Gk._kArr_l[qttilde])[1];
            lower_level += _Gk(wtilde-_Gk._precomp_qn[wttilde],ktilde-_Gk._kArr_l[qttilde])(0,0) * _Gk(wbar-_Gk._precomp_qn[wttilde],kbar-_Gk._kArr_l[qttilde])(1,1);
        }
    }
    
    lower_level *= SPINDEG*_Gk._u/(_Gk._beta*_Gk._Nk);
    bubble = -1.0*lower_level;
    lower_level += 1.0;
    return std::make_tuple(_Gk._u/lower_level,bubble);
}

std::tuple< std::complex<double>,std::complex<double> > ThreadWrapper::gamma_oneD_spsp_IPT(double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar) const{
    std::complex<double> lower_level(0.0,0.0), bubble(0.0,0.0);
    for (size_t wttilde=0; wttilde<_splInline.iqn_array.size(); wttilde++){
        for (size_t qttilde=0; qttilde<_splInline.k_array.size(); qttilde++){
            lower_level += buildGK1D_IPT(wtilde-_splInline.iqn_array[wttilde],ktilde-_splInline.k_array[qttilde])[0]*buildGK1D_IPT(wbar-_splInline.iqn_array[wttilde],kbar-_splInline.k_array[qttilde])[1];
        }
    }
    
    lower_level *= SPINDEG*GreenStuff::U/(GreenStuff::beta*GreenStuff::N_k);
    bubble = -1.0*bubble;
    lower_level += 1.0;
    return std::make_tuple(GreenStuff::U/lower_level,bubble);
}

std::complex<double> ThreadWrapper::gamma_oneD_spsp_full_lower(double kp,double kbar,std::complex<double> iknp,std::complex<double> wbar) const{
    std::complex<double> lower_level(0.0,0.0);
    for (size_t iknpp=0; iknpp<_Gk._size; iknpp++){
        for (size_t kpp=0; kpp<_Gk._kArr_l.size(); kpp++){
            lower_level+=_Gk(_Gk._precomp_wn[iknpp]+iknp-wbar,_Gk._kArr_l[kpp]+kp-kbar)(0,0)*_Gk(_Gk._precomp_wn[iknpp],_Gk._kArr_l[kpp])(1,1);
        }
    }
    lower_level*=SPINDEG*_Gk._u/(_Gk._beta*_Gk._Nk);
    lower_level+=1.0;

    return _Gk._u/lower_level;
}

std::complex<double> ThreadWrapper::gamma_oneD_spsp_full_lower_IPT(double kp,double kbar,std::complex<double> iknp,std::complex<double> wbar) const{
    std::complex<double> lower_level(0.0,0.0);
    for (size_t iknpp=static_cast<size_t>(_splInline.iwn_array.size()/2); iknpp<_splInline.iwn_array.size(); iknpp++){
        for (size_t kpp=0; kpp<_splInline.k_array.size(); kpp++){
            lower_level+=buildGK1D_IPT(_splInline.iwn_array[iknpp]+iknp-wbar,_splInline.k_array[kpp]+kp-kbar)[0] * buildGK1D_IPT(_splInline.iwn_array[iknpp],_splInline.k_array[kpp])[1];
        }
    }
    lower_level*=SPINDEG*GreenStuff::U/(GreenStuff::beta*GreenStuff::N_k);
    lower_level+=1.0;

    return GreenStuff::U/lower_level;
}

std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > ThreadWrapper::gamma_oneD_spsp_full_middle_plotting(double ktilde,double kbar,std::complex<double> wbar,std::complex<double> wtilde) const{
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    std::complex<double> middle_level_tmp(0.0,0.0), middle_level_inf_tmp(0.0,0.0), middle_level_corr_tmp(0.0,0.0);
    for (size_t kp=0; kp<_Gk._kArr_l.size(); kp++){
        for (size_t iknp=0; iknp<_Gk._size; iknp++){
            middle_level_inf_tmp+= _Gk(_Gk._precomp_wn[iknp],_Gk._kArr_l[kp]
            )(0,0)*gamma_oneD_spsp_full_lower(_Gk._kArr_l[kp],kbar,_Gk._precomp_wn[iknp],wbar
            )*_Gk(_Gk._precomp_wn[iknp]-_q._iwn,_Gk._kArr_l[kp]-_q._qx)(0,0);
        }
    }
    middle_level_inf_tmp*=SPINDEG/(_Gk._Nk*_Gk._beta);
    middle_level_corr_tmp+=middle_level_inf_tmp;
    middle_level_inf_tmp+=std::get<1>(gamma_oneD_spsp(ktilde,wtilde,kbar,wbar));
    middle_level_tmp-=middle_level_inf_tmp;
    middle_level_tmp+=1.0;

    return std::make_tuple(GreenStuff::U/middle_level_tmp,middle_level_corr_tmp,middle_level_inf_tmp);
}

std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > ThreadWrapper::gamma_oneD_spsp_full_middle_plotting_IPT(double ktilde,double kbar,std::complex<double> wbar,std::complex<double> wtilde) const{
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    std::complex<double> middle_level_tmp(0.0,0.0), middle_level_inf_tmp(0.0,0.0), middle_level_corr_tmp(0.0,0.0);
    for (size_t kp=0; kp<_splInline.k_array.size(); kp++){
        for (size_t iknp=static_cast<size_t>(_splInline.iwn_array.size()/2); iknp<_splInline.iwn_array.size(); iknp++){
            middle_level_inf_tmp+= buildGK1D_IPT(_splInline.iwn_array[iknp],_splInline.k_array[kp]
            )[0]*gamma_oneD_spsp_full_lower_IPT(_splInline.k_array[kp],kbar,_splInline.iwn_array[iknp],wbar
            )*buildGK1D_IPT(_splInline.iwn_array[iknp]-_q._iwn,_splInline.k_array[kp]-_q._qx)[0];
        }
    }
    middle_level_inf_tmp*=SPINDEG/(GreenStuff::N_k*GreenStuff::beta);
    middle_level_corr_tmp+=middle_level_inf_tmp;
    middle_level_inf_tmp+=std::get<1>(gamma_oneD_spsp_IPT(ktilde,wtilde,kbar,wbar));
    middle_level_tmp-=middle_level_inf_tmp;
    middle_level_tmp+=1.0;

    return std::make_tuple(GreenStuff::U/middle_level_tmp,middle_level_corr_tmp,middle_level_inf_tmp);
}

#elif DIM == 2
void ThreadWrapper::operator()(solver_prototype sp, size_t kbarx_m_tildex, size_t kbary_m_tildey, bool is_jj, bool is_full) const{
/* In 2D, calculating the weights implies computing whilst dismissing the translational invariance of the vertex function (Gamma). */
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    std::complex<double> tmp_val_kt_kb(0.0,0.0), tmp_val_weights(0.0,0.0), tmp_val_tot_sus(0.0,0.0), tmp_val_mid_lev(0.0,0.0);
    std::complex<double> tmp_val_corr(0.0,0.0), val_jj(0.0,0.0);
    double kbarx;
    std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > tupOfValsFull;
    std::tuple< std::complex<double>,std::complex<double> > tupOfVals;
    switch(sp){
    case solver_prototype::HF_prot:
        // Looking whether the calculations concern the full calculation considering the corrections or not.
        for (size_t wtilde=0; wtilde<_Gk._size; wtilde++){
            for (size_t wbar=0; wbar<_Gk._size; wbar++){
                if (!is_full){
                    tupOfVals = gamma_twoD_spsp(_Gk._kArr_l[kbarx_m_tildex],_Gk._kArr_l[kbary_m_tildey],_Gk._precomp_wn[wtilde],_Gk._precomp_wn[wbar]);
                    tmp_val_kt_kb += std::get<0>(tupOfVals);
                    tmp_val_mid_lev += std::get<1>(tupOfVals);
                }
                else{
                    tupOfValsFull = gamma_twoD_spsp_full_middle_plotting(_Gk._kArr_l[kbarx_m_tildex],_Gk._kArr_l[kbary_m_tildey],_Gk._precomp_wn[wbar],_Gk._precomp_wn[wtilde]);
                    tmp_val_kt_kb += std::get<0>(tupOfValsFull);
                    tmp_val_corr += std::get<1>(tupOfValsFull);
                    tmp_val_mid_lev += std::get<2>(tupOfValsFull);
                }
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
        matMidLev(kbary_m_tildey,kbarx_m_tildex) = tmp_val_mid_lev*(1.0/_Gk._beta)*(1.0/_Gk._beta);
        if (is_full)
            matCorr(kbary_m_tildey,kbarx_m_tildex) = tmp_val_corr*(1.0/_Gk._beta)*(1.0/_Gk._beta);
        if (world_rank != root_process){ // Saving into vector of tuples..
            vecGammaSlaves->push_back( std::make_tuple( kbary_m_tildey,kbarx_m_tildex, tmp_val_kt_kb*(1.0/_Gk._beta)*(1.0/_Gk._beta) ) );
            vecWeightsSlaves->push_back( std::make_tuple( kbary_m_tildey,kbarx_m_tildex, tmp_val_weights*(1.0/_Gk._beta)*(1.0/_Gk._beta) ) );
            vecMidLevSlaves->push_back( std::make_tuple( kbary_m_tildey,kbarx_m_tildex,tmp_val_mid_lev*(1.0/_Gk._beta)*(1.0/_Gk._beta) ) );
            if (is_full)
                vecCorrSlaves->push_back( std::make_tuple( kbary_m_tildey,kbarx_m_tildex,tmp_val_corr*(1.0/_Gk._beta)*(1.0/_Gk._beta) ) );
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
                if (!is_full){
                    tupOfVals = gamma_twoD_spsp_IPT(_splInline.k_array[kbarx_m_tildex],_splInline.k_array[kbary_m_tildey],_splInline.iwn_array[wtilde],_splInline.iwn_array[wbar]);
                    tmp_val_kt_kb += std::get<0>(tupOfVals);
                    tmp_val_mid_lev += std::get<1>(tupOfVals);
                }
                else{
                    tupOfValsFull = gamma_twoD_spsp_full_middle_plotting_IPT(_splInline.k_array[kbarx_m_tildex],_splInline.k_array[kbary_m_tildey],_splInline.iwn_array[wbar],_splInline.iwn_array[wtilde]);
                    tmp_val_kt_kb += std::get<0>(tupOfValsFull);
                    tmp_val_corr += std::get<1>(tupOfValsFull);
                    tmp_val_mid_lev += std::get<2>(tupOfValsFull);
                }
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
        matMidLev(kbary_m_tildey,kbarx_m_tildex) = tmp_val_mid_lev*(1.0/GreenStuff::beta)*(1.0/GreenStuff::beta);
        if (is_full)
            matCorr(kbary_m_tildey,kbarx_m_tildex) = tmp_val_corr*(1.0/GreenStuff::beta)*(1.0/GreenStuff::beta);
        if (world_rank != root_process){
            vecGammaSlaves->push_back( std::make_tuple( kbary_m_tildey,kbarx_m_tildex,tmp_val_kt_kb*(1.0/GreenStuff::beta)*(1.0/GreenStuff::beta) ) );
            vecWeightsSlaves->push_back( std::make_tuple( kbary_m_tildey,kbarx_m_tildex,tmp_val_weights*(1.0/GreenStuff::beta)*(1.0/GreenStuff::beta) ) );
            vecMidLevSlaves->push_back( std::make_tuple( kbary_m_tildey,kbarx_m_tildex,tmp_val_mid_lev*(1.0/GreenStuff::beta)*(1.0/GreenStuff::beta) ) );
            if (is_full)
                vecCorrSlaves->push_back( std::make_tuple( kbary_m_tildey,kbarx_m_tildex,tmp_val_corr*(1.0/GreenStuff::beta)*(1.0/GreenStuff::beta) ) );
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

std::tuple< std::complex<double>,std::complex<double> > ThreadWrapper::gamma_twoD_spsp(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar) const{
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    std::complex<double> lower_level(0.0,0.0), bubble(0.0,0.0);
    for (size_t wttilde=0; wttilde<_Gk._size; wttilde++){
        for (size_t qttildey=0; qttildey<_Gk._kArr_l.size(); qttildey++){
            for (size_t qttildex=0; qttildex<_Gk._kArr_l.size(); qttildex++){ // the change of variable only applies to k-space, due to periodicity modulo 2pi.
                lower_level += buildGK2D((wtilde-_Gk._precomp_qn[wttilde]),_Gk._kArr_l[qttildex],_Gk._kArr_l[qttildey])[0]*buildGK2D((wbar-_Gk._precomp_qn[wttilde]),(_Gk._kArr_l[qttildex]+kbarx_m_tildex),(_Gk._kArr_l[qttildey]+kbary_m_tildey))[1];
            }
        }
    }
    lower_level *= SPINDEG*_Gk._u/(_Gk._beta*_Gk._Nk*_Gk._Nk); //factor 2 for the spin and minus sign added
    bubble = -1.0*lower_level;
    lower_level += 1.0;
    return std::make_tuple(_Gk._u/lower_level,bubble);
}

std::tuple< std::complex<double>,std::complex<double> > ThreadWrapper::gamma_twoD_spsp_IPT(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar) const{
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    std::complex<double> lower_level(0.0,0.0), bubble(0.0,0.0);
    for (size_t wttilde=0; wttilde<_splInline.iqn_array.size(); wttilde++){
        for (size_t qttildey=0; qttildey<_splInline.k_array.size(); qttildey++){
            for (size_t qttildex=0; qttildex<_splInline.k_array.size(); qttildex++){ // the change of variable only applies to k-space, due to periodicity modulo 2pi.
                lower_level += buildGK2D_IPT((wtilde-_splInline.iqn_array[wttilde]),_splInline.k_array[qttildex],_splInline.k_array[qttildey])[0]*buildGK2D_IPT((wbar-_splInline.iqn_array[wttilde]),(_splInline.k_array[qttildex]+kbarx_m_tildex),(_splInline.k_array[qttildey]+kbary_m_tildey))[1];
            }
        }
    }
    lower_level *= SPINDEG*GreenStuff::U/(GreenStuff::beta*GreenStuff::N_k*GreenStuff::N_k); //factor 2 for the spin and minus sign added
    bubble = -1.0*lower_level; 
    lower_level += 1.0;
    return std::make_tuple(GreenStuff::U/lower_level,bubble);
}

std::complex<double> ThreadWrapper::gamma_twoD_spsp_full_lower(double kpx,double kpy,double kbarx,double kbary,std::complex<double> iknp,std::complex<double> wbar) const{
    std::complex<double> lower_level(0.0,0.0);
    for (size_t iknpp=0; iknpp<_Gk._size; iknpp++){
        for (size_t kppx=0; kppx<_Gk._kArr_l.size(); kppx++){
            for (size_t kppy=0; kppy<_Gk._kArr_l.size(); kppy++){
                lower_level += _Gk(_Gk._precomp_wn[iknpp]+iknp-wbar,_Gk._kArr_l[kppx]+kpx-kbarx,_Gk._kArr_l[kppy]+kpy-kbary)[0]*_Gk(_Gk._precomp_wn[iknpp],_Gk._kArr_l[kppx],_Gk._kArr_l[kppy])[1];
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

std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > ThreadWrapper::gamma_twoD_spsp_full_middle_plotting(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wbar,std::complex<double> wtilde) const{
    /* This function uses the externally linked matrices matCorr and matMidLev */
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    std::complex<double> middle_level_tmp(0.0,0.0), middle_level_inf_tmp(0.0,0.0), middle_level_corr_tmp(0.0,0.0);
    double kbarx, kbary;
    for (size_t kpx=0; kpx<_Gk._kArr_l.size(); kpx++){
        for (size_t kpy=0; kpy<_Gk._kArr_l.size(); kpy++){
            for (size_t ikpn=0; ikpn<_Gk._size; ikpn++){
                for (size_t ktildex=0; ktildex<_Gk._kArr_l.size(); ktildex++){
                    kbarx=_Gk._kArr_l[ktildex]-kbarx_m_tildex;
                    for (size_t ktildey=0; ktildey<_Gk._kArr_l.size(); ktildey++){
                        kbary=_Gk._kArr_l[ktildey]-kbary_m_tildey;
                        middle_level_inf_tmp+=_Gk(_Gk._precomp_wn[ikpn],_Gk._kArr_l[kpx],_Gk._kArr_l[kpy]
                        )(0,0) * gamma_twoD_spsp_full_lower(_Gk._kArr_l[kpx],_Gk._kArr_l[kpy],kbarx,kbary,_Gk._precomp_wn[ikpn],wbar
                        ) * _Gk(_Gk._precomp_wn[ikpn]-_qq._iwn,_Gk._kArr_l[kpx]-_qq._qx,_Gk._kArr_l[kpy]-_qq._qy)(0,0);
                    }
                }
                middle_level_inf_tmp*=1.0/_Gk._Nk*1.0/_Gk._Nk;
            }
        }
    }
    middle_level_inf_tmp*=SPINDEG/(_Gk._Nk*_Gk._Nk*_Gk._beta);
    middle_level_corr_tmp+=middle_level_inf_tmp; // This defines the correction term.
    middle_level_inf_tmp+=std::get<1>(gamma_twoD_spsp(kbarx_m_tildex,kbary_m_tildey,wtilde,wbar)); // This defines the full portion of the denominator.
    middle_level_tmp-=middle_level_inf_tmp;
    middle_level_tmp+=1.0;

    return std::make_tuple(GreenStuff::U/middle_level_tmp,middle_level_corr_tmp,middle_level_inf_tmp);
}

// Should be some tuple returned...otherwise the sum over the Matsubara frequencies isn't performed.
std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > ThreadWrapper::gamma_twoD_spsp_full_middle_plotting_IPT(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wbar,std::complex<double> wtilde) const{
    /* This function uses the externally linked matrices matCorr and matMidLev */
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    std::complex<double> middle_level_tmp(0.0,0.0), middle_level_inf_tmp(0.0,0.0), middle_level_corr_tmp(0.0,0.0);
    double kbarx, kbary;
    for (size_t kpx=0; kpx<_splInline.k_array.size(); kpx++){
        for (size_t kpy=0; kpy<_splInline.k_array.size(); kpy++){
            for (size_t ikpn=static_cast<size_t>(_splInline.iwn_array.size()/2); ikpn<_splInline.iwn_array.size(); ikpn++){
                for (size_t ktildex=0; ktildex<_splInline.k_array.size(); ktildex++){
                    kbarx=_splInline.k_array[ktildex]-kbarx_m_tildex;
                    for (size_t ktildey=0; ktildey<_splInline.k_array.size(); ktildey++){
                        kbary=_splInline.k_array[ktildey]-kbary_m_tildey;
                        middle_level_inf_tmp+=_Gk(_splInline.iwn_array[ikpn],_splInline.k_array[kpx],_splInline.k_array[kpy]
                        )(0,0) * gamma_twoD_spsp_full_lower_IPT(_splInline.k_array[kpx],_splInline.k_array[kpy],kbarx,kbary,_splInline.iwn_array[ikpn],wbar
                        ) * _Gk(_splInline.iwn_array[ikpn]-_qq._iwn,_splInline.k_array[kpx]-_qq._qx,_splInline.k_array[kpy]-_qq._qy)(0,0);
                    }
                }
                middle_level_inf_tmp*=1.0/GreenStuff::N_k*1.0/GreenStuff::N_k;
            }
        }
    }
    middle_level_inf_tmp*=SPINDEG/(GreenStuff::N_k*GreenStuff::N_k*GreenStuff::beta);
    middle_level_corr_tmp+=middle_level_inf_tmp; // This defines the correction term.
    middle_level_inf_tmp+=std::get<1>(gamma_twoD_spsp_IPT(kbarx_m_tildex,kbary_m_tildey,wtilde,wbar)); // This defines the full portion of the denominator.
    middle_level_tmp-=middle_level_inf_tmp;
    middle_level_tmp+=1.0;

    return std::make_tuple(GreenStuff::U/middle_level_tmp,middle_level_corr_tmp,middle_level_inf_tmp);
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

void get_vector_mpi(size_t totSize,bool is_jj,bool is_full,solver_prototype sp,std::vector<mpistruct_t>* vec_root_process){
    size_t idx=0;
    for (size_t lkt=0; lkt<vecK.size(); lkt++){
        for (size_t lkb=0; lkb<vecK.size(); lkb++){
            mpistruct_t MPIObj;
            MPIObj._lkt=lkt;
            MPIObj._lkb=lkb;
            MPIObj._is_jj=is_jj;
            MPIObj._is_full=is_full;
            MPIObj._sp=sp;
            vec_root_process->at(idx) = MPIObj;
            idx++;
        }
    }       
}

void fetch_data_from_slaves(int an_id,MPI_Status& status,bool is_full,int ierr,size_t num_elements_per_proc,size_t sizeOfTuple){
    char chars_to_receive[50];
    int sizeOfGamma, sizeOfWeights, sizeOfTotSus, sizeOfMidLev, sizeOfCorr, sender;
    std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecGammaTmp = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >(num_elements_per_proc);
    std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecWeightsTmp = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >(num_elements_per_proc);
    std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecTotSusTmp = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >(num_elements_per_proc);
    std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecMidLevTmp = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >(num_elements_per_proc);
    std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecCorrTmp = nullptr;
    if (is_full)
        vecCorrTmp = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >(num_elements_per_proc);
    // Should send the sizes of the externally linked vectors of tuples to be able to receive. That is why the need to probe...
    MPI_Probe(an_id,RETURN_DATA_TAG_GAMMA,MPI_COMM_WORLD,&status);
    MPI_Get_count(&status,MPI_BYTE,&sizeOfGamma);
    MPI_Probe(an_id,RETURN_DATA_TAG_WEIGHTS,MPI_COMM_WORLD,&status);
    MPI_Get_count(&status,MPI_BYTE,&sizeOfWeights);
    MPI_Probe(an_id,RETURN_DATA_TAG_TOT_SUS,MPI_COMM_WORLD,&status);
    MPI_Get_count(&status,MPI_BYTE,&sizeOfTotSus);
    MPI_Probe(an_id,RETURN_DATA_TAG_MID_LEV,MPI_COMM_WORLD,&status);
    MPI_Get_count(&status,MPI_BYTE,&sizeOfMidLev);
    if (is_full){
        MPI_Probe(an_id,RETURN_DATA_TAG_CORR,MPI_COMM_WORLD,&status);
        MPI_Get_count(&status,MPI_BYTE,&sizeOfCorr);
    }
    
    ierr = MPI_Recv( chars_to_receive, 50, MPI_CHAR, an_id,
            RETURN_DATA_TAG, MPI_COMM_WORLD, &status);
    if (is_full){
        ierr = MPI_Recv( (void*)(vecCorrTmp->data()), sizeOfCorr, MPI_BYTE, an_id,
                RETURN_DATA_TAG_CORR, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    ierr = MPI_Recv( (void*)(vecMidLevTmp->data()), sizeOfMidLev, MPI_BYTE, an_id,
            RETURN_DATA_TAG_MID_LEV, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    ierr = MPI_Recv( (void*)(vecTotSusTmp->data()), sizeOfTotSus, MPI_BYTE, an_id,
            RETURN_DATA_TAG_TOT_SUS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    ierr = MPI_Recv( (void*)(vecWeightsTmp->data()), sizeOfWeights, MPI_BYTE, an_id,
            RETURN_DATA_TAG_WEIGHTS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    ierr = MPI_Recv( (void*)(vecGammaTmp->data()), sizeOfGamma, MPI_BYTE, an_id,
            RETURN_DATA_TAG_GAMMA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    sender = status.MPI_SOURCE;
    printf("Slave process %i returned\n", sender);
    printf("%s\n",chars_to_receive);
    /* Now the data received from the other processes have to be stored in their arma::Mats on root process */
    size_t kt,kb,ii;
    if (is_full){
        for (ii=0; ii<sizeOfCorr/sizeOfTuple; ii++){
            kb=std::get<0>(vecCorrTmp->at(ii));
            kt=std::get<1>(vecCorrTmp->at(ii));
            matCorr(kb,kt)=std::get<2>(vecCorrTmp->at(ii));
        }
    }
    for (ii=0; ii<sizeOfMidLev/sizeOfTuple; ii++){
        kb=std::get<0>(vecMidLevTmp->at(ii));
        kt=std::get<1>(vecMidLevTmp->at(ii));
        matMidLev(kb,kt)=std::get<2>(vecMidLevTmp->at(ii));
    }
    for (ii=0; ii<sizeOfGamma/sizeOfTuple; ii++){
        kb=std::get<0>(vecGammaTmp->at(ii));
        kt=std::get<1>(vecGammaTmp->at(ii));
        matGamma(kb,kt)=std::get<2>(vecGammaTmp->at(ii));
    }
    for (ii=0; ii<sizeOfWeights/sizeOfTuple; ii++){
        kb=std::get<0>(vecWeightsTmp->at(ii));
        kt=std::get<1>(vecWeightsTmp->at(ii));
        matWeigths(kb,kt)=std::get<2>(vecWeightsTmp->at(ii));
    }
    for (ii=0; ii<sizeOfTotSus/sizeOfTuple; ii++){
        kb=std::get<0>(vecTotSusTmp->at(ii));
        kt=std::get<1>(vecTotSusTmp->at(ii));
        matTotSus(kb,kt)=std::get<2>(vecTotSusTmp->at(ii));
    }
    delete vecGammaTmp; delete vecWeightsTmp;
    delete vecTotSusTmp; delete vecMidLevTmp;
    if (is_full)
        delete vecCorrTmp;
}