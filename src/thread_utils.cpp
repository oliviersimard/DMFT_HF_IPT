#include "thread_utils.hpp"

using namespace ThreadFunctor;

arma::Mat< std::complex<double> > matGamma; // Matrices (and tensor) used in case parallel. Initialized in the main.
arma::Mat< std::complex<double> > matWeigths;
arma::Mat< std::complex<double> > matTotSus;
arma::Mat< std::complex<double> > matCorr;
arma::Mat< std::complex<double> > matMidLev;
std::complex<double>**** gamma_tensor;
std::complex<double>**** gamma_full_tensor; // Only useful when computing the full ladder diagrams. (1D)
std::vector< gamma_tensor_content >* ThreadWrapper::vecGammaTensorContent = new std::vector< gamma_tensor_content >();
std::vector< gamma_tensor_content >* ThreadWrapper::vecGammaFullTensorContent = new std::vector< gamma_tensor_content >(); // Only useful when computing the full ladder diagrams. (1D)
int root_process=0;
std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecGammaSlaves = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >();
std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecWeightsSlaves = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >();
std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecTotSusSlaves = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >();
std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecCorrSlaves = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >();
std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecMidLevSlaves = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >();

#if DIM == 1
void ThreadWrapper::operator()(size_t ktilde, size_t kbar, bool is_jj, bool is_full, size_t j, solver_prototype sp) const{
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    std::complex<double> tmp_val_kt_kb(0.0,0.0),tmp_val_weights(0.0,0.0),tmp_val_tot_sus(0.0,0.0),
    tmp_val_mid_lev(0.0,0.0),tmp_val_corr(0.0,0.0);
    std::complex<double> tmp_val_kt_kb_tmp, tmp_val_weights_tmp;
    std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > tupOfValsFull;
    std::tuple< std::complex<double>,std::complex<double> > tupOfVals;
    switch(sp){
    case solver_prototype::HF_prot:
        for (size_t wbar=0; wbar<_Gk._size; wbar++){
            for (size_t wtilde=0; wtilde<_Gk._size; wtilde++){
                tmp_val_weights_tmp = _Gk(_Gk._precomp_wn[wtilde],_Gk._kArr_l[ktilde]
                                    )(0,0)*_Gk(_Gk._precomp_wn[wtilde]-_q._iwn,_Gk._kArr_l[ktilde]-_q._qx
                                    )(0,0)*_Gk(_Gk._precomp_wn[wbar]-_q._iwn,_Gk._kArr_l[kbar]-_q._qx
                                    )(1,1)*_Gk(_Gk._precomp_wn[wbar],_Gk._kArr_l[kbar]
                                    )(1,1);
                if (!is_full){
                    if (j==0){
                        tupOfVals = gamma_oneD_spsp(_Gk._kArr_l[ktilde],_Gk._precomp_wn[wtilde],_Gk._kArr_l[kbar],_Gk._precomp_wn[wbar]);
                        tmp_val_kt_kb_tmp = std::get<0>(tupOfVals);
                        gamma_tensor_content GammaTObj(ktilde,wtilde,kbar,wbar,tmp_val_kt_kb_tmp);
                        vecGammaTensorContent->push_back(std::move(GammaTObj));
                        tmp_val_mid_lev += std::get<1>(tupOfVals);
                        tmp_val_kt_kb += tmp_val_kt_kb_tmp;
                        tmp_val_tot_sus += tmp_val_kt_kb_tmp*tmp_val_weights_tmp;
                    } else{
                        tmp_val_tot_sus += gamma_tensor[ktilde][wtilde][kbar][wbar]*tmp_val_weights_tmp;
                    }
                } else{
                    tupOfValsFull = gamma_oneD_spsp_full_middle_plotting(ktilde,kbar,wbar,wtilde,j);
                    tmp_val_kt_kb_tmp = std::get<0>(tupOfValsFull);
                    tmp_val_corr += std::get<1>(tupOfValsFull);
                    tmp_val_mid_lev += std::get<2>(tupOfValsFull);
                    tmp_val_kt_kb += tmp_val_kt_kb_tmp;
                    tmp_val_tot_sus += tmp_val_kt_kb_tmp*tmp_val_weights_tmp;
                }
                tmp_val_weights += tmp_val_weights_tmp;
                if ((wtilde==0) && (wbar==0)){
                    std::cout << "Process id: " << world_rank << "\n";
                    std::cout << "ktilde: " << _Gk._kArr_l[ktilde] << "\n";
                    std::cout << "kbar: " << _Gk._kArr_l[kbar] << "\n";
                    std::cout << "Tot Sus: " << tmp_val_tot_sus << "\n";
                    std::cout << "Weights: " << tmp_val_weights << std::endl;
                }
            } 
        }
        save_data_to_local_extern_matrix_instances(tmp_val_kt_kb,tmp_val_weights,tmp_val_mid_lev,tmp_val_corr,tmp_val_tot_sus,
                    ktilde,kbar,is_jj,is_full,world_rank,j);
        break;
    case solver_prototype::IPT2_prot:
        for (size_t wbar=static_cast<size_t>(_splInline._iwn_array.size()/2); wbar<_splInline._iwn_array.size(); wbar++){
            for (size_t wtilde=static_cast<size_t>(_splInline._iwn_array.size()/2); wtilde<_splInline._iwn_array.size(); wtilde++){
                tmp_val_weights_tmp = buildGK1D_IPT(
                                    _splInline._iwn_array[wtilde],_splInline._k_array[ktilde]
                                    )[0]*buildGK1D_IPT(
                                    _splInline._iwn_array[wtilde]-_q._iwn,_splInline._k_array[ktilde]-_q._qx
                                    )[0]*buildGK1D_IPT(
                                    _splInline._iwn_array[wbar]-_q._iwn,_splInline._k_array[kbar]-_q._qx
                                    )[1]*buildGK1D_IPT(
                                    _splInline._iwn_array[wbar],_splInline._k_array[kbar]
                                    )[1];
                if (!is_full){
                    if (j==0){
                        tupOfVals = gamma_oneD_spsp_IPT(_splInline._k_array[ktilde],_splInline._iwn_array[wtilde],_splInline._k_array[kbar],_splInline._iwn_array[wbar]);
                        tmp_val_kt_kb_tmp = std::get<0>(tupOfVals);
                        gamma_tensor_content GammaTObj(ktilde,wtilde%static_cast<size_t>(_splInline._iwn_array.size()/2),kbar,wbar%static_cast<size_t>(_splInline._iwn_array.size()/2),tmp_val_kt_kb_tmp);
                        vecGammaTensorContent->push_back(std::move(GammaTObj));
                        tmp_val_mid_lev += std::get<1>(tupOfVals);
                        tmp_val_kt_kb += tmp_val_kt_kb_tmp;
                        tmp_val_tot_sus += tmp_val_kt_kb_tmp*tmp_val_weights_tmp;
                    } else{
                        tmp_val_tot_sus += gamma_tensor[ktilde][wtilde%static_cast<size_t>(_splInline._iwn_array.size()/2)][kbar][wbar%static_cast<size_t>(_splInline._iwn_array.size()/2)]*tmp_val_weights_tmp;
                    }
                } else{
                    tupOfValsFull = gamma_oneD_spsp_full_middle_plotting_IPT(ktilde,kbar,wbar,wtilde,j);
                    tmp_val_kt_kb_tmp = std::get<0>(tupOfValsFull);
                    tmp_val_corr += std::get<1>(tupOfValsFull);
                    tmp_val_mid_lev += std::get<2>(tupOfValsFull);
                    tmp_val_kt_kb += tmp_val_kt_kb_tmp;
                    tmp_val_tot_sus += tmp_val_kt_kb_tmp*tmp_val_weights_tmp;
                }
                tmp_val_weights += tmp_val_weights_tmp;
                if ((wtilde==static_cast<size_t>(_splInline._iwn_array.size()/2)) && (wbar==static_cast<size_t>(_splInline._iwn_array.size()/2))){
                    std::cout << "Process id: " << world_rank << "\n";
                    std::cout << "ktilde: " << _splInline._k_array[ktilde] << "\n";
                    std::cout << "kbar: " << _splInline._k_array[kbar] << "\n";
                    std::cout << "Tot Sus IPT: " << tmp_val_tot_sus << "\n";
                    std::cout << "Weights: " << tmp_val_weights << std::endl;
                }
            } 
        }
        save_data_to_local_extern_matrix_instancesIPT(tmp_val_kt_kb,tmp_val_weights,tmp_val_mid_lev,tmp_val_corr,tmp_val_tot_sus,
                    ktilde,kbar,is_jj,is_full,world_rank,j);
        break;
    }
}

std::tuple< std::complex<double>,std::complex<double> > ThreadWrapper::gamma_oneD_spsp(double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar) const{
    std::complex<double> lower_level(0.0,0.0), bubble;
    for (size_t wttilde=0; wttilde<_Gk._size; wttilde++){
        for (size_t qttilde=0; qttilde<_Gk._kArr_l.size(); qttilde++){
            if ( ( qttilde==0 ) || ( qttilde==(_Gk._kArr_l.size()-1) ) )
                lower_level += 0.5*_Gk(wtilde-_Gk._precomp_qn[wttilde],ktilde-_Gk._kArr_l[qttilde])(0,0) * _Gk(wbar-_Gk._precomp_qn[wttilde],kbar-_Gk._kArr_l[qttilde])(1,1);
            else
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
    for (size_t qttilde=0; qttilde<_splInline._k_array.size(); qttilde++){
        if ( (qttilde==0) || (qttilde==_splInline._k_array.size()-1) ){
            for (size_t wttilde=0; wttilde<iqnArr_l.size(); wttilde++){
                lower_level += 0.5*buildGk1D_IPT(wtilde-iqnArr_l[wttilde],ktilde-_splInline._k_array[qttilde])*buildGk1D_IPT(wbar-iqnArr_l[wttilde],kbar-_splInline._k_array[qttilde]);
            }
        }else{
            for (size_t wttilde=0; wttilde<iqnArr_l.size(); wttilde++){
                lower_level += buildGk1D_IPT(wtilde-iqnArr_l[wttilde],ktilde-_splInline._k_array[qttilde])*buildGk1D_IPT(wbar-iqnArr_l[wttilde],kbar-_splInline._k_array[qttilde]);
            }
        }
    }
    
    lower_level *= SPINDEG*GreenStuff::U/(GreenStuff::beta*GreenStuff::N_k);
    bubble = -1.0*lower_level;
    lower_level += 1.0;
    return std::make_tuple(GreenStuff::U/lower_level,bubble);
}

std::complex<double> ThreadWrapper::gamma_oneD_spsp_full_lower(double kp,double kbar,std::complex<double> iknp,std::complex<double> wbar) const{
    std::complex<double> lower_level(0.0,0.0);
    for (size_t iknpp=0; iknpp<_Gk._size; iknpp++){
        for (size_t kpp=0; kpp<_Gk._kArr_l.size(); kpp++){
            if ( ( kpp==0 ) || ( kpp==(_Gk._kArr_l.size()-1) ) )
                lower_level += 0.5*_Gk(_Gk._precomp_wn[iknpp]+iknp-wbar,_Gk._kArr_l[kpp]+kp-kbar)(0,0)*_Gk(_Gk._precomp_wn[iknpp],_Gk._kArr_l[kpp])(1,1);
            else
                lower_level += _Gk(_Gk._precomp_wn[iknpp]+iknp-wbar,_Gk._kArr_l[kpp]+kp-kbar)(0,0)*_Gk(_Gk._precomp_wn[iknpp],_Gk._kArr_l[kpp])(1,1);
        }
    }
    lower_level*=SPINDEG*_Gk._u/(_Gk._beta*_Gk._Nk);
    lower_level+=1.0;

    return _Gk._u/lower_level;
}

std::complex<double> ThreadWrapper::gamma_oneD_spsp_full_lower_IPT(double kp,double kbar,std::complex<double> iknp,std::complex<double> wbar) const{
    std::complex<double> lower_level(0.0,0.0);
    for (size_t iknpp=static_cast<size_t>(_splInline._iwn_array.size()/2); iknpp<_splInline._iwn_array.size(); iknpp++){
        for (size_t kpp=0; kpp<_splInline._k_array.size(); kpp++){
            if ( (kpp==0) || (kpp==_splInline._k_array.size()-1) )
                lower_level += 0.5*buildGK1D_IPT(_splInline._iwn_array[iknpp]+iknp-wbar,_splInline._k_array[kpp]+kp-kbar)[0] * buildGK1D_IPT(_splInline._iwn_array[iknpp],_splInline._k_array[kpp])[1];
            else
                lower_level += buildGK1D_IPT(_splInline._iwn_array[iknpp]+iknp-wbar,_splInline._k_array[kpp]+kp-kbar)[0] * buildGK1D_IPT(_splInline._iwn_array[iknpp],_splInline._k_array[kpp])[1];
        }
    }
    lower_level*=SPINDEG*GreenStuff::U/(GreenStuff::beta*GreenStuff::N_k);
    lower_level+=1.0;

    return GreenStuff::U/lower_level;
}

std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > ThreadWrapper::gamma_oneD_spsp_full_middle_plotting(size_t ktilde,size_t kbar,size_t wbar,size_t wtilde,size_t j) const{
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    std::complex<double> middle_level_tmp(0.0,0.0), middle_level_inf_tmp(0.0,0.0), middle_level_corr_tmp(0.0,0.0);
    std::complex<double> tmp_middle_level_inf_tmp, tmp_bubble_without_corr_tmp;
    for (size_t kp=0; kp<_Gk._kArr_l.size(); kp++){
        for (size_t iknp=0; iknp<_Gk._size; iknp++){
            if (j==0){ // Saving lower-most component of the full ladder susceptibility.
                tmp_middle_level_inf_tmp = gamma_oneD_spsp_full_lower(_Gk._kArr_l[kp],_Gk._kArr_l[kbar],_Gk._precomp_wn[iknp],_Gk._precomp_wn[wbar]);
                if (ktilde==0 && wtilde==0){
                    gamma_tensor_content GammaTObj(kp,iknp,kbar,wbar,tmp_middle_level_inf_tmp);
                    vecGammaFullTensorContent->push_back(std::move(GammaTObj));
                }
            } else{
                tmp_middle_level_inf_tmp = gamma_full_tensor[kbar][wbar][kp][iknp];
            }
            middle_level_corr_tmp+=tmp_middle_level_inf_tmp; // Extracting the lower level
            if ( ( kp==0 ) || ( kp==(_Gk._kArr_l.size()-1) ) ){
                middle_level_inf_tmp += 0.5*_Gk(_Gk._precomp_wn[iknp],_Gk._kArr_l[kp]
                )(0,0)*tmp_middle_level_inf_tmp*_Gk(_Gk._precomp_wn[iknp]-_q._iwn,_Gk._kArr_l[kp]-_q._qx)(0,0);
            } else{
                middle_level_inf_tmp += _Gk(_Gk._precomp_wn[iknp],_Gk._kArr_l[kp]
                )(0,0)*tmp_middle_level_inf_tmp*_Gk(_Gk._precomp_wn[iknp]-_q._iwn,_Gk._kArr_l[kp]-_q._qx)(0,0);
            }
        }
    }
    middle_level_inf_tmp*=SPINDEG/(_Gk._Nk*_Gk._beta);
    middle_level_corr_tmp*=SPINDEG/(_Gk._Nk*_Gk._beta);
    if (j==0){ // Saving the bare denominator of the full susceptibility.
        tmp_bubble_without_corr_tmp = std::get<1>(gamma_oneD_spsp(_Gk._kArr_l[ktilde],_Gk._precomp_wn[wtilde],_Gk._kArr_l[kbar],_Gk._precomp_wn[wbar]));
        gamma_tensor_content GammaTObj(ktilde,wtilde,kbar,wbar,tmp_bubble_without_corr_tmp);
        vecGammaTensorContent->push_back(std::move(GammaTObj));
    } else{
        tmp_bubble_without_corr_tmp = gamma_tensor[ktilde][wtilde][kbar][wbar];
    }
    middle_level_inf_tmp+=tmp_bubble_without_corr_tmp;
    middle_level_tmp-=middle_level_inf_tmp;
    middle_level_tmp+=1.0;

    return std::make_tuple(GreenStuff::U/middle_level_tmp,middle_level_corr_tmp,middle_level_inf_tmp);
}

std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > ThreadWrapper::gamma_oneD_spsp_full_middle_plotting_IPT(size_t ktilde,size_t kbar,size_t wbar,size_t wtilde,size_t j) const{
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    std::complex<double> middle_level_tmp(0.0,0.0), middle_level_inf_tmp(0.0,0.0), middle_level_corr_tmp(0.0,0.0);
    std::complex<double> tmp_middle_level_inf_tmp, tmp_bubble_without_corr_tmp;
    for (size_t kp=0; kp<_splInline._k_array.size(); kp++){
        for (size_t iknp=static_cast<size_t>(_splInline._iwn_array.size()/2); iknp<_splInline._iwn_array.size(); iknp++){
            if (j==0){ // One slice of ktilde-wtilde is necessary.
                tmp_middle_level_inf_tmp = gamma_oneD_spsp_full_lower_IPT(_splInline._k_array[kp],_splInline._k_array[kbar],_splInline._iwn_array[iknp],_splInline._iwn_array[wbar]);
                if (ktilde==0 && wtilde==static_cast<size_t>(_splInline._iwn_array.size()/2)){
                    gamma_tensor_content GammaTObj(kp,iknp%static_cast<size_t>(_splInline._iwn_array.size()/2),kbar,wbar%static_cast<size_t>(_splInline._iwn_array.size()/2),tmp_middle_level_inf_tmp);
                    vecGammaFullTensorContent->push_back(std::move(GammaTObj));
                }
            } else{ // Then one has to fetch the stored values...
                tmp_middle_level_inf_tmp = gamma_full_tensor[kbar][wbar%static_cast<size_t>(_splInline._iwn_array.size()/2)][kp][iknp%static_cast<size_t>(_splInline._iwn_array.size()/2)];
            }
            middle_level_corr_tmp += tmp_middle_level_inf_tmp; // Extracting the lower level
            if ( (kp==0) || (kp==(_splInline._k_array.size()-1)) ){
                middle_level_inf_tmp += 0.5*buildGK1D_IPT(_splInline._iwn_array[iknp],_splInline._k_array[kp]
                )[0]*tmp_middle_level_inf_tmp*buildGK1D_IPT(_splInline._iwn_array[iknp]-_q._iwn,_splInline._k_array[kp]-_q._qx)[0];
            } else{
                middle_level_inf_tmp += buildGK1D_IPT(_splInline._iwn_array[iknp],_splInline._k_array[kp]
                )[0]*tmp_middle_level_inf_tmp*buildGK1D_IPT(_splInline._iwn_array[iknp]-_q._iwn,_splInline._k_array[kp]-_q._qx)[0];
            }
        }
    }
    middle_level_inf_tmp*=SPINDEG/(GreenStuff::N_k*GreenStuff::beta);
    middle_level_corr_tmp*=SPINDEG/(GreenStuff::N_k*GreenStuff::beta);
    if (j==0){
        tmp_bubble_without_corr_tmp = std::get<1>(gamma_oneD_spsp_IPT(_splInline._k_array[ktilde],_splInline._iwn_array[wtilde],_splInline._k_array[kbar],_splInline._iwn_array[wbar]));
        gamma_tensor_content GammaTObj(ktilde,wtilde%static_cast<size_t>(_splInline._iwn_array.size()/2),kbar,wbar%static_cast<size_t>(_splInline._iwn_array.size()/2),tmp_bubble_without_corr_tmp);
        vecGammaTensorContent->push_back(std::move(GammaTObj));
    } else{
        tmp_bubble_without_corr_tmp = gamma_tensor[ktilde][wtilde%static_cast<size_t>(_splInline._iwn_array.size()/2)][kbar][wbar%static_cast<size_t>(_splInline._iwn_array.size()/2)];
    }
    middle_level_inf_tmp+=tmp_bubble_without_corr_tmp;
    middle_level_tmp-=middle_level_inf_tmp;
    middle_level_tmp+=1.0;

    return std::make_tuple(GreenStuff::U/middle_level_tmp,middle_level_corr_tmp,middle_level_inf_tmp);
}

#elif DIM == 2
void ThreadWrapper::operator()(solver_prototype sp, size_t kbarx_m_tildex, size_t kbary_m_tildey, bool is_jj, bool is_full, size_t j) const{
/* In 2D, calculating the weights implies computing whilst dismissing the translational invariance of the vertex function (Gamma). */
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    std::complex<double> tmp_val_kt_kb(0.0,0.0), tmp_val_weights(0.0,0.0), tmp_val_tot_sus(0.0,0.0), tmp_val_mid_lev(0.0,0.0);
    std::complex<double> tmp_val_corr(0.0,0.0);
    std::complex<double> tmp_val_kt_kb_tmp, tmp_val_weights_tmp;
    std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > tupOfValsFull;
    std::tuple< std::complex<double>,std::complex<double> > tupOfVals;
    switch(sp){
    case solver_prototype::HF_prot:
        // Looking whether the calculations concern the full calculation considering the corrections or not.
        for (size_t wtilde=0; wtilde<_Gk._size; wtilde++){
            for (size_t wbar=0; wbar<_Gk._size; wbar++){
                tmp_val_weights_tmp = getWeightsHF(_Gk._kArr_l[kbarx_m_tildex],_Gk._kArr_l[kbary_m_tildey],wtilde,wbar);
                if (!is_full){
                    if (j==0){
                        tupOfVals = gamma_twoD_spsp(_Gk._kArr_l[kbarx_m_tildex],_Gk._kArr_l[kbary_m_tildey],_Gk._precomp_wn[wtilde],_Gk._precomp_wn[wbar]);
                        tmp_val_kt_kb_tmp = std::get<0>(tupOfVals);
                        gamma_tensor_content GammaTObj(kbarx_m_tildex,wtilde,kbary_m_tildey,wbar,tmp_val_kt_kb_tmp);
                        vecGammaTensorContent->push_back(std::move(GammaTObj));
                        tmp_val_mid_lev += std::get<1>(tupOfVals);
                        tmp_val_kt_kb+=tmp_val_kt_kb_tmp;
                        tmp_val_tot_sus+=tmp_val_kt_kb_tmp*tmp_val_weights_tmp;
                    } else{
                        tmp_val_tot_sus += gamma_tensor[kbarx_m_tildex][wtilde][kbary_m_tildey][wbar]*tmp_val_weights_tmp;
                    }
                } else{
                    tupOfValsFull = gamma_twoD_spsp_full_middle_plotting(_Gk._kArr_l[kbarx_m_tildex],_Gk._kArr_l[kbary_m_tildey],_Gk._precomp_wn[wbar],_Gk._precomp_wn[wtilde]);
                    tmp_val_kt_kb_tmp = std::get<0>(tupOfValsFull);
                    tmp_val_corr += std::get<1>(tupOfValsFull);
                    tmp_val_mid_lev += std::get<2>(tupOfValsFull);
                    tmp_val_kt_kb += tmp_val_kt_kb_tmp;
                    tmp_val_tot_sus += tmp_val_kt_kb_tmp*tmp_val_weights_tmp;
                }
                tmp_val_weights+=tmp_val_weights_tmp;
                if ((wtilde==0) && (wbar==0)){
                    std::cout << "Process id: " << world_rank << "\n";
                    std::cout << "kbarx_m_tildex: " << _Gk._kArr_l[kbarx_m_tildex] << "\n";
                    std::cout << "kbary_m_tildey: " << _Gk._kArr_l[kbary_m_tildey] << "\n";
                    std::cout << "Tot Sus: " << tmp_val_tot_sus << "\n";
                    std::cout << "Weights: " << tmp_val_weights << std::endl;
                }
            } 
        }
        save_data_to_local_extern_matrix_instances(tmp_val_kt_kb,tmp_val_weights,tmp_val_mid_lev,tmp_val_corr,tmp_val_tot_sus,
                    kbarx_m_tildex,kbary_m_tildey,is_jj,is_full,world_rank,j);
        break;
    case solver_prototype::IPT2_prot:
        for (size_t wtilde=static_cast<size_t>(_splInline._iwn_array.size()/2); wtilde<_splInline._iwn_array.size(); wtilde++){
            for (size_t wbar=static_cast<size_t>(_splInline._iwn_array.size()/2); wbar<_splInline._iwn_array.size(); wbar++){
                tmp_val_weights_tmp = getWeightsIPT(_splInline._k_array[kbarx_m_tildex],_splInline._k_array[kbary_m_tildey],wtilde,wbar);
                if (!is_full){
                    if (j==0){
                        tupOfVals = gamma_twoD_spsp_IPT(_splInline._k_array[kbarx_m_tildex],_splInline._k_array[kbary_m_tildey],_splInline._iwn_array[wtilde],_splInline._iwn_array[wbar]);
                        tmp_val_kt_kb_tmp = std::get<0>(tupOfVals);
                        gamma_tensor_content GammaTObj(kbarx_m_tildex,wtilde%static_cast<size_t>(_splInline._iwn_array.size()/2),kbary_m_tildey,wbar%static_cast<size_t>(_splInline._iwn_array.size()/2),tmp_val_kt_kb_tmp);
                        vecGammaTensorContent->push_back(std::move(GammaTObj));
                        tmp_val_mid_lev += std::get<1>(tupOfVals);
                        tmp_val_kt_kb += tmp_val_kt_kb_tmp;
                        tmp_val_tot_sus += tmp_val_kt_kb_tmp*tmp_val_weights_tmp;
                    } else{
                        tmp_val_tot_sus += gamma_tensor[kbarx_m_tildex][wtilde%static_cast<size_t>(_splInline._iwn_array.size()/2)][kbary_m_tildey][wbar%static_cast<size_t>(_splInline._iwn_array.size()/2)]*tmp_val_weights_tmp;
                    }
                }
                else{
                    tupOfValsFull = gamma_twoD_spsp_full_middle_plotting_IPT(_splInline._k_array[kbarx_m_tildex],_splInline._k_array[kbary_m_tildey],_splInline._iwn_array[wbar],_splInline._iwn_array[wtilde]);
                    tmp_val_kt_kb_tmp = std::get<0>(tupOfValsFull);
                    tmp_val_corr += std::get<1>(tupOfValsFull);
                    tmp_val_mid_lev += std::get<2>(tupOfValsFull);
                    tmp_val_kt_kb += tmp_val_kt_kb_tmp;
                    tmp_val_tot_sus += tmp_val_kt_kb_tmp*tmp_val_weights_tmp;
                }
                tmp_val_weights+=tmp_val_weights_tmp;
                if ((wtilde==static_cast<size_t>(_splInline._iwn_array.size()/2)) && (wbar==static_cast<size_t>(_splInline._iwn_array.size()/2))){
                    std::cout << "Process id: " << world_rank << "\n";
                    std::cout << "kbarx_m_tildex: " << _splInline._k_array[kbarx_m_tildex] << "\n";
                    std::cout << "kbary_m_tildey: " << _splInline._k_array[kbary_m_tildey] << "\n";
                    std::cout << "Tot Sus: " << tmp_val_tot_sus << "\n";
                    std::cout << "Weights: " << tmp_val_weights << std::endl;
                }
            } 
        }
        save_data_to_local_extern_matrix_instancesIPT(tmp_val_kt_kb,tmp_val_weights,tmp_val_mid_lev,tmp_val_corr,tmp_val_tot_sus,
                    kbarx_m_tildex,kbary_m_tildey,is_jj,is_full,world_rank,j);
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
                if ( ( (qttildey==0) || (qttildey==(_Gk._kArr_l.size()-1)) ) || ( (qttildex==0) || (qttildex==(_Gk._kArr_l.size()-1)) ) ){
                    if ( ( (qttildey==0) || (qttildey==(_Gk._kArr_l.size()-1)) ) && ( (qttildex==0) || (qttildex==(_Gk._kArr_l.size()-1)) ) ){
                        lower_level += 0.25*_Gk((wtilde-_Gk._precomp_qn[wttilde]),_Gk._kArr_l[qttildex],_Gk._kArr_l[qttildey])(0,0)*_Gk((wbar-_Gk._precomp_qn[wttilde]),(_Gk._kArr_l[qttildex]+kbarx_m_tildex),(_Gk._kArr_l[qttildey]+kbary_m_tildey))(1,1);
                    } else{
                        lower_level += 0.5*_Gk((wtilde-_Gk._precomp_qn[wttilde]),_Gk._kArr_l[qttildex],_Gk._kArr_l[qttildey])(0,0)*_Gk((wbar-_Gk._precomp_qn[wttilde]),(_Gk._kArr_l[qttildex]+kbarx_m_tildex),(_Gk._kArr_l[qttildey]+kbary_m_tildey))(1,1);
                    }
                } else{
                    lower_level += _Gk((wtilde-_Gk._precomp_qn[wttilde]),_Gk._kArr_l[qttildex],_Gk._kArr_l[qttildey])(0,0)*_Gk((wbar-_Gk._precomp_qn[wttilde]),(_Gk._kArr_l[qttildex]+kbarx_m_tildex),(_Gk._kArr_l[qttildey]+kbary_m_tildey))(1,1);
                }
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
    for (size_t wttilde=0; wttilde<_splInline._iqn_array.size(); wttilde++){
        for (size_t qttildey=0; qttildey<_splInline._k_array.size(); qttildey++){
            for (size_t qttildex=0; qttildex<_splInline._k_array.size(); qttildex++){ // the change of variable only applies to k-space, due to periodicity modulo 2pi.
                if ( ( (qttildey==0) || (qttildey==(static_cast<size_t>(_splInline._k_array.size()-1))) ) || ( (qttildex==0) || (qttildex==(static_cast<size_t>(_splInline._k_array.size()-1))) ) ){
                    if ( ( (qttildey==0) || (qttildey==(static_cast<size_t>(_splInline._k_array.size()-1))) ) && ( (qttildex==0) || (qttildex==(static_cast<size_t>(_splInline._k_array.size()-1))) ) ){
                        lower_level += 0.25*buildGK2D_IPT((wtilde-_splInline._iqn_array[wttilde]),_splInline._k_array[qttildex],_splInline._k_array[qttildey])[0]*buildGK2D_IPT((wbar-_splInline._iqn_array[wttilde]),(_splInline._k_array[qttildex]+kbarx_m_tildex),(_splInline._k_array[qttildey]+kbary_m_tildey))[1];
                    } else{
                        lower_level += 0.5*buildGK2D_IPT((wtilde-_splInline._iqn_array[wttilde]),_splInline._k_array[qttildex],_splInline._k_array[qttildey])[0]*buildGK2D_IPT((wbar-_splInline._iqn_array[wttilde]),(_splInline._k_array[qttildex]+kbarx_m_tildex),(_splInline._k_array[qttildey]+kbary_m_tildey))[1];
                    }
                } else{
                    lower_level += buildGK2D_IPT((wtilde-_splInline._iqn_array[wttilde]),_splInline._k_array[qttildex],_splInline._k_array[qttildey])[0]*buildGK2D_IPT((wbar-_splInline._iqn_array[wttilde]),(_splInline._k_array[qttildex]+kbarx_m_tildex),(_splInline._k_array[qttildey]+kbary_m_tildey))[1];
                }
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
                if ( ( (kppy==0) || (kppy==(_Gk._kArr_l.size()-1)) ) || ( (kppx==0) || (kppx==(_Gk._kArr_l.size()-1)) ) ){
                    if ( ( (kppy==0) || (kppy==(_Gk._kArr_l.size()-1)) ) && ( (kppx==0) || (kppx==(_Gk._kArr_l.size()-1)) ) ){
                        lower_level += 0.25*_Gk(_Gk._precomp_wn[iknpp]+iknp-wbar,_Gk._kArr_l[kppx]+kpx-kbarx,_Gk._kArr_l[kppy]+kpy-kbary)(0,0)*_Gk(_Gk._precomp_wn[iknpp],_Gk._kArr_l[kppx],_Gk._kArr_l[kppy])(1,1);
                    } else{
                        lower_level += 0.5*_Gk(_Gk._precomp_wn[iknpp]+iknp-wbar,_Gk._kArr_l[kppx]+kpx-kbarx,_Gk._kArr_l[kppy]+kpy-kbary)(0,0)*_Gk(_Gk._precomp_wn[iknpp],_Gk._kArr_l[kppx],_Gk._kArr_l[kppy])(1,1);
                    }
                } else{
                    lower_level += _Gk(_Gk._precomp_wn[iknpp]+iknp-wbar,_Gk._kArr_l[kppx]+kpx-kbarx,_Gk._kArr_l[kppy]+kpy-kbary)(0,0)*_Gk(_Gk._precomp_wn[iknpp],_Gk._kArr_l[kppx],_Gk._kArr_l[kppy])(1,1);
                }
            }
        }
    }
    lower_level *= SPINDEG*_Gk._u/(_Gk._beta*(_Gk._Nk)*(_Gk._Nk)); /// No minus sign at ground level. Factor 2 for spin.
    lower_level += 1.0;
    return _Gk._u/lower_level; // Means that we have to multiply the middle level of this component by the two missing Green's functions.
}

std::complex<double> ThreadWrapper::gamma_twoD_spsp_full_lower_IPT(double kpx,double kpy,double kbarx,double kbary,std::complex<double> iknp,std::complex<double> wbar) const{
    std::complex<double> lower_level(0.0,0.0);
    for (size_t iknpp=static_cast<size_t>(_splInline._iwn_array.size()/2); iknpp<_splInline._iwn_array.size(); iknpp++){
        for (size_t kppx=0; kppx<_splInline._k_array.size(); kppx++){
            for (size_t kppy=0; kppy<_splInline._k_array.size(); kppy++){
                if ( ( (kppy==0) || (kppy==(static_cast<size_t>(_splInline._k_array.size()-1))) ) || ( (kppx==0) || (kppx==(static_cast<size_t>(_splInline._k_array.size()-1))) ) ){
                    if ( ( (kppy==0) || (kppy==(static_cast<size_t>(_splInline._k_array.size()-1))) ) && ( (kppx==0) || (kppx==(static_cast<size_t>(_splInline._k_array.size()-1))) ) ){
                        lower_level += 0.25*buildGK2D_IPT(_splInline._iwn_array[iknpp]+iknp-wbar,_splInline._k_array[kppx]+kpx-kbarx,_splInline._k_array[kppy]+kpy-kbary)[0]*buildGK2D_IPT(_splInline._iwn_array[iknpp],_splInline._k_array[kppx],_splInline._k_array[kppy])[1];
                    } else{
                        lower_level += 0.5*buildGK2D_IPT(_splInline._iwn_array[iknpp]+iknp-wbar,_splInline._k_array[kppx]+kpx-kbarx,_splInline._k_array[kppy]+kpy-kbary)[0]*buildGK2D_IPT(_splInline._iwn_array[iknpp],_splInline._k_array[kppx],_splInline._k_array[kppy])[1];
                    }
                } else{
                    lower_level += buildGK2D_IPT(_splInline._iwn_array[iknpp]+iknp-wbar,_splInline._k_array[kppx]+kpx-kbarx,_splInline._k_array[kppy]+kpy-kbary)[0]*buildGK2D_IPT(_splInline._iwn_array[iknpp],_splInline._k_array[kppx],_splInline._k_array[kppy])[1];
                }
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
    for (size_t kpx=0; kpx<_splInline._k_array.size(); kpx++){
        for (size_t kpy=0; kpy<_splInline._k_array.size(); kpy++){
            for (size_t ikpn=static_cast<size_t>(_splInline._iwn_array.size()/2); ikpn<_splInline._iwn_array.size(); ikpn++){
                for (size_t ktildex=0; ktildex<_splInline._k_array.size(); ktildex++){
                    kbarx=_splInline._k_array[ktildex]-kbarx_m_tildex;
                    for (size_t ktildey=0; ktildey<_splInline._k_array.size(); ktildey++){
                        kbary=_splInline._k_array[ktildey]-kbary_m_tildey;
                        middle_level_inf_tmp+=_Gk(_splInline._iwn_array[ikpn],_splInline._k_array[kpx],_splInline._k_array[kpy]
                        )(0,0) * gamma_twoD_spsp_full_lower_IPT(_splInline._k_array[kpx],_splInline._k_array[kpy],kbarx,kbary,_splInline._iwn_array[ikpn],wbar
                        ) * _Gk(_splInline._iwn_array[ikpn]-_qq._iwn,_splInline._k_array[kpx]-_qq._qx,_splInline._k_array[kpy]-_qq._qy)(0,0);
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
            if ( ( (ktildey==0) || (ktildey==(_Gk._kArr_l.size()-1)) ) || ( (ktildex==0) || (ktildex==(_Gk._kArr_l.size()-1)) ) ){
                if ( ( (ktildey==0) || (ktildey==(_Gk._kArr_l.size()-1)) ) && ( (ktildex==0) || (ktildex==(_Gk._kArr_l.size()-1)) ) ){
                    tmp_val_weigths += 0.25*_Gk(wtilde,_Gk._kArr_l[ktildex],_Gk._kArr_l[ktildey])(0,0)*_Gk(wtilde-_qq._iwn,_Gk._kArr_l[ktildex]-_qq._qx,_Gk._kArr_l[ktildey]-_qq._qy
                    )(0,0)*_Gk(wbar-_qq._iwn,kbarx-_qq._qx,kbary-_qq._qy)(1,1)*_Gk(wbar,kbarx,kbary)(1,1);
                } else{
                    tmp_val_weigths += 0.5*_Gk(wtilde,_Gk._kArr_l[ktildex],_Gk._kArr_l[ktildey])(0,0)*_Gk(wtilde-_qq._iwn,_Gk._kArr_l[ktildex]-_qq._qx,_Gk._kArr_l[ktildey]-_qq._qy
                    )(0,0)*_Gk(wbar-_qq._iwn,kbarx-_qq._qx,kbary-_qq._qy)(1,1)*_Gk(wbar,kbarx,kbary)(1,1);
                }
            } else{
                tmp_val_weigths += _Gk(wtilde,_Gk._kArr_l[ktildex],_Gk._kArr_l[ktildey])(0,0)*_Gk(wtilde-_qq._iwn,_Gk._kArr_l[ktildex]-_qq._qx,_Gk._kArr_l[ktildey]-_qq._qy
                )(0,0)*_Gk(wbar-_qq._iwn,kbarx-_qq._qx,kbary-_qq._qy)(1,1)*_Gk(wbar,kbarx,kbary)(1,1);
            }
        }      
    }
    tmp_val_weigths*=1.0/_Gk._Nk/_Gk._Nk;
    return tmp_val_weigths;
}

std::complex<double> ThreadWrapper::getWeightsIPT(double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar) const{
    std::complex<double> tmp_val_weigths(0.0,0.0);
    double kbary, kbarx;
    for (size_t ktildey=0; ktildey<_splInline._k_array.size(); ktildey++){
        kbary = _splInline._k_array[ktildey] - kbary_m_tildey;
        for (size_t ktildex=0; ktildex<_splInline._k_array.size(); ktildex++){
            kbarx = _splInline._k_array[ktildex] - kbarx_m_tildex;
            tmp_val_weigths += buildGK2D_IPT(wtilde,ktildex,ktildey)[0] * buildGK2D_IPT(wtilde-_qq._iwn,ktildex-_qq._qx,ktildey-_qq._qy
                                )[0] * buildGK2D_IPT(wbar-_qq._iwn,kbarx-_qq._qx,kbary-_qq._qy)[1] * buildGK2D_IPT(wbar,kbarx,kbary)[1];
        }      
    }
    tmp_val_weigths*=1.0/GreenStuff::N_k/GreenStuff::N_k;
    return tmp_val_weigths;
}

#endif /* DIM */

std::complex<double> ThreadWrapper::lindhard_functionIPT(bool is_jj, std::ofstream& ofS, const std::string& strOutput, int world_rank) const{
    std::complex<double> bubble(0.0,0.0); // G_aver is used to output the averaged bubble function over the lattice.
    const Integrals integralsObj;
    ofS.open(strOutput, std::ios::app | std::ios::out);
    #if DIM == 1
    std::cout << "world rank: " << world_rank << ", q_x: " << _q._qx << " and q_iwn: " << _q._iwn << std::endl;
    std::ofstream testing_Self;
    if (_q._iwn==*(iqnArr_l.begin())){
        testing_Self.open("test_spline_bubble_iqn_"+std::to_string(_q._iwn.imag())+".dat", std::ofstream::app | std::ofstream::out);
    }
    std::function<std::complex<double>(double,std::complex<double>)> chi_spsp = [&](double kx, std::complex<double> iwn){
        return buildGK1D_IPT(iwn,kx)[0]*buildGK1D_IPT(iwn+_q._iwn,kx+_q._qx)[1];;
    };
    std::function<std::complex<double>(double,std::complex<double>)> chi_jj = [&](double kx, std::complex<double> iwn){
        return (-2.0*std::sin(kx))*buildGK1D_IPT(iwn,kx)[0]*buildGK1D_IPT(iwn+_q._iwn,kx+_q._qx)[1]*(-2.0*std::sin(kx));
    };
    for (size_t wttilde=0; wttilde<_splInline._iwn_array.size(); wttilde++){ //static_cast<size_t>(_splInline._iwn_array.size()/2)
        if (!is_jj){
            bubble+=1./(2.*M_PI)*integralsObj.I1D(chi_spsp,-M_PI,M_PI,_splInline._iwn_array[wttilde]);
        } else{
            bubble+=1./(2.*M_PI)*integralsObj.I1D(chi_jj,-M_PI,M_PI,_splInline._iwn_array[wttilde]);
        }
        // for (size_t qttilde=0; qttilde<_splInline._k_array.size(); qttilde++){
        //     if (!is_jj){
        //         if ( (qttilde==0) || (qttilde==(_splInline._k_array.size()-1)) ){
        //             bubble += 0.5*buildGK1D_IPT(_splInline._iwn_array[wttilde],_splInline._k_array[qttilde])[0]*buildGK1D_IPT(_splInline._iwn_array[wttilde]+_q._iwn,_splInline._k_array[qttilde]+_q._qx)[1];
        //         }
        //         else{
        //             bubble += buildGK1D_IPT(_splInline._iwn_array[wttilde],_splInline._k_array[qttilde])[0]*buildGK1D_IPT(_splInline._iwn_array[wttilde]+_q._iwn,_splInline._k_array[qttilde]+_q._qx)[1];
        //         }
        //     }
        //     else{
        //         if ( (qttilde==0) || (qttilde==(_splInline._k_array.size()-1)) )
        //             bubble += 0.5*(-2.0*std::sin(_splInline._k_array[qttilde]))*buildGK1D_IPT(_splInline._iwn_array[wttilde],_splInline._k_array[qttilde])[0]*buildGK1D_IPT(_splInline._iwn_array[wttilde]-_q._iwn,_splInline._k_array[qttilde]-_q._qx)[1]*(-2.0*std::sin(_splInline._k_array[qttilde]));
        //         else
        //             bubble += (-2.0*std::sin(_splInline._k_array[qttilde]))*buildGK1D_IPT(_splInline._iwn_array[wttilde],_splInline._k_array[qttilde])[0]*buildGK1D_IPT(_splInline._iwn_array[wttilde]-_q._iwn,_splInline._k_array[qttilde]-_q._qx)[1]*(-2.0*std::sin(_splInline._k_array[qttilde])); // Do I have to add _q._qx to the current??
        //     }
        // }
        if (_q._iwn==*(iqnArr_l.begin())){
            testing_Self << (_splInline._iwn_array[wttilde]-_q._iwn).imag() << "\t\t" << _splInline.calculateSpline( (_splInline._iwn_array[wttilde]-_q._iwn).imag() ).imag() << "\n";
        }
    }
    if (_q._iwn==*(iqnArr_l.begin())){
        testing_Self.close();
    }
    #elif DIM == 2
    for (size_t wttilde=static_cast<size_t>(_splInline._iwn_array.size()/2); wttilde<_splInline._iwn_array.size(); wttilde++){
        for (size_t qttildex=0; qttildex<_splInline._k_array.size(); qttildex++){
            for (size_t qttildey=0; qttildey<_splInline._k_array.size(); qttildey++){
                if (!is_jj){
                    if ( ( (qttildex==0) || (qttildex==_splInline._iwn_array.size()-1) ) || ( (qttildey==0) || (qttildey==_splInline._iwn_array.size()-1) ) ){
                        if ( ( (qttildex==0) || (qttildex==_splInline._iwn_array.size()-1) ) && ( (qttildey==0) || (qttildey==_splInline._iwn_array.size()-1) ) ){
                            bubble += 0.25*buildGK2D_IPT(_splInline._iwn_array[wttilde],_splInline._k_array[qttildex],_splInline._k_array[qttildey])[0]*buildGK2D_IPT(_splInline._iwn_array[wttilde]+_qq._iwn,_splInline._k_array[qttildex]+_qq._qx,_splInline._k_array[qttildey]+_qq._qy)[1];
                        } else{
                            bubble += 0.5*buildGK2D_IPT(_splInline._iwn_array[wttilde],_splInline._k_array[qttildex],_splInline._k_array[qttildey])[0]*buildGK2D_IPT(_splInline._iwn_array[wttilde]+_qq._iwn,_splInline._k_array[qttildex]+_qq._qx,_splInline._k_array[qttildey]+_qq._qy)[1];
                        }
                    } else{
                        bubble += buildGK2D_IPT(_splInline._iwn_array[wttilde],_splInline._k_array[qttildex],_splInline._k_array[qttildey])[0]*buildGK2D_IPT(_splInline._iwn_array[wttilde]+_qq._iwn,_splInline._k_array[qttildex]+_qq._qx,_splInline._k_array[qttildey]+_qq._qy)[1];
                    }
                }
                else{
                    if ( ( (qttildex==0) || (qttildex==_splInline._iwn_array.size()-1) ) || ( (qttildey==0) || (qttildey==_splInline._iwn_array.size()-1) ) ){
                        if ( ( (qttildex==0) || (qttildex==_splInline._iwn_array.size()-1) ) && ( (qttildey==0) || (qttildey==_splInline._iwn_array.size()-1) ) ){
                            bubble += 0.25*(-2.0*std::sin(_splInline._k_array[qttildex]))*buildGK2D_IPT(_splInline._iwn_array[wttilde],_splInline._k_array[qttildex],_splInline._k_array[qttildey])[0]*buildGK2D_IPT(_splInline._iwn_array[wttilde]+_qq._iwn,_splInline._k_array[qttildex]+_qq._qx,_splInline._k_array[qttildey]+_qq._qy)[1]*(-2.0*std::sin(_splInline._k_array[qttildex]));
                        } else{
                            bubble += 0.5*(-2.0*std::sin(_splInline._k_array[qttildex]))*buildGK2D_IPT(_splInline._iwn_array[wttilde],_splInline._k_array[qttildex],_splInline._k_array[qttildey])[0]*buildGK2D_IPT(_splInline._iwn_array[wttilde]+_qq._iwn,_splInline._k_array[qttildex]+_qq._qx,_splInline._k_array[qttildey]+_qq._qy)[1]*(-2.0*std::sin(_splInline._k_array[qttildex]));
                        }
                    } else {
                        bubble += (-2.0*std::sin(_splInline._k_array[qttildex]))*buildGK2D_IPT(_splInline._iwn_array[wttilde],_splInline._k_array[qttildex],_splInline._k_array[qttildey])[0]*buildGK2D_IPT(_splInline._iwn_array[wttilde]+_qq._iwn,_splInline._k_array[qttildex]+_qq._qx,_splInline._k_array[qttildey]+_qq._qy)[1]*(-2.0*std::sin(_splInline._k_array[qttildex]));
                    }
                }
            }
        }
    }
    #endif
    bubble *= -1.0*SPINDEG/(GreenStuff::beta); //*GreenStuff::N_k
    std::cout << "non-interacting bubble: " << bubble << std::endl;
    #if DIM == 1
    if ( _q._iwn == *(iqnArr_l.begin()) )
        ofS << "/iwn" << "\t\t" << "iwn re" << "\t\t" << "iwn im" << "\n";
    ofS << _q._iwn.imag() << "\t\t" << bubble.real() << "\t\t" << bubble.imag() << "\n";
    #elif DIM == 2
    if ( _qq._iwn.imag()==0.0 )
        ofS << "/iwn" << "\t\t" << "iwn re" << "\t\t" << "iwn im" << "\n";
    ofS << _qq._iwn.imag() << "\t\t" << bubble.real() << "\t\t" << bubble.imag() << "\n";
    #endif
    ofS.close();
    return bubble;
}

std::complex<double> ThreadWrapper::lindhard_function(bool is_jj, std::ofstream& ofS, const std::string& strOutput, int world_rank) const{
    std::complex<double> bubble(0.0,0.0);
    ofS.open(strOutput, std::ios::app | std::ios::out);
    #if DIM == 1
    for (size_t wttilde=0; wttilde<_Gk._size; wttilde++){
        for (size_t qttilde=0; qttilde<_Gk._kArr_l.size(); qttilde++){
            if (!is_jj){
                if ( ( qttilde==0 ) || ( qttilde==(_Gk._kArr_l.size()-1) ) )
                    bubble += 0.5*_Gk(_Gk._precomp_wn[wttilde],_Gk._kArr_l[qttilde])(0,0)*_Gk(_Gk._precomp_wn[wttilde]+_q._iwn,_Gk._kArr_l[qttilde]+_q._qx)(0,0);
                else
                    bubble += _Gk(_Gk._precomp_wn[wttilde],_Gk._kArr_l[qttilde])(0,0)*_Gk(_Gk._precomp_wn[wttilde]+_q._iwn,_Gk._kArr_l[qttilde]+_q._qx)(0,0);
            }
            else{
                if ( ( qttilde==0 ) || ( qttilde==(_Gk._kArr_l.size()-1) ) )
                    bubble += 0.5*(-2.0*std::sin(_splInline._k_array[qttilde]))*_Gk(_Gk._precomp_wn[wttilde],_Gk._kArr_l[qttilde])(0,0)*_Gk(_Gk._precomp_wn[wttilde]+_q._iwn,_Gk._kArr_l[qttilde]+_q._qx)(0,0)*(-2.0*std::sin(_splInline._k_array[qttilde]));
                else
                    bubble += (-2.0*std::sin(_splInline._k_array[qttilde]))*_Gk(_Gk._precomp_wn[wttilde],_Gk._kArr_l[qttilde])(0,0)*_Gk(_Gk._precomp_wn[wttilde]+_q._iwn,_Gk._kArr_l[qttilde]+_q._qx)(0,0)*(-2.0*std::sin(_splInline._k_array[qttilde]));
            }
        }   
    }
    #elif DIM == 2 /* since one only considers the first-nearest neighbor dispersion relation */
    for (size_t wttilde=0; wttilde<_Gk._size; wttilde++){
        for (size_t qttildex=0; qttildex<_Gk._kArr_l.size(); qttildex++){
            for (size_t qttildey=0; qttildey<_Gk._kArr_l.size(); qttildey++){
                if (!is_jj){
                    if ( ( (qttildey==0) || (qttildey==(_Gk._kArr_l.size()-1)) ) || ( (qttildex==0) || (qttildex==(_Gk._kArr_l.size()-1)) ) ){
                        if ( ( (qttildey==0) || (qttildey==(_Gk._kArr_l.size()-1)) ) && ( (qttildex==0) || (qttildex==(_Gk._kArr_l.size()-1)) ) ){
                            bubble += 0.25*_Gk(_Gk._precomp_wn[wttilde],_Gk._kArr_l[qttildex],_Gk._kArr_l[qttildey])(0,0)*_Gk(_Gk._precomp_wn[wttilde]+_qq._iwn,_Gk._kArr_l[qttildex]+_qq._qx,_Gk._kArr_l[qttildey]+_qq._qy)(0,0);
                        } else{
                            bubble += 0.5*_Gk(_Gk._precomp_wn[wttilde],_Gk._kArr_l[qttildex],_Gk._kArr_l[qttildey])(0,0)*_Gk(_Gk._precomp_wn[wttilde]+_qq._iwn,_Gk._kArr_l[qttildex]+_qq._qx,_Gk._kArr_l[qttildey]+_qq._qy)(0,0);
                        }
                    } else{
                        bubble += _Gk(_Gk._precomp_wn[wttilde],_Gk._kArr_l[qttildex],_Gk._kArr_l[qttildey])(0,0)*_Gk(_Gk._precomp_wn[wttilde]+_qq._iwn,_Gk._kArr_l[qttildex]+_qq._qx,_Gk._kArr_l[qttildey]+_qq._qy)(0,0);
                    }
                }
                else
                    bubble += (-2.0*std::sin(_splInline._k_array[qttildex]))*_Gk(_Gk._precomp_wn[wttilde],_Gk._kArr_l[qttildex],_Gk._kArr_l[qttildey])(0,0)*_Gk(_Gk._precomp_wn[wttilde]+_qq._iwn,_Gk._kArr_l[qttildex]+_qq._qx,_Gk._kArr_l[qttildey]+_qq._qy)(0,0)*(-2.0*std::sin(_splInline._k_array[qttildex]));
            }
        }
    }
    #endif
    bubble *= -1.0*SPINDEG*1.0/(GreenStuff::beta*GreenStuff::N_k);
    #if DIM == 1
    if ( _q._iwn.imag()==0.0 )
        ofS << "/iwn" << "\t\t" << "iwn re" << "\t\t" << "iwn im" << "\n";
    ofS << _q._iwn.imag() << "\t\t" << bubble.real() << "\t\t" << bubble.imag() << "\n";
    #elif DIM == 2
    if ( _qq._iwn.imag()==0.0 )
        ofS << "/iwn" << "\t\t" << "iwn re" << "\t\t" << "iwn im" << "\n";
    ofS << _qq._iwn.imag() << "\t\t" << bubble.real() << "\t\t" << bubble.imag() << "\n";
    #endif
    ofS.close();
    return bubble;
}

void ThreadWrapper::save_data_to_local_extern_matrix_instancesIPT(std::complex<double> tmp_val_kt_kb,std::complex<double> tmp_val_weights,std::complex<double> tmp_val_mid_lev,std::complex<double> tmp_val_corr,
                        std::complex<double> tmp_val_tot_sus,size_t k1,size_t k2,bool is_jj,bool is_full,int world_rank,size_t j) const{
    std::complex<double> val_jj(0.0,0.0);
    double beta_div = (1.0/GreenStuff::beta/GreenStuff::beta);
    matWeigths(k2,k1) = tmp_val_weights*beta_div;
    if (is_full){
        if (j==0){ // Because matCorr is "homogeneous" throughout the processes afterwards.. (only part not depending on q)
            matCorr(k2,k1) = tmp_val_corr*beta_div;
            if (world_rank != root_process){
                vecCorrSlaves->push_back( std::make_tuple( k2,k1,tmp_val_corr*beta_div ) );
            }
        }
        matGamma(k2,k1) = tmp_val_kt_kb*beta_div;
        matMidLev(k2,k1) = tmp_val_mid_lev*beta_div;
        if (world_rank != root_process){
            vecGammaSlaves->push_back( std::make_tuple( k2,k1,tmp_val_kt_kb*beta_div ) );
            vecMidLevSlaves->push_back( std::make_tuple( k2,k1,tmp_val_mid_lev*beta_div ) );
        }
    } else{
        if (j==0){
            matGamma(k2,k1) = tmp_val_kt_kb*beta_div;
            matMidLev(k2,k1) = tmp_val_mid_lev*beta_div;
            if (world_rank != root_process){
                vecGammaSlaves->push_back( std::make_tuple( k2,k1,tmp_val_kt_kb*beta_div ) );
                vecMidLevSlaves->push_back( std::make_tuple( k2,k1,tmp_val_mid_lev*beta_div ) );
            }
        }
    }
    if (world_rank != root_process){
        vecWeightsSlaves->push_back( std::make_tuple( k2,k1,tmp_val_weights*beta_div ) );
    }
    if (!is_jj){
        matTotSus(k2,k1) = beta_div*tmp_val_tot_sus; // These matrices are static variables.
        if (world_rank != root_process)
            vecTotSusSlaves->push_back( std::make_tuple( k2,k1,beta_div*tmp_val_tot_sus ) );
    } else if (is_jj){
        #if DIM == 1
        val_jj = -1.0*beta_div*(-2.0*std::sin(_splInline._k_array[k1]))*tmp_val_tot_sus*(-2.0*std::sin(_splInline._k_array[k2]));
        matTotSus(k2,k1) = val_jj;
        #elif DIM == 2
        double kbarx;
        for (size_t ktildex=0; ktildex<_splInline._k_array.size(); ktildex++){
            kbarx = (_splInline._k_array[ktildex]-_splInline._k_array[k1]); // It brakets within [-2pi,2pi].
            val_jj += -1.0*(-2.0*std::sin(_splInline._k_array[ktildex]))*tmp_val_tot_sus*(-2.0*std::sin(kbarx));
        }
        matTotSus(k2,k1) = val_jj*(1.0/GreenStuff::N_k)*beta_div;
        #endif
        if (world_rank != root_process)
            vecTotSusSlaves->push_back( std::make_tuple( k2,k1,val_jj ) );
    }
}

void ThreadWrapper::save_data_to_local_extern_matrix_instances(std::complex<double> tmp_val_kt_kb,std::complex<double> tmp_val_weights,std::complex<double> tmp_val_mid_lev,std::complex<double> tmp_val_corr,std::complex<double> tmp_val_tot_sus,
                        size_t k1,size_t k2,bool is_jj,bool is_full,int world_rank,size_t j) const{
    std::complex<double> val_jj(0.0,0.0);
    double beta_div = 1.0/_Gk._beta/_Gk._beta;
    matWeigths(k2,k1) = tmp_val_weights*beta_div;
    if (is_full){
        if (j==0){
            matCorr(k2,k1) = tmp_val_corr*beta_div;
            if (world_rank != root_process)
                vecCorrSlaves->push_back( std::make_tuple( k2,k1,tmp_val_corr*beta_div ) );
        }
        matGamma(k2,k1) = tmp_val_kt_kb*beta_div;
        matMidLev(k2,k1) = tmp_val_mid_lev*beta_div;
        if (world_rank != root_process){ // Saving into vector of tuples..
            vecGammaSlaves->push_back( std::make_tuple( k2,k1, tmp_val_kt_kb*beta_div ) );
            vecMidLevSlaves->push_back( std::make_tuple( k2,k1,tmp_val_mid_lev*beta_div ) );
        }
    } else{
        if (j==0){
            matGamma(k2,k1) = tmp_val_kt_kb*beta_div;
            matMidLev(k2,k1) = tmp_val_mid_lev*beta_div;
            if (world_rank != root_process){
                vecGammaSlaves->push_back( std::make_tuple( k2,k1, tmp_val_kt_kb*beta_div ) );
                vecMidLevSlaves->push_back( std::make_tuple( k2,k1,tmp_val_mid_lev*beta_div ) );
            }
        }
    }
    if (world_rank != root_process){ // Saving into vector of tuples..
        vecWeightsSlaves->push_back( std::make_tuple( k2,k1, tmp_val_weights*beta_div ) );
    }
    if (!is_jj){
        matTotSus(k2,k1) = beta_div*tmp_val_tot_sus; // These matrices are static variables.
        if (world_rank != root_process)
            vecTotSusSlaves->push_back( std::make_tuple( k2,k1,beta_div*tmp_val_tot_sus ) );
    } else if (is_jj){ // kbarx could also be kbary
        #if DIM == 1
        val_jj = -1.0*beta_div*(-2.0*std::sin(_Gk._kArr_l[k1]))*tmp_val_tot_sus*(-2.0*std::sin(_Gk._kArr_l[k2]));
        matTotSus(k2,k1) = val_jj;
        #elif DIM == 2
        double kbarx;
        for (size_t ktildex=0; ktildex<_Gk._kArr_l.size(); ktildex++){
            kbarx = (_Gk._kArr_l[ktildex]-_Gk._kArr_l[k1]); // It brakets within [-2pi,2pi].
            val_jj += -1.0*(-2.0*std::sin(_Gk._kArr_l[ktildex]))*tmp_val_tot_sus*(-2.0*std::sin(kbarx));
        }
        matTotSus(k2,k1)=(1.0/_Gk._Nk)*val_jj*beta_div;
        #endif
        if (world_rank != root_process)
            vecTotSusSlaves->push_back( std::make_tuple( k2,k1,val_jj ) );
    }
}

void ThreadFunctor::get_vector_mpi(size_t totSize,bool is_jj,bool is_full,solver_prototype sp,std::vector<mpistruct_t>* vec_root_process){
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

void ThreadFunctor::fetch_data_from_slaves(int an_id,MPI_Status& status,bool is_full,int ierr,size_t num_elements_per_proc,size_t sizeOfTuple,size_t j){
    char chars_to_receive[50];
    int sizeOfGamma, sizeOfWeights, sizeOfTotSus, sizeOfMidLev, sizeOfCorr, sender;
    std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecMidLevTmp, *vecGammaTmp;
    std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecCorrTmp = nullptr;
    std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecWeightsTmp = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >(num_elements_per_proc);
    std::vector< std::tuple< size_t,size_t,std::complex<double> > >* vecTotSusTmp = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >(num_elements_per_proc);
    if (is_full){
        vecMidLevTmp = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >(num_elements_per_proc);
        vecGammaTmp = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >(num_elements_per_proc);
        if (j==0)
            vecCorrTmp = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >(num_elements_per_proc);
    } else{
        if (j==0){
            vecMidLevTmp = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >(num_elements_per_proc);
            vecGammaTmp = new std::vector< std::tuple< size_t,size_t,std::complex<double> > >(num_elements_per_proc);
        }
    }
    // Should send the sizes of the externally linked vectors of tuples to be able to receive. That is why the need to probe...
    if (is_full){
        if (j==0){
            MPI_Probe(an_id,RETURN_DATA_TAG_CORR,MPI_COMM_WORLD,&status);
            MPI_Get_count(&status,MPI_BYTE,&sizeOfCorr);
        }
        MPI_Probe(an_id,RETURN_DATA_TAG_MID_LEV,MPI_COMM_WORLD,&status);
        MPI_Get_count(&status,MPI_BYTE,&sizeOfMidLev);
        MPI_Probe(an_id,RETURN_DATA_TAG_GAMMA,MPI_COMM_WORLD,&status);
        MPI_Get_count(&status,MPI_BYTE,&sizeOfGamma);
    } else{
        if (j==0){
            MPI_Probe(an_id,RETURN_DATA_TAG_MID_LEV,MPI_COMM_WORLD,&status);
            MPI_Get_count(&status,MPI_BYTE,&sizeOfMidLev);
            MPI_Probe(an_id,RETURN_DATA_TAG_GAMMA,MPI_COMM_WORLD,&status);
            MPI_Get_count(&status,MPI_BYTE,&sizeOfGamma);
        }
    }
    MPI_Probe(an_id,RETURN_DATA_TAG_WEIGHTS,MPI_COMM_WORLD,&status);
    MPI_Get_count(&status,MPI_BYTE,&sizeOfWeights);
    MPI_Probe(an_id,RETURN_DATA_TAG_TOT_SUS,MPI_COMM_WORLD,&status);
    MPI_Get_count(&status,MPI_BYTE,&sizeOfTotSus);
    
    ierr = MPI_Recv( chars_to_receive, 50, MPI_CHAR, an_id,
            RETURN_DATA_TAG, MPI_COMM_WORLD, &status);
    if (is_full){
        if (j==0){
            ierr = MPI_Recv( (void*)(vecCorrTmp->data()), sizeOfCorr, MPI_BYTE, an_id,
                    RETURN_DATA_TAG_CORR, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        ierr = MPI_Recv( (void*)(vecMidLevTmp->data()), sizeOfMidLev, MPI_BYTE, an_id,
                RETURN_DATA_TAG_MID_LEV, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        ierr = MPI_Recv( (void*)(vecGammaTmp->data()), sizeOfGamma, MPI_BYTE, an_id,
                RETURN_DATA_TAG_GAMMA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else{
        if (j==0){
            ierr = MPI_Recv( (void*)(vecMidLevTmp->data()), sizeOfMidLev, MPI_BYTE, an_id,
                RETURN_DATA_TAG_MID_LEV, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            ierr = MPI_Recv( (void*)(vecGammaTmp->data()), sizeOfGamma, MPI_BYTE, an_id,
                RETURN_DATA_TAG_GAMMA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    ierr = MPI_Recv( (void*)(vecTotSusTmp->data()), sizeOfTotSus, MPI_BYTE, an_id,
            RETURN_DATA_TAG_TOT_SUS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    ierr = MPI_Recv( (void*)(vecWeightsTmp->data()), sizeOfWeights, MPI_BYTE, an_id,
            RETURN_DATA_TAG_WEIGHTS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    sender = status.MPI_SOURCE;
    printf("Slave process %i returned\n", sender);
    printf("%s\n",chars_to_receive);
    /* Now the data received from the other processes have to be stored in their arma::Mats on root process */
    size_t kt,kb,ii;
    if (is_full){
        if (j==0){
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
    } else{
        if (j==0){
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
        }
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
    if (is_full){
        delete vecGammaTmp;
        delete vecMidLevTmp;
        if (j==0){
            delete vecCorrTmp;
        }
    } else{
        if (j==0){
            delete vecGammaTmp;
            delete vecMidLevTmp;
        }
    }
    delete vecWeightsTmp;
    delete vecTotSusTmp;
}

void ThreadWrapper::fetch_data_gamma_tensor_alltogether(size_t totSizeGammaTensor,int ierr, std::vector<int>* vec_counts, std::vector<int>* vec_disps, 
                            std::vector<int>* vec_counts_full, std::vector<int>* vec_disps_full,bool is_full){
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    MPI_Datatype gamma_tensor_content_type;
    std::vector< gamma_tensor_content >* tmpFullGammaGathered = nullptr;
    std::vector< gamma_tensor_content >* tmpGammaGathered = new std::vector< gamma_tensor_content >(totSizeGammaTensor); // Needs to host the data from all the processes.
    if (is_full)
        tmpFullGammaGathered = new std::vector< gamma_tensor_content >(totSizeGammaTensor);
    size_t kt, wt, kb, wb;
    create_mpi_data_struct(gamma_tensor_content_type); // Generates the custom MPI data type.
    std::cout << "totSizeGammaTensor: " << totSizeGammaTensor << std::endl;
    // std::cout << "size in bytes of vec_gamma_tensor_content for process " << world_rank << " : " << sizeof(gamma_tensor_content)*vecGammaFullTensorContent->size() << std::endl;
    ierr = MPI_Allgatherv((void*)(vecGammaTensorContent->data()),vecGammaTensorContent->size(),gamma_tensor_content_type,
                (void*)(tmpGammaGathered->data()),(vec_counts->data()),(vec_disps->data()),gamma_tensor_content_type,MPI_COMM_WORLD);
    std::cout << "size in bytes of gamma_tensor_content " << sizeof(gamma_tensor_content) << std::endl;
    std::cout << "The world rank is " << world_rank << std::endl;
    // std::cout << "The size of tmpFullGammaGathered is " << tmpFullGammaGathered->size() << std::endl;
    // std::cout << "The size of vecGammaFullTensorContent is " << vecGammaFullTensorContent->size() << std::endl;
    assert(MPI_SUCCESS==ierr);
    if (is_full){
        ierr = MPI_Allgatherv((void*)(vecGammaFullTensorContent->data()),vecGammaFullTensorContent->size(),gamma_tensor_content_type,
                (void*)(tmpFullGammaGathered->data()),(vec_counts_full->data()),(vec_disps_full->data()),gamma_tensor_content_type,MPI_COMM_WORLD);
        assert(MPI_SUCCESS==ierr);
    }
    // if (world_rank==1){
    //     for (auto el : *tmpFullGammaGathered){
    //         std::cout << el._ktilde << "," << el._kbar << "," << el._wtilde << "," << el._wbar << std::endl;
    //     }
    // }
    // Filling up the tensor used in determining the susceptibilities (for iqn > 0)
    size_t num=0;
    for (size_t l=0; l<totSizeGammaTensor; l++){
        kt=tmpGammaGathered->at(l)._ktilde;
        wt=tmpGammaGathered->at(l)._wtilde;
        kb=tmpGammaGathered->at(l)._kbar;
        wb=tmpGammaGathered->at(l)._wbar;
        gamma_tensor[kt][wt][kb][wb]=tmpGammaGathered->at(l)._gamma;
        num++;
    }
    if (is_full){
        num=0;
        for (size_t l=0; l<totSizeGammaTensor; l++){
            kb=tmpFullGammaGathered->at(l)._kbar;
            wb=tmpFullGammaGathered->at(l)._wbar;
            kt=tmpFullGammaGathered->at(l)._ktilde; // This is rather kp
            wt=tmpFullGammaGathered->at(l)._wtilde; // This is rather iknp
            gamma_full_tensor[kb][wb][kt][wt]=tmpFullGammaGathered->at(l)._gamma;
            num++;
        }
    }
    MPI_Type_free(&gamma_tensor_content_type); // Releasing type
    delete tmpGammaGathered; delete vec_counts;
    delete vecGammaTensorContent; delete vec_disps;
    if (is_full){
        delete vecGammaFullTensorContent; delete vec_counts_full;
        delete tmpFullGammaGathered; delete vec_disps_full;
    }
}

void ThreadFunctor::send_messages_to_root_process(bool is_full, int ierr, size_t sizeOfTuple, char* chars_to_send, size_t j){
    if (is_full){
        ierr = MPI_Send( (void*)(vecMidLevSlaves->data()), sizeOfTuple*vecMidLevSlaves->size(), MPI_BYTE, root_process, RETURN_DATA_TAG_MID_LEV, MPI_COMM_WORLD);
        ierr = MPI_Send( (void*)(vecGammaSlaves->data()), sizeOfTuple*vecGammaSlaves->size(), MPI_BYTE, root_process, RETURN_DATA_TAG_GAMMA, MPI_COMM_WORLD);
        if (j==0){
            ierr = MPI_Send( (void*)(vecCorrSlaves->data()), sizeOfTuple*vecCorrSlaves->size(), MPI_BYTE, root_process, RETURN_DATA_TAG_CORR, MPI_COMM_WORLD);
            vecCorrSlaves->clear();
        }
        vecGammaSlaves->clear(); vecMidLevSlaves->clear();
    } else{
        if (j==0){
            ierr = MPI_Send( (void*)(vecMidLevSlaves->data()), sizeOfTuple*vecMidLevSlaves->size(), MPI_BYTE, root_process, RETURN_DATA_TAG_MID_LEV, MPI_COMM_WORLD);
            ierr = MPI_Send( (void*)(vecGammaSlaves->data()), sizeOfTuple*vecGammaSlaves->size(), MPI_BYTE, root_process, RETURN_DATA_TAG_GAMMA, MPI_COMM_WORLD);
            vecGammaSlaves->clear(); vecMidLevSlaves->clear();
        }
    }
    ierr = MPI_Send( (void*)(vecWeightsSlaves->data()), sizeOfTuple*vecWeightsSlaves->size(), MPI_BYTE, root_process, RETURN_DATA_TAG_WEIGHTS, MPI_COMM_WORLD);
    ierr = MPI_Send( (void*)(vecTotSusSlaves->data()), sizeOfTuple*vecTotSusSlaves->size(), MPI_BYTE, root_process, RETURN_DATA_TAG_TOT_SUS, MPI_COMM_WORLD);
    ierr = MPI_Send( chars_to_send, 50, MPI_CHAR, root_process, RETURN_DATA_TAG, MPI_COMM_WORLD);
    vecWeightsSlaves->clear(); vecTotSusSlaves->clear();
}

void ThreadFunctor::create_mpi_data_struct(MPI_Datatype& gamma_tensor_content_type){
    int lengths[5]={ 1, 1, 1, 1, 1 };
    MPI_Aint offsets[5]={ offsetof(gamma_tensor_content,_ktilde), offsetof(gamma_tensor_content,_wtilde), 
                        offsetof(gamma_tensor_content,_kbar), offsetof(gamma_tensor_content,_wbar), offsetof(gamma_tensor_content,_gamma) };
    MPI_Datatype types[5]={ MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_CXX_DOUBLE_COMPLEX };
    MPI_Type_struct(5,lengths,offsets,types,&gamma_tensor_content_type);
    MPI_Type_commit(&gamma_tensor_content_type);
}

void ThreadFunctor::fill_up_counts_disps(std::vector<int>* vec_counts_full, std::vector<int>* vec_disps_full, size_t num_elems_per_proc, size_t sizeOfElMPI_Allgatherv_full){
    assert(vec_counts_full->size()==vec_disps_full->size());
    int num_of_k_bars = vecK.size(), an_id = 0, remaining_num_of_kbars, disps = 0;
    bool is_finished_dispatch=false;
    while (an_id<vec_counts_full->size()){
        if (!is_finished_dispatch){
            if (an_id==0) { // One only needs to select the first array for which wtilde=ktilde=0. Hence, the dispatch of ktilde=0 across the processes is done carefully.
                remaining_num_of_kbars = num_of_k_bars - (int)(num_elems_per_proc+1); // Need to watch out for the size_t->int conversion.
            }else{
                remaining_num_of_kbars = num_of_k_bars - (int)num_elems_per_proc;
            }
            if (remaining_num_of_kbars==0) {
                vec_counts_full->at(an_id) = num_of_k_bars*(int)sizeOfElMPI_Allgatherv_full;
                vec_disps_full->at(an_id) = (an_id!=0) ? disps*(int)sizeOfElMPI_Allgatherv_full : 0;
                disps+=num_of_k_bars;
                // Cannot dispatch further.
                is_finished_dispatch=true;
            } else if (remaining_num_of_kbars>0) {
                vec_counts_full->at(an_id) = (an_id!=0) ? (int)(num_elems_per_proc*sizeOfElMPI_Allgatherv_full) : (int)((num_elems_per_proc+1)*sizeOfElMPI_Allgatherv_full);
                vec_disps_full->at(an_id) = (an_id!=0) ? disps*(int)sizeOfElMPI_Allgatherv_full : 0;
                disps += (an_id!=0) ? (int)num_elems_per_proc : (int)(num_elems_per_proc+1);
            } else {
                vec_counts_full->at(an_id) = (int)(num_of_k_bars*sizeOfElMPI_Allgatherv_full);
                vec_disps_full->at(an_id) = disps*(int)sizeOfElMPI_Allgatherv_full;
                // Cannot dispatch further.
                is_finished_dispatch=true;
            }
            num_of_k_bars=remaining_num_of_kbars;
            std::cout << "num_of_k_bars: " << num_of_k_bars << std::endl;
        } else { // All the relevant ktilde=0 values have been dispatched.
            vec_counts_full->at(an_id) = 0;
            vec_disps_full->at(an_id) = disps*(int)sizeOfElMPI_Allgatherv_full;
        }
        an_id++;
    }
}