#ifndef Susceptibilities_H_
#define Susceptibilities_H_

#include<tuple>
#include<algorithm>
#include "IPT2nd3rdorderSingle2.hpp"
#define MULT_N_TAU 2

template<typename T>
inline void calculateSusceptibilities(T&,const IPT2::SplineInline< std::complex<double> >&,const std::string&,const std::string&,const bool&,const bool&);

// Make this class a template such that it hosts HF::FunctorBuildGk and IPT::DMFTproc objects
template<class T>
class Susceptibility{ // The current-current susceptibility uses the +U in gamma, multiplied by the currents thereafter.
    public:
        /* 1D methods */
        #if DIM == 1
        std::complex<double> get_weights(T&,double,std::complex<double>,double,std::complex<double>,HF::K_1D) const;
        std::complex<double> get_weights(T&,double,std::complex<double>,double,std::complex<double>,HF::K_1D,const IPT2::SplineInline< std::complex<double> >&) const;
        std::complex<double> gamma_oneD_spsp(T&,double,std::complex<double>,double,std::complex<double>) const;
        std::complex<double> gamma_oneD_spsp(T&,double,std::complex<double>,double,std::complex<double>,const IPT2::SplineInline< std::complex<double> >&) const;
        std::complex<double> gamma_oneD_spsp_full_lower(T&,double,double,std::complex<double>,std::complex<double>) const;
        std::complex<double> gamma_oneD_spsp_full_lower(T&,double,double,std::complex<double>,std::complex<double>,const IPT2::SplineInline< std::complex<double> >&) const;
        std::tuple< std::complex<double>, std::complex<double> > gamma_oneD_spsp_jj_plotting(T&,double,std::complex<double>,double,std::complex<double>,HF::K_1D,const bool&) const;
        std::tuple< std::complex<double>, std::complex<double> > gamma_oneD_spsp_jj_plotting(T&,double,std::complex<double>,double,std::complex<double>,HF::K_1D,const IPT2::SplineInline< std::complex<double> >&,const bool&) const;
        std::tuple< std::complex<double>, std::complex<double> > gamma_oneD_spsp_crossed_plotting(T&,double,std::complex<double>,double,std::complex<double>,HF::K_1D) const;
        std::tuple< std::complex<double>, std::complex<double> > gamma_oneD_spsp_crossed_plotting(T&,double,std::complex<double>,double,std::complex<double>,HF::K_1D,const IPT2::SplineInline< std::complex<double> >&) const;
        std::tuple< std::complex<double>, std::complex<double>, std::complex<double> > gamma_oneD_spsp_jj_full_middle_plotting(T&,double,std::complex<double>,double,std::complex<double>,HF::K_1D,const bool&) const;
        std::tuple< std::complex<double>, std::complex<double>, std::complex<double> > gamma_oneD_spsp_jj_full_middle_plotting(T&,double,std::complex<double>,double,std::complex<double>,HF::K_1D,const IPT2::SplineInline< std::complex<double> >&,const bool&) const;
        /* 2D methods */
        #elif DIM == 2
        std::complex<double> getWeights(T& Type,HF::K_2D qq,double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar) const;
        std::complex<double> getWeights(T& Type,HF::K_2D qq,double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar,const IPT2::SplineInline< std::complex<double> >& splInline) const;
        std::complex<double> gamma_twoD_spsp(T& Type,double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar) const;
        std::complex<double> gamma_twoD_spsp(T& Type,double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar,const IPT2::SplineInline< std::complex<double> >& splInline) const;
        std::complex<double> gamma_twoD_spsp_full_lower(T& Type,double kpx,double kpy,double kbarx,double kbary,std::complex<double> iknp,std::complex<double> wbar) const;
        std::complex<double> gamma_twoD_spsp_full_lower(T& Type,double kpx,double kpy,double kbarx,double kbary,std::complex<double> iknp,std::complex<double> wbar,const IPT2::SplineInline< std::complex<double> >& splInline) const;
        std::tuple< std::complex<double>, std::complex<double> > gamma_twoD_spsp_jj_plotting(T& Type,double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar,const bool&) const;
        std::tuple< std::complex<double>, std::complex<double> > gamma_twoD_spsp_jj_plotting(T& Type,double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar,const IPT2::SplineInline< std::complex<double> >& splInline,const bool& is_jj) const;
        std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > gamma_twoD_spsp_jj_full_middle_plotting(T& Type,double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wbar,std::complex<double> wtilde,HF::K_2D qq,const bool& is_jj) const;
        std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > gamma_twoD_spsp_jj_full_middle_plotting(T& Type,double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wbar,std::complex<double> wtilde,HF::K_2D qq,const IPT2::SplineInline< std::complex<double> >& splInline,const bool& is_jj) const;
        #endif
};

/*************************************************************** 1D methods ***************************************************************/
/* Functions entering the full spin susceptibility. */
#if DIM == 1
template<>
inline std::complex<double> Susceptibility<HF::FunctorBuildGk>::gamma_oneD_spsp_full_lower(HF::FunctorBuildGk& Gk,double kp,double kbar,std::complex<double> iknp,std::complex<double> wbar) const{
    std::complex<double> lower_level(0.0,0.0);
    for (size_t iknpp=0; iknpp<Gk._size; iknpp++){
        for (size_t kpp=0; kpp<vecK.size(); kpp++){
            lower_level += Gk(Gk._precomp_wn[iknpp]+iknp-wbar,vecK[kpp]+kp-kbar)(0,0)*Gk(Gk._precomp_wn[iknpp],vecK[kpp])(1,1);
        }
    }
    lower_level *= SPINDEG*Gk._u/(Gk._beta*(Gk._Nk)); /// No minus sign at ground level. Factor 2 for spin.
    //std::cout << "Lowest level (kbar): " << Gk._u/(1.0+lower_level) << "\n";
    //std::cout << "Nk: " << Gk._Nk << " " << Gk._beta << " " << Gk._u << "\n";
    lower_level += 1.0;
    return Gk._u/lower_level; // Means that we have to multiply the middle level of this component by the two missing Green's functions.
}
template<class T>
inline std::complex<double> Susceptibility<T>::gamma_oneD_spsp_full_lower(T& Type,double kp,double kbar,std::complex<double> iknp,std::complex<double> wbar,const IPT2::SplineInline< std::complex<double> >& splInline) const{
    static_assert(std::is_same<T, IPT2::DMFTproc>::value, "Designed only for IPT2::DMFTproc!");
    std::complex<double> lower_level(0.0,0.0);
    for (size_t iknpp=static_cast<size_t>(iwnArr_l.size()/2); iknpp<iwnArr_l.size(); iknpp++){ // Have to be careful, because iwnArr_l contains also negative Matsubara frequencies.
        for (size_t kpp=0; kpp<vecK.size(); kpp++){ // In the PM state, there is no difference between the up and down spin projections. Building the GFs on the spot.
            lower_level += 1.0/( (iwnArr_l[iknpp]+iknp-wbar) + GreenStuff::mu - epsilonk(vecK[kpp]+kp-kbar) - splInline.calculateSpline( (iwnArr_l[iknpp]+iknp-wbar).imag() ) ) * 1.0/( iwnArr_l[iknpp] + GreenStuff::mu - epsilonk(vecK[kpp]) - splInline.calculateSpline( iwnArr_l[iknpp].imag() ) );
        }
    }
    lower_level *= SPINDEG*GreenStuff::U/(GreenStuff::beta*GreenStuff::N_k); /// No minus sign at ground level. Factor 2 for spin.
    //std::cout << "Lowest level (kbar): " << Gk._u/(1.0+lower_level) << "\n";
    //std::cout << "Nk: " << Gk._Nk << " " << Gk._beta << " " << Gk._u << "\n";
    lower_level += 1.0;
    return GreenStuff::U/lower_level; // Means that we have to multiply the middle level of this component by the two missing Green's functions.
}

template<>
inline std::complex<double> Susceptibility<HF::FunctorBuildGk>::gamma_oneD_spsp(HF::FunctorBuildGk& Gk,double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar) const{ // q's contain bosonic Matsubara frequencies.
    std::complex<double> lower_level(0.0,0.0);
    for (size_t wttilde=0; wttilde<Gk._size; wttilde++){
        for (size_t qttilde=0; qttilde<vecK.size(); qttilde++){
            lower_level += Gk(wtilde-Gk._precomp_qn[wttilde],ktilde-vecK[qttilde])(0,0)*Gk(wbar-Gk._precomp_qn[wttilde],kbar-vecK[qttilde])(1,1);
        }
    }
    lower_level *= -SPINDEG*Gk._u/(Gk._beta*(Gk._Nk)); // Factor 2 for spin summation.
    return lower_level;
}

template<class T>
inline std::complex<double> Susceptibility<T>::gamma_oneD_spsp(T& Type,double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar,const IPT2::SplineInline< std::complex<double> >& splInline) const{ // q's contain bosonic Matsubara frequencies.
    static_assert(std::is_same<T, IPT2::DMFTproc>::value, "Designed only for IPT2::DMFTproc!");
    std::complex<double> lower_level(0.0,0.0);
    for (size_t wttilde=0; wttilde<iqnArr_l.size(); wttilde++){
        for (size_t qttilde=0; qttilde<vecK.size(); qttilde++){
            lower_level += 1.0/( (wtilde-iqnArr_l[wttilde]) + GreenStuff::mu - epsilonk(ktilde-vecK[qttilde]) - splInline.calculateSpline( (wtilde-iqnArr_l[wttilde]).imag() ) ) * 1.0/( (wbar-iqnArr_l[wttilde]) + GreenStuff::mu - epsilonk(kbar-vecK[qttilde]) - splInline.calculateSpline( (wbar-iqnArr_l[wttilde]).imag() ) );
        }
    }
    lower_level *= -SPINDEG*GreenStuff::U/(GreenStuff::beta*GreenStuff::N_k); // Factor 2 for spin summation.
    return lower_level;
}

template<>
inline std::tuple< std::complex<double>, std::complex<double> > Susceptibility<HF::FunctorBuildGk>::gamma_oneD_spsp_jj_plotting(HF::FunctorBuildGk& Gk,double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar,HF::K_1D q,const bool& is_jj) const{
    std::complex<double> lower_level(0.0,0.0), chi_bubble(0.0,0.0);
    for (size_t wttilde=0; wttilde<Gk._size; wttilde++){
        for (size_t qttilde=0; qttilde<vecK.size(); qttilde++){
            lower_level+=Gk(wtilde+q._iwn-Gk._precomp_qn[wttilde],ktilde+q._qx-vecK[qttilde])(0,0)*Gk(wbar+q._iwn-Gk._precomp_qn[wttilde],kbar+q._qx-vecK[qttilde])(1,1);
        }
    }
    lower_level*=-SPINDEG*Gk._u/(Gk._beta*Gk._Nk);
    chi_bubble=lower_level;
    lower_level*=-1.0;
    lower_level+=1.0;
    if (is_jj)
        return std::make_tuple( -1.0*(-2.0*std::sin(ktilde))*Gk._u/lower_level*(-2.0*std::sin(kbar)), chi_bubble );
    else
        return std::make_tuple(Gk._u/lower_level,chi_bubble);
}

template<>
inline std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > Susceptibility<HF::FunctorBuildGk>::gamma_oneD_spsp_jj_full_middle_plotting(HF::FunctorBuildGk& Gk,double ktilde,std::complex<double> wbar,double kbar,std::complex<double> wtilde,HF::K_1D q,const bool& is_jj) const{
/* Uses gamma_oneD_spsp to compute the infinite-ladder-down part and gamma_oneD_spsp_lower to compute corrections stemming from infinite delta G/delta phi. */    
    std::complex<double> middle_level(0.0,0.0),middle_level_inf(0.0,0.0),middle_level_corr(0.0,0.0);
    for (size_t kp=0; kp<vecK.size(); kp++){
        for (size_t iknp=0; iknp<Gk._size; iknp++){
            middle_level_inf += Gk(Gk._precomp_wn[iknp],vecK[kp]
            )(0,0) * gamma_oneD_spsp_full_lower(Gk,vecK[kp],kbar,Gk._precomp_wn[iknp],wbar
            ) * Gk(Gk._precomp_wn[iknp]-q._iwn,vecK[kp]-q._qx
            )(0,0);

        }
    }
    middle_level_inf*=(SPINDEG/((Gk._Nk)*Gk._beta)); // Factor 2 for spin.
    middle_level_corr+=middle_level_inf;
    middle_level_inf+=gamma_oneD_spsp(Gk,ktilde,wtilde,kbar,wbar);
    middle_level-=middle_level_inf;
    middle_level+=1.0;
    if (is_jj)
        return std::make_tuple( -1.0*(-2.0*std::sin(ktilde))*Gk._u/middle_level*(-2.0*std::sin(kbar)), middle_level_inf, middle_level_corr );
    else
        return std::make_tuple(Gk._u/middle_level,middle_level_inf,middle_level_corr);
}

template<class T>
inline std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > Susceptibility<T>::gamma_oneD_spsp_jj_full_middle_plotting(T& Type,double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar,HF::K_1D q,const IPT2::SplineInline< std::complex<double> >& splInline,const bool& is_jj) const{
/* Uses gamma_oneD_spsp to compute the infinite-ladder-down part and gamma_oneD_spsp_lower to compute corrections stemming from infinite delta G/delta phi. */    
    static_assert(std::is_same<T, IPT2::DMFTproc>::value, "Designed only for IPT2::DMFTproc!");
    std::complex<double> middle_level(0.0,0.0),middle_level_inf(0.0,0.0),middle_level_corr(0.0,0.0);
    for (size_t kp=0; kp<vecK.size(); kp++){
        //std::cout << "kp: " << kp << "\n";
        for (size_t iknp=static_cast<size_t>(iwnArr_l.size()/2); iknp<iwnArr_l.size(); iknp++){
            middle_level_inf += 1.0/( iwnArr_l[iknp] + GreenStuff::mu - epsilonk(vecK[kp]) - splInline.calculateSpline( iwnArr_l[iknp].imag() ) 
            ) * gamma_oneD_spsp_full_lower(Type,vecK[kp],kbar,iwnArr_l[iknp],wbar,splInline
            ) * 1.0/( (iwnArr_l[iknp]-q._iwn) + GreenStuff::mu - epsilonk(vecK[kp]-q._qx) - splInline.calculateSpline( (iwnArr_l[iknp]-q._iwn).imag() )
            );
        }
    }
    middle_level_inf*=SPINDEG/(GreenStuff::N_k*GreenStuff::beta); // Factor 2 for spin.
    middle_level_corr+=middle_level_inf;
    middle_level_inf+=gamma_oneD_spsp(Type,ktilde,wtilde,kbar,wbar,splInline);
    middle_level-=middle_level_inf;
    middle_level+=1.0;
    if (is_jj)
        return std::make_tuple( -1.0*(-2.0*std::sin(ktilde))*GreenStuff::U/middle_level*(-2.0*std::sin(kbar)), middle_level_inf, middle_level_corr );
    else
        return std::make_tuple(GreenStuff::U/middle_level,middle_level_inf,middle_level_corr);
}

template<class T>
inline std::tuple< std::complex<double>, std::complex<double> > Susceptibility<T>::gamma_oneD_spsp_jj_plotting(T& Type,double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar,HF::K_1D q,const IPT2::SplineInline< std::complex<double> >& splInline,const bool& is_jj) const{ // q's contain bosonic Matsubara frequencies.
    static_assert(std::is_same<T, IPT2::DMFTproc>::value, "Designed only for IPT2::DMFTproc!");
    std::complex<double> lower_level(0.0,0.0), chi_bubble(0.0,0.0); // chi_bubble represents the lower bubble in the vertex function, not the total vertex function.
    for (size_t wttilde=0; wttilde<iqnArr_l.size(); wttilde++){
        for (size_t qttilde=0; qttilde<vecK.size(); qttilde++){
            lower_level += 1.0/( (wtilde+q._iwn-iqnArr_l[wttilde]) + GreenStuff::mu - epsilonk(ktilde+q._qx-vecK[qttilde]) - splInline.calculateSpline( (wtilde+q._iwn-iqnArr_l[wttilde]).imag() ) ) * 1.0/( (wbar+q._iwn-iqnArr_l[wttilde]) + GreenStuff::mu - epsilonk(kbar+q._qx-vecK[qttilde]) - splInline.calculateSpline( (wbar+q._iwn-iqnArr_l[wttilde]).imag() ) );
        }
    }
    lower_level *= -SPINDEG*GreenStuff::U/(GreenStuff::beta*GreenStuff::N_k); //factor 2 for the spin and minus sign added
    chi_bubble = lower_level; 
    lower_level *= -1.0;
    lower_level += 1.0;
    if (is_jj)
        return std::make_tuple( -1.0*(-2.0*std::sin(ktilde))*GreenStuff::U/lower_level*(-2.0*std::sin(kbar)), chi_bubble );
    else
        return std::make_tuple(GreenStuff::U/lower_level,chi_bubble);
}   

template<>
inline std::tuple< std::complex<double>,std::complex<double> > Susceptibility<HF::FunctorBuildGk>::gamma_oneD_spsp_crossed_plotting(HF::FunctorBuildGk& Gk,double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar, HF::K_1D q) const{ // q's contain bosonic Matsubara frequencies.
    std::complex<double> lower_level(0.0,0.0),chi_bubble(0.0,0.0);
    for (size_t wttilde=0; wttilde<Gk._size; wttilde++){
        for (size_t qttilde=0; qttilde<vecK.size(); qttilde++){
            lower_level += Gk(wtilde+wbar-Gk._precomp_qn[wttilde]-q._iwn,ktilde+kbar-vecK[qttilde]-q._qx)(0,0)*Gk(Gk._precomp_qn[wttilde],vecK[qttilde])(1,1);
        }
    }
    lower_level *= -SPINDEG*Gk._u/(Gk._beta*Gk._Nk); // Factor 2 for spin summation.
    chi_bubble=lower_level;
    lower_level*=-1.0;
    lower_level+=1.0;
    return std::make_tuple(Gk._u/lower_level,chi_bubble);
}

template<class T>
inline std::tuple< std::complex<double>,std::complex<double> > Susceptibility<T>::gamma_oneD_spsp_crossed_plotting(T& Type,double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar,HF::K_1D q,const IPT2::SplineInline< std::complex<double> >& splInline) const{ // q's contain bosonic Matsubara frequencies.
    static_assert(std::is_same<T, IPT2::DMFTproc>::value, "Designed only for IPT2::DMFTproc!");
    std::complex<double> lower_level(0.0,0.0),chi_bubble(0.0,0.0);
    for (size_t wttilde=0; wttilde<iqnArr_l.size(); wttilde++){
        for (size_t qttilde=0; qttilde<vecK.size(); qttilde++){
            lower_level += 1.0/( (wtilde+wbar-q._iwn-iqnArr_l[wttilde]) + GreenStuff::mu - epsilonk(ktilde+kbar-vecK[qttilde]-q._qx) - splInline.calculateSpline( (wtilde+wbar-q._iwn-iqnArr_l[wttilde]).imag() ) ) * 1.0/( iqnArr_l[wttilde] + GreenStuff::mu - epsilonk(vecK[qttilde]) - splInline.calculateSpline( iqnArr_l[wttilde].imag() ) );//Gk(Gk._precomp_qn[wttilde],vecK[qttilde])(1,1);
        }
    }
    lower_level *= -SPINDEG*GreenStuff::U/(GreenStuff::beta*GreenStuff::N_k); // Factor 2 for spin summation.
    chi_bubble=lower_level;
    lower_level*=-1.0;
    lower_level+=1.0;
    return std::make_tuple(GreenStuff::U/lower_level,chi_bubble);
}

template<>
inline std::complex<double> Susceptibility<HF::FunctorBuildGk>::get_weights(HF::FunctorBuildGk& Gk,double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar,HF::K_1D q) const{
    return Gk(wtilde,ktilde)(0,0)*Gk(wtilde+q._iwn,ktilde+q._qx)(0,0)*Gk(wbar,kbar)(1,1)*Gk(wbar+q._iwn,kbar+q._qx)(1,1);
}

template<class T>
inline std::complex<double> Susceptibility<T>::get_weights(T& Type,double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar,HF::K_1D q,const IPT2::SplineInline< std::complex<double> >& splInline) const{
    static_assert(std::is_same<T, IPT2::DMFTproc>::value, "Designed only for IPT2::DMFTproc!");
    std::complex<double> weightsTmp=1.0/( (wtilde) + GreenStuff::mu - epsilonk(ktilde) - splInline.calculateSpline( (wtilde).imag() ) ) * 1.0/( (wtilde+q._iwn) + GreenStuff::mu - epsilonk(ktilde+q._qx) - splInline.calculateSpline( (wtilde+q._iwn).imag() ) 
    ) * 1.0/( (wbar) + GreenStuff::mu - epsilonk(kbar) - splInline.calculateSpline( (wbar).imag() ) ) * 1.0/( (wbar+q._iwn) + GreenStuff::mu - epsilonk(kbar+q._qx) - splInline.calculateSpline( (wbar+q._iwn).imag() ) );
    return weightsTmp;
}
/*************************************************************** 2D methods ***************************************************************/
#elif DIM == 2
template<>
inline std::tuple< std::complex<double>, std::complex<double> > Susceptibility<HF::FunctorBuildGk>::gamma_twoD_spsp_jj_plotting(HF::FunctorBuildGk& Gk,double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar,const bool& is_jj) const{
    std::complex<double> lower_level(0.0,0.0), chi_bubble(0.0,0.0); // chi_bubble represents the lower bubble in the vertex function, not the total vertex function.
    for (size_t wttilde=0; wttilde<Gk._size; wttilde++){
        for (size_t qttildey=0; qttildey<vecK.size(); qttildey++){
            for (size_t qttildex=0; qttildex<vecK.size(); qttildex++){ // the change of variable only applies to k-space, due to periodicity modulo 2pi.
                lower_level += Gk((wtilde-Gk._precomp_qn[wttilde]),vecK[qttildex],vecK[qttildey])(0,0)*Gk((wbar-Gk._precomp_qn[wttilde]),(vecK[qttildex]+kbarx_m_tildex),(vecK[qttildey]+kbary_m_tildey))(1,1);
            }
        }
    }
    lower_level *= -SPINDEG*Gk._u/(Gk._beta*Gk._Nk*Gk._Nk); //factor 2 for the spin and minus sign added
    chi_bubble = lower_level; 
    lower_level *= -1.0;
    lower_level += 1.0;
    if (is_jj){
        double current_vertices=0.0,kbarx;
        for (size_t ktildex=0; ktildex<vecK.size(); ktildex++){ // Many combinations of ktildex (y) and kbarx (y) give within [-pi,pi] the same difference. 
            kbarx = vecK[ktildex] - kbarx_m_tildex;
            current_vertices+=-2.0*std::sin(vecK[ktildex])*-2.0*std::sin(kbarx);
        }
        current_vertices*=1.0/Gk._Nk;
        return std::make_tuple(-1.0*current_vertices*Gk._u/lower_level,chi_bubble);
    }
    else
        return std::make_tuple(Gk._u/lower_level,chi_bubble);
}

template<class T>
inline std::tuple< std::complex<double>, std::complex<double> > Susceptibility<T>::gamma_twoD_spsp_jj_plotting(T& Type,double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar,const IPT2::SplineInline< std::complex<double> >& splInline,const bool& is_jj) const{
    static_assert(std::is_same<T,IPT2::DMFTproc>::value, "Must be IPT solvers passed in.");
    std::complex<double> lower_level(0.0,0.0), chi_bubble(0.0,0.0); // chi_bubble represents the lower bubble in the vertex function, not the total vertex function.
    for (size_t wttilde=0; wttilde<iqnArr_l.size(); wttilde++){
        for (size_t qttildey=0; qttildey<vecK.size(); qttildey++){
            for (size_t qttildex=0; qttildex<vecK.size(); qttildex++){ // the change of variable only applies to k-space, due to periodicity modulo 2pi.
                lower_level += 1.0/( (wtilde-iqnArr_l[wttilde]) + GreenStuff::mu - epsilonk(vecK[qttildex],vecK[qttildey]) - splInline.calculateSpline( (wtilde-iqnArr_l[wttilde]).imag() ) ) * 1.0/( (wbar-iqnArr_l[wttilde]) + GreenStuff::mu - epsilonk((vecK[qttildex]+kbarx_m_tildex),(vecK[qttildey]+kbary_m_tildey)) - splInline.calculateSpline( (wbar-iqnArr_l[wttilde]).imag() ) );
            }
        }
    }
    lower_level *= -SPINDEG*GreenStuff::U/(GreenStuff::beta*GreenStuff::N_k*GreenStuff::N_k); //factor 2 for the spin and minus sign added
    chi_bubble = lower_level; 
    lower_level *= -1.0;
    lower_level += 1.0;
    if (is_jj){
        double current_vertices=0.0, kbarx;
        for (size_t ktildex=0; ktildex<vecK.size(); ktildex++){ // Many combinations of ktildex (y) and kbarx (y) give within [-pi,pi] the same difference. 
            kbarx = vecK[ktildex] - kbarx_m_tildex;
            current_vertices+=-2.0*std::sin(vecK[ktildex])*-2.0*std::sin(kbarx);
        }
        current_vertices*=1.0/GreenStuff::N_k;
        return std::make_tuple(-1.0*current_vertices*GreenStuff::U/lower_level,chi_bubble);
    }
    else
        return std::make_tuple(GreenStuff::U/lower_level,chi_bubble);
}

template<>
inline std::complex<double> Susceptibility<HF::FunctorBuildGk>::gamma_twoD_spsp(HF::FunctorBuildGk& Gk,double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar) const{
    std::complex<double> lower_level(0.0,0.0);
    for (size_t wttilde=0; wttilde<Gk._size; wttilde++){
        for (size_t qttildey=0; qttildey<vecK.size(); qttildey++){
            for (size_t qttildex=0; qttildex<vecK.size(); qttildex++){ // the change of variable only applies to k-space, due to periodicity modulo 2pi.
                lower_level += Gk((wtilde-Gk._precomp_qn[wttilde]),vecK[qttildex],vecK[qttildey])(0,0)*Gk((wbar-Gk._precomp_qn[wttilde]),(vecK[qttildex]+kbarx_m_tildex),(vecK[qttildey]+kbary_m_tildey))(1,1);
            }
        }
    }
    lower_level *= -SPINDEG*Gk._u/(Gk._beta*Gk._Nk*Gk._Nk); /// Removed minus sign
    return lower_level;
}

template<class T>
inline std::complex<double> Susceptibility<T>::gamma_twoD_spsp(T& Type,double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar,const IPT2::SplineInline< std::complex<double> >& splInline) const{
    static_assert(std::is_same<T,IPT2::DMFTproc>::value, "Must be IPT solvers passed in.");
    std::complex<double> lower_level(0.0,0.0);
    for (size_t wttilde=0; wttilde<iqnArr_l.size(); wttilde++){
        for (size_t qttildey=0; qttildey<vecK.size(); qttildey++){
            for (size_t qttildex=0; qttildex<vecK.size(); qttildex++){ // the change of variable only applies to k-space, due to periodicity modulo 2pi.
                lower_level += 1.0/( (wtilde-iqnArr_l[wttilde]) + GreenStuff::mu - epsilonk(vecK[qttildex],vecK[qttildey]) - splInline.calculateSpline( (wtilde-iqnArr_l[wttilde]).imag() ) ) * 1.0/( (wbar-iqnArr_l[wttilde]) + GreenStuff::mu - epsilonk((vecK[qttildex]+kbarx_m_tildex),(vecK[qttildey]+kbary_m_tildey)) - splInline.calculateSpline( (wbar-iqnArr_l[wttilde]).imag() ) );
            }
        }
    }
    lower_level *= -SPINDEG*GreenStuff::U/(GreenStuff::beta*GreenStuff::N_k*GreenStuff::N_k); /// Removed minus sign
    return lower_level;
}

template<>
inline std::complex<double> Susceptibility<HF::FunctorBuildGk>::gamma_twoD_spsp_full_lower(HF::FunctorBuildGk& Gk,double kpx,double kpy,double kbarx,double kbary,std::complex<double> iknp,std::complex<double> wbar) const{
    std::complex<double> lower_level(0.0,0.0);
    for (size_t iknpp=0; iknpp<Gk._size; iknpp++){
        for (size_t kppx=0; kppx<vecK.size(); kppx++){
            for (size_t kppy=0; kppy<vecK.size(); kppy++){
                lower_level += Gk(Gk._precomp_wn[iknpp]+iknp-wbar,vecK[kppx]+kpx-kbarx,vecK[kppy]+kpy-kbary)(0,0)*Gk(Gk._precomp_wn[iknpp],vecK[kppx],vecK[kppy])(1,1);
            }
        }
    }
    lower_level *= SPINDEG*Gk._u/(Gk._beta*(Gk._Nk)*(Gk._Nk)); /// No minus sign at ground level. Factor 2 for spin.
    lower_level += 1.0;
    return Gk._u/lower_level; // Means that we have to multiply the middle level of this component by the two missing Green's functions.
}

template<class T>
inline std::complex<double> Susceptibility<T>::gamma_twoD_spsp_full_lower(T& Type,double kpx,double kpy,double kbarx,double kbary,std::complex<double> iknp,std::complex<double> wbar,const IPT2::SplineInline< std::complex<double> >& splInline) const{
    static_assert(std::is_same<T,IPT2::DMFTproc>::value, "Must be IPT solvers passed in.");
    std::complex<double> lower_level(0.0,0.0);
    for (size_t iknpp=static_cast<size_t>(iwnArr_l.size()/2); iknpp<iwnArr_l.size(); iknpp++){
        for (size_t kppx=0; kppx<vecK.size(); kppx++){
            for (size_t kppy=0; kppy<vecK.size(); kppy++){
                lower_level += 1.0/( (iwnArr_l[iknpp]+iknp-wbar) + GreenStuff::mu - epsilonk(vecK[kppx]+kpx-kbarx,vecK[kppy]+kpy-kbary) - splInline.calculateSpline( (iwnArr_l[iknpp]+iknp-wbar).imag() ) ) * 1.0/( (iwnArr_l[iknpp]) + GreenStuff::mu - epsilonk(vecK[kppx],vecK[kppy]) - splInline.calculateSpline( (iwnArr_l[iknpp]).imag() ) );
            }
        }
    }
    lower_level *= SPINDEG*GreenStuff::U/(GreenStuff::beta*(GreenStuff::N_k)*(GreenStuff::N_k)); /// No minus sign at ground level. Factor 2 for spin.
    lower_level += 1.0;
    return GreenStuff::U/lower_level; // Means that we have to multiply the middle level of this component by the two missing Green's functions.
}

template<>
inline std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > Susceptibility<HF::FunctorBuildGk>::gamma_twoD_spsp_jj_full_middle_plotting(HF::FunctorBuildGk& Gk,double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wbar,std::complex<double> wtilde,HF::K_2D q,const bool& is_jj) const{
/* Uses gamma_twoD_spsp to compute the infinite-ladder-down part and gamma_twoD_spsp_lower to compute corrections stemming from infinite delta G/delta phi. */    
    std::complex<double> middle_level(0.0,0.0),middle_level_inf(0.0,0.0),middle_level_corr(0.0,0.0);
    double kbarx, kbary;
    for (size_t kpx=0; kpx<vecK.size(); kpx++){
        std::cout << "kpx: " << kpx << "\n";
        for (size_t kpy=0; kpy<vecK.size(); kpy++){
            std::cout << "kpy: " << kpy << "\n";
            for (size_t iknp=0; iknp<Gk._size; iknp++){
                for (size_t ktildex=0; ktildex<vecK.size(); ktildex++){
                    kbarx = vecK[ktildex] - kbarx_m_tildex;
                    for (size_t ktildey=0; ktildey<vecK.size(); ktildey++){
                        kbary = vecK[ktildey] - kbary_m_tildey;
                        middle_level_inf += Gk(Gk._precomp_wn[iknp],vecK[kpx],vecK[kpy]
                        )(0,0) * gamma_twoD_spsp_full_lower(Gk,vecK[kpx],vecK[kpy],kbarx,kbary,Gk._precomp_wn[iknp],wbar
                        ) * Gk(Gk._precomp_wn[iknp]-q._iwn,vecK[kpx]-q._qx,vecK[kpy]-q._qy
                        )(0,0);
                    }
                }
                middle_level_inf*=1.0/Gk._Nk/Gk._Nk;
            }
        }
    }
    middle_level_inf*=(SPINDEG/((Gk._Nk)*(Gk._Nk)*Gk._beta)); // Factor 2 for spin.
    middle_level_corr+=middle_level_inf;
    middle_level_inf+=gamma_twoD_spsp(Gk,kbarx_m_tildex,kbary_m_tildey,wtilde,wbar);
    middle_level-=middle_level_inf;
    middle_level+=1.0;
    if (is_jj){
        double current_vertices=0.0,kbarx;
        for (size_t ktildex=0; ktildex<vecK.size(); ktildex++){ // Many compbinations of ktildex (y) and kbarx (y) give within [-pi,pi] give the same difference. 
            kbarx = vecK[ktildex] - kbarx_m_tildex;
            current_vertices+=-2.0*std::sin(vecK[ktildex])*-2.0*std::sin(kbarx);
        }
        current_vertices*=1.0/Gk._Nk;
        return std::make_tuple(-1.0*current_vertices*Gk._u/middle_level,middle_level_inf,middle_level_corr);
    }
    else
        return std::make_tuple(Gk._u/middle_level,middle_level_inf,middle_level_corr);
}

template<class T>
inline std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > Susceptibility<T>::gamma_twoD_spsp_jj_full_middle_plotting(T& Type,double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wbar,std::complex<double> wtilde,HF::K_2D q,const IPT2::SplineInline< std::complex<double> >& splInline,const bool& is_jj) const{
/* Uses gamma_twoD_spsp to compute the infinite-ladder-down part and gamma_twoD_spsp_lower to compute corrections stemming from infinite delta G/delta phi. */    
    static_assert(std::is_same<T,IPT2::DMFTproc>::value, "Must be IPT solvers passed in.");
    std::complex<double> middle_level(0.0,0.0),middle_level_inf(0.0,0.0),middle_level_corr(0.0,0.0);
    double kbarx, kbary;
    for (size_t kpx=0; kpx<vecK.size(); kpx++){
        std::cout << "kpx: " << kpx << "\n";
        for (size_t kpy=0; kpy<vecK.size(); kpy++){
            std::cout << "kpy: " << kpy << "\n";
            for (size_t iknp=static_cast<size_t>(iwnArr_l.size()/2); iknp<iwnArr_l.size(); iknp++){
                for (size_t ktildex=0; ktildex<vecK.size(); ktildex++){
                    kbarx = vecK[ktildex] - kbarx_m_tildex;
                    for (size_t ktildey=0; ktildey<vecK.size(); ktildey++){
                        kbary = vecK[ktildey] - kbary_m_tildey;
                        middle_level_inf += 1.0/( iwnArr_l[iknp] + GreenStuff::mu - epsilonk(vecK[kpx],vecK[kpy]) - splInline.calculateSpline( (iwnArr_l[iknp]).imag() )
                        ) * gamma_twoD_spsp_full_lower(Type,vecK[kpx],vecK[kpy],kbarx,kbary,iwnArr_l[iknp],wbar,splInline
                        ) * 1.0/( (iwnArr_l[iknp]-q._iwn) + GreenStuff::mu - epsilonk(vecK[kpx]-q._qx,vecK[kpy]-q._qy) - splInline.calculateSpline( (iwnArr_l[iknp]-q._iwn).imag() ) );
                    }
                }
                middle_level_inf*=1.0/GreenStuff::N_k/GreenStuff::N_k;
            }
        }
    }
    middle_level_inf*=SPINDEG/(GreenStuff::N_k*GreenStuff::N_k*GreenStuff::beta); // Factor 2 for spin.
    middle_level_corr+=middle_level_inf;
    middle_level_inf+=gamma_twoD_spsp(Type,kbarx_m_tildex,kbary_m_tildey,wtilde,wbar,splInline);
    middle_level-=middle_level_inf;
    middle_level+=1.0;
    if (is_jj){
        double current_vertices=0.0,kbarx;
        for (size_t ktildex=0; ktildex<vecK.size(); ktildex++){ // Many compbinations of ktildex (y) and kbarx (y) give within [-pi,pi] give the same difference. 
            kbarx = vecK[ktildex] - kbarx_m_tildex;
            current_vertices+=-2.0*std::sin(vecK[ktildex])*-2.0*std::sin(kbarx);
        }
        current_vertices*=1.0/GreenStuff::N_k;
        return std::make_tuple(-1.0*current_vertices*GreenStuff::U/middle_level,middle_level_inf,middle_level_corr);
    }
    else
        return std::make_tuple(GreenStuff::U/middle_level,middle_level_inf,middle_level_corr);
}

template<>
inline std::complex<double> Susceptibility<HF::FunctorBuildGk>::getWeights(HF::FunctorBuildGk& Gk,HF::K_2D qq,double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar) const{
    std::complex<double> tmp_val_weigths(0.0,0.0);
    double kbary, kbarx;
    for (size_t ktildey=0; ktildey<vecK.size(); ktildey++){
        kbary = vecK[ktildey] - kbary_m_tildey;
        for (size_t ktildex=0; ktildex<vecK.size(); ktildex++){
            kbarx = vecK[ktildex] - kbarx_m_tildex;
            tmp_val_weigths += Gk(wtilde,vecK[ktildex],vecK[ktildey])(0,0)*Gk(wtilde+qq._iwn,vecK[ktildex]+qq._qx,vecK[ktildey]+qq._qy
                )(0,0)*Gk(wbar+qq._iwn,kbarx+qq._qx,kbary+qq._qy)(1,1)*Gk(wbar,kbarx,kbary)(1,1);
        }      
    }
    tmp_val_weigths*=1.0/vecK.size()/vecK.size();
    return tmp_val_weigths;
}

template<class T>
inline std::complex<double> Susceptibility<T>::getWeights(T& Type,HF::K_2D qq,double kbarx_m_tildex,double kbary_m_tildey,std::complex<double> wtilde,std::complex<double> wbar,const IPT2::SplineInline< std::complex<double> >& splInline) const{
    static_assert(std::is_same<T, IPT2::DMFTproc>::value, "Designed only for IPT2::DMFTproc!");
    std::complex<double> tmp_val_weigths(0.0,0.0);
    double kbary, kbarx;
    for (size_t ktildey=0; ktildey<vecK.size(); ktildey++){
        kbary = vecK[ktildey] - kbary_m_tildey;
        for (size_t ktildex=0; ktildex<vecK.size(); ktildex++){
            kbarx = vecK[ktildex] - kbarx_m_tildex;
            tmp_val_weigths += 1.0/( (wtilde) + GreenStuff::mu - epsilonk(ktildex,ktildey) - splInline.calculateSpline( (wtilde).imag() ) ) * 1.0/( (wtilde+qq._iwn) + GreenStuff::mu - epsilonk(ktildex+qq._qx,ktildey+qq._qy) - splInline.calculateSpline( (wtilde+qq._iwn).imag() )
                ) * 1.0/( (wbar+qq._iwn) + GreenStuff::mu - epsilonk(kbarx+qq._qx,kbary+qq._qy) - splInline.calculateSpline( (wbar+qq._iwn).imag() ) ) * 1.0/( (wbar) + GreenStuff::mu - epsilonk(kbarx,kbary) - splInline.calculateSpline( (wbar).imag() ) );;
        }      
    }
    tmp_val_weigths*=1.0/vecK.size()/vecK.size();
    return tmp_val_weigths;
}

#endif /* DIM */
/******************************* Common methods *******************************/

template<>
inline void calculateSusceptibilities< IPT2::DMFTproc >(IPT2::DMFTproc& sublatt1,const IPT2::SplineInline< std::complex<double> >& splInlineObj,const std::string& pathToDir,const std::string& customDirName,const bool& is_full,const bool& is_jj){
    Susceptibility<IPT2::DMFTproc> susObj;
    std::ofstream outputChispspGamma, outputChispspWeights, outputChispspBubble, outputChispspBubbleCorr;
    std::string trailingStr = is_full ? "_full" : "";
    std::string frontStr = is_jj ? "jj_" : "";
    std::string strOutputChispspGamma(pathToDir+customDirName+"/susceptibilities/ChispspGamma_IPT2_MULT_NTAU_"+std::to_string(MULT_N_TAU)+"_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+trailingStr+".dat");
    std::string strOutputChispspWeights(pathToDir+customDirName+"/susceptibilities/ChispspWeights_IPT2_MULT_NTAU_"+std::to_string(MULT_N_TAU)+"_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+trailingStr+".dat");
    std::string strOutputChispspBubble(pathToDir+customDirName+"/susceptibilities/ChispspBubble_IPT2_MULT_NTAU_"+std::to_string(MULT_N_TAU)+"_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+trailingStr+".dat");
    std::string strOutputChispspBubbleCorr(pathToDir+customDirName+"/susceptibilities/ChispspBubbleCorr_IPT2_MULT_NTAU_"+std::to_string(MULT_N_TAU)+"_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(GreenStuff::U)+"_beta_"+std::to_string(GreenStuff::beta)+"_N_tau_"+std::to_string(GreenStuff::N_tau)+"_Nk_"+std::to_string(GreenStuff::N_k)+trailingStr+".dat");
    #if DIM == 1
    for (size_t ktilde=0; ktilde<vecK.size(); ktilde++){
        std::cout << "ktilde: " << ktilde << "\n";
    #elif DIM == 2
    for (size_t kbary_m_tildey=0; kbary_m_tildey<vecK.size(); kbary_m_tildey++){
        std::cout << "ktildey_m_bary: " << kbary_m_tildey << "\n";
    #endif
    #if DIM == 1
        for (size_t kbar=0; kbar<vecK.size(); kbar++){
            //std::cout << "kbar: " << kbar << "\n";
    #elif DIM == 2
        for (size_t kbarx_m_tildex=0; kbarx_m_tildex<vecK.size(); kbarx_m_tildex++){
            //std::cout << "ktildex_m_barx: " << kbarx_m_tildex << "\n";
    #endif
            std::complex<double> tmp_val_kt_kb(0.0,0.0), tmp_val_kt_kb_bubble(0.0,0.0), tmp_val_weights(0.0,0.0), tmp_val_bubble_corr(0.0,0.0);
            for (size_t wtilde=static_cast<size_t>(iwnArr_l.size()/2); wtilde<iwnArr_l.size(); wtilde++){
                for (size_t wbar=static_cast<size_t>(iwnArr_l.size()/2); wbar<iwnArr_l.size(); wbar++){
                    std::tuple< std::complex<double>, std::complex<double>, std::complex<double> > gammaStuffFull;
                    std::tuple< std::complex<double>, std::complex<double> > gammaStuff;
                    if ( (wtilde==static_cast<size_t>(iwnArr_l.size()/2)) && (wbar==static_cast<size_t>(iwnArr_l.size()/2)) ){
                        if (!is_full){
                            #if DIM == 1
                            gammaStuff=susObj.gamma_oneD_spsp_jj_plotting(sublatt1,vecK[ktilde],iwnArr_l[wtilde],vecK[kbar],iwnArr_l[wbar],HF::K_1D(0.0,std::complex<double>(0.0,0.0)),splInlineObj,is_jj);
                            #elif DIM == 2
                            gammaStuff=susObj.gamma_twoD_spsp_jj_plotting(sublatt1,vecK[kbarx_m_tildex],vecK[kbary_m_tildey],iwnArr_l[wtilde],iwnArr_l[wbar],splInlineObj,is_jj);
                            #endif
                            
                            tmp_val_kt_kb+=std::get<0>(gammaStuff);
                            tmp_val_kt_kb_bubble+=std::get<1>(gammaStuff);
                        } else{
                            
                            #if DIM == 1
                            gammaStuffFull=susObj.gamma_oneD_spsp_jj_full_middle_plotting(sublatt1,vecK[ktilde],iwnArr_l[wtilde],vecK[kbar],iwnArr_l[wbar],HF::K_1D(0.0,std::complex<double>(0.0,0.0)),splInlineObj,is_jj);
                            #elif DIM == 2
                            gammaStuffFull=susObj.gamma_twoD_spsp_jj_full_middle_plotting(sublatt1,vecK[kbarx_m_tildex],vecK[kbary_m_tildey],iwnArr_l[wbar],iwnArr_l[wtilde],HF::K_2D(0.0,0.0,std::complex<double>(0.0,0.0)),splInlineObj,is_jj);
                            #endif
                            
                            tmp_val_kt_kb+=std::get<0>(gammaStuffFull);
                            tmp_val_kt_kb_bubble+=std::get<1>(gammaStuffFull);
                            tmp_val_bubble_corr+=std::get<2>(gammaStuffFull);
                        }
                        #if DIM == 1
                        tmp_val_weights+=susObj.get_weights(sublatt1,vecK[ktilde],iwnArr_l[wtilde],vecK[kbar],iwnArr_l[wbar],HF::K_1D(0.0,std::complex<double>(0.0,0.0)),splInlineObj);
                        #elif DIM == 2
                        tmp_val_weights+=susObj.getWeights(sublatt1,HF::K_2D(0.0,0.0,std::complex<double>(0.0,0.0)),vecK[kbarx_m_tildex],vecK[kbary_m_tildey],iwnArr_l[wtilde],iwnArr_l[wbar],splInlineObj);
                        #endif
                    }
                }
            }
            // tmp_val_weights*=1.0/GreenStuff::beta/GreenStuff::beta;
            // tmp_val_kt_kb*=1.0/GreenStuff::beta/GreenStuff::beta;
            // tmp_val_kt_kb_bubble*=1.0/GreenStuff::beta/GreenStuff::beta;
            outputChispspWeights.open(strOutputChispspWeights, std::ofstream::in | std::ofstream::app);
            outputChispspGamma.open(strOutputChispspGamma, std::ofstream::in | std::ofstream::app);
            outputChispspBubble.open(strOutputChispspBubble, std::ofstream::in | std::ofstream::app);
            outputChispspWeights << tmp_val_weights << " ";
            outputChispspGamma << tmp_val_kt_kb << " ";
            outputChispspBubble << tmp_val_kt_kb_bubble << " ";
            if (is_full){
                // tmp_val_bubble_corr*=1.0/GreenStuff::beta/GreenStuff::beta;
                outputChispspBubbleCorr.open(strOutputChispspBubbleCorr, std::ofstream::in | std::ofstream::app);
                outputChispspBubbleCorr << tmp_val_bubble_corr << " ";
                outputChispspBubbleCorr.close();
            }
            outputChispspWeights.close();
            outputChispspGamma.close();
            outputChispspBubble.close();
        }
        outputChispspWeights.open(strOutputChispspWeights, std::ofstream::in | std::ofstream::app);
        outputChispspGamma.open(strOutputChispspGamma, std::ofstream::in | std::ofstream::app);
        outputChispspBubble.open(strOutputChispspBubble, std::ofstream::in | std::ofstream::app);
        outputChispspWeights << "\n";
        outputChispspGamma << "\n";
        outputChispspBubble << "\n";
        if (is_full){
            outputChispspBubbleCorr.open(strOutputChispspBubbleCorr, std::ofstream::in | std::ofstream::app);
            outputChispspBubbleCorr << "\n";
            outputChispspBubbleCorr.close();
        }
        outputChispspWeights.close();
        outputChispspGamma.close();
        outputChispspBubble.close();
    }
}

template<>
inline void calculateSusceptibilities<HF::FunctorBuildGk>(HF::FunctorBuildGk& Gk,const std::string& pathToDir,const std::string& customDirName,const bool& is_full,const bool& is_jj){
    std::ofstream outputChispspGamma, outputChispspWeights, outputChispspBubble, outputChispspBubbleCorr;
    std::string trailingStr = is_full ? "_full" : ""; // Important for the data is loaded with colormap.py.
    std::string frontStr = is_jj ? "jj_" : "";
    std::string strOutputChispspGamma(pathToDir+customDirName+"/susceptibilities/ChispspGamma_HF_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+trailingStr+".dat");
    std::string strOutputChispspWeights(pathToDir+customDirName+"/susceptibilities/ChispspWeights_HF_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+trailingStr+".dat");
    std::string strOutputChispspBubble(pathToDir+customDirName+"/susceptibilities/ChispspBubble_HF_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+trailingStr+".dat");
    std::string strOutputChispspBubbleCorr(pathToDir+customDirName+"/susceptibilities/ChispspBubbleCorr_HF_"+frontStr+std::to_string(DIM)+"D_U_"+std::to_string(Gk._u)+"_beta_"+std::to_string(Gk._beta)+"_N_tau_"+std::to_string(Gk._size)+"_Nk_"+std::to_string(Gk._Nk)+trailingStr+".dat");
    Susceptibility< HF::FunctorBuildGk > susObj;
    #if DIM == 1
    for (size_t ktilde=0; ktilde<vecK.size(); ktilde++){
        std::cout << "ktilde: " << ktilde << "\n";
    #elif DIM == 2
    for (size_t kbary_m_tildey=0; kbary_m_tildey<vecK.size(); kbary_m_tildey++){
        std::cout << "ktildey_m_bary: " << kbary_m_tildey << "\n";
    #endif
    #if DIM == 1
        for (size_t kbar=0; kbar<vecK.size(); kbar++){
            std::cout << "kbar: " << kbar << "\n";
    #elif DIM == 2
        for (size_t kbarx_m_tildex=0; kbarx_m_tildex<vecK.size(); kbarx_m_tildex++){
            //std::cout << "ktildex_m_barx: " << kbarx_m_tildex << "\n";
    #endif
            std::complex<double> tmp_val_kt_kb(0.0,0.0), tmp_val_kt_kb_bubble(0.0,0.0), tmp_val_weights(0.0,0.0), tmp_val_bubble_corr(0.0,0.0);
            for (size_t wtilde=static_cast<size_t>(iwnArr_l.size()/2); wtilde<iwnArr_l.size(); wtilde++){
                for (size_t wbar=static_cast<size_t>(iwnArr_l.size()/2); wbar<iwnArr_l.size(); wbar++){
                    std::tuple< std::complex<double>, std::complex<double>, std::complex<double> > gammaStuffFull;
                    std::tuple< std::complex<double>, std::complex<double> > gammaStuff;
                    if ( (wtilde==static_cast<size_t>(iwnArr_l.size()/2)) && (wbar==static_cast<size_t>(iwnArr_l.size()/2)) ){
                        if (!is_full){
                            #if DIM == 1
                            gammaStuff=susObj.gamma_oneD_spsp_jj_plotting( Gk,vecK[ktilde],iwnArr_l[wtilde],vecK[kbar],iwnArr_l[wbar],HF::K_1D(0.0,std::complex<double>(0.0,0.0)),is_jj );
                            #elif DIM == 2
                            gammaStuff=susObj.gamma_twoD_spsp_jj_plotting( Gk,vecK[kbarx_m_tildex],vecK[kbary_m_tildey],iwnArr_l[wtilde],iwnArr_l[wbar],is_jj );
                            #endif
                            tmp_val_kt_kb+=std::get<0>(gammaStuff);
                            tmp_val_kt_kb_bubble+=std::get<1>(gammaStuff);
                        } else{
                            #if DIM == 1
                            gammaStuffFull=susObj.gamma_oneD_spsp_jj_full_middle_plotting( Gk,vecK[ktilde],iwnArr_l[wtilde],vecK[kbar],iwnArr_l[wbar],HF::K_1D(0.0,std::complex<double>(0.0,0.0)),is_jj );
                            #elif DIM == 2
                            gammaStuffFull=susObj.gamma_twoD_spsp_jj_full_middle_plotting( Gk,vecK[kbarx_m_tildex],vecK[kbary_m_tildey],iwnArr_l[wbar],iwnArr_l[wtilde],HF::K_2D(0.0,0.0,std::complex<double>(0.0,0.0)),is_jj );
                            #endif
        
                            tmp_val_kt_kb+=std::get<0>(gammaStuffFull);
                            tmp_val_kt_kb_bubble+=std::get<1>(gammaStuffFull);
                            tmp_val_bubble_corr+=std::get<2>(gammaStuffFull);
                        }
                        #if DIM == 1
                        tmp_val_weights+=susObj.get_weights( Gk,vecK[ktilde],iwnArr_l[wtilde],vecK[kbar],iwnArr_l[wbar],HF::K_1D(0.0,std::complex<double>(0.0,0.0)) );
                        #elif DIM == 2
                        tmp_val_weights+=susObj.getWeights( Gk,HF::K_2D(0.0,0.0,std::complex<double>(0.0,0.0)),vecK[kbarx_m_tildex],vecK[kbary_m_tildey],iwnArr_l[wtilde],iwnArr_l[wbar] );
                        #endif
                    }
                }
            }
            // tmp_val_weights*=1.0/Gk._beta/Gk._beta;
            // tmp_val_kt_kb*=1.0/Gk._beta/Gk._beta;
            // tmp_val_kt_kb_bubble*=1.0/Gk._beta/Gk._beta;
            outputChispspWeights.open(strOutputChispspWeights, std::ofstream::in | std::ofstream::app);
            outputChispspGamma.open(strOutputChispspGamma, std::ofstream::in | std::ofstream::app);
            outputChispspBubble.open(strOutputChispspBubble, std::ofstream::in | std::ofstream::app);
            outputChispspWeights << tmp_val_weights << " ";
            outputChispspGamma << tmp_val_kt_kb << " ";
            outputChispspBubble << tmp_val_kt_kb_bubble << " ";
            if (is_full){
                // tmp_val_bubble_corr*=1.0/Gk._beta/Gk._beta;
                outputChispspBubbleCorr.open(strOutputChispspBubbleCorr, std::ofstream::in | std::ofstream::app);
                outputChispspBubbleCorr << tmp_val_bubble_corr << " ";
                outputChispspBubbleCorr.close();
            }
            outputChispspWeights.close();
            outputChispspGamma.close();
            outputChispspBubble.close();
        }
        outputChispspWeights.open(strOutputChispspWeights, std::ofstream::in | std::ofstream::app);
        outputChispspGamma.open(strOutputChispspGamma, std::ofstream::in | std::ofstream::app);
        outputChispspBubble.open(strOutputChispspBubble, std::ofstream::in | std::ofstream::app);
        outputChispspWeights << "\n";
        outputChispspGamma << "\n";
        outputChispspBubble << "\n";
        if (is_full){
            outputChispspBubbleCorr.open(strOutputChispspBubbleCorr, std::ofstream::in | std::ofstream::app);
            outputChispspBubbleCorr << "\n";
            outputChispspBubbleCorr.close();
        }
        outputChispspWeights.close();
        outputChispspGamma.close();
        outputChispspBubble.close();
    }
}


#endif /* end of SUsceptibilities_H_ */