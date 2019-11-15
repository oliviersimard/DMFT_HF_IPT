#ifndef Susceptibilities_H_
#define Susceptibilities_H_

#include<tuple>
#include<algorithm>
#include "IPT2nd3rdorderSingle2.hpp"

// Make this class a template such that it hosts HF::FunctorBuildGk and IPT::DMFTproc objects
template<class T>
class Susceptibility{ // The current-current susceptibility uses the +U in gamma, multiplied by the currents thereafter.
    public:
        std::complex<double> get_weights(T&,double,std::complex<double>,double,std::complex<double>,HF::K_1D) const;
        std::complex<double> get_weights(T&,double,std::complex<double>,double,std::complex<double>,HF::K_1D,const IPT2::SplineInline< std::complex<double> >&) const;
        std::complex<double> gamma_oneD_spsp(T&,double,std::complex<double>,double,std::complex<double>) const;
        std::complex<double> gamma_oneD_spsp(T&,double,std::complex<double>,double,std::complex<double>,const IPT2::SplineInline< std::complex<double> >&) const;
        std::complex<double> gamma_oneD_spsp_full_lower(T&,double,double,std::complex<double>,std::complex<double>) const;
        std::complex<double> gamma_oneD_spsp_full_lower(T&,double,double,std::complex<double>,std::complex<double>,const IPT2::SplineInline< std::complex<double> >&) const;
        std::tuple< std::complex<double>, std::complex<double>, std::complex<double> > gamma_oneD_spsp_full_middle_plotting(T&,double,std::complex<double>,double,std::complex<double>,HF::K_1D) const;
        std::tuple< std::complex<double>, std::complex<double>, std::complex<double> > gamma_oneD_spsp_full_middle_plotting(T&,double,std::complex<double>,double,std::complex<double>,HF::K_1D,const IPT2::SplineInline< std::complex<double> >&) const;
        std::tuple< std::complex<double>, std::complex<double> > gamma_oneD_spsp_plotting(T&,double,std::complex<double>,double,std::complex<double>,HF::K_1D) const;
        std::tuple< std::complex<double>, std::complex<double> > gamma_oneD_spsp_plotting(T&,double,std::complex<double>,double,std::complex<double>,HF::K_1D,const IPT2::SplineInline< std::complex<double> >&) const;
        std::tuple< std::complex<double>, std::complex<double> > gamma_oneD_spsp_crossed_plotting(T&,double,std::complex<double>,double,std::complex<double>,HF::K_1D) const;
        std::tuple< std::complex<double>, std::complex<double> > gamma_oneD_spsp_crossed_plotting(T&,double,std::complex<double>,double,std::complex<double>,HF::K_1D,const IPT2::SplineInline< std::complex<double> >&) const;
};

/*************************************************************** 1D methods ***************************************************************/
/* Functions entering the full spin susceptibility. */
template<>
inline std::complex<double> Susceptibility<HF::FunctorBuildGk>::gamma_oneD_spsp_full_lower(HF::FunctorBuildGk& Gk,double kp,double kbar,std::complex<double> iknp,std::complex<double> wbar) const{
    std::complex<double> lower_level(0.0,0.0);
    for (size_t iknpp=0; iknpp<Gk._size; iknpp++){
        for (size_t kpp=0; kpp<Gk._kArr_l.size(); kpp++){
            lower_level += Gk(Gk._precomp_wn[iknpp]+iknp-wbar,Gk._kArr_l[kpp]+kp-kbar)(0,0)*Gk(Gk._precomp_wn[iknpp],Gk._kArr_l[kpp])(1,1);
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
    for (int wttilde=0; wttilde<Gk._size; wttilde++){
        for (size_t qttilde=0; qttilde<Gk._kArr_l.size(); qttilde++){
            lower_level += Gk(wtilde-Gk._precomp_qn[wttilde],ktilde-Gk._kArr_l[qttilde])(0,0)*Gk(wbar-Gk._precomp_qn[wttilde],kbar-Gk._kArr_l[qttilde])(1,1);
        }
    }
    lower_level *= -SPINDEG*Gk._u/(Gk._beta*(Gk._Nk)); // Factor 2 for spin summation.
    return lower_level;
}

template<class T>
inline std::complex<double> Susceptibility<T>::gamma_oneD_spsp(T& Type,double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar,const IPT2::SplineInline< std::complex<double> >& splInline) const{ // q's contain bosonic Matsubara frequencies.
    static_assert(std::is_same<T, IPT2::DMFTproc>::value, "Designed only for IPT2::DMFTproc!");
    std::complex<double> lower_level(0.0,0.0);
    for (int wttilde=static_cast<size_t>(iwnArr_l.size()/2); wttilde<iwnArr_l.size(); wttilde++){
        for (size_t qttilde=0; qttilde<vecK.size(); qttilde++){
            lower_level += 1.0/( (wtilde-iqnArr_l[wttilde]) + GreenStuff::mu - epsilonk(ktilde-vecK[qttilde]) - splInline.calculateSpline( (wtilde-iqnArr_l[wttilde]).imag() ) ) * 1.0/( (wbar-iqnArr_l[wttilde]) + GreenStuff::mu - epsilonk(kbar-vecK[qttilde]) - splInline.calculateSpline( (wbar-iqnArr_l[wttilde]).imag() ) );
        }
    }
    lower_level *= -SPINDEG*GreenStuff::U/(GreenStuff::beta*GreenStuff::N_k); // Factor 2 for spin summation.
    return lower_level;
}

template<>
inline std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > Susceptibility<HF::FunctorBuildGk>::gamma_oneD_spsp_full_middle_plotting(HF::FunctorBuildGk& Gk,double ktilde,std::complex<double> wbar,double kbar,std::complex<double> wtilde,HF::K_1D q) const{
/* Uses gamma_oneD_spsp to compute the infinite-ladder-down part and gamma_oneD_spsp_lower to compute corrections stemming from infinite delta G/delta phi. */    
    std::complex<double> middle_level(0.0,0.0),middle_level_inf(0.0,0.0),middle_level_corr(0.0,0.0);
    for (size_t kp=0; kp<Gk._kArr_l.size(); kp++){
        for (size_t iknp=0; iknp<Gk._size; iknp++){
            middle_level_inf += Gk(Gk._precomp_wn[iknp],Gk._kArr_l[kp]
            )(0,0) * gamma_oneD_spsp_full_lower(Gk,Gk._kArr_l[kp],kbar,Gk._precomp_wn[iknp],wbar
            ) * Gk(Gk._precomp_wn[iknp]-q._iwn,Gk._kArr_l[kp]-q._qx
            )(0,0);

        }
    }
    middle_level_inf*=(SPINDEG/((Gk._Nk)*Gk._beta)); // Factor 2 for spin.
    middle_level_corr+=middle_level_inf;
    middle_level_inf+=gamma_oneD_spsp(Gk,ktilde,wtilde,kbar,wbar);
    middle_level-=middle_level_inf;
    middle_level+=1.0;
    return std::make_tuple(Gk._u/middle_level,middle_level_inf,middle_level_corr);
}

template<class T>
inline std::tuple< std::complex<double>,std::complex<double>,std::complex<double> > Susceptibility<T>::gamma_oneD_spsp_full_middle_plotting(T& Type,double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar,HF::K_1D q,const IPT2::SplineInline< std::complex<double> >& splInline) const{
/* Uses gamma_oneD_spsp to compute the infinite-ladder-down part and gamma_oneD_spsp_lower to compute corrections stemming from infinite delta G/delta phi. */    
    static_assert(std::is_same<T, IPT2::DMFTproc>::value, "Designed only for IPT2::DMFTproc!");
    std::complex<double> middle_level(0.0,0.0),middle_level_inf(0.0,0.0),middle_level_corr(0.0,0.0);
    for (size_t kp=0; kp<vecK.size(); kp++){
        std::cout << "kp: " << kp << "\n";
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
    return std::make_tuple(GreenStuff::U/middle_level,middle_level_inf,middle_level_corr);
}

template<>
inline std::tuple< std::complex<double>, std::complex<double> > Susceptibility<HF::FunctorBuildGk>::gamma_oneD_spsp_plotting(HF::FunctorBuildGk& Gk,double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar,HF::K_1D q) const{ // q's contain bosonic Matsubara frequencies.
    std::complex<double> lower_level(0.0,0.0), chi_bubble(0.0,0.0); // chi_bubble represents the lower bubble in the vertex function, not the total vertex function.
    for (int wttilde=0; wttilde<Gk._size; wttilde++){
        for (size_t qttilde=0; qttilde<Gk._kArr_l.size(); qttilde++){
            lower_level += Gk(wtilde+q._iwn-Gk._precomp_qn[wttilde],ktilde+q._qx-Gk._kArr_l[qttilde])(0,0)*Gk(wbar+q._iwn-Gk._precomp_qn[wttilde],kbar+q._qx-Gk._kArr_l[qttilde])(1,1);
            //lower_level += Gk(-(wtilde+q._iwn-Gk._precomp_qn[wttilde]),-Gk._kArr_l[qttilde])(0,0)*Gk(-(wbar+q._iwn-Gk._precomp_qn[wttilde]),-(Gk._kArr_l[qttilde]+kbar-ktilde))(1,1);
        }
    }
    lower_level *= -SPINDEG*Gk._u/(Gk._beta*Gk._Nk); //factor 2 for the spin and minus sign added
    chi_bubble = lower_level; 
    lower_level *= -1.0;
    lower_level += 1.0;
    return std::make_tuple(Gk._u/lower_level,chi_bubble);
}

template<class T>
inline std::tuple< std::complex<double>, std::complex<double> > Susceptibility<T>::gamma_oneD_spsp_plotting(T& Type,double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar,HF::K_1D q,const IPT2::SplineInline< std::complex<double> >& splInline) const{ // q's contain bosonic Matsubara frequencies.
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
    return std::make_tuple(GreenStuff::U/lower_level,chi_bubble);
}

template<>
inline std::tuple< std::complex<double>,std::complex<double> > Susceptibility<HF::FunctorBuildGk>::gamma_oneD_spsp_crossed_plotting(HF::FunctorBuildGk& Gk,double ktilde,std::complex<double> wtilde,double kbar,std::complex<double> wbar, HF::K_1D q) const{ // q's contain bosonic Matsubara frequencies.
    std::complex<double> lower_level(0.0,0.0),chi_bubble(0.0,0.0);
    for (int wttilde=0; wttilde<Gk._size; wttilde++){
        for (size_t qttilde=0; qttilde<Gk._kArr_l.size(); qttilde++){
            lower_level += Gk(wtilde+wbar-Gk._precomp_qn[wttilde]-q._iwn,ktilde+kbar-Gk._kArr_l[qttilde]-q._qx)(0,0)*Gk(Gk._precomp_qn[wttilde],Gk._kArr_l[qttilde])(1,1);
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
            lower_level += 1.0/( (wtilde+wbar-q._iwn-iqnArr_l[wttilde]) + GreenStuff::mu - epsilonk(ktilde+kbar-vecK[qttilde]-q._qx) - splInline.calculateSpline( (wtilde+wbar-q._iwn-iqnArr_l[wttilde]).imag() ) ) * 1.0/( iqnArr_l[wttilde] + GreenStuff::mu - epsilonk(vecK[qttilde]) - splInline.calculateSpline( iqnArr_l[wttilde].imag() ) );//Gk(Gk._precomp_qn[wttilde],Gk._kArr_l[qttilde])(1,1);
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


#endif /* end of SUsceptibilities_H_ */