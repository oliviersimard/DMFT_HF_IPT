import numpy as np
from IPT_Utils import Sublattice, check_converged, check_converged_AFM
import matplotlib.pyplot as plt
import cubicspline as cs
import linearspline as ls

# Max number of iterations in DMFT loop
MAX_ITER = 100

if __name__ == "__main__":
    #parameters
    beta=5.0
    U=12.0
    Ntau=4096 # Must be even, due to mirroring of the Matsubara frequencies
    hyb_c=2.0 # leading moment of Hyb function is hyb_c/iwn
    mu=U/2.0 # Chemical potential on the impurity
    mu0=0.0 # Chemical potential for the Weiss Green's function
    Nk=500

    # Initializing static variables that apply for the whole system (both sublattices)
    Sublattice._beta=beta
    Sublattice._U=U
    Sublattice._Ntau=Ntau
    Sublattice._hyb_c=hyb_c
    Sublattice._Nk=Nk
    Sublattice.set_static_attrs()
    
    # Relevant containers to start with
    Hyb1 = np.ndarray((2*Ntau,2,2,),dtype=complex)
    Gloc = np.ndarray((2*Ntau,2,2,),dtype=complex)
    SE = np.ndarray((2*Ntau,2,2,),dtype=complex)
    SE_tau = np.ndarray((2*Ntau+1,2,2,),dtype=float)
    Gimp_tau = np.ndarray((2*Ntau+1,2,2,),dtype=float)
    # Initial Hyb
    for i,iwn in enumerate(Sublattice._iwn_arr):
        Hyb1[i,0,0] = hyb_c/iwn - 0.01 # up
        Hyb1[i,1,1] = hyb_c/iwn + 0.01 # down
    G01 = np.ndarray((2*Ntau,2,2,),dtype=complex)
    # Initial Weiss Green's function
    for i,iwn in enumerate(Sublattice._iwn_arr):
        G01[i,0,0] = 1.0/( iwn + mu0 - Hyb1[i,0,0] )
        G01[i,1,1] = 1.0/( iwn + mu0 - Hyb1[i,1,1] )

    # Constructor of one of the sublattice. It sets the various arrays used in the DMFT procedure.
    sublatt1=Sublattice(mu,mu0,Hyb1,G01,Gloc,SE,SE_tau,Gimp_tau)
    
    it=0 # DMFT iteration initialization
    h=0.5 # initial spin splitting. Relevant for AFM case.
    Weiss_tmp = sublatt1._G0
    # Set this parameter to True to set up calculations allowing for AFM. Set to False to compute in PM state only.
    AFM = False
    is_converged = False
    # Target particle density. Notice that fixing the chemical potentials mu=U/2 and mu0=0 leads the density values slightly different to
    # half-filling at low T. This is more stable when using cubic spline, because fixing chemical potentials leads to half-filling for wider
    # range of parameters.
    n_target = 0.5
    while not is_converged and it<MAX_ITER:
        print("iter: ", it)
        if AFM:
            # AFM case
            if it>1:
                h=0.0
            sublatt1.update_densities_AFM(it,n_target) # To disable chemical potential root finding, comment this line
            sublatt1.DMFT_AFM(h) 
            sublatt1.save_Gloc_AFM(it)
            sublatt1.save_SE_AFM(it)
            if it>1:
                is_converged = check_converged_AFM(Weiss_tmp,sublatt1._G0,it)
        else:
            # PM case
            sublatt1.update_densities(it,n_target) # To disable chemical potential root finding, comment this line
            sublatt1.DMFT()
            sublatt1.dbl_occupancy()
            sublatt1.save_Gloc(it)
            sublatt1.save_SE(it)
            if it>1:
                is_converged = check_converged(Weiss_tmp,sublatt1._G0,it)
        Weiss_tmp = sublatt1._G0
        it+=1

