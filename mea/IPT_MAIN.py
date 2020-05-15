import numpy as np
from IPT_Utils import Sublattice, check_converged, check_converged_AFM
import matplotlib.pyplot as plt
from copy import deepcopy

# Max number of iterations in DMFT loop
MAX_ITER = 100

if __name__ == "__main__":
    ############################### PARAMETERS ####################################
    dim=1 # Dimension can either be 1 or 2
    betamin=25.0; betamax=27.0; betastep=2.0 # Inverse temperature
    Umin=2.0; Umax=3.0; Ustep=1.0 # Hubbard U
    Ntau=1024 # Must be even, due to mirroring of the Matsubara frequencies
    mu0=0.0 # Chemical potential for the Weiss Green's function. Relevant only in the PM case
    Nk=201 # k-space grid
    h=0.1 # initial spin splitting. Relevant for AFM case.
    type_spline = "cubic" # Type of spline used in the impurity solver
    # Set this parameter to True to set up calculations allowing for AFM. Set to False to compute in PM state only.
    AFM = True
    # Target particle density. Notice that fixing the chemical potentials mu=U/2 and mu0=0 leads the density values slightly different to
    # half-filling at low T. This is more stable when using cubic spline, because fixing chemical potentials leads to half-filling for wider
    # range of parameters.
    n_target = 0.5
    ############################### PARAMETERS ####################################

    if dim==1:
        hyb_c=2.0 # leading moment of Hyb function is hyb_c/iwn
    elif dim==2:
        hyb_c=4.0

    # Initializing static variables that apply for the whole system (both sublattices)
    Sublattice._Ntau=Ntau
    Sublattice._hyb_c=hyb_c
    Sublattice._Nk=Nk
    if AFM:
        Sublattice._h=h
    else:
        Sublattice._h=0.0

    # Relevant containers to start with
    Hyb1 = np.ndarray((2*Ntau,2,2,),dtype=complex)
    Gloc = np.ndarray((2*Ntau,2,2,),dtype=complex)
    SE = np.ndarray((2*Ntau,2,2,),dtype=complex)
    SE_tau = np.ndarray((2*Ntau+1,2,2,),dtype=float)
    Gloc_tau = np.ndarray((2*Ntau+1,2,2,),dtype=float)

    for U in np.linspace(Umin,Umax,Ustep,endpoint=True):
        for beta in np.linspace(betamin,betamax,betastep,endpoint=True):
            print("U: ", U, "beta: ", beta)
            Sublattice._beta=beta
            Sublattice._U=U
        
            Sublattice.set_static_attrs()
            mu=U/2.0 # Chemical potential on the impurity
            # Initial Hyb
            for i,iwn in enumerate(Sublattice._iwn_arr):
                Hyb1[i,0,0] = hyb_c/iwn # up
                Hyb1[i,1,1] = hyb_c/iwn # down
            G01 = np.ndarray((2*Ntau,2,2,),dtype=complex)
            # Initial Weiss Green's function
            for i,iwn in enumerate(Sublattice._iwn_arr):
                G01[i,0,0] = 1.0/( iwn - Sublattice._h + mu0 - Hyb1[i,0,0] )
                G01[i,1,1] = 1.0/( iwn + Sublattice._h + mu0 - Hyb1[i,1,1] )

            # Constructor of one of the sublattice. It sets the various arrays used in the DMFT procedure.
            sublatt1=Sublattice(mu,mu0,Hyb1,G01,Gloc,Gloc_tau,SE,SE_tau)
            
            it=0 # DMFT iteration initialization
            Weiss_tmp = deepcopy(sublatt1._G0)
            is_converged = False
            if AFM:
                sublatt1.set_mu_to_0()
            while not is_converged and it<MAX_ITER:
                print("\n"+"*"*25+" iter: ", it, "*"*25)
                print("beta: ", Sublattice._beta, " U: ", Sublattice._U, " mu: ", sublatt1._mu, " h: ", Sublattice._h, " hyb_c: ", Sublattice._hyb_c)
                if AFM:
                    # AFM case
                    if it>1:
                        Sublattice._h=0.0
                    sublatt1.DMFT_AFM(dim,type_spline) 
                    #sublatt1.save_Gloc_AFM(it,dim,type_spline)
                    #sublatt1.save_SE_AFM(it,dim,type_spline)
                    sublatt1.save_Gloc_tau_AFM(it,dim,type_spline)
                    if it>1:
                        is_converged = check_converged_AFM(Weiss_tmp,sublatt1._G0,it)
                else:
                    # PM case
                    sublatt1.update_densities(it,n_target) # To disable chemical potential root finding, comment this line
                    sublatt1.DMFT(dim,type_spline)
                    sublatt1.dbl_occupancy() # Computes the double occupancy
                    sublatt1.save_Gloc(it,dim,type_spline)
                    sublatt1.save_SE(it,dim,type_spline)
                    if it>1:
                        is_converged = check_converged(Weiss_tmp,sublatt1._G0,it)
                Weiss_tmp = deepcopy(sublatt1._G0)
                it+=1

            Sublattice.reset_static_attrs()
