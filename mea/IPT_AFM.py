import numpy as np
from IPT_Utils import Sublattice
import matplotlib.pyplot as plt
import cubicspline as cs
import linearspline as ls


MAX_ITER = 100
MIN_TOL = 1e-6

def check_converged_AFM(G_0_last : np.ndarray, G_0_current : np.ndarray) -> bool:
    assert G_0_current.shape[0]==G_0_last.shape[0], "Inputted ndarrays must have the same lengths."
    G_0_up_diff=0.0; G_0_down_diff=0.0
    for l in range(G_0_current.shape[0]):
        G_0_up_diff += np.abs(G_0_last[l,0,0]-G_0_current[l,0,0])
        G_0_down_diff += np.abs(G_0_last[l,1,1]-G_0_current[l,1,1])

    if G_0_up_diff<MIN_TOL and G_0_down_diff<MIN_TOL:
        return True
    else:
        return False

def check_converged(G_0_last : np.ndarray, G_0_current : np.ndarray, it : int) -> bool:
    assert G_0_current.shape[0]==G_0_last.shape[0], "Inputted ndarrays must have the same lengths."
    G_0_diff=0.0
    for l in range(G_0_current.shape[0]):
        G_0_diff += np.abs(G_0_last[l,0,0]-G_0_current[l,0,0])

    if G_0_diff<MIN_TOL and it>10:
        return True
    else:
        return False

if __name__ == "__main__":
    #parameters
    beta=5.0
    U=4.0
    Ntau=4096 # Must be even, due to mirroring of the Matsubara frequencies
    hyb_c=2.0
    mu=U/2.0
    mu0=0.0
    Nk=999
    is_converged=False

    # Initializing static variables
    Sublattice._beta=beta
    Sublattice._U=U
    Sublattice._Ntau=Ntau
    Sublattice._hyb_c=hyb_c
    Sublattice._Nk=Nk
    Sublattice.set_static_attrs()
    
    # Relevant containers
    Hyb1 = np.ndarray((2*Ntau,2,2,),dtype=complex)
    Gloc = np.ndarray((2*Ntau,2,2,),dtype=complex)
    SE = np.ndarray((2*Ntau,2,2,),dtype=complex)
    # Initial Hyb
    for i,iwn in enumerate(Sublattice._iwn_arr):
        Hyb1[i,0,0] = hyb_c/iwn - 0.01 # up
        Hyb1[i,1,1] = hyb_c/iwn + 0.01 # down
    G01 = np.ndarray((2*Ntau,2,2,),dtype=complex)
    for i,iwn in enumerate(Sublattice._iwn_arr):
        G01[i,0,0] = 1.0/( iwn + mu0 - Hyb1[i,0,0] )
        G01[i,1,1] = 1.0/( iwn + mu0 - Hyb1[i,1,1] )

    # Now constructor. It builds various arrays
    sublatt1=Sublattice(mu,mu0,Hyb1,G01,Gloc,SE)
    
    it=0
    h=0.5
    Weiss_tmp = sublatt1._G0
    AFM = False
    n_target = 0.5
    while not is_converged and it<MAX_ITER:
        print("iter: ", it)
        if AFM:
            if it>1:
                h=0.0
            sublatt1.DMFT_AFM(h) 
            sublatt1.save_Gloc_AFM(it)
            if it>1:
                is_converged = check_converged_AFM(Weiss_tmp,sublatt1._G0)
        else:
            sublatt1.update_densities(it,n_target)
            sublatt1.DMFT()
            sublatt1.save_Gloc(it)
            sublatt1.save_SE(it)
            if it>1:
                is_converged = check_converged(Weiss_tmp,sublatt1._G0,it)
        Weiss_tmp = sublatt1._G0
        it+=1

