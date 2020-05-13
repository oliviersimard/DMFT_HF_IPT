import numpy as np
from scipy.integrate import simps
from copy import deepcopy
import matplotlib.pyplot as plt
from sys import exit

# Max number of iterations in DMFT loop
MAX_ITER = 100
MIN_TOL = 1e-4

def epsilonk(k : float) -> float:
    """
    This function represents the nearest-neighbour dispersion relation in 1D.

    Parameters:
        k (float): momentum
    
    Returns:
        (float): dispersion relation at inputted momentum
    """
    return -2.0*np.cos(k)

def linear_spline_tau_to_iwn(G_tau, beta : float):
    """
    Function computing the fermionic Matsubara frequency representation of inputted imaginary-time object using a linear spline.

    Parameters:
        G_tau (array): imaginary-time defined object to be transformed
        beta (float): inverse temperature

    Returns:
        G_iwn (array): fermionic Matsubara frequency representation of the inputted imaginary-time object
    """
    M = len(G_tau)
    NN = M//2
    assert np.mod(M,2)==1, "The imaginary-time length has got to be odd. (iwn has to be even for mirroring reasons.)"
    *_,delta_tau = np.linspace(0,beta,M,retstep=True)
    # beta value is discarded, because not needed
    S_p = np.empty(M-1,dtype=complex)
    G_iwn = np.empty(M-1,dtype=complex)
    iwn_array = np.array([1.j*(2.0*n+1.0)*np.pi/beta for n in range(-NN,NN)],dtype=complex)
    # Filling up the kernel array that will enter FFT
    for ip in range(M-1):
        S_p[ip] =  np.exp( 1.j * np.pi * ip / (M-1) ) * G_tau[ip]
    
    # Fourier transforming
    IFFT_data = np.fft.ifft(S_p)
    ## Mirroring, because using negative Matsubara frequencies to start with
    IFFT_data = np.concatenate((IFFT_data[NN:],IFFT_data[:NN]))

    for i,iwn in enumerate(iwn_array):
        G_iwn[i] = 1.0/iwn + (-1.0 + np.exp(-iwn*delta_tau))/( ((iwn)**2)*delta_tau ) \
             + 2.0/( delta_tau * ((iwn)**2) ) * ( np.cos(iwn.imag*delta_tau) - 1.0 ) * M * IFFT_data[i]

    return G_iwn

def get_iwn_to_tau(G_iwn, beta : float, type_of="positive"):
    """
    This function computes transformation from iwn to tau for fermionic functions.

    Parameters:
        G_iwn (array): function defined in fermionic Matsubara frequencies to transform to imaginary time.
        beta (float): inverse temperature

    Returns:
        tau_final_G (array): function defined in imaginary time (tau)
    """
    MM = len(G_iwn) # N = M/2
    tau_final_G = np.zeros(MM+1,dtype=float)
    # FFT
    if type_of=="positive": # G(tau)
        tau_resolved_G = np.fft.fft(G_iwn)
    elif type_of=="negative": # G(-tau)
        tau_resolved_G = np.fft.fft(np.conj(G_iwn))

    for i in range(MM):
        tau_final_G[i] = ( np.exp( -1.j * np.pi * i * ( 1.0/(MM) - 1.0 ) )*tau_resolved_G[i] ).real

    for i in range(MM):
        tau_final_G[MM] += ( np.exp( -1.j * np.pi * (1.0-(MM)) )*G_iwn[i] ).real
    tau_final_G *= (1./beta)

    return tau_final_G

if __name__ == "__main__":
    #parameters
    beta=5.0
    U=4.0
    Ntau=4096 # Must be even, due to mirroring of the Matsubara frequencies
    hyb_c=2.0 # leading moment of Hyb function is hyb_c/iwn
    mu=U/2.0 # Chemical potential on the impurity
    mu0=0.0 # Chemical potential for the Weiss Green's function
    Nk=501

    delta_tau = beta/Ntau
    iwn_array = np.empty((2*Ntau,),dtype=complex)
    tau_array = np.empty((2*Ntau+1,),dtype=complex)
    iwn_array = np.array([1.0j*(2.0*n+1.0)*np.pi/beta for n in range(-Ntau,Ntau)],dtype=complex)
    for l in range(2*Ntau+1):
        tau_array[l] = l*delta_tau
    k_array = np.array([-np.pi+l*(2.0*np.pi/(Nk-1)) for l in range(Nk)],dtype=float)
    # Relevant containers to start with
    Hyb1 = np.ndarray((2*Ntau,2,2,),dtype=complex)
    G01 = np.ndarray((2*Ntau,2,2,),dtype=complex)
    # Initial Hyb
    for i,iwn in enumerate(iwn_array):
        Hyb1[i,0,0] = hyb_c/iwn # up
    # Initial Weiss Green's function
    for i,iwn in enumerate(iwn_array):
        G01[i,0,0] = 1.0/( iwn + mu0 - Hyb1[i,0,0] )
    # Other containers relevant after initialization
    Gloc = np.ndarray((2*Ntau,2,2,),dtype=complex) # Local Green's function (iwn)
    Gloc_tau = np.ndarray((2*Ntau+1,2,2,),dtype=float) # Local Green's function (tau)
    G0_tau = np.ndarray((2*Ntau+1,2,2,),dtype=float) # Weiss Green's function (tau)
    G0_m_tau = np.ndarray((2*Ntau+1,2,2,),dtype=float) # Weiss Green's function (-tau)
    SE = np.ndarray((2*Ntau,2,2,),dtype=complex) # self-energy (iwn)
    SE_tau = np.ndarray((2*Ntau+1,2,2,),dtype=float) # self-energy (tau)
    
    it=0 # DMFT iteration initialization
    Weiss_tmp = deepcopy(G01)
    G0 = deepcopy(G01) # Weiss Green's function (iwn)
    Hyb = deepcopy(Hyb1) # hybridisation function (iwn)
    is_converged = False
    while not is_converged and it<MAX_ITER:
        print("iter: ", it)
        # AFM case
        if it>1:
            h=0.0
        # subtracting the leadind tail
        plt.figure(0)
        plt.plot(list(map(lambda x: x.imag,iwn_array)),list(map(lambda x: x.imag,G0[:,0,0])),marker='P',ms=2.5,c="black")
        for j,iwn in enumerate(iwn_array):
            G0[j,0,0] -= 1.0/( iwn )
        G0_tau[:,0,0] = get_iwn_to_tau(G0[:,0,0],beta) #up
        G0_m_tau[:,0,0] = get_iwn_to_tau(G0[:,0,0],beta,type_of="negative") #up
        for i in range(len(tau_array)):
            G0_tau[i,0,0] += -0.5 #up
            G0_m_tau[i,0,0] += 0.5
        # Weiss Green's function densities used for Hartree term
        n0_up = -1.0*G0_tau[-1,0,0]
        print("n0_up: ", n0_up)
        # Self-energy impurity
        for i in range(SE_tau.shape[0]):
            SE_tau[i,0,0] = G0_tau[i,0,0]*G0_m_tau[i,0,0]*G0_tau[i,0,0] #up
        # Fourier transforming back the self-energy to iwn
        SE[:,0,0] = U*(1.0-n0_up) - U*U*linear_spline_tau_to_iwn(SE_tau[:,0,0],beta) #up

        # Computing the local densities
        for i,iwn in enumerate(iwn_array):
            Gloc[i,0,0] = 1.0/( iwn + mu - Hyb[i,0,0] - SE[i,0,0] ) - 1.0/( iwn ) # up
        Gloc_tau[:,0,0] = get_iwn_to_tau(Gloc[:,0,0],beta)
        for i in range(len(tau_array)):
            Gloc_tau[i,0,0] += -0.5
        n_up = -1.0*Gloc_tau[-1,0,0]
        print("impurity n_up: ", n_up)

        # DMFT
        # Not as stable if using quad to integrate. Better to use simps, even though much longer
        for i,iwn in enumerate(iwn_array):
            k_def_G_latt_up = np.empty((Nk,),dtype=complex)
            for j,k in enumerate(k_array):
                k_def_G_latt_up[j] = 1.0/( iwn + mu - SE[i,0,0] - epsilonk(k) )
            Gloc[i,0,0] = 1.0/(2*np.pi)*simps(k_def_G_latt_up,k_array) #up
        # update the hybdridisation function and set Weiss Green's function for the next iteration
        for i,iwn in enumerate(iwn_array):
            Hyb[i,0,0] = iwn + mu - SE[i,0,0] - 1.0/Gloc[i,0,0] # up
            G0[i,0,0] = 1.0/( iwn + mu0 - Hyb[i,0,0] ) # up

        filename = "data_test_IPT/Green_loc_1D_U_{0:.5f}".format(U)+"_beta_{0:.5f}".format(beta)+"_N_tau_{0:d}".format(Ntau)+"_Nit_{0:d}".format(it)+".dat"
        with open(filename,"w+") as floc:
            for i in range(len(iwn_array)):
                if i==0:
                    floc.write("iwn\t\tRe Gloc up\t\tIm Gloc up\n")
                floc.write("{0:.8f}\t\t{1:.8f}\t\t{2:.8f}\n".format(iwn_array[i].imag,Gloc[i,0,0].real,Gloc[i,0,0].imag))
        floc.close()

        filename = "data_test_IPT/Self_energy_1D_U_{0:.5f}".format(U)+"_beta_{0:.5f}".format(beta)+"_N_tau_{0:d}".format(Ntau)+"_Nit_{0:d}".format(it)+".dat"
        with open(filename,"w+") as fse:
            for i in range(len(iwn_array)):
                if i==0:
                    fse.write("iwn\t\tRe SE up\t\tIm SE up\n")
                fse.write("{0:.8f}\t\t{1:.8f}\t\t{2:.8f}\n".format(iwn_array[i].imag,SE[i,0,0].real,SE[i,0,0].imag))
        fse.close()
        
        if it>1:
            G_0_up_diff = 0.0
            for l in range(G0.shape[0]):
                G_0_up_diff += np.abs(Weiss_tmp[l,0,0]-G0[l,0,0])
            print("G0_diff up: ", G_0_up_diff)
            if G_0_up_diff<MIN_TOL:
                is_converged=True
        Weiss_tmp = deepcopy(G0)
        it+=1

