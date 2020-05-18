import numpy as np
from scipy.integrate import simps
from copy import deepcopy
import matplotlib.pyplot as plt
import linearspline as ls
import cubicspline as cs
import os
from time import sleep

# Max number of iterations in DMFT loop
MAX_ITER = 100
MIN_TOL = 1e-4
# Max number of iterations in the false position root finding method
MAX_ITER_ROOT = 40

def epsilonk(k : float) -> float:
    """
    This function represents the nearest-neighbour dispersion relation in 1D.

    Parameters:
        k (float): momentum
    
    Returns:
        (float): dispersion relation at inputted momentum
    """
    return -2.0*np.cos(k)

def epsilonk_2D(kx : float,ky : float) -> float:
    """
    This function represents the nearest-neighbour dispersion relation in 1D.

    Parameters:
        k (float): momentum
    
    Returns:
        (float): dispersion relation at inputted momentum
    """
    return -2.0*(np.cos(kx)+np.cos(ky))

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
    ##################################### parameters ############################################
    type_spline = "cubic" # can either be linear or cubic
    dim=2 # Can be one or two
    #beta=2.0
    #U=2.0
    Ntau=1024  # Must be even, due to mirroring of the Matsubara frequencies. In 2d, set at least 512
    Nk=201 # In 2d, decrease this to around 51
    ##################################### parameters ############################################
    if dim==1:
        hyb_c=2.0 # leading moment of Hyb function is hyb_c/iwn
    elif dim==2:
        hyb_c=4.0
    else:
        raise ValueError("Wrong inputted dimension of calculations.")
    mu=0.0 # Chemical potential on the impurity
    #mu0=U/2.0
    k_array = np.array([-np.pi/2.0+l*(1.0*np.pi/(Nk-1)) for l in range(Nk)],dtype=float)
    k_array_full = np.array([-np.pi+l*(2.0*np.pi/(Nk-1)) for l in range(Nk)],dtype=float)
    
    # Relevant containers to start with
    Hyb1 = np.ndarray((2*Ntau,2,2,),dtype=complex)
    G01 = np.ndarray((2*Ntau,2,2,),dtype=complex)
    # Other containers relevant after initialization
    Gloc = np.ndarray((2*Ntau,2,2,),dtype=complex) # Local Green's function (iwn)
    Gloc_tau = np.ndarray((2*Ntau+1,2,2,),dtype=float) # Local Green's function (tau)
    G0_tau = np.ndarray((2*Ntau+1,2,2,),dtype=float) # Weiss Green's function (tau)
    G0_m_tau = np.ndarray((2*Ntau+1,2,2,),dtype=float) # Weiss Green's function (-tau)
    SE = np.ndarray((2*Ntau,2,2,),dtype=complex) # self-energy (iwn)
    SE_tau = np.ndarray((2*Ntau+1,2,2,),dtype=float) # self-energy (tau)
    #Sigma_Hartree_2nd = np.ndarray((2*Ntau+1,2,2,),dtype=float) # self-energy second-order Hartree
    
    for U in np.arange(3.0,3.1,2.0):
        for beta in np.arange(30.0,30.1,1.0):
            print("U: ", U, " and beta: ", beta)
            it=0 # DMFT iteration initialization
            is_converged = False
            h=0.1 # initial spin splitting. Relevant for AFM case.
            delta_tau = beta/(2*Ntau)
            tau_array = np.empty((2*Ntau+1,),dtype=float)
            iwn_array = np.array([1.0j*(2.0*n+1.0)*np.pi/beta for n in range(-Ntau,Ntau)],dtype=complex)
            for l in range(2*Ntau+1):
                tau_array[l] = l*delta_tau
            # Initial Hyb
            for i,iwn in enumerate(iwn_array):
                Hyb1[i,0,0] = hyb_c/iwn # up
                Hyb1[i,1,1] = hyb_c/iwn # down
            # Initial Weiss Green's function
            for i,iwn in enumerate(iwn_array):
                G01[i,0,0] = 1.0/( iwn + mu - h - Hyb1[i,0,0] )
                G01[i,1,1] = 1.0/( iwn + mu + h - Hyb1[i,1,1] )
            Weiss_tmp = deepcopy(G01)
            G0 = deepcopy(G01) # Weiss Green's function (iwn)
            Hyb = deepcopy(Hyb1) # hybridisation function (iwn)
            while not is_converged and it<MAX_ITER:
                if it>1:
                    h=0.0
                print("\n"+"*"*25+" iter: ", it,"*"*25)
                print("beta: ", beta, " U: ", U, " mu: ", mu, " h: ", h, " hyb_c: ", hyb_c)
                # AFM case
                # subtracting the leadind tail
                for j,iwn in enumerate(iwn_array):
                    G0[j,0,0] -= 1.0/( iwn )
                    G0[j,1,1] -= 1.0/( iwn )
                G0_tau[:,0,0] = get_iwn_to_tau(G0[:,0,0],beta) #up
                G0_m_tau[:,0,0] = get_iwn_to_tau(G0[:,0,0],beta,type_of="negative") #up
                G0_tau[:,1,1] = get_iwn_to_tau(G0[:,1,1],beta) #down
                G0_m_tau[:,1,1] = get_iwn_to_tau(G0[:,1,1],beta,type_of="negative") #down
                for i in range(len(tau_array)):
                    G0_tau[i,0,0] += -0.5 #up
                    G0_tau[i,1,1] += -0.5 #down
                    G0_m_tau[i,0,0] += 0.5 #up
                    G0_m_tau[i,1,1] += 0.5 #down
                for j,iwn in enumerate(iwn_array):
                    G0[j,0,0] += 1.0/( iwn )
                    G0[j,1,1] += 1.0/( iwn )
                # Weiss Green's function densities used for Hartree term
                n0_up = -1.0*G0_tau[-1,0,0]
                n0_down = -1.0*G0_tau[-1,1,1]
                print("n0_up: ", n0_up, " and n0_down: ", n0_down)
                # Computing the iwn SE for up and down spins
                if type_spline=="cubic":
                    Sigma_up_iwn, Sigma_down_iwn = cs.compute_Sigma_iwn_cubic_spline(G0,G0_tau,G0_m_tau,delta_tau,U,h,hyb_c,mu,beta,tau_array,iwn_array,SE_tau)
                elif type_spline=="linear":
                    for i in range(SE_tau.shape[0]):
                        SE_tau[i,0,0] = -1.0*U*U*G0_tau[i,0,0]*G0_m_tau[i,1,1]*G0_tau[i,1,1] #up
                        SE_tau[i,1,1] = -1.0*U*U*G0_tau[i,1,1]*G0_m_tau[i,0,0]*G0_tau[i,0,0] #down
                    Sigma_up_iwn = ls.linear_spline_Sigma_tau_to_iwn(SE_tau[:,0,0],beta)
                    Sigma_down_iwn = ls.linear_spline_Sigma_tau_to_iwn(SE_tau[:,1,1],beta)
                # 2nd order Hartree term
                # Convolution
                Sigma_Hartree_2nd_up = 0.0; Sigma_Hartree_2nd_down = 0.0
                for i in range(2*Ntau+1):
                    Sigma_Hartree_2nd_up += -1.0*delta_tau*G0_tau[i,1,1]*G0_tau[2*Ntau-i,1,1]#delta_tau*G0_tau[j-i,1,1]*G0_tau[2*Ntau-(j-i),1,1]
                    Sigma_Hartree_2nd_down += -1.0*delta_tau*G0_tau[i,0,0]*G0_tau[2*Ntau-i,0,0]#delta_tau*G0_tau[j-i,0,0]*G0_tau[2*Ntau-(j-i),0,0]
                Sigma_Hartree_2nd_up *= U*U*(n0_up-0.5)
                Sigma_Hartree_2nd_down *= U*U*(n0_down-0.5)
                SE[:,0,0] = U*(n0_down-0.5) + Sigma_up_iwn + Sigma_Hartree_2nd_up #up
                SE[:,1,1] = U*(n0_up-0.5) + Sigma_down_iwn + Sigma_Hartree_2nd_down #down
                ########################## plotting #######################################
                # plt.figure(0)
                # plt.title(r"$\operatorname{Im}\Sigma^{(2)}_{\sigma} (i\omega_n)$ vs $i\omega_n$")
                # plt.plot(list(map(lambda x: x.imag,iwn_array)),list(map(lambda x: x.imag,Sigma_up_iwn)),marker='o',ms=2.5,c='red',label=r"$\sigma=\uparrow$")
                # plt.plot(list(map(lambda x: x.imag,iwn_array)),list(map(lambda x: x.imag,Sigma_down_iwn)),marker='v',ms=2.5,c='green',label=r"$\sigma=\downarrow$")
                # plt.xlabel(r"$i\omega_n$")
                # plt.legend()
                # plt.figure(1)
                # plt.title(r"$\Sigma^{(2)}_{\sigma} (\tau)$ vs $\tau$")
                # plt.plot(tau_array,SE_tau[:,0,0],marker='d',ms=2.5,c='red',label=r"$\sigma=\uparrow$")
                # plt.plot(tau_array,SE_tau[:,1,1],marker='s',ms=2.5,c='green',label=r"$\sigma=\downarrow$")
                # plt.xlabel(r"$\tau$")
                # plt.legend()
                # plt.figure(2)
                # plt.title(r"$\operatorname{Im}\Sigma^{tot}_{\sigma} (i\omega_n)$ vs $i\omega_n$")
                # plt.plot(list(map(lambda x: x.imag,iwn_array)),list(map(lambda x: x.imag,SE[:,0,0])),marker='d',ms=2.5,c='red',label=r"$\sigma=\uparrow$")
                # plt.plot(list(map(lambda x: x.imag,iwn_array)),list(map(lambda x: x.imag,SE[:,1,1])),marker='s',ms=2.5,c='green',label=r"$\sigma=\downarrow$")
                # plt.xlabel(r"$i\omega_n$")
                # plt.legend()
                # plt.show()
                ########################### plotting ######################################
                # Computing the local densities
                for i,iwn in enumerate(iwn_array):
                    Gloc[i,0,0] = 1.0/( iwn + mu - h - Hyb[i,0,0] - SE[i,0,0] ) - 1.0/( iwn ) # up
                    Gloc[i,1,1] = 1.0/( iwn + mu + h - Hyb[i,1,1] - SE[i,1,1] ) - 1.0/( iwn ) # down
                Gloc_tau[:,0,0] = get_iwn_to_tau(Gloc[:,0,0],beta)
                Gloc_tau[:,1,1] = get_iwn_to_tau(Gloc[:,1,1],beta)
                for i in range(len(tau_array)):
                    Gloc_tau[i,0,0] += -0.5
                    Gloc_tau[i,1,1] += -0.5
                n_up = -1.0*Gloc_tau[-1,0,0]
                n_down = -1.0*Gloc_tau[-1,1,1]
                print("impurity n_up: ", n_up, " and n_down: ", n_down)
                for i,iwn in enumerate(iwn_array):
                    Gloc[i,0,0] += 1.0/( iwn ) # up
                    Gloc[i,1,1] += 1.0/( iwn ) # down
                # DMFT
                # Not as stable if using quad to integrate. Better to use simps, even though much longer
                G_alpha_beta = np.ndarray((2,2,),dtype=complex)
                G_alpha_beta_inv = np.ndarray((2*Ntau,2,2,),dtype=complex)
                GAA_latt_up_k = np.empty((Nk,),dtype=complex)
                GAA_latt_down_k = np.empty((Nk,),dtype=complex)
                GAB_latt_up_k = np.empty((Nk,),dtype=complex)
                #GAB_latt_down_k = np.empty((Nk,),dtype=complex)
                for i,iwn in enumerate(iwn_array):
                    if dim==1:
                        for j,k in enumerate(k_array):
                            GAA_latt_up_k[j] = 1.0/( iwn + mu - h - SE[i,0,0] - epsilonk(k)**2/( iwn + mu + h - SE[i,1,1] ) )
                            GAA_latt_down_k[j] = 1.0/( iwn + mu + h - SE[i,1,1] - epsilonk(k)**2/( iwn + mu - h - SE[i,0,0] ) )
                            GAB_latt_up_k[j] = epsilonk(k)/( ( iwn + mu - h - SE[i,0,0] )*( iwn + mu + h - SE[i,1,1] ) - epsilonk(k)**2 )
                            #GAB_latt_down_k[j] = epsilonk(k)/( ( iwn + mu - U*(n0_up) - SE_iwn_down[i] )*( iwn + mu - U*(1.0-n0_up) + np.conj(SE_iwn_down[i]) ) - epsilonk(k)**2 )
                        GAA_up_loc = 1.0/(1.0*np.pi)*np.trapz(GAA_latt_up_k,k_array) #up
                        GAA_down_loc = 1.0/(1.0*np.pi)*np.trapz(GAA_latt_down_k,k_array) #down
                        GAB_up_loc = 1.0/(1.0*np.pi)*np.trapz(GAB_latt_up_k,k_array) # off-diagonal terms
                    elif dim==2:
                        for j,kx in enumerate(k_array):
                            GAA_latt_up_k_tmp = np.empty((Nk,),dtype=complex)
                            GAA_latt_down_k_tmp = np.empty((Nk,),dtype=complex)
                            GAB_latt_up_k_tmp = np.empty((Nk,),dtype=complex)
                            for l,ky in enumerate(k_array):
                                GAA_latt_up_k_tmp[l] = 1.0/( iwn + mu - h - SE[i,0,0] - epsilonk_2D(kx,ky)**2/( iwn + mu + h - SE[i,1,1] ) )
                                GAA_latt_down_k_tmp[l] = 1.0/( iwn + mu + h - SE[i,1,1] - epsilonk_2D(kx,ky)**2/( iwn + mu - h - SE[i,0,0] ) )
                                GAB_latt_up_k_tmp[l] = epsilonk_2D(kx,ky)/( ( iwn + mu - h - SE[i,0,0] )*( iwn + mu + h - SE[i,1,1] ) - epsilonk_2D(kx,ky)**2 )
                            
                            GAA_latt_up_k[j] = np.trapz(GAA_latt_up_k_tmp,k_array)
                            GAA_latt_down_k[j] = np.trapz(GAA_latt_down_k_tmp,k_array)
                            GAB_latt_up_k[j] = np.trapz(GAB_latt_up_k_tmp,k_array)
                        
                        GAA_up_loc = 1.0/(np.pi**2)*np.trapz(GAA_latt_up_k,k_array) #up
                        GAA_down_loc = 1.0/(np.pi**2)*np.trapz(GAA_latt_down_k,k_array) #down
                        GAB_up_loc = 1.0/(np.pi**2)*np.trapz(GAB_latt_up_k,k_array) # off-diagonal terms
                    # solving for inverse Gloc up and down
                    G_alpha_beta[0,0]=GAA_up_loc; G_alpha_beta[0,1]=GAB_up_loc
                    G_alpha_beta[1,0]=GAB_up_loc; G_alpha_beta[1,1]=GAA_down_loc
                    G_alpha_beta_inv[i,:,:] = np.linalg.inv(G_alpha_beta)
            
                # update the hybdridisation function and set Weiss Green's function for the next iteration
                alpha = 0.1
                for i,iwn in enumerate(iwn_array):
                    Hyb[i,0,0] = (1.0-alpha)*(iwn + mu - h - SE[i,0,0] - G_alpha_beta_inv[i,0,0]) + alpha*(Hyb[i,0,0]) # up
                    G0[i,0,0] = 1.0/( iwn + mu - h - Hyb[i,0,0] ) # up Have to search for mu away from half-filling
                    Hyb[i,1,1] = (1.0-alpha)*(iwn + mu + h - SE[i,1,1] - G_alpha_beta_inv[i,1,1]) + alpha*(Hyb[i,1,1]) # down
                    G0[i,1,1] = 1.0/( iwn + mu + h - Hyb[i,1,1] ) # down
                
                # Saving files at each iteration
                directory_container = "{0:d}D_U_{1:.5f}_beta_{2:.5f}_n_{3:.5f}_Ntau_{4:d}".format(dim,U,beta,0.5,Ntau)
                if not os.path.isdir("./data_test_IPT/"+directory_container):
                    os.mkdir("./data_test_IPT/"+directory_container)

                filename = "Green_loc_{0}D".format(dim)+"_AFM_"+type_spline+"_U_{0:.5f}".format(U)+"_beta_{0:.5f}".format(beta)+"_N_tau_{0:d}".format(Ntau)+"_h_{0:.5f}".format(h)+"_Nit_{0:d}".format(it)+".dat"
                with open(filename,"w+") as floc:
                    for i in range(len(iwn_array)):
                        if i==0:
                            floc.write("iwn\t\tRe Gloc up\t\tIm Gloc up\t\tRe Gloc down\t\tIm Gloc down\n")
                        floc.write("{0:.8f}\t\t{1:.8f}\t\t{2:.8f}\t\t{3:.8f}\t\t{4:.8f}\n".format(iwn_array[i].imag,Gloc[i,0,0].real,Gloc[i,0,0].imag,Gloc[i,1,1].real,Gloc[i,1,1].imag))
                floc.close()

                filename = "./data_test_IPT/"+directory_container+"/Green_Weiss_tau_{0}D".format(dim)+"_AFM_"+type_spline+"_U_{0:.5f}".format(U)+"_beta_{0:.5f}".format(beta)+"_N_tau_{0:d}".format(Ntau)+"_h_{0:.5f}".format(h)+"_Nit_{0:d}".format(it)+".dat"
                with open(filename,"w+") as floc_tau:
                    for i in range(len(tau_array)):
                        if i==0:
                            floc_tau.write("tau\t\tGloc up\t\tGloc down\n")
                        floc_tau.write("{0:.8f}\t\t{1:.8f}\t\t{2:.8f}\n".format(tau_array[i],G0_tau[i,0,0],G0_tau[i,1,1]))
                floc_tau.close()

                # filename = "data_test_IPT/Self_energy_{0}D".format(dim)+"_AFM_"+type_spline+"_U_{0:.5f}".format(U)+"_beta_{0:.5f}".format(beta)+"_N_tau_{0:d}".format(Ntau)+"_h_{0:.5f}".format(h)+"_Nit_{0:d}".format(it)+".dat"
                # with open(filename,"w+") as fse:
                #     for i in range(len(iwn_array)):
                #         if i==0:
                #             fse.write("iwn\t\tRe SE up\t\tIm SE up\t\tRe SE down\t\tIm SE down\n")
                #         fse.write("{0:.8f}\t\t{1:.8f}\t\t{2:.8f}\t\t{3:.8f}\t\t{4:.8f}\n".format(iwn_array[i].imag,SE[i,0,0].real,SE[i,0,0].imag,SE[i,1,1].real,SE[i,1,1].imag))
                # fse.close()
                
                if it>1:
                    G_0_up_diff = 0.0; G_0_down_diff = 0.0
                    for l in range(G0.shape[0]):
                        G_0_up_diff += np.abs(Weiss_tmp[l,0,0]-G0[l,0,0])
                        G_0_down_diff += np.abs(Weiss_tmp[l,1,1]-G0[l,1,1])
                    print("G0_diff up: ", G_0_up_diff," and G0_diff down: ",G_0_down_diff)
                    if G_0_up_diff<MIN_TOL and G_0_down_diff<MIN_TOL:
                        is_converged=True
                Weiss_tmp = deepcopy(G0)
                it+=1