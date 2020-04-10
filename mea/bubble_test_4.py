import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import cubicspline as cs
import linearspline as ls
from math import isclose
import h5py


def disp_rel(k : float, mu : float) -> float:
    return -2.0*np.cos(k) - mu

def velocities(k : float) -> float:
    return 2.0*np.sin(k)

def Green_funct_iwn(iwn : complex, k : float, mu : float) -> complex:
    return 1.0/(iwn - disp_rel(k,mu))

def fermi_dirac(k : float, beta : float, mu : float) -> float:
    return 1.0/(1.0+np.exp(beta*disp_rel(k,mu)))

def Green_funct_tau(tau : float, k : float, beta : float, mu : float) -> float:
    return -1.0*( np.exp( disp_rel(k,mu) * (beta-tau) ) * fermi_dirac(k,beta,mu) )

def get_iwn_to_tau(G_iwn, beta : float):
    MM = len(G_iwn) # N = M/2
    tau_final_G = np.zeros(MM+1,dtype=float)
    # FFT
    tau_resolved_G = np.fft.fft(G_iwn)
    for i in range(MM):
        tau_final_G[i] = ( (1./beta)*np.exp( -1.j * np.pi * i *( 1.0/(MM) - 1.0 ) )*tau_resolved_G[i] ).real

    for i in range(MM):
        tau_final_G[MM] += ( (1./beta)*np.exp( -1.j * np.pi * (1.0-(MM)) )*G_iwn[i] ).real

    return tau_final_G

if __name__ == "__main__":
    is_jj = True
    beta = 10.0
    N_k = 351
    N_q = 51
    N_tau = 557 ## Should be odd
    MM = N_tau//2
    U = 0.0
    mu = U/2.0
    # Saving imaginary-time array of the bubble
    k_array, k_step = np.linspace(0.0,2.0*np.pi,N_k,retstep=True)
    # Incoming momentum
    q_array, q_step = np.linspace(0.0,2.0*np.pi,N_q,retstep=True)
    k_array+=1.0*k_step

    #iqn_array = np.array([1.0j*(2.0*n)*np.pi/beta for n in range(-MM+1,MM)],dtype=complex)
    iwn_array = np.array([1.0j*(2.0*n+1.0)*np.pi/beta for n in range(-MM,MM)],dtype=complex)
    beta_array, beta_step = np.linspace(0.0,beta,N_tau,retstep=True)
    # Bubble grid lying in (tau,r)-space
    G_iwn_k = np.empty((N_k,len(iwn_array)),dtype=complex)
    G_tau_k = np.empty((N_k,N_tau),dtype=float)
    G_tau_k_q = np.empty((N_k,N_tau),dtype=float)
    GG_tau_k = np.empty((N_k,N_tau),dtype=float)
    GG_tau_iqn = np.empty((N_k,len(iwn_array)),dtype=complex)
    # for i,k in enumerate(k_array):
    #     G_iwn_k[i,:] = np.array([Green_funct_iwn(iwn,k) - 1.0/iwn for iwn in iwn_array],dtype=complex)

    # ## Getting G_tau
    # for i,k in enumerate(k_array):
    #     G_tau_k[i,:] = get_iwn_to_tau(G_iwn_k[i,:],beta) - 0.5
    for l,q in enumerate(q_array):
        print("q: ", q)
        for i,k in enumerate(k_array):
            G_tau_k[i,:] = np.array([Green_funct_tau(tau,k,beta,mu) for tau in beta_array],dtype=float)
            G_tau_k_q[i,:] = np.array([Green_funct_tau(tau,k+q,beta,mu) for tau in beta_array],dtype=float)
            # plt.figure(0)
            # plt.title(r"$G(\tau)$ vs $\tau$ for $\beta={0}$ and $q={1:.4f}$".format(beta,q))
            # plt.xlabel(r"$\tau$")
            # if np.mod(i,3)==0:
            #     plt.plot(beta_array,G_tau_k[i,:])

            # plt.figure(1)
            # plt.title(r"$G(-\tau)$ vs $\tau$ for $\beta={0}$ and $q={1:.4f}$".format(beta,q))
            # plt.xlabel(r"$\tau$")
            # if np.mod(i,3)==0:
            #     plt.yscale('log')
            #     plt.plot(beta_array,-1.0*G_tau_k[i,::-1])

        for j,k in enumerate(k_array):
            if not is_jj:
                for i in range(len(beta_array)):
                    GG_tau_k[j,i] = (-1.0)*(-2.0)*G_tau_k[j,i]*G_tau_k_q[j,-1-i]
            else:
                for i in range(len(beta_array)):
                    GG_tau_k[j,i] = (-1.0)*(-2.0)*G_tau_k[j,i]*G_tau_k_q[j,-1-i]*(velocities(k)**2)
                # plt.figure(2)
                # plt.title(r"$\chi_{jj}(\tau)$ for"+r" $\beta={0}$ and $q={1:.4f}$".format(beta,q))
                # if np.mod(j,30)==0:
                #     plt.plot(beta_array,GG_tau_k[j,:],label="{0}".format(k))
                # plt.legend()

        iqn_array = []
        for j,k in enumerate(k_array):
            # cs.Cubic_spline(beta_step,GG_tau_k[j,:])
            # der_GG = cs.Cubic_spline.get_derivative(beta_array,GG_tau_k[j,:])
            # cs.Cubic_spline.building_matrix_components(der_GG[0],der_GG[-1])
            # # Next step is to do LU decomposition
            # cs.Cubic_spline.tridiagonal_LU_decomposition()
            # # And then LU solving. From the b coefficients, the a's and c's can be deduced.
            # cs.Cubic_spline.tridiagonal_LU_solve()
            # # Getting m_a and m_c from the b's
            # cs.Cubic_spline.construct_coeffs(beta_array)
            # cs.FermionsvsBosons(beta,beta_array)
            # cubic_Spl_bb_iqn, iqn_array = cs.FermionsvsBosons.bosonic_corr_funct()
            # GG_tau_iqn[j,:] = cubic_Spl_bb_iqn
            linear_spline_GG_iqn, iqn_array = ls.linear_spline_tau_to_iqn_corr(GG_tau_k[j,:],beta)
            GG_tau_iqn[j,:] = linear_spline_GG_iqn
            # if np.mod(l,10)==0:
            #     plt.figure(3)
            #     plt.title(r"$\chi_{jj}(iq_n)$ "+r"for $q={0:.4f}$".format(q))
            #     plt.plot(list(map(lambda x: x.real,linear_spline_GG_iqn)))
            ####
            # Resetting the class singletons
            # cs.Cubic_spline.reset()
            # cs.FermionsvsBosons.reset()

        bubble_GG_iqn = np.empty((len(iqn_array),),dtype=complex)
        bubble_GG_iqn_cubic = np.empty((len(iqn_array),),dtype=complex)
        for i in range(len(iqn_array)):
            bubble_GG_iqn[i] = 1.0/(2.0*np.pi) * simps(GG_tau_iqn[:,i],k_array)
            bubble_GG_iqn_cubic[i] = 1.0/(2.0*np.pi) * simps(GG_tau_iqn[:,i],k_array)

        # print("len(iwn_array): ", len(iwn_array))
        # print("iwn_array: ", iwn_array)
        qn_array = np.array(list(map(lambda x: x.imag,iqn_array)))
        bubble_GG_iqn_real = np.array(list(map(lambda x: x.real,bubble_GG_iqn)))
        bubble_GG_iqn_imag = np.array(list(map(lambda x: x.imag,bubble_GG_iqn)))

        if np.mod(l,10)==0:
            plt.figure(4)
            plt.title(r"$\operatorname{Re}\chi^0(i\omega_n)$ for "+r"$\beta={0}, N_k={1}$".format(beta,N_k) + r", $N_{\omega_n}=$"+r"${0}$".format(N_tau))
            plt.plot(qn_array,bubble_GG_iqn_real,marker='*')
            plt.ylabel(r"$\operatorname{Re}\chi^0(i\omega_n)$")
            plt.xlabel(r"$i\omega_n$")
            plt.figure(5)
            plt.title(r"$\operatorname{Im}\chi^0(i\omega_n)$ for "+r"$\beta={0}, N_k={1}$".format(beta,N_k) + r", $N_{\omega_n}=$"+r"${0}$".format(N_tau))
            plt.plot(qn_array,bubble_GG_iqn_imag,marker='*',c="red")
            plt.ylabel(r"$\operatorname{Im}\chi^0(i\omega_n)$")
            plt.xlabel(r"$i\omega_n$")
            

        with h5py.File("bb_U_{0}_beta_{1}_Ntau_{2}_Nk_{3}_Nq_{4}_isjj_{5}.hdf5".format(U,beta,N_tau,N_k,N_q,is_jj),"a") as hf:
            hf.create_dataset("q_{0:.5f}".format(q), (N_tau-1,), dtype=np.complex, data=bubble_GG_iqn)
            
        hf.close()
    
    plt.show()
