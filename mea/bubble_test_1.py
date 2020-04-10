import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps


def disp_rel(k : float):
    return -2.0*np.cos(k)

def velocities(k : float):
    return 2.0*np.sin(k)

def Green_funct_iwn(iwn : complex, k : float):
    return 1.0/(iwn - disp_rel(k))

if __name__ == "__main__":
    is_jj = True
    beta = 40.0
    N_k = 252
    N_tau = 355 ## Should be odd
    MM = N_tau//2 + 1
    # Saving imaginary-time array of the bubble
    k_array, k_step = np.linspace(0,2.0*np.pi,N_k,retstep=True)
    k_array+=2.0*k_step

    iqn_array = np.array([1.0j*(2.0*n)*np.pi/beta for n in range(-MM+1,MM)],dtype=np.clongdouble)
    iwn_array = np.array([1.0j*(2.0*n+1.0)*np.pi/beta for n in range(-MM,MM)],dtype=np.clongdouble)
    # Bubble grid lying in (tau,r)-space
    q=0.0001
    bubble_non_interacting_iqn_tot_arr = np.empty((N_tau,),dtype=np.clongdouble)
    for i,iqn in enumerate(iqn_array):
        print("iqn: ", iqn)
        bubble_non_interacting_kk=0.0
        for j,k in enumerate(k_array):
            bubble_non_interacting_iwn=0.0
            if not is_jj:
                if j==0 or (j==len(k_array)-1):
                    for iwn in iwn_array:
                        bubble_non_interacting_iwn += 0.5*(-2.0)*(-1.0)*Green_funct_iwn(iwn,k)*Green_funct_iwn(iwn+iqn,k+q)
                else:
                    for iwn in iwn_array:
                        bubble_non_interacting_iwn += (-2.0)*(-1.0)*Green_funct_iwn(iwn,k)*Green_funct_iwn(iwn+iqn,k+q)
            else:
                if j==0 or (j==len(k_array)-1):
                    for iwn in iwn_array:
                        bubble_non_interacting_iwn += 0.5*(-2.0)*(-1.0)*Green_funct_iwn(iwn,k)*Green_funct_iwn(iwn+iqn,k+q)*(velocities(k)**2)
                else:
                    for iwn in iwn_array:
                        bubble_non_interacting_iwn += (-2.0)*(-1.0)*Green_funct_iwn(iwn,k)*Green_funct_iwn(iwn+iqn,k+q)*(velocities(k)**2)
            bubble_non_interacting_kk += (1.0/beta)*bubble_non_interacting_iwn
        bubble_non_interacting_kk *= 1.0/(N_k-1.0)
        print("bubble val: ", bubble_non_interacting_kk)
        bubble_non_interacting_iqn_tot_arr[i] = bubble_non_interacting_kk

    bubble_non_interacting_iqn_tot_arr = np.concatenate( ( np.append(bubble_non_interacting_iqn_tot_arr[MM:], bubble_non_interacting_iqn_tot_arr[MM-1]), bubble_non_interacting_iqn_tot_arr[:MM-1] ) ,axis=0 )
    qn_array = list(map(lambda x: x.imag,iqn_array))
    bubble_non_interacting_q_0_real = np.asanyarray(list(map(lambda x: x.real,bubble_non_interacting_iqn_tot_arr[:])))
    bubble_non_interacting_q_0_imag = np.asanyarray(list(map(lambda x: x.imag,bubble_non_interacting_iqn_tot_arr[:])))

    plt.figure(0)
    plt.title(r"$\operatorname{Re}\chi^0(i\omega_n)$ for "+r"$\beta={0}, N_k={1}$".format(beta,N_k) + r", $N_{\omega_n}=$"+r"${0}$".format(N_tau))
    plt.plot(qn_array,bubble_non_interacting_q_0_real,marker='*')
    plt.ylabel(r"$\operatorname{Re}\chi^0(i\omega_n)$")
    plt.xlabel(r"$i\omega_n$")
    plt.figure(1)
    plt.title(r"$\operatorname{Im}\chi^0(i\omega_n)$ for "+r"$\beta={0}, N_k={1}$".format(beta,N_k) + r", $N_{\omega_n}=$"+r"${0}$".format(N_tau))
    plt.plot(qn_array,bubble_non_interacting_q_0_imag,marker='*',c="red")
    plt.ylabel(r"$\operatorname{Im}\chi^0(i\omega_n)$")
    plt.xlabel(r"$i\omega_n$")
    plt.show()

    with open("bubble_iqn_1.dat","w") as f:
        for i,qn in enumerate(qn_array):
            if i==0:
                f.write("/\n")
            f.write("%f\t\t%f\t\t%f\n" % (qn,bubble_non_interacting_q_0_real[i],bubble_non_interacting_q_0_imag[i])) # Setting im part to 0 to see...
    f.close()
