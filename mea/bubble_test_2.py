import numpy as np
import matplotlib.pyplot as plt

def fermi_dist(beta : float, energy : float):
    nk = 1.0/(1.0 + np.exp(beta*energy))
    return nk

def Green_funct_taum(beta : float, energy : float, tau : float):
    return 0.5*np.exp(energy*tau)*fermi_dist(beta,energy) + 0.5*np.exp(-energy*tau)*(1.0-fermi_dist(beta,energy))

def Fourier_transform_iqn(list_q_vs_taum, beta_step : float, beta : float):
    # The elements of the bubble tensor have already been transformed to q-space
    M=len(list_q_vs_taum)
    assert np.mod(M,2)==1, "The imaginary-time array length has to be odd."
    bubble_iqn_for_q = np.empty(M,dtype=complex)
    NN=M//2
    iqn_array = np.array([1.0j*(2.*n)*np.pi/beta for n in range(-NN,NN+1)],dtype=complex)
    assert M==len(iqn_array), "The number of bosonic Matsubara frequencies has to be N_tau-1 to exclude tau=beta."
    fft_array = np.fft.fft(list_q_vs_taum) ## Have to exclude beta
    for i,iqn in enumerate(iqn_array):
        if iqn.imag==0.0:
            bubble_iqn_for_q[i] = beta_step*np.sum(list_q_vs_taum)
        else:
            bubble_iqn_for_q[i] = ( 2.0/(beta_step*(iqn**2)*beta) ) * (np.cos(iqn.imag*beta_step)-1)*fft_array[i]

    bubble_iqn_for_q = np.concatenate( ( np.append(bubble_iqn_for_q[NN+1:], bubble_iqn_for_q[NN]), bubble_iqn_for_q[:NN] ) ,axis=0 )
    return bubble_iqn_for_q, iqn_array


if __name__ == "__main__":
    beta = 50.0
    N_tau = 128 ## Should be even
    epsilon = 0.4
    # Saving imaginary-time array of the bubble
    beta_array, beta_step = np.linspace(0,beta,N_tau,retstep=True)

    bubble_non_interacting=np.empty((N_tau,),dtype=complex)
    Green_funct_taum = np.array([Green_funct_taum(beta,epsilon,tau) for tau in beta_array],dtype=float)
    plt.figure(0)
    plt.plot(beta_array,Green_funct_taum)
    # Bubble grid lying in (tau,r)-space
    for i in range(N_tau):
        bubble_non_interacting[i] = (-1.0)*2.0*(-1.0)*Green_funct_taum[i]*Green_funct_taum[-1-i]
    plt.figure(1)
    plt.title(r"$\chi^0(\tau)$ for "+r"$\epsilon={0}, \beta={1}$".format(epsilon,beta))
    plt.plot(beta_array,bubble_non_interacting,c="red",marker='.',markersize=2.0)
    plt.ylabel(r"$\chi^0(\tau)$")
    plt.xlabel(r"$\tau$")

    bubble_non_interacting = np.delete(bubble_non_interacting,(N_tau-1),axis=0) # deleting the last row corresponding to tau=beta
    
    bubble_non_interacting, iqn_array = Fourier_transform_iqn(bubble_non_interacting,beta_step,beta)
    
    bubble_non_interacting_real = np.asarray(list(map(lambda x: x.real, bubble_non_interacting)))
    qn_array = np.asanyarray(list(map(lambda x: x.imag, iqn_array)))

    with open("test_iqn_bubble_2.dat","w") as f:
        for i in range(len(bubble_non_interacting)):
            if i==0:
                f.write("/\n")
            f.write("{0.imag}\t\t{1.real}\t\t{1.imag}\n".format(iqn_array[i],bubble_non_interacting[i]))

    f.close()
    plt.figure(2)
    plt.title(r"$\operatorname{Re}\chi^0(iq_n)$ for" + r" $\epsilon={0}, \beta={1}$, ".format(epsilon,beta) + r"$N_{\tau}$=" + r"${0}$".format(N_tau))
    plt.plot(qn_array,bubble_non_interacting_real,c="green",marker='*',markersize=2.0)
    plt.xlabel(r"Bosonic frequency $q_n$")
    plt.ylabel(r"$\chi^0(iq_n)$")
    plt.show()
    print("bubble iqn: ", bubble_non_interacting)