import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps


def disp_rel(k : float):
    return -2.0*np.cos(k)

def fermi_dist(beta : float, k : float):
    nk = 1.0/(1.0 + np.exp(beta*disp_rel(k)))
    return nk

def Green_funct_taum(beta : float, k : float, tau : float):
    return np.exp(disp_rel(k)*tau)*fermi_dist(beta,k)

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
            bubble_iqn_for_q[i] = ( 2.0/(beta_step*(iqn**2)*beta) ) * (np.cos(iqn.imag*beta_step)-1.0)*fft_array[i]

    # Need to mirror due to negative Matsubara frequency component
    bubble_iqn_for_q = np.concatenate( ( np.append(bubble_iqn_for_q[NN+1:], bubble_iqn_for_q[NN]), bubble_iqn_for_q[:NN] ) ,axis=0 )
    return bubble_iqn_for_q, iqn_array


if __name__ == "__main__":
    beta = 30.0
    N_k = 512
    N_tau = 128 ## Should be even
    # Saving imaginary-time array of the bubble
    beta_array, beta_step = np.linspace(0,beta,N_tau,retstep=True)
    k_array, k_step = np.linspace(0,2.0*np.pi,N_k,retstep=True)
    k_array+=3.0*k_step

    bubble_non_interacting=np.empty((N_tau,N_k),dtype=complex)
    # Bubble grid lying in (tau,r)-space
    q=0.2
    for j,k in enumerate(k_array):
        if j==0 or (j==len(k_array)-1):
            for i in range(N_tau):
                bubble_non_interacting[i,j] = 0.5*(-2.0)*(-1.0)*Green_funct_taum(beta,k,beta_array[i])*Green_funct_taum(beta,k+q,beta_array[-1]-beta_array[i])
        else:
            for i in range(N_tau):
                bubble_non_interacting[i,j] = (-2.0)*(-1.0)*Green_funct_taum(beta,k,beta_array[i])*Green_funct_taum(beta,k+q,beta_array[-1]-beta_array[i])

    bubble_non_interacting_k_summed=np.empty((N_tau,),dtype=complex)
    for i,tau in enumerate(beta_array):
        # if i==1:
        #     print("bubble: ", bubble_non_interacting[i,:])    
        k_sum = 1.0/(N_k-1.0)*np.sum(bubble_non_interacting[i,:])
        if i==1:
            print("k_sum: ", k_sum)
        bubble_non_interacting_k_summed[i] = k_sum
    
    del bubble_non_interacting
        # k_Green_funct_taum = [Green_funct_taum(beta,k,tau) for k in k_array]
        # k_summed_Green = 1.0/(2.0*np.pi)*simps(k_Green_funct_taum,k_array)
        # bubble_non_interacting[i] = k_summed_Green
    
    plt.figure(0)
    plt.plot(beta_array,bubble_non_interacting_k_summed,marker='*') #bubble in (tau,r) space
    #print("bubble_non_interacting before: ", bubble_non_interacting_k_summed)
    bubble_non_interacting_k_summed = np.delete(bubble_non_interacting_k_summed,(N_tau-1),axis=0) # deleting the last row corresponding to tau=beta
    #print("bubble_non_interacting after: ", bubble_non_interacting_k_summed)
    iqn_Green_funct_q, iqn_array = Fourier_transform_iqn(bubble_non_interacting_k_summed,beta_step,beta)
    bubble_non_interacting_k_summed = iqn_Green_funct_q

    qn_array = list(map(lambda x: x.imag,iqn_array))
    bubble_non_interacting_q_0_real = np.asanyarray(list(map(lambda x: x.real,bubble_non_interacting_k_summed[:])))
    bubble_non_interacting_q_0_imag = np.asanyarray(list(map(lambda x: x.imag,bubble_non_interacting_k_summed[:])))


    plt.figure(1)
    plt.plot(qn_array,bubble_non_interacting_q_0_real,marker='*')
    plt.show()

    with open("bubble_iqn_1.dat","w") as f:
        for i,qn in enumerate(qn_array):
            if i==0:
                f.write("/\n")
            f.write("%f\t\t%f\t\t%f\n" % (qn,bubble_non_interacting_q_0_real[i],bubble_non_interacting_q_0_imag[i]))
    f.close()
