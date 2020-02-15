import numpy as np

def linear_spline_tau_to_iwn(G_tau, beta : float):
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

def linear_spline_tau_to_iqn_corr(corr_funct, beta : float):
    """The correlation function (corr_funct) is a function defined over tau-space. Since the number of fermionic Matsubara frequencies (MF) 
    is even (for mirroring during FFTs, because negative MFs are used), the number of imaginary-time points has gotten to be odd.
    """
    M = len(corr_funct)
    assert np.mod(M,2)==1, "The imaginary-time length has got to be odd. (iwn has to be even for mirroring reasons.)"
    *_, delta_tau = np.linspace(0,beta,M,retstep=True)
    K_iqn = np.empty(M-1,dtype=complex)
    iqn_array = np.array([1.0j*(2.0*n)*np.pi/beta for n in range(M-1)],dtype=complex)
    # Fourier transforming
    IFFT_data = np.fft.ifft(corr_funct)

    for i,iqn in enumerate(iqn_array):
        if i==0:
            K_iqn[i] = delta_tau*np.sum(corr_funct)
        else:
            K_iqn[i] = 2.0/( delta_tau * ((iqn)**2) ) * ( np.cos(iqn.imag*delta_tau) - 1.0 ) * M * IFFT_data[i]

    return K_iqn, iqn_array