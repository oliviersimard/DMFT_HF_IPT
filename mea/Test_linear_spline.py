import numpy as np
import matplotlib.pyplot as plt


def partial_DFT(input):
    """
    The input consists of the tau-defined self-energy. The tau function has one element less than iwn output.
    """
    MM = len(input)
    output = np.empty((MM,),dtype=complex)
    for n in range(MM):  # For each output element
        s = complex(0)
        for l in range(1,MM):  # For each input element, leaving out the first element in tau-defined array object
            angle = 2.0j * np.pi * l * n / (MM)
            s += input[l] * np.exp(angle)
        output[n] = s
    
    return output


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
             + 2.0/( delta_tau * ((iwn)**2) ) * ( np.cos(iwn.imag*delta_tau) - 1.0 ) * (M-1) * IFFT_data[i]

    return G_iwn

def linear_spline_Sigma_tau_to_iwn(SE_tau, beta : float):
    """
    Function computing the fermionic Matsubara frequency representation of inputted imaginary-time object using a linear spline.

    Parameters:
        SE_tau (array): imaginary-time defined self-energy to be transformed
        beta (float): inverse temperature

    Returns:
        SE_iwn (array): fermionic Matsubara frequency representation of the inputted imaginary-time self-energy
    """
    M = len(SE_tau)
    NN = M//2
    assert np.mod(M,2)==1, "The imaginary-time length has got to be odd. (iwn has to be even for mirroring reasons.)"
    *_,delta_tau = np.linspace(0,beta,M,retstep=True)
    SE_0 = SE_tau[0]; SE_beta = SE_tau[-1]
    # beta value is discarded, because not needed
    S_p = np.empty(M-1,dtype=complex)
    SE_iwn = np.empty(M-1,dtype=complex)
    iwn_array = np.array([1.j*(2.0*n+1.0)*np.pi/beta for n in range(-NN,NN)],dtype=complex)
    # Filling up the kernel array that will enter FFT
    for ip in range(M-1):
        S_p[ip] =  np.exp( 1.j * np.pi * ip / (M-1) ) * SE_tau[ip]
    
    # Fourier transforming
    IFFT_data_m = partial_DFT(S_p)
    IFFT_data = np.fft.ifft(S_p)
    ## Mirroring, because using negative Matsubara frequencies to start with
    IFFT_data_m = np.concatenate((IFFT_data_m[NN:],IFFT_data_m[:NN]))
    IFFT_data = np.concatenate((IFFT_data[NN:],IFFT_data[:NN]))

    for i,iwn in enumerate(iwn_array):
        SE_iwn[i] = ( -SE_beta - SE_0 )/iwn - (SE_beta*(np.exp(-iwn*delta_tau)-1.0))/( delta_tau * ((iwn)**2) ) \
        + (-1.0 + np.exp(-iwn*delta_tau))/( ((iwn)**2)*delta_tau ) * IFFT_data_m[i] \
        + (np.exp(iwn*delta_tau)-1.0)/( delta_tau * ((iwn)**2) ) * (M-1) * IFFT_data[i]

    return SE_iwn



if __name__=="__main__":
    N_tau = 2000
    beta = 50
    hyb_c = 2
    mu0 = 0.0
    U=2.0
    mu=U/2.0
    
    iwn_array = np.array([1.j*(2*n+1)*np.pi/beta for n in range(-N_tau,N_tau)],dtype=complex)
    tau_array = np.array([l*beta/(2*N_tau) for l in range(2*N_tau+1)],dtype=float)
    wn = np.array(list(map(lambda x: x.imag,iwn_array)))
    G_test_up = np.empty((2*N_tau,),dtype=complex)
    G_test_down = np.empty((2*N_tau,),dtype=complex)
    Sigma_up_tau = np.empty((2*N_tau+1,),dtype=complex)
    for i,iwn in enumerate(iwn_array):
        G_test_up[i] = 1.0/( iwn + mu + mu0 - hyb_c/iwn ) - 1.0/iwn
        G_test_down[i] = 1.0/( iwn + mu - mu0 - hyb_c/iwn ) - 1.0/iwn

    G_test_up_tau = get_iwn_to_tau(G_test_up,beta)
    G_test_down_tau = get_iwn_to_tau(G_test_down,beta)
    G_test_down_m_tau = get_iwn_to_tau(G_test_down,beta,type_of="negative")

    for i in range(len(tau_array)):
        G_test_up_tau[i] += -0.5
        G_test_down_tau[i] += -0.5
        G_test_down_m_tau[i] += 0.5
        Sigma_up_tau[i] = -1.0*U*U*G_test_up_tau[i]*G_test_down_m_tau[i]*G_test_down_tau[i]
    plt.figure(3)
    plt.plot(tau_array,G_test_up_tau,marker='P',ms=2.5,c='red')
    plt.plot(tau_array,G_test_down_tau,marker='d',ms=2.5,c='grey')
    plt.plot(tau_array,G_test_down_m_tau,marker='x',ms=2.5,c='violet')
    
    for i,iwn in enumerate(iwn_array):
        G_test_up[i] += 1.0/iwn
        G_test_down[i] += 1.0/iwn

    G_test_up_back = linear_spline_tau_to_iwn(G_test_up_tau,beta)
    G_test_down_back = linear_spline_tau_to_iwn(G_test_down_tau,beta)
    Sigma_up_iwn = linear_spline_Sigma_tau_to_iwn(Sigma_up_tau,beta)

    plt.figure(2)
    plt.plot(wn,list(map(lambda x: x.imag,Sigma_up_iwn)),marker='s',ms=2.5,c="cyan")

    lol=[]
    for i in range(len(iwn_array)):
        lol.append(Sigma_up_iwn[i]*iwn_array[i])
    plt.figure(5)
    plt.plot(wn,list(map(lambda x: x.imag, lol)),marker='o',ms=2.5)

    for i,iwn in enumerate(iwn_array):
        Sigma_up_iwn[i] -= (U**2/4.0)/iwn
        Sigma_up_iwn[i] *= iwn
    
    print(Sigma_up_iwn)

    Sigma_up_back_tau = get_iwn_to_tau(Sigma_up_iwn,beta)
    for i in range(len(tau_array)):
        Sigma_up_back_tau[i] += -0.5*(U**2/4.0)
    

    G_test_up_im = np.array(list(map(lambda x: x.imag,G_test_up)))
    G_test_up_back_im = np.array(list(map(lambda x: x.imag,G_test_up_back)))
    G_test_down_im = np.array(list(map(lambda x: x.imag,G_test_down)))
    G_test_down_back_im = np.array(list(map(lambda x: x.imag,G_test_down_back)))
    plt.figure(0)
    plt.plot(wn,G_test_up_im,marker='v',ms=2.5,c="red",label="up before")
    plt.plot(wn,G_test_up_back_im,marker='v',ms=2.5,c="green",label="up after")
    plt.plot(wn,G_test_down_im,marker='v',ms=2.5,c="black",label="down before")
    plt.plot(wn,G_test_down_back_im,marker='v',ms=2.5,c="blue",label="down after")
    plt.legend()
    plt.figure(1)
    plt.plot(tau_array,Sigma_up_tau,marker='o',ms=2.5,c="blue")
    plt.plot(tau_array,Sigma_up_back_tau,marker='>',ms=2.5,c='black')
    plt.show()
    