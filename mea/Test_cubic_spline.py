import numpy as np
import matplotlib.pyplot as plt
import cubicspline as cs


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


if __name__=="__main__":
    N_tau = 6000
    beta = 5
    hyb_c = 2
    mu0 = 0.3
    U=2.0
    mu=U/2.0
    delta_tau = beta/(2.0*N_tau)
    
    iwn_array = np.array([1.j*(2*n+1)*np.pi/beta for n in range(-N_tau,N_tau)],dtype=complex)
    tau_array = np.array([l*beta/(2*N_tau) for l in range(2*N_tau+1)],dtype=float)
    wn = np.array(list(map(lambda x: x.imag,iwn_array)))
    G_test_up = np.empty((2*N_tau,),dtype=complex)
    G_test_down = np.empty((2*N_tau,),dtype=complex)
    Sigma_up_tau = np.empty((2*N_tau+1,),dtype=float)
    Sigma_down_tau = np.empty((2*N_tau+1,),dtype=float)
    for i,iwn in enumerate(iwn_array):
        G_test_up[i] = 1.0/( iwn + mu + mu0 - hyb_c/iwn ) - 1.0/iwn
        G_test_down[i] = 1.0/( iwn + mu - mu0 - hyb_c/iwn ) - 1.0/iwn

    G_test_up_tau = get_iwn_to_tau(G_test_up,beta)
    G_test_down_tau = get_iwn_to_tau(G_test_down,beta)
    G_test_down_m_tau = get_iwn_to_tau(G_test_down,beta,type_of="negative")
    G_test_up_m_tau = get_iwn_to_tau(G_test_up,beta,type_of="negative")

    for i in range(len(tau_array)):
        G_test_up_tau[i] += -0.5
        G_test_down_tau[i] += -0.5
        G_test_down_m_tau[i] += 0.5
        G_test_up_m_tau[i] += 0.5
        Sigma_up_tau[i] = -1.0*U*U*G_test_up_tau[i]*G_test_down_m_tau[i]*G_test_down_tau[i]
        Sigma_down_tau[i] = -1.0*U*U*G_test_down_tau[i]*G_test_up_m_tau[i]*G_test_up_tau[i]
    plt.figure(0)
    plt.plot(tau_array,G_test_up_tau,marker='P',ms=2.5,c='red')
    plt.plot(tau_array,G_test_down_tau,marker='d',ms=2.5,c='grey')
    plt.plot(tau_array,G_test_down_m_tau,marker='x',ms=2.5,c='violet')
    
    for i,iwn in enumerate(iwn_array):
        G_test_up[i] += 1.0/iwn
        G_test_down[i] += 1.0/iwn
    
    cs.Cubic_spline(delta_tau,G_test_up_tau)
    der_G = cs.Cubic_spline.get_derivative_4th_order(tau_array,G_test_up_tau)
    cs.Cubic_spline.building_matrix_components(der_G[0],der_G[-1])
    cs.Cubic_spline.tridiagonal_LU_decomposition()
    cs.Cubic_spline.tridiagonal_LU_solve()
    # Getting m_a and m_c from the b's
    cs.Cubic_spline.construct_coeffs(tau_array)

    rand_arr = np.random.rand(2000)*beta
    interplot = []
    for el in rand_arr:
        interplot.append(cs.Cubic_spline.get_spline(tau_array,el))

    plt.figure(1)
    plt.scatter(rand_arr,interplot,c="orange",marker='o')
    plt.plot(tau_array,G_test_up_tau,marker='d',ms=2.5,c='cyan')
    #plt.plot(wn_arr,im_G_iwn)

    # Instantiation of FermionsvsBosons
    cs.FermionsvsBosons(beta,tau_array)
    cubic_Spl_G_iwn, iwn_array_pos = cs.FermionsvsBosons.fermionic_propagator()

    wn_array_pos = np.array(list(map(lambda x: x.imag,iwn_array_pos)))
    wn_array = np.array(list(map(lambda x: x.imag,iwn_array)))
    cubic_Spl_G_iwn_real = np.array(list(map(lambda x: x.real,cubic_Spl_G_iwn)))
    cubic_Spl_G_iwn_imag = np.array(list(map(lambda x: x.imag,cubic_Spl_G_iwn)))
    plt.figure(2)
    plt.plot(wn_array_pos,cubic_Spl_G_iwn_imag,marker='>',ms=2.5,c="indigo")
    plt.plot(wn_array,list(map(lambda x: x.imag,G_test_up)),marker='o',ms=2.5,c="red")
    plt.show()

    cs.Cubic_spline.reset()
    cs.FermionsvsBosons.reset()
    ######################################################## SE spline #######################################################################

    cs.Cubic_spline(delta_tau,Sigma_up_tau)
    der_G = cs.Cubic_spline.get_derivative_6th_order(tau_array,Sigma_up_tau)
    cs.Cubic_spline.building_matrix_components(der_G[0],der_G[-1])
    cs.Cubic_spline.tridiagonal_LU_decomposition()
    cs.Cubic_spline.tridiagonal_LU_solve()
    # Getting m_a and m_c from the b's
    cs.Cubic_spline.construct_coeffs(tau_array)

    # Instantiation of FermionsvsBosons
    cs.FermionsvsBosons(beta,tau_array)
    Sigma_up_iwn, iwn_array_pos = cs.FermionsvsBosons.fermionic_propagator()
    Sigma_up_iwn_real = np.array(list(map(lambda x: x.real,Sigma_up_iwn)))
    Sigma_up_iwn_imag = np.array(list(map(lambda x: x.imag,Sigma_up_iwn)))

    cs.Cubic_spline.reset()
    cs.FermionsvsBosons.reset()
    ################################################### SE #################################################################

    cs.Cubic_spline(delta_tau,Sigma_down_tau)
    der_G = cs.Cubic_spline.get_derivative_6th_order(tau_array,Sigma_down_tau)
    cs.Cubic_spline.building_matrix_components(der_G[0],der_G[-1])
    cs.Cubic_spline.tridiagonal_LU_decomposition()
    cs.Cubic_spline.tridiagonal_LU_solve()
    # Getting m_a and m_c from the b's
    cs.Cubic_spline.construct_coeffs(tau_array)

    rand_arr = np.random.rand(2000)*beta
    interplot = []
    for el in rand_arr:
        interplot.append(cs.Cubic_spline.get_spline(tau_array,el))

    plt.figure(5)
    plt.plot(tau_array,Sigma_up_tau,marker='d',ms=2.5,c='red')
    plt.plot(tau_array,Sigma_down_tau,marker='d',ms=2.5,c='green')
    plt.scatter(rand_arr,interplot,c="grey",marker='o')

    # Instantiation of FermionsvsBosons
    cs.FermionsvsBosons(beta,tau_array)
    Sigma_down_iwn, iwn_array_pos = cs.FermionsvsBosons.fermionic_propagator()
    Sigma_down_iwn_real = np.array(list(map(lambda x: x.real,Sigma_down_iwn)))
    Sigma_down_iwn_imag = np.array(list(map(lambda x: x.imag,Sigma_down_iwn)))
    plt.figure(6)
    plt.plot(wn_array_pos,Sigma_down_iwn_imag,marker='<',ms=2.5,c="green")
    plt.plot(wn_array_pos,Sigma_up_iwn_imag,marker='>',ms=2.5,c="blue")
    plt.show()
    # for i,iwn in enumerate(iwn_array_pos):
    #     Sigma_up_iwn[i] -= (U**2/4.0)/iwn
    #     print(Sigma_up_iwn[i]*iwn)

    # # for i,iwn in enumerate(iwn_array_pos):
    # #     Sigma_up_iwn[i] -= (U**2/4.0)/iwn

    # Sigma_up_back_tau = get_iwn_to_tau(Sigma_up_iwn,beta)
    # for i in range(len(tau_array)):
    #     Sigma_up_back_tau[i] += -0.5*(U**2/4.0)
    
    # plt.figure(5)
    # plt.plot(tau_array,Sigma_up_tau,marker='o',ms=2.5,c="blue")
    # plt.plot(tau_array,Sigma_up_back_tau,marker='>',ms=2.5,c='black')
    # plt.show()
    