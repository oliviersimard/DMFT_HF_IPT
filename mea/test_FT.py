import matplotlib.pyplot as plt
from sys import exit
from scipy.integrate import simps
from cubicspline import np
import cubicspline as cs
import linearspline as ls

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

def disp_rel(k : float) -> float:
    return -2.0*np.cos(k)

def get_iwn_to_tau(G_iwn, beta : float):
    MM = len(G_iwn) # N = M/2
    tau_final_G = np.zeros(MM+1,dtype=float)
    # FFT
    tau_resolved_G = np.fft.fft(G_iwn)
    for i in range(MM):
        tau_final_G[i] = ( (1./beta)*np.exp( -1.j * np.pi * i *( 1.0/(MM) - 1.0 ) )*tau_resolved_G[i] ).real

    for i in range(MM):
        tau_final_G[MM] += ( (1./beta)*np.exp( -1.j * np.pi * (1.0-(MM)) )*tau_resolved_G[i] ).real

    return tau_final_G

if __name__ == "__main__":
    # Choose whether you want to load the self-energy (False) or the local Green's function (True).
    is_G_loc = True
    #is_raw_summation = False

    if is_G_loc:
        # file containing the data to be inputted as G_iwn
        loading_G_iwn = np.genfromtxt("Green_loc_1D_U_10.000000_beta_60.000000_n_0.500000_N_tau_512_Nit_34.dat",dtype=float,skip_header=1,usecols=(0,1,2))
        beta = 60.0 # Change this according to the parameters loaded
        U = 10.0 # Change this according to the parameters loaded
        wn_arr = loading_G_iwn[:,0]
        re_G_iwn = loading_G_iwn[:,1]
        im_G_iwn = loading_G_iwn[:,2]
        beta_array_G, delta_beta = np.linspace(0.0,beta,len(wn_arr)+1,retstep=True)
        #print("beta_array_G: ", beta_array_G)
        assert len(re_G_iwn)==len(im_G_iwn), "The lengths have to be the same."
        G_iwn = np.array([complex(re_G_iwn[i],im_G_iwn[i]) for i in range(len(re_G_iwn))],dtype=complex)

        # Substracting tail of G
        for i,wn in enumerate(wn_arr):
            G_iwn[i] -= 1.0/(1.j*wn) #+ ( 2.0 + (U**2)/4.0 )/( (1.j*wn)**3 )  # Only odd contributions at half-filling

        # From iwn to tau
        tau_final_G = get_iwn_to_tau(G_iwn,beta)
        for i in range(len(tau_final_G)):
            tau_final_G[i] -= 0.5 #- ( 2.0 + (U**2)/4.0 )*beta_array_G[i]*(beta-beta_array_G[i])/4.0
        
        GG_tau = np.empty((len(beta_array_G),),dtype=float)
        # Computing the imaginary-time Green's function
        for i in range(len(beta_array_G)):
            GG_tau[i] = (-2.0)*(-1.0)*tau_final_G[i]*tau_final_G[-1-i]

        K_iqn, iqn_array = ls.linear_spline_tau_to_iqn_corr(GG_tau,beta)

        ############################################## Cubic spline for GG #################################################
        # Instantiation of Cubic_spline
        cs.Cubic_spline(delta_beta,GG_tau)

        der_GG = cs.Cubic_spline.get_derivative(beta_array_G,GG_tau)
        #der_G = cs.Cubic_spline.get_derivative(wn_arr,im_G_iwn)

        # plt.figure(0)
        # plt.plot(beta_array_G,der_GG,marker='*')
        #plt.plot(wn_arr,der_G,marker='*')
        print("GG der: ", der_GG[-1])
        cs.Cubic_spline.building_matrix_components(der_GG[0],der_GG[-1])
        #cs.Cubic_spline.building_matrix_components(der_G[0],der_G[-1])
        # Next step is to do LU decomposition
        cs.Cubic_spline.tridiagonal_LU_decomposition()
        # And then LU solving. From the b coefficients, the a's and c's can be deduced.
        cs.Cubic_spline.tridiagonal_LU_solve()
        # Getting m_a and m_c from the b's
        cs.Cubic_spline.construct_coeffs(beta_array_G)
        #cs.Cubic_spline.construct_coeffs(wn_arr)
        # Checking the actual cubic spline
        rand_arr = np.random.rand(2000)*beta
        interplot = []
        for el in rand_arr:
            interplot.append(cs.Cubic_spline.get_spline(beta_array_G,el))
        
        plt.figure(1)
        #plt.plot(beta_array_G,cs.Cubic_spline._ma,marker='*')
        plt.scatter(rand_arr,interplot,marker='o')
        plt.plot(beta_array_G,GG_tau)
        #plt.plot(wn_arr,im_G_iwn)

        # Instantiation of FermionsvsBosons
        cs.FermionsvsBosons(beta,beta_array_G)
        cubic_Spl_bb_iqn, iqn_array = cs.FermionsvsBosons.bosonic_corr_funct()

        qn_array = np.asarray(list(map(lambda x: x.imag,iqn_array)))
        cubic_Spl_bb_iqn_real = np.asarray(list(map(lambda x: x.real,cubic_Spl_bb_iqn)))
        cubic_Spl_bb_iqn_imag = np.asarray(list(map(lambda x: x.imag,cubic_Spl_bb_iqn)))
        plt.figure(2)
        plt.plot(qn_array,cubic_Spl_bb_iqn_real,c="cyan",marker='o')
        
        K_iqn_real = np.asarray(list(map(lambda x: x.real,K_iqn)))
        K_iqn_imag = np.asarray(list(map(lambda x: x.imag,K_iqn)))
        plt.figure(3)
        plt.plot(qn_array,K_iqn_real,c="blue",marker='o')
        ################################################# End of cubic spline for GG ################################################
        cs.Cubic_spline.reset()
        ############################################## Cubic spline for G #################################################
        # Instantiation of Cubic_spline
        cs.Cubic_spline(delta_beta,tau_final_G)

        der_G = cs.Cubic_spline.get_derivative(beta_array_G,tau_final_G)
        #der_G = cs.Cubic_spline.get_derivative(wn_arr,im_G_iwn)

        print("G_der: ", der_G[-1])
        plt.figure(4)
        plt.plot(beta_array_G,der_G,marker='*')
        #plt.plot(wn_arr,der_G,marker='*')
        
        cs.Cubic_spline.building_matrix_components(der_G[0],der_G[-1])
        #cs.Cubic_spline.building_matrix_components(der_G[0],der_G[-1])
        # Next step is to do LU decomposition
        cs.Cubic_spline.tridiagonal_LU_decomposition()
        # And then LU solving. From the b coefficients, the a's and c's can be deduced.
        cs.Cubic_spline.tridiagonal_LU_solve()
        # Getting m_a and m_c from the b's
        cs.Cubic_spline.construct_coeffs(beta_array_G)
        #cs.Cubic_spline.construct_coeffs(wn_arr)
        # Checking the actual cubic spline
        rand_arr = np.random.rand(2000)*beta
        interplot = []
        for el in rand_arr:
            interplot.append(cs.Cubic_spline.get_spline(beta_array_G,el))

        plt.figure(5)
        plt.scatter(rand_arr,interplot,c="orange",marker='o')
        #plt.plot(beta_array_G,der_G)
        #plt.plot(wn_arr,im_G_iwn)

        # Instantiation of FermionsvsBosons
        cs.FermionsvsBosons(beta,beta_array_G)
        cubic_Spl_G_iwn, iwn_array = cs.FermionsvsBosons.fermionic_propagator()
        
        wn_array = np.asarray(list(map(lambda x: x.imag,iwn_array)))
        cubic_Spl_G_iwn_real = np.asarray(list(map(lambda x: x.real,cubic_Spl_G_iwn)))
        cubic_Spl_G_iwn_imag = np.asarray(list(map(lambda x: x.imag,cubic_Spl_G_iwn)))
        plt.figure(6)
        plt.plot(wn_array,cubic_Spl_G_iwn_imag,c="brown",marker='o')
        ################################################# End of cubic spline for G ################################################

        # From tau to iwn
        iwn_final_G = ls.linear_spline_tau_to_iwn(tau_final_G,beta) # Can either be tau_final_G or t_from_p externally loaded...
        
        iwn_final_G_re = list(map(lambda x: x.real, iwn_final_G))
        iwn_final_G_im = list(map(lambda x: x.imag, iwn_final_G))
        
        plt.figure(7)
        plt.title(r"Real part of $\mathcal{G}(i\omega_n)$")
        plt.plot(wn_arr,iwn_final_G_re,marker='.',c="green",label="linear spline")
        plt.plot(wn_arr,re_G_iwn,marker='*',c="red",label="original")
        plt.xlabel(r"$i\omega_n$")
        plt.ylabel(r"$\operatorname{Re}\mathcal{G}(i\omega_n)$")
        plt.legend()
        plt.figure(8)
        plt.title(r"Imaginary part of $\mathcal{G}(i\omega_n)$")
        plt.plot(wn_arr,iwn_final_G_im,marker='.',c="green",label="linear spline")
        plt.plot(wn_arr,im_G_iwn,marker='*',c="red",label="original")
        plt.xlabel(r"$i\omega_n$")
        plt.ylabel(r"$\operatorname{Im}\mathcal{G}(i\omega_n)$")
        plt.legend()
        plt.show()

        beta_array = np.linspace(0,beta,len(tau_final_G))
        with open("test_FT.dat","w") as f:
            for i,G_tau in enumerate(tau_final_G):
                #print("G_tau is: ", G_tau)
                f.write("%f  %f\n" % (beta_array[i],G_tau))
        f.close()

        with open("test_bb.dat", "w") as f2:
            for i in range(len(K_iqn)):
                if i==0:
                    f2.write("/\n")
                f2.write("%f\t\t%f\t\t%f\n" % (qn_array[i],K_iqn[i].real,K_iqn[i].imag))

        f2.close()

        with open("test_bb_cubic.dat", "w") as f2:
            for i in range(len(cubic_Spl_bb_iqn)):
                if i==0:
                    f2.write("/\n")
                f2.write("%f\t\t%f\t\t%f\n" % (qn_array[i],cubic_Spl_bb_iqn_real[i],cubic_Spl_bb_iqn_imag[i]))

        f2.close()
    else:
        # file containing the data to be inputted as Sigma_iwn
        wn, sigma_iwn_re, sigma_iwn_im = np.genfromtxt("Self_energy_1D_U_14.000000_beta_40.000000_n_0.500000_N_tau_1024_Nit_13.dat",dtype=float,skip_header=1,usecols=(0,1,2),unpack=True)
        sigma_iwn = sigma_iwn_re + 1.j*sigma_iwn_im
        iwn_array = 1.j*wn
        beta = 40.0 # Change this according to the parameters loaded
        U = 14.0 # Change this according to the parameters loaded
        mu = U/2.0
        N_k = 100
        beta_array_self = np.linspace(0.0,beta,len(iwn_array)+1)
        k_array, k_step = np.linspace(0.0,2*np.pi,N_k,retstep=True)
        k_array += k_step
        # Building up the Green's function from the self-energy using the dispersion relation
        bubble_GG_k_iqn = np.empty((N_k,len(iwn_array)),dtype=complex) # Assumption that iqn is same length as iwn...
        bubble_GG_iqn = np.empty((len(iwn_array),),dtype=complex)
        G_k_iwn = np.empty((N_k,len(iwn_array)),dtype=complex)

        # Building the lattice Green's function
        for i,k in enumerate(k_array):
            G_k_iwn[i,:] = np.array([ ( 1.0/( iwn_array[j] + mu - disp_rel(k) - sigma_iwn[j] ) ) for j in range(len(iwn_array)) ],dtype=complex)

        plt.figure(0)
        for n in range(0,N_k,N_k//10):
            G_k_iwn_imag = np.asarray(list(map(lambda x: x.imag,G_k_iwn[n,:])))
            plt.plot( wn, G_k_iwn_imag )

        # Substracting the tail
        for j,iwn in enumerate(iwn_array):
            G_k_iwn[:,j] -= 1.0/(iwn) #+ ( 2.0 + (U**2)/4.0 )/( (iwn)**3 )  # Only odd contributions at half-filling

        # Transforming the iwn Green's functions into tau Green's functions
        G_tau_for_k = np.empty((N_k,len(beta_array_self)),dtype=float)
        for i,k in enumerate(k_array):
            G_tau_for_k[i,:] = get_iwn_to_tau(G_k_iwn[i,:],beta) - 0.5 #+ ( 2.0 + (U**2)/4.0 )*beta_array_self[i]*(beta-beta_array_self[i])/4.0

        plt.figure(1)
        for n in range(0,N_k,N_k//10):
            plt.plot( beta_array_self, G_tau_for_k[n,:] )

        iqn_array = []
        for i,k in enumerate(k_array):
            GG_tau_for_k = np.empty((len(beta_array_self),),dtype=float)
            for j in range(len(beta_array_self)):
                GG_tau_for_k[j] = (-2.0)*(-1.0)*G_tau_for_k[i,j]*G_tau_for_k[i,-1-j] # Eventually put the velocities as prefactors 
            GG_iqn_k, iqn_array = ls.linear_spline_tau_to_iqn_corr(GG_tau_for_k,beta)
            del GG_tau_for_k
            bubble_GG_k_iqn[i,:] = GG_iqn_k

        for i in range(len(iwn_array)):
            bubble_GG_iqn[i] = 1.0/(2.0*np.pi) * simps(bubble_GG_k_iqn[:,i],k_array)

        bubble_GG_iqn_real = np.asarray(list(map(lambda x: x.real,bubble_GG_iqn)))
        plt.figure(2)
        plt.plot(bubble_GG_iqn_real)
        plt.show()

        with open("test_bb.dat", "w") as f2:
            for i in range(len(bubble_GG_iqn)):
                if i==0:
                    f2.write("/\n")
                if cs.isclose(bubble_GG_iqn[i].real,0.0,abs_tol=1e-6): # To remove the tail that spoils the results
                    break
                f2.write("%f\t\t%f\t\t%f\n" % (iqn_array[i].imag,bubble_GG_iqn[i].real,bubble_GG_iqn[i].imag))

        f2.close()