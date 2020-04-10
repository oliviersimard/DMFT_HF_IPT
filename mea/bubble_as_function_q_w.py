import matplotlib.pyplot as plt
from sys import exit
from scipy.integrate import simps
from cubicspline import np
import cubicspline as cs
import linearspline as ls
import h5py

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

def disp_rel(k : float) -> float:
    return -2.0*np.cos(k)

def velocity_squared(k : float) -> float:
    return 4.0*(np.sin(k)**2)

def velocity(k : float) -> float:
    return 2.0*np.sin(k)
    
if __name__=="__main__":

    # file containing the data to be inputted as Sigma_iwn
    wn, sigma_iwn_re, sigma_iwn_im = np.genfromtxt("./data/Self_energy_1D_U_2.000000_beta_50.000000_n_0.500000_N_tau_4096_Nit_5.dat",dtype=float,skip_header=1,usecols=(0,1,2),unpack=True)
    sigma_iwn = sigma_iwn_re + 1.j*sigma_iwn_im
    iwn_array = 1.j*wn
    beta = 50.0 # Change this according to the parameters loaded
    U = 2.0 # Change this according to the parameters loaded
    mu = U/2.0
    N_k = 300
    N_q = 51
    is_jj = False
    N_tau = len(iwn_array)+1
    beta_array_self, delta_beta = np.linspace(0.0,beta,len(iwn_array)+1,retstep=True)
    k_array, k_step = np.linspace(0.0,2.0*np.pi,N_k,retstep=True)
    q_array, q_step = np.linspace(0.0,2.0*np.pi,N_q,retstep=True)
    #k_array += k_step ## Shifting the k-grid
    # Building up the Green's function from the self-energy using the dispersion relation
    bubble_GG_k_iqn = np.empty((N_k,len(iwn_array)),dtype=complex) # Assumption that iqn is same length as iwn...
    bubble_GG_k_iqn_cubic = np.empty((N_k,len(iwn_array)),dtype=complex)
    bubble_GG_iqn = np.empty((len(iwn_array),),dtype=complex)
    bubble_GG_iqn_cubic = np.empty((len(iwn_array),),dtype=complex)
    G_k_iwn = np.empty((N_k,len(iwn_array)),dtype=complex)
    G_k_iwn_2 = np.empty((N_k,len(iwn_array)),dtype=complex)
    G_k_iwn_test = np.empty((N_k,len(iwn_array)),dtype=complex)
    dG_k_iwn = np.empty((N_k,len(iwn_array)),dtype=complex)
    G_k_q_iwn = np.empty((N_k,len(iwn_array)),dtype=complex)
    sum_rule_val_iqn_0_vs_q = []
    sum_rule_val_iqn_0 = 0.0

    # Building the lattice Green's function
    for i,k in enumerate(k_array):
        G_k_iwn[i,:] = np.array([ ( 1.0/( iwn_array[j] + mu - disp_rel(k) - sigma_iwn[j] ) ) for j in range(N_tau-1) ],dtype=complex)
        G_k_iwn_2[i,:] = G_k_iwn[i,:]
        dG_k_iwn[i,:] = G_k_iwn[i,:]
        G_k_iwn_test[i,:] = G_k_iwn[i,:]
            
    plt.figure(0)
    for j,iwn in enumerate(iwn_array):
        for l,k in enumerate(k_array):
            G_k_iwn_test[l,j] -= 1.0/(iwn) + disp_rel(k)/(iwn*iwn) #+ ( 2.0 + 0.25*(U**2) )/( iwn*iwn*iwn )  # Only odd contributions at half-filling
        G_k_iwn_test[:,j] *= (iwn**3)
    G_k_iwn_imag = np.asarray(list(map(lambda x: x.real,G_k_iwn_test[0,:])))
    print("limit: ", U**2/4.0 + disp_rel(k_array[0])**2)
    plt.plot( wn, G_k_iwn_imag )
    
    # Substracting the tail
    for j,iwn in enumerate(iwn_array):
        for l,k in enumerate(k_array):
            G_k_iwn[l,j] -= 1.0/(iwn) + disp_rel(k)/(iwn*iwn) + (U**2/4.0 + disp_rel(k)**2)/(iwn**3)#+ ( 2.0 + 0.25*(U**2) )/( iwn*iwn*iwn )  # Only odd contributions at half-filling
            dG_k_iwn[l,j] -= 1.0/(iwn) + disp_rel(k)/(iwn*iwn) + (U**2/4.0 + disp_rel(k)**2)/(iwn**3)
        dG_k_iwn[:,j] *= -1.0*iwn
        G_k_iwn_2[:,j] -= 1.0/(iwn) # Only odd contributions at half-filling
    
    # Transforming the iwn Green's functions into tau Green's functions
    G_tau_for_k = np.empty((N_k,len(beta_array_self)),dtype=float)
    G_tau_for_k_2 = np.empty((N_k,len(beta_array_self)),dtype=float)
    dG_tau_for_k = np.empty((N_k,len(beta_array_self)),dtype=float)
    plt.figure(4)
    #plt.yscale('log')
    for i,k in enumerate(k_array):
        G_tau_for_k[i,:] = cs.get_iwn_to_tau(G_k_iwn[i,:],beta)
        G_tau_for_k_2[i,:] = cs.get_iwn_to_tau(G_k_iwn_2[i,:],beta)
        dG_tau_for_k[i,:] = cs.get_iwn_to_tau(dG_k_iwn[i,:],beta,type_of="Derivative")
        for j,tau in enumerate(beta_array_self):
            G_tau_for_k[i,j] += - 0.5 - 0.25*(beta-2.0*tau)*disp_rel(k) + 0.25*tau*(beta-tau)*(U**2/4.0 + disp_rel(k)**2) #+ 0.25*( 2.0 + 0.25*(U**2) )*tau*(beta-tau)
            G_tau_for_k_2[i,j] += -0.5
            dG_tau_for_k[i,j] += 0.5*disp_rel(k) + 0.5*( beta/2.0 - tau )*( U**2/4.0 + disp_rel(k)**2 )
        dG_tau_for_k[i,N_tau-1] = disp_rel(k)-mu+U*0.5 - 1.0*dG_tau_for_k[i,0]
        sum_rule_val_iqn_0_vs_q.append(4.0*np.cos(k)*(-G_tau_for_k[i,-1]))
        if np.mod(i,50)==0:
            der_G_vs_k_4 = cs.Cubic_spline.get_derivative_4th_order(beta_array_self,G_tau_for_k[i,:])
            print("diff_1: ", dG_tau_for_k[i,-1]+dG_tau_for_k[i,0], " and ", disp_rel(k))
            print("diff_2: ", der_G_vs_k_4[-1]+der_G_vs_k_4[0], " and ", disp_rel(k))
            plt.plot(beta_array_self,dG_tau_for_k[i,:],marker='v')
            plt.plot(beta_array_self,der_G_vs_k_4,marker='.',label="{0}".format(k_array[i]))
            #plt.plot(-G_tau_for_k[i,::-1],marker='v',ms=1.0)
            #plt.plot(-G_tau_for_k_2[i,::-1],marker='.',ms=1.0)
    plt.legend()
    
    sum_rule_val_iqn_0 = 1.0/N_k*sum(sum_rule_val_iqn_0_vs_q)
    print("Sum rule: ", sum_rule_val_iqn_0)
    
    # plt.figure(1)
    # for n in range(0,N_k,N_k//10):
    #     #der_G_vs_k_6 = cs.Cubic_spline.get_derivative_6th_order(beta_array_self,G_tau_for_k_2[n,:])
    #     der_G_vs_k_4 = cs.Cubic_spline.get_derivative_4th_order(beta_array_self,G_tau_for_k[n,:])
    #     #plt.plot( beta_array_self, G_tau_for_k[n,:], marker='o',label="{0}".format(k_array[n]))
    #     #plt.plot( beta_array_self, der_G_vs_k_6, marker='o',label="{0}".format(k_array[n]))
    #     plt.plot( beta_array_self, der_G_vs_k_4, marker='.',label="{0}".format(k_array[n]))
    # plt.legend()

    iqn_array = []
    for l,q in enumerate(q_array):
        print("q: ", q, " for iter: ", l)
        # Computing the G(k+q) Green's function
        for i,k in enumerate(k_array):
            G_k_q_iwn[i,:] = np.array([ ( 1.0/( iwn_array[j] + mu - disp_rel(k+q) - sigma_iwn[j] ) ) for j in range(N_tau-1) ],dtype=complex)

        for i,iwn in enumerate(iwn_array):
            for l,k in enumerate(k_array):
                G_k_q_iwn[l,i] -= 1.0/(iwn) + disp_rel(k+q)/(iwn*iwn) + (U**2/4.0 + disp_rel(k+q)**2)/(iwn**3)

        G_tau_for_k_q = np.empty((N_k,len(beta_array_self)),dtype=float)
        for i,k in enumerate(k_array):
            G_tau_for_k_q[i,:] = cs.get_iwn_to_tau(G_k_q_iwn[i,:],beta)
            for j,tau in enumerate(beta_array_self):
                G_tau_for_k_q[i,j] += - 0.5 - 0.25*(beta-2.0*tau)*disp_rel(k+q) + 0.25*tau*(beta-tau)*(U**2/4.0 + disp_rel(k+q)**2)
        
        for i,k in enumerate(k_array):
            GG_tau_for_k = np.empty((len(beta_array_self),),dtype=float)
            if is_jj:
                for j in range(len(beta_array_self)):
                    GG_tau_for_k[j] = velocity(k)*velocity(k)*(-2.0)*(-1.0)*G_tau_for_k_q[i,j]*G_tau_for_k[i,-1-j]
            else:
                for j in range(len(beta_array_self)):
                    GG_tau_for_k[j] = (-2.0)*(-1.0)*G_tau_for_k_q[i,j]*G_tau_for_k[i,-1-j]
            #### Cubic spline ####
            cs.Cubic_spline(delta_beta,GG_tau_for_k)
            der_GG_self = cs.Cubic_spline.get_derivative_4th_order(beta_array_self,GG_tau_for_k)
            cs.Cubic_spline.building_matrix_components(der_GG_self[0],der_GG_self[-1])
            # Next step is to do LU decomposition
            cs.Cubic_spline.tridiagonal_LU_decomposition()
            # And then LU solving. From the b coefficients, the a's and c's can be deduced.
            cs.Cubic_spline.tridiagonal_LU_solve()
            # Getting m_a and m_c from the b's
            cs.Cubic_spline.construct_coeffs(beta_array_self)
            cs.FermionsvsBosons(beta,beta_array_self)
            print("k: ", k)
            cubic_Spl_bb_iqn,iqn_array = cs.FermionsvsBosons.bosonic_corr_funct()
            bubble_GG_k_iqn_cubic[i,:] = cubic_Spl_bb_iqn
            ####
            del GG_tau_for_k
            # Resetting the class singletons
            cs.Cubic_spline.reset()
            cs.FermionsvsBosons.reset()
        
        for i in range(len(iwn_array)):
            #bubble_GG_iqn[i] = 1.0/(2.0*np.pi) * simps(bubble_GG_k_iqn[:,i],k_array)
            bubble_GG_iqn_cubic[i] = 1.0/(2.0*np.pi) * simps(bubble_GG_k_iqn_cubic[:,i],k_array)

        with h5py.File("bb_U_{0}_beta_{1}_Ntau_{2}_Nk_{3}_Nq_{4}_isjj_{5}.hdf5".format(U,beta,N_tau,N_k,N_q,is_jj),"a") as fhdf:
            fhdf.create_dataset("q_{0:.5f}".format(q), (N_tau-1,), dtype=np.complex, data=bubble_GG_iqn_cubic)

        fhdf.close()

    #bubble_GG_iqn_real = np.asarray(list(map(lambda x: x.real,bubble_GG_iqn)))
    bubble_GG_iqn_cubic_real = np.asarray(list(map(lambda x: x.real,bubble_GG_iqn_cubic)))
    plt.figure(3)
    #plt.plot(bubble_GG_iqn_real,c="black",marker='*',label="linear spline")
    plt.plot(bubble_GG_iqn_cubic_real,c="red",marker='.',label="cubic spline")
    plt.legend()
    plt.show()
