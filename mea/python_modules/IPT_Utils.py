import numpy as np
from linearspline import linear_spline_tau_to_iwn
from cubicspline import get_iwn_to_tau
from scipy.integrate import simps, quad
from sys import exit
import matplotlib.pyplot as plt

MAX_ITER_ROOT = 40

def epsilonk(k : float) -> float:
    return -2.0*np.cos(k)

def Hsuperior(tau : float, mu0 : float, hyb_c : float, beta : float) -> float:
    z1 = -0.5*mu0 + 0.5*np.sqrt(mu0*mu0+4.0*hyb_c)
    z2 = -0.5*mu0 - 0.5*np.sqrt(mu0*mu0+4.0*hyb_c)

    return -z1/(z1-z2)*1.0/(np.exp(z1*(tau-beta))+np.exp(z1*tau)) - z2/(z2-z1)*1.0/(np.exp(z2*(tau-beta))+np.exp(z2*tau))

def Hinferior(tau : float, mu0 : float, hyb_c : float, beta : float) -> float:
    z1 = -0.5*mu0 + 0.5*np.sqrt(mu0*mu0+4.0*hyb_c)
    z2 = -0.5*mu0 - 0.5*np.sqrt(mu0*mu0+4.0*hyb_c)
    
    return z1/(z1-z2)*1.0/(np.exp(z1*(beta-tau))+np.exp(-z1*tau)) + z2/(z2-z1)*1.0/(np.exp(z2*(beta-tau))+np.exp(-z2*tau))

def get_iwn_to_tau_hyb(G_iwn, beta : float, hyb_c : float, opt="positive"):
    MM = len(G_iwn) # N = M/2
    tau_final_G = np.zeros(MM+1,dtype=float)
    # FFT
    if opt=="positive":
        tau_resolved_G = np.fft.fft(G_iwn)
        for i in range(MM):
            tau=i*beta/MM
            tau_final_G[i] = ( (1./beta)*np.exp( -1.j * np.pi * i * ( 1.0/(MM) - 1.0 ) )*tau_resolved_G[i] + Hsuperior(tau,0.0,hyb_c,beta) ).real

        tau_final_G[MM] = -1.0-tau_final_G[0]
    elif opt=="negative":
        tau_resolved_G = np.fft.fft(np.conj(G_iwn))
        for i in range(MM):
            tau=i*beta/MM
            tau_final_G[i] = ( (1./beta)*np.exp( -1.j * np.pi * i * ( 1.0/(MM) - 1.0 ) )*tau_resolved_G[i] + Hinferior(tau,0.0,hyb_c,beta) ).real

        tau_final_G[MM] = 1.0-tau_final_G[0]

    return tau_final_G

def false_position_method(funct, a : float, b : float, tol : float) -> float:
    if funct(a)*funct(b)>0.0:
        raise ValueError("The function domain doesn't properly bracket the function...change a and b.")
    c = 10.0*tol
    it=1
    while np.abs(funct(c))>tol and it<MAX_ITER_ROOT:
        c = (a-b)*funct(a)/(funct(b)-funct(a)) + a
        if np.abs(funct(c))<=tol:
            break
        elif np.abs(funct(c))>tol:
            if funct(a)*funct(c)<0.0:
                b=c
            elif funct(b)*funct(c)<0.0:
                a=c
            else:
                raise Exception("Bad behaviour of the function.")
        it+=1
    
    return c

class Sublattice(object):

    #static variables
    _beta=0.0
    _U=0.0
    _Ntau=0.0
    _hyb_c=0.0
    _Nk=0.0
    _delta_tau = 0.0
    _iwn_arr=np.array([],dtype=complex)
    _tau_arr=np.array([],dtype=float)

    @staticmethod
    def set_static_attrs():
        # Work on static variables
        Sublattice._delta_tau = Sublattice._beta/Sublattice._Ntau
        for n in range(-Sublattice._Ntau,Sublattice._Ntau):
            Sublattice._iwn_arr = np.append(Sublattice._iwn_arr,complex(0.0,(2.0*n+1.0)*np.pi/Sublattice._beta))
        for l in range(2*Sublattice._Ntau+1):
            Sublattice._tau_arr = np.append(Sublattice._tau_arr,l*Sublattice._delta_tau)
        
        assert len(Sublattice._iwn_arr)==len(Sublattice._tau_arr)-1, "Length of Matsubara array has to be shorter by one element compared to tau array."


    # Static variables must be initiated before calling the constructor
    # Hybridisation function, Self energy, Weiss Green's function and Local Green's function containers in constructor
    def __init__(self, mu : float, mu0 : float, Hyb : np.ndarray, G0 : np.ndarray, Gloc : np.ndarray, SE : np.ndarray):
        self._mu = mu
        self._mu0 = mu0
        self._Hyb = Hyb
        self._G0 = G0
        self._Gloc = Gloc
        self._SE_iwn = SE

    def update_self_energy_AFM(self) -> None:
        # First get G0(tau)
        G0_tau = np.ndarray((2*Sublattice._Ntau+1,2,2,),dtype=float)
        #G0_m_tau = np.ndarray((2*Sublattice._Ntau+1,2,2,),dtype=float)
        # subtracting the leadind tail
        for j,iwn in enumerate(Sublattice._iwn_arr):
            self._G0[j,0,0] -= 1.0/( iwn )
            self._G0[j,1,1] -= 1.0/( iwn )
        G0_tau[:,0,0] = get_iwn_to_tau(self._G0[:,0,0],Sublattice._beta) #up
        G0_tau[:,1,1] = get_iwn_to_tau(self._G0[:,1,1],Sublattice._beta) #down
        # G0_m_tau[:,0,0] = get_iwn_to_tau(np.conj(self._G0[:,0,0]),Sublattice._beta) #up
        # G0_m_tau[:,1,1] = get_iwn_to_tau(np.conj(self._G0[:,1,1]),Sublattice._beta) #down
        for j in range(len(Sublattice._tau_arr)):
            G0_tau[j,0,0] += -0.5
            G0_tau[j,1,1] += -0.5
            # G0_m_tau[j,0,0] += Hinferior(tau,0.0,Sublattice._hyb_c,Sublattice._beta)
            # G0_m_tau[j,1,1] += Hinferior(tau,0.0,Sublattice._hyb_c,Sublattice._beta)
        plt.plot(Sublattice._tau_arr,G0_tau[:,0,0],marker='v',ms=2.5,c="red")
        # plt.plot(Sublattice._tau_arr,G0_m_tau[:,0,0],marker='o',ms=2.5,c="black")
        plt.show()
        # Weiss Green's function densities used for Hartree term
        n0_up = -1.0*G0_tau[-1,0,0]
        n0_down = -1.0*G0_tau[-1,1,1]
        print("n0_up: ", n0_up, " and n0_down: ", n0_down)
        SE_tau = np.ndarray((2*Sublattice._Ntau+1,2,2,),dtype=complex)
        # Self-energy impurity
        for i in range(SE_tau.shape[0]):
            SE_tau[i,0,0] = -G0_tau[i,0,0]*G0_tau[-1-i,1,1]*G0_tau[i,1,1] #up
            SE_tau[i,1,1] = -G0_tau[i,1,1]*G0_tau[-1-i,0,0]*G0_tau[i,0,0] #down
        # Fourier transforming back the self-energy to iwn
        self._SE_iwn[:,0,0] = Sublattice._U*(1.0-n0_up) - Sublattice._U*Sublattice._U*linear_spline_tau_to_iwn(SE_tau[:,0,0],Sublattice._beta) #up
        self._SE_iwn[:,1,1] = Sublattice._U*n0_up - Sublattice._U*Sublattice._U*linear_spline_tau_to_iwn(SE_tau[:,1,1],Sublattice._beta) #down

    def update_self_energy(self) -> None:
        # First get G0(tau)
        G0_tau = np.ndarray((2*Sublattice._Ntau+1,2,2,),dtype=float)
        G0_m_tau = np.ndarray((2*Sublattice._Ntau+1,2,2,),dtype=float)
        # subtracting the leadind tail
        for j,iwn in enumerate(Sublattice._iwn_arr):
            self._G0[j,0,0] -= 1.0/( iwn - Sublattice._hyb_c/iwn )
        G0_tau[:,0,0] = get_iwn_to_tau_hyb(self._G0[:,0,0],Sublattice._beta,Sublattice._hyb_c) #up
        G0_m_tau[:,0,0] = get_iwn_to_tau_hyb(self._G0[:,0,0],Sublattice._beta,Sublattice._hyb_c,opt="negative") #up
        # plt.plot(Sublattice._tau_arr,G0_tau[:,0,0],marker='v',ms=2.5,c="red")
        # plt.plot(Sublattice._tau_arr,G0_m_tau[:,0,0],marker='v',ms=2.5,c="black")
        # plt.show()
        # Weiss Green's function densities used for Hartree term
        n0_up = -1.0*G0_tau[-1,0,0]
        print("n0_up: ", n0_up)
        SE_tau = np.ndarray((2*Sublattice._Ntau+1,2,2,),dtype=complex)
        # Self-energy impurity
        for i in range(SE_tau.shape[0]):
            SE_tau[i,0,0] = G0_tau[i,0,0]*G0_m_tau[i,0,0]*G0_tau[i,0,0] #up
        # Fourier transforming back the self-energy to iwn
        self._SE_iwn[:,0,0] = Sublattice._U*(1.0-n0_up) - Sublattice._U*Sublattice._U*linear_spline_tau_to_iwn(SE_tau[:,0,0],Sublattice._beta) #up

    def DMFT_AFM(self, h : float) -> None:
        # update the self-energy with the updated objects from last iteration
        self.update_self_energy_AFM()
        # computing the physical particle density
        G_imp_iwn = np.ndarray((2*Sublattice._Ntau,2,2,),dtype=complex)
        G_imp_tau = np.ndarray((2*Sublattice._Ntau+1,2,2,),dtype=float)
        for i,iwn in enumerate(Sublattice._iwn_arr):
            G_imp_iwn[i,0,0] = 1.0/( iwn + self._mu - h - self._Hyb[i,0,0] - self._SE_iwn[i,0,0] ) - 1.0/iwn
            G_imp_iwn[i,1,1] = 1.0/( iwn + self._mu + h - self._Hyb[i,1,1] - self._SE_iwn[i,1,1] ) - 1.0/iwn
        G_imp_tau[:,0,0] = get_iwn_to_tau(G_imp_iwn[:,0,0],Sublattice._beta)
        G_imp_tau[:,1,1] = get_iwn_to_tau(G_imp_iwn[:,1,1],Sublattice._beta)
        del G_imp_iwn
        for j in range(G_imp_tau.shape[0]):
            G_imp_tau[j,0,0] -= 0.5
            G_imp_tau[j,1,1] -= 0.5
        print("impurity n_up: ", -1.0*G_imp_tau[-1,0,0], " and n_down: ", -1.0*G_imp_tau[-1,1,1])
        #k_array spanning from -pi/2.0 to pi/2.0
        k_array = np.array([-np.pi/2.0+l*(1.0*np.pi/(Sublattice._Nk-1)) for l in range(Sublattice._Nk)],dtype=float)
        for i,iwn in enumerate(Sublattice._iwn_arr):
            k_def_G_latt_up = np.empty((Sublattice._Nk,),dtype=complex)
            k_def_G_latt_down = np.empty((Sublattice._Nk,),dtype=complex)
            for j,k in enumerate(k_array):
                k_def_G_latt_up[j] = 1.0/( iwn + self._mu - h - self._SE_iwn[i,0,0] - epsilonk(k)**2/( iwn + self._mu + h - self._SE_iwn[i,1,1] ) )
                k_def_G_latt_down[j] = 1.0/( iwn + self._mu + h - self._SE_iwn[i,1,1] - epsilonk(k)**2/( iwn + self._mu - h - self._SE_iwn[i,0,0] ) )
            self._Gloc[i,0,0] = 1.0/np.pi*simps(k_def_G_latt_up,k_array) #up
            self._Gloc[i,1,1] = 1.0/np.pi*simps(k_def_G_latt_down,k_array) #down
        # update the hybdridisation function and set Weiss Green's function for the next iteration
        for i,iwn in enumerate(Sublattice._iwn_arr):
            self._Hyb[i,0,0] = iwn + self._mu - h - self._SE_iwn[i,0,0] - 1.0/self._Gloc[i,0,0]
            self._G0[i,0,0] = 1.0/( iwn - self._Hyb[i,0,0] )
            self._Hyb[i,1,1] = iwn + self._mu + h - self._SE_iwn[i,1,1] - 1.0/self._Gloc[i,1,1]
            self._G0[i,1,1] = 1.0/( iwn - self._Hyb[i,1,1] )
    
    def DMFT(self) -> None:
        # update the self-energy with the updated objects from last iteration
        self.update_self_energy()
        # computing the physical particle density
        G_imp_iwn = np.ndarray((2*Sublattice._Ntau,2,2,),dtype=complex)
        G_imp_tau = np.ndarray((2*Sublattice._Ntau+1,2,2,),dtype=float)
        for i,iwn in enumerate(Sublattice._iwn_arr):
            G_imp_iwn[i,0,0] = 1.0/( iwn + self._mu - self._Hyb[i,0,0] - self._SE_iwn[i,0,0] ) - 1.0/( iwn - Sublattice._hyb_c/iwn - Sublattice._U**2/(4.0*iwn) )
        G_imp_tau[:,0,0] = get_iwn_to_tau_hyb(G_imp_iwn[:,0,0],Sublattice._beta,Sublattice._hyb_c+Sublattice._U**2/4.0)
        del G_imp_iwn
        # for j,tau in enumerate(Sublattice._tau_arr):
        #     G_imp_tau[j,0,0] += -0.5
        print("impurity n_up: ", -1.0*G_imp_tau[-1,0,0])
        #k_array spanning from -pi/2.0 to pi/2.0
        for i,iwn in enumerate(Sublattice._iwn_arr):
            funct_integrate_re = lambda k: ( 1.0/( iwn + self._mu - self._SE_iwn[i,0,0] - epsilonk(k) ) ).real
            funct_integrate_im = lambda k: ( 1.0/( iwn + self._mu - self._SE_iwn[i,0,0] - epsilonk(k) ) ).imag
            self._Gloc[i,0,0] = 1.0/(2.0*np.pi)*( quad(funct_integrate_re,-np.pi,np.pi)[0] + 1.0j*quad(funct_integrate_im,-np.pi,np.pi)[0] )

        # update the hybdridisation function and set Weiss Green's function for the next iteration
        for i,iwn in enumerate(Sublattice._iwn_arr):
            self._Hyb[i,0,0] = iwn + self._mu - self._SE_iwn[i,0,0] - 1.0/self._Gloc[i,0,0]
            self._G0[i,0,0] = 1.0/( iwn + self._mu0 - self._Hyb[i,0,0] )

    @staticmethod
    def update_mu(arr_mu : np.ndarray, mu : float) -> float:
        nn=0.0
        for i,iwn in enumerate(Sublattice._iwn_arr):
            nn += ( 1./( arr_mu[i,0,0] + mu ) - 1./iwn ).real
        nn*=1./Sublattice._beta
        nn+=0.5

        return nn
            
    def update_densities(self, it : float, n_target : float) -> None:
        n0_mu0 = np.ndarray((2*Sublattice._Ntau,2,2,),dtype=complex)
        n_mu = np.ndarray((2*Sublattice._Ntau,2,2,),dtype=complex)
        for i,iwn in enumerate(Sublattice._iwn_arr):
            n_mu[i,0,0] = iwn - self._Hyb[i,0,0] - self._SE_iwn[i,0,0]
            n0_mu0[i,0,0] = iwn - self._Hyb[i,0,0]
        
        get_new_n0 = lambda mu0: Sublattice.update_mu(n0_mu0,mu0)-n_target
        new_mu0 = false_position_method(get_new_n0,-20.0,20.0,0.001)
        get_new_n = lambda mu: Sublattice.update_mu(n_mu,mu)-n_target
        new_mu = false_position_method(get_new_n,-20.0,20.0,0.001)
        print("new mu: ", new_mu, " and new mu0: ", new_mu0)
        # Updating the chemical potentials according to the target density
        self._mu0 = new_mu0
        if it>1:
            self._mu = new_mu

    
    def save_Gloc_AFM(self, it : int) -> None:
        filename = "data_test_IPT/Green_loc_1D_AFM_U_{0:.5f}".format(Sublattice._U)+"_beta_{0:.5f}".format(Sublattice._beta)+"_N_tau_{0:d}".format(Sublattice._Ntau)+"_Nit_{0:d}".format(it)+".dat"
        with open(filename,"w+") as f:
            for i in range(len(Sublattice._iwn_arr)):
                if i==0:
                    f.write("iwn\t\tRe Gloc up\t\tIm Gloc up\t\tRe Gloc down\t\tIm Gloc down\n")
                f.write("{0:.8f}\t\t{1:.8f}\t\t{2:.8f}\t\t{3:.8f}\t\t{4:.8f}\n".format(Sublattice._iwn_arr[i].imag,self._Gloc[i,0,0].real,self._Gloc[i,0,0].imag,self._Gloc[i,1,1].real,self._Gloc[i,1,1].imag))
        f.close()
    
    def save_Gloc(self, it : int) -> None:
        filename = "data_test_IPT/Green_loc_1D_U_{0:.5f}".format(Sublattice._U)+"_beta_{0:.5f}".format(Sublattice._beta)+"_N_tau_{0:d}".format(Sublattice._Ntau)+"_Nit_{0:d}".format(it)+".dat"
        with open(filename,"w+") as f:
            for i in range(len(Sublattice._iwn_arr)):
                if i==0:
                    f.write("iwn\t\tRe Gloc up\t\tIm Gloc up\n")
                f.write("{0:.8f}\t\t{1:.8f}\t\t{2:.8f}\n".format(Sublattice._iwn_arr[i].imag,self._Gloc[i,0,0].real,self._Gloc[i,0,0].imag))
        f.close()

    def save_SE(self, it : int) -> None:
        filename = "data_test_IPT/Self_energy_1D_U_{0:.5f}".format(Sublattice._U)+"_beta_{0:.5f}".format(Sublattice._beta)+"_N_tau_{0:d}".format(Sublattice._Ntau)+"_Nit_{0:d}".format(it)+".dat"
        with open(filename,"w+") as f:
            for i in range(len(Sublattice._iwn_arr)):
                if i==0:
                    f.write("iwn\t\tRe SE up\t\tIm SE up\n")
                f.write("{0:.8f}\t\t{1:.8f}\t\t{2:.8f}\n".format(Sublattice._iwn_arr[i].imag,self._SE_iwn[i,0,0].real,self._SE_iwn[i,0,0].imag))
        f.close()
