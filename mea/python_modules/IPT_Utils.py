import numpy as np
from linearspline import linear_spline_tau_to_iwn, linear_spline_Sigma_tau_to_iwn
import cubicspline as cs
from scipy.integrate import simps, quad
from sys import exit
import matplotlib.pyplot as plt
from time import sleep
import os
import scipy.linalg as sla

# Max number of iterations in the false position root finding method
MAX_ITER_ROOT = 40
# Min tolerance threshold to assess convergence of the solution
MIN_TOL = 1e-4
# quad abs_tol
MIN_QUAD_TOL = 1e-5

def epsilonk_1D(k : float) -> float:
    """
    This function represents the nearest-neighbour dispersion relation in 1D.

    Parameters:
        k (float): momentum
    
    Returns:
        (float): dispersion relation at inputted momentum
    """
    return -2.0*np.cos(k)

def epsilonk_2D(kx : float,ky : float) -> float:
    """
    This function represents the nearest-neighbour dispersion relation in 1D.

    Parameters:
        kx (float): momentum in x direction
        ky (float): momentum in y direction
    
    Returns:
        (float): dispersion relation at inputted momentum
    """
    return -2.0*(np.cos(kx)+np.cos(ky))

def Hsuperior(tau : float, mu0 : float, hyb_c : float, beta : float) -> float:
    """
    Function representing the analytical continuation G_0(tau) (0<tau<beta) of the asymptotic form of the Weiss Green's function defined as:
    G_0(iw_n) = 1.0/( iwn + mu0 - hyb_c/iwn )

    Parameters:
        tau (float): imaginary time component (0<tau<beta)
        mu0 (float): Weiss chemical potential (should be 0 at half-filling)
        hyb_c (float): one-dimensional integral in k-space over square of dispersion relation (see Notes)
        beta (float): inverse temperature

    Returns:
        (float): analytical continuation of the asymptotic Weiss Green's function evaluated at given tau
    """
    z1 = -0.5*mu0 + 0.5*np.sqrt(mu0*mu0+4.0*hyb_c)
    z2 = -0.5*mu0 - 0.5*np.sqrt(mu0*mu0+4.0*hyb_c)

    return -z1/(z1-z2)*1.0/(np.exp(z1*(tau-beta))+np.exp(z1*tau)) - z2/(z2-z1)*1.0/(np.exp(z2*(tau-beta))+np.exp(z2*tau))

def Hinferior(tau : float, mu0 : float, hyb_c : float, beta : float) -> float:
    """
    Function representing the analytical continuation G_0(-tau) (0<tau<beta) of the asymptotic form of the Weiss Green's function defined as:
    G_0(iw_n) = 1.0/( iwn + mu0 - hyb_c/iwn ). It is important to stress that this is for G_0(-tau).

    Parameters:
        tau (float): imaginary time component (0<tau<beta)
        mu0 (float): Weiss chemical potential (should be 0 at half-filling)
        hyb_c (float): one-dimensional integral in k-space over square of dispersion relation (see Notes)
        beta (float): inverse temperature

    Returns:
        (float): analytical continuation of the asymptotic Weiss Green's function evaluated at given tau
    """
    z1 = -0.5*mu0 + 0.5*np.sqrt(mu0*mu0+4.0*hyb_c)
    z2 = -0.5*mu0 - 0.5*np.sqrt(mu0*mu0+4.0*hyb_c)
    
    return z1/(z1-z2)*1.0/(np.exp(z1*(beta-tau))+np.exp(-z1*tau)) + z2/(z2-z1)*1.0/(np.exp(z2*(beta-tau))+np.exp(-z2*tau))

def get_iwn_to_tau_hyb(G_iwn, beta : float, mu0 : float, hyb_c : float, opt="positive"):
    """
    Function representing the analytical continuation G_0(-tau) (0<tau<beta) of the asymptotic form of the Weiss Green's function defined as:
    G_0(iw_n) = 1.0/( iwn + mu0 - hyb_c/iwn ). It is important to stress that this is for G_0(-tau).

    Parameters:
        G_iwn (array): object defined in fermionic Matsubara frequencies to be represented in imaginary time (0<tau<beta)
        beta (float): inverse temperature
        mu0 (float): Weiss chemical potential (should be 0 at half-filling)
        hyb_c (float): one-dimensional integral in k-space over square of dispersion relation (see Notes)
        opt (str): can either be "positive" or "negative": "positive" means that the fermionic object G_iwn is transmuted
        into G(tau) after Fourier transformation. Negative means that G_iwn becomes G(-tau) after Fourier transformation.

    Returns:
        tau_final_G (array): container holding the data in imaginary time 
    """
    MM = len(G_iwn) # N = M/2
    tau_final_G = np.zeros(MM+1,dtype=float)
    # FFT
    if opt=="positive":
        tau_resolved_G = np.fft.fft(G_iwn)
        for i in range(MM):
            tau=i*beta/MM
            tau_final_G[i] = ( (1./beta)*np.exp( -1.j * np.pi * i * ( 1.0/(MM) - 1.0 ) )*tau_resolved_G[i] + Hsuperior(tau,mu0,hyb_c,beta) ).real

        tau_final_G[MM] = -1.0-tau_final_G[0]
    elif opt=="negative":
        tau_resolved_G = np.fft.fft(np.conj(G_iwn))
        for i in range(MM):
            tau=i*beta/MM
            tau_final_G[i] = ( (1./beta)*np.exp( -1.j * np.pi * i * ( 1.0/(MM) - 1.0 ) )*tau_resolved_G[i] + Hinferior(tau,mu0,hyb_c,beta) ).real

        tau_final_G[MM] = 1.0-tau_final_G[0]

    return tau_final_G

def false_position_method(funct, a : float, b : float, tol : float) -> float:
    """
    False Position Method finding root of function having one free parameter.

    Parameters:
        funct (function): function object whose root is to be found
        a (float): lower boundary of the domain of search
        b (float): upper boundary of the domain of search
        tol (float): tolerance threshold to assess whether the root has been found or not

    Returns:
        c (float): abscisse value whose image is 0 (root) 
    """
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

def check_converged_AFM(G_0_last : np.ndarray, G_0_current : np.ndarray, it : int) -> bool:
    """ 
    This function evaluates whether the Weiss Green's function has converged by comparing
    the two latest ones in the AFM case scenario.

    Parameters:
        G_0_last (ndarray): Weiss Green's function at iteration before it.
        G_0_current (ndarray): Weiss Green's function at iteration before it.
        it (int): iteration it
    
    Returns:
        (bool): boolean value assessing whether the Weiss Green's function at iteration it has converged or not.
    """
    assert G_0_current.shape[0]==G_0_last.shape[0], "Inputted ndarrays must have the same lengths."
    G_0_up_diff=0.0; G_0_down_diff=0.0
    for l in range(G_0_current.shape[0]):
        G_0_up_diff += np.abs(G_0_last[l,0,0]-G_0_current[l,0,0])
        G_0_down_diff += np.abs(G_0_last[l,1,1]-G_0_current[l,1,1])

    print("G_0_up_diff at it ", it, " is ", G_0_up_diff, " and G_0_down_diff is ", G_0_down_diff)
    if G_0_up_diff<MIN_TOL and G_0_down_diff<MIN_TOL and it>2:
        return True
    else:
        return False

def check_converged(G_0_last : np.ndarray, G_0_current : np.ndarray, it : int) -> bool:
    """ 
    This function evaluates whether the Weiss Green's function has converged by comparing
    the two latest ones in the PM case scenario.

    Parameters:
        G_0_last (ndarray): Weiss Green's function at iteration before it.
        G_0_current (ndarray): Weiss Green's function at iteration before it.
        it (int): iteration it
    
    Returns:
        (bool): boolean value assessing whether the Weiss Green's function at iteration it has converged or not.
    """
    assert G_0_current.shape[0]==G_0_last.shape[0], "Inputted ndarrays must have the same lengths."
    G_0_diff=0.0
    for l in range(G_0_current.shape[0]):
        G_0_diff += np.abs(G_0_last[l,0,0]-G_0_current[l,0,0])
    
    print("G_0_diff at it ", it, " is ", G_0_diff)
    if G_0_diff<MIN_TOL and it>2:
        return True
    else:
        return False

def broyden_good(x, y, f_equations, J_equations, tol=10e-10, maxIters=50):
    steps_taken = 0
 
    A, B, f = f_equations(x,y)
    J = J_equations(x,y)
 
    while np.linalg.norm(f,2) > tol and steps_taken < maxIters:
 
        s = sla.solve(J,-1*f)
 
        x = x + s[0]
        y = y + s[1]
        A, B, newf = f_equations(x,y)
        z = newf - f
 
        J = J + (np.outer ((z - np.dot(J,s)),s)) / (np.dot(s,s))
 
        f = newf
        steps_taken += 1
 
    return steps_taken, x, y, A, B

class Sublattice(object):
    """
    Minimal class holding member variables and methods to go about solving Anderson impurity model for DMFT-IPT in both PM and AFM
    scenarios.
    """
    #static variables' initialisation
    _h=0.0
    _beta=0.0
    _U=0.0
    # Half the total number of fermionic Matsubara frequencies (even number)
    _Ntau=0.0
    _hyb_c=0.0
    # k-space discretization (not really used, mainly for testing)
    _Nk=0.0
    # tau step
    _delta_tau=0.0
    # fermionic Matsubara frequency container
    _iwn_arr=np.array([],dtype=complex)
    # imaginary-time container (0<=tau<=beta)
    _tau_arr=np.array([],dtype=float)
    # k-space container (original Brillouin zone)
    _k_arr=np.array([],dtype=float)
    # k-space container (Reduced AFM Brillouin zone)
    _k_AFM_arr=np.array([],dtype=float)

    # Luttinger sum rule
    ITER_LUTTINGER_SUM_RULE = 0 # <---------------------- added for testing

    @staticmethod
    def set_static_attrs():
        """
        This static method sets some of the static variables living in Sublattice class space. It sets for instance "_delta_tau",
        "_iwn_arr", and "_tau_arr", having set previously all the remaining static variables in main.
        """
        Sublattice._delta_tau = Sublattice._beta/(2.0*Sublattice._Ntau)
        for n in range(-Sublattice._Ntau,Sublattice._Ntau):
            Sublattice._iwn_arr = np.append(Sublattice._iwn_arr,complex(0.0,(2.0*n+1.0)*np.pi/Sublattice._beta))
        for l in range(2*Sublattice._Ntau+1):
            Sublattice._tau_arr = np.append(Sublattice._tau_arr,l*Sublattice._delta_tau)
        for i in range(Sublattice._Nk):
            Sublattice._k_arr = np.append(Sublattice._k_arr,-np.pi+i*(2.0*np.pi/(Sublattice._Nk-1.0)))
        for i in range(Sublattice._Nk):
            Sublattice._k_AFM_arr = np.append(Sublattice._k_AFM_arr,-np.pi/2.0+i*(1.0*np.pi/(Sublattice._Nk-1.0)))
        
        assert len(Sublattice._iwn_arr)==len(Sublattice._tau_arr)-1, "Length of Matsubara array has to be shorter by one element compared to tau array."

    @staticmethod
    def reset_static_attrs():
        """
        This static method resets some of the static variables living in Sublattice class space. It resets for instance "_delta_tau",
        "_iwn_arr", and "_tau_arr", having set previously all the remaining unchanging static variables in main.
        """
        Sublattice._delta_tau = 0.0
        Sublattice._iwn_arr = np.array([],dtype=complex)
        Sublattice._tau_arr = np.array([],dtype=float)
        Sublattice._k_arr = np.array([],dtype=float)
        Sublattice._k_AFM_arr = np.array([],dtype=float)

    # Static variables must be initiated before calling the constructor
    # Hybridisation function, Self energy, Weiss Green's function and Local Green's function containers in constructor
    def __init__(self, mu : float, mu0 : float, Hyb : np.ndarray, G0 : np.ndarray, Gloc : np.ndarray, Gloc_tau : np.ndarray, SE : np.ndarray, SE_tau : np.ndarray, n_up=0.5, n_down=0.5):
        """
        Constructor of Sublattice class.
        """
        self._mu = mu
        self._mu0 = mu0
        self._Hyb = Hyb # Hybridisation (iwn)
        self._G0 = G0 # Weiss Green (iwn)
        self._Gloc = Gloc # Local Green (iwn)
        self._Gloc_tau = Gloc_tau # Local Green (tau)
        self._SE = SE # Self energy (iwn)
        self._SE_tau = SE_tau # Self energy (tau)
        self._n_up = n_up # up-spin density
        self._n_down = n_down # down-spin density
        self._n0_up = n_up # up-spin Weiss density
        self._n0_down = n_down # down-spin Weiss density
    
    def set_mu_to_0(self):
        """
            Sets the impurity chemical potential to 0 in the AFM case, since the chemical potential enters the 
            definition of the Hartree term.
        """
        self._mu=0.0

    def update_self_energy_AFM(self,type_spline : str) -> None:
        """
        Method updating the IPT impurity self-energy in AFM scenario. The Weiss Green's function is transformed into
        G_0(tau) and G_0(-tau) to compute the self-energy Sigma(tau) for the two spins. Then, Sigma(tau) is transformed back 
        into Sigma(iwn).

        Parameters:
            None
        
        Returns:
            self._SE (ndarray): updated self-energy in fermionic Matsubara frequencies.
        """
        # First get G0(tau)
        G0_tau = np.ndarray((2*Sublattice._Ntau+1,2,2,),dtype=float)
        G0_m_tau = np.ndarray((2*Sublattice._Ntau+1,2,2,),dtype=float)
        # subtracting the leadind tail
        for j,iwn in enumerate(Sublattice._iwn_arr):
            self._G0[j,0,0] -= 1.0/( iwn )
            self._G0[j,1,1] -= 1.0/( iwn )
        G0_tau[:,0,0] = cs.get_iwn_to_tau(self._G0[:,0,0],Sublattice._beta) #up
        G0_m_tau[:,0,0] = cs.get_iwn_to_tau(self._G0[:,0,0],Sublattice._beta,opt="negative") #up
        G0_tau[:,1,1] = cs.get_iwn_to_tau(self._G0[:,1,1],Sublattice._beta) #down
        G0_m_tau[:,1,1] = cs.get_iwn_to_tau(self._G0[:,1,1],Sublattice._beta,opt="negative") #down
        
        for i in range(len(Sublattice._tau_arr)):
            G0_tau[i,0,0] += -0.5 #up
            G0_tau[i,1,1] += -0.5 #down
            G0_m_tau[i,0,0] += 0.5 #up
            G0_m_tau[i,1,1] += 0.5 #down
        for j,iwn in enumerate(Sublattice._iwn_arr):
            self._G0[j,0,0] += 1.0/( iwn )
            self._G0[j,1,1] += 1.0/( iwn )

        # Weiss Green's function densities used for Hartree term
        self._n0_up = -1.0*G0_tau[-1,0,0]
        self._n0_down = -1.0*G0_tau[-1,1,1]
        print("n0_up: ", self._n0_up, " and n0_down: ", self._n0_down)
        # Self-energy impurity
        if type_spline=="cubic":
            Sigma_up_iwn, Sigma_down_iwn = cs.compute_Sigma_iwn_cubic_spline(self._G0,G0_tau,G0_m_tau,Sublattice._delta_tau,Sublattice._U, \
                Sublattice._h,Sublattice._hyb_c,self._mu,Sublattice._beta,Sublattice._tau_arr,Sublattice._iwn_arr,self._SE_tau)
        elif type_spline=="linear":
            for i in range(self._SE_tau.shape[0]):
                self._SE_tau[i,0,0] = -1.0*Sublattice._U**2*G0_tau[i,0,0]*G0_m_tau[i,1,1]*G0_tau[i,1,1] #up
                self._SE_tau[i,1,1] = -1.0*Sublattice._U**2*G0_tau[i,1,1]*G0_m_tau[i,0,0]*G0_tau[i,0,0] #down
            Sigma_up_iwn = linear_spline_Sigma_tau_to_iwn(self._SE_tau[:,0,0],Sublattice._beta)
            Sigma_down_iwn = linear_spline_Sigma_tau_to_iwn(self._SE_tau[:,1,1],Sublattice._beta)
        # 2nd order Hartree term
        # Convolution
        Sigma_Hartree_2nd_up = 0.0; Sigma_Hartree_2nd_down = 0.0
        for i in range(2*Sublattice._Ntau+1):
            Sigma_Hartree_2nd_up += -1.0*Sublattice._delta_tau*G0_tau[i,1,1]*G0_tau[2*Sublattice._Ntau-i,1,1]#delta_tau*G0_tau[j-i,1,1]*G0_tau[2*Ntau-(j-i),1,1]
            Sigma_Hartree_2nd_down += -1.0*Sublattice._delta_tau*G0_tau[i,0,0]*G0_tau[2*Sublattice._Ntau-i,0,0]#delta_tau*G0_tau[j-i,0,0]*G0_tau[2*Ntau-(j-i),0,0]
        Sigma_Hartree_2nd_up *= Sublattice._U*Sublattice._U*(self._n0_up-0.5)
        Sigma_Hartree_2nd_down *= Sublattice._U*Sublattice._U*(self._n0_down-0.5)
        self._SE[:,0,0] = Sublattice._U*(self._n0_down-0.5) + Sigma_up_iwn + Sigma_Hartree_2nd_up #up
        self._SE[:,1,1] = Sublattice._U*(self._n0_up-0.5) + Sigma_down_iwn + Sigma_Hartree_2nd_down #down
        # print("SE up")
        # for i in range(len(Sublattice._iwn_arr)):
        #     print(self._SE[i,0,0])
        # print("SE down")
        # for i in range(len(Sublattice._iwn_arr)):
        #     print(self._SE[i,1,1])
        # exit(0)

    def update_self_energy(self,type_spline : str) -> None:
        """
        Method updating the IPT impurity self-energy in PM scenario. Basically, the Weiss Green's function is transformed into
        G_0(tau) and G_0(-tau) to compute the self-energy Sigma(tau). Then, Sigma(tau) is transformed into Sigma(iwn).

        Parameters:
            type_spline (str): determines the type of spline to be used in the impurity solver.
        
        Returns:
            self._SE (ndarray): updated self-energy in fermionic Matsubara frequencies.
        """
        # First get G0(tau)
        G0_tau = np.ndarray((2*Sublattice._Ntau+1,2,2,),dtype=float)
        G0_m_tau = np.ndarray((2*Sublattice._Ntau+1,2,2,),dtype=float)
        # subtracting the leading tail
        for j,iwn in enumerate(Sublattice._iwn_arr):
            self._G0[j,0,0] -= 1.0/( iwn + self._mu0 - Sublattice._hyb_c/iwn ) # 1.0/iwn
        G0_tau[:,0,0] = get_iwn_to_tau_hyb(self._G0[:,0,0],Sublattice._beta,self._mu0,Sublattice._hyb_c) #up cs.get_iwn_to_tau(self._G0[:,0,0],Sublattice._beta)
        G0_m_tau[:,0,0] = get_iwn_to_tau_hyb(self._G0[:,0,0],Sublattice._beta,self._mu0,Sublattice._hyb_c,opt="negative") #up cs.get_iwn_to_tau(self._G0[:,0,0],Sublattice._beta,opt="negative") #
        # for j in range(len(Sublattice._tau_arr)):
        #     G0_tau[j,0,0] += -0.5
        #     G0_m_tau[j,0,0] += 0.5
        for l,iwn in enumerate(Sublattice._iwn_arr):
            self._G0[l,0,0] += 1.0/( iwn + self._mu0 - Sublattice._hyb_c/iwn )
        # Weiss Green's function densities used for Hartree term
        self._n0_up = -1.0*G0_tau[-1,0,0]
        print("n0_up: ", self._n0_up)
        # Self-energy impurity
        for i in range(self._SE_tau.shape[0]):
            self._SE_tau[i,0,0] = G0_tau[i,0,0]*G0_m_tau[i,0,0]*G0_tau[i,0,0] #up
        A = self._n_up*(1.0-self._n_up) / ( self._n0_up*(1.0-self._n0_up) )
        B = ( (1.0-self._n_up)*Sublattice._U + self._mu0 - self._mu ) / ( self._n0_up*(1.0-self._n0_up)*Sublattice._U**2 )
        print("A: ", A, " B: ", B)
        if type_spline=="linear":
            SE_2nd_up = -1.0*Sublattice._U*Sublattice._U*linear_spline_Sigma_tau_to_iwn(self._SE_tau[:,0,0],Sublattice._beta)
        elif type_spline=="cubic":
            cs.Cubic_spline(Sublattice._delta_tau,self._SE_tau[:,0,0])
            # getting the boundary conditions
            der_G0_up = cs.Cubic_spline.get_derivative_FFT_G0(self._G0[:,0,0],Sublattice._iwn_arr,Sublattice._tau_arr,0.0,self._mu0,Sublattice._hyb_c)
            der_G0_m_up = cs.Cubic_spline.get_derivative_FFT_G0(self._G0[:,0,0],Sublattice._iwn_arr,Sublattice._tau_arr,0.0,self._mu0,Sublattice._hyb_c,opt="negative")
            left_der_up = 2.0*der_G0_up[0]*G0_m_tau[0,0,0] + G0_tau[0,0,0]*der_G0_m_up[0]*G0_tau[0,0,0]
            right_der_up = 2.0*der_G0_up[-1]*G0_m_tau[-1,0,0] + G0_tau[-1,0,0]*der_G0_m_up[-1]*G0_tau[-1,0,0]
            
            cs.Cubic_spline.building_matrix_components(left_der_up,right_der_up)
            cs.Cubic_spline.tridiagonal_LU_decomposition()
            cs.Cubic_spline.tridiagonal_LU_solve()
            # Getting m_a and m_c from the b's
            cs.Cubic_spline.construct_coeffs(Sublattice._tau_arr)
            cs.FermionsvsBosons(Sublattice._beta,Sublattice._tau_arr)
            SE_2nd_up, *_ = cs.FermionsvsBosons.fermionic_propagator()
            SE_2nd_up *= -1.0*Sublattice._U*Sublattice._U
            # cleaning
            cs.FermionsvsBosons.reset()
            cs.Cubic_spline.reset()
        
        for i in range(len(Sublattice._iwn_arr)):
            self._SE[i,0,0] = Sublattice._U*(1.0-self._n0_up) + A*SE_2nd_up[i]/( 1.0 - B*SE_2nd_up[i] ) #up
    
    def update_self_energy_doped(self,type_spline : str) -> None:
        """
        Method updating the IPT impurity self-energy in PM scenario. Basically, the Weiss Green's function is transformed into
        G_0(tau) and G_0(-tau) to compute the self-energy Sigma(tau). Then, Sigma(tau) is transformed into Sigma(iwn).

        Parameters:
            type_spline (str): determines the type of spline to be used in the impurity solver.
        
        Returns:
            self._SE (ndarray): updated self-energy in fermionic Matsubara frequencies.
        """
        # First get G0(tau)
        G0_tau = np.ndarray((2*Sublattice._Ntau+1,2,2,),dtype=float)
        G0_m_tau = np.ndarray((2*Sublattice._Ntau+1,2,2,),dtype=float)
        # derivative of Sigma w.r.t tau
        der_Sigma_diwn = np.ndarray((2*Sublattice._Ntau,2,2,),dtype=complex)
        G_der_Sigma_diwn = np.ndarray((2*Sublattice._Ntau,2,2,),dtype=complex)
        # subtracting the leading tail
        for j,iwn in enumerate(Sublattice._iwn_arr):
            self._G0[j,0,0] -= 1.0/( iwn + self._mu0 - Sublattice._hyb_c/iwn ) # 1.0/iwn
        G0_tau[:,0,0] = get_iwn_to_tau_hyb(self._G0[:,0,0],Sublattice._beta,self._mu0,Sublattice._hyb_c) #up cs.get_iwn_to_tau(self._G0[:,0,0],Sublattice._beta)
        G0_m_tau[:,0,0] = get_iwn_to_tau_hyb(self._G0[:,0,0],Sublattice._beta,self._mu0,Sublattice._hyb_c,opt="negative") #up cs.get_iwn_to_tau(self._G0[:,0,0],Sublattice._beta,opt="negative") #
        
        for l,iwn in enumerate(Sublattice._iwn_arr):
            self._G0[l,0,0] += 1.0/( iwn + self._mu0 - Sublattice._hyb_c/iwn )
        # Weiss Green's function densities used for Hartree term
        self._n0_up = -1.0*G0_tau[-1,0,0]
        print("n0_up: ", self._n0_up)
        # Self-energy impurity
        tau_dSE_dtau = np.empty((self._SE_tau.shape[0],),dtype=float) # <-------------------- added for test
        for i in range(self._SE_tau.shape[0]):
            self._SE_tau[i,0,0] = G0_tau[i,0,0]*G0_m_tau[i,0,0]*G0_tau[i,0,0] #up
            
        if type_spline=="linear":
            SE_2nd_up = -1.0*Sublattice._U*Sublattice._U*linear_spline_Sigma_tau_to_iwn(self._SE_tau[:,0,0],Sublattice._beta)
            
        elif type_spline=="cubic":
            cs.Cubic_spline(Sublattice._delta_tau,self._SE_tau[:,0,0])
            # getting the boundary conditions
            der_G0_up = cs.Cubic_spline.get_derivative_FFT_G0(self._G0[:,0,0],Sublattice._iwn_arr,Sublattice._tau_arr,0.0,self._mu0,Sublattice._hyb_c)
            der_G0_m_up = cs.Cubic_spline.get_derivative_FFT_G0(self._G0[:,0,0],Sublattice._iwn_arr,Sublattice._tau_arr,0.0,self._mu0,Sublattice._hyb_c,opt="negative")
            left_der_up = 2.0*der_G0_up[0]*G0_m_tau[0,0,0] + G0_tau[0,0,0]*der_G0_m_up[0]*G0_tau[0,0,0]
            right_der_up = 2.0*der_G0_up[-1]*G0_m_tau[-1,0,0] + G0_tau[-1,0,0]*der_G0_m_up[-1]*G0_tau[-1,0,0]
            for ii in range(2*Sublattice._Ntau+1):
                tau_dSE_dtau[ii] = -1.0*Sublattice._U*Sublattice._U*Sublattice._tau_arr[ii]*(2.0*der_G0_up[ii]*G0_m_tau[ii,0,0] + G0_tau[ii,0,0]*der_G0_m_up[ii]*G0_tau[ii,0,0])
            
            linear_spline_tau_SE_dtau = linear_spline_Sigma_tau_to_iwn(tau_dSE_dtau,Sublattice._beta)
            cs.Cubic_spline.building_matrix_components(left_der_up,right_der_up)
            cs.Cubic_spline.tridiagonal_LU_decomposition()
            cs.Cubic_spline.tridiagonal_LU_solve()
            # Getting m_a and m_c from the b's
            cs.Cubic_spline.construct_coeffs(Sublattice._tau_arr)
            cs.FermionsvsBosons(Sublattice._beta,Sublattice._tau_arr)
            SE_2nd_up, *_ = cs.FermionsvsBosons.fermionic_propagator()
            SE_2nd_up *= -1.0*Sublattice._U*Sublattice._U
            # cleaning
            cs.FermionsvsBosons.reset()
            cs.Cubic_spline.reset()

        for jj,iwn in enumerate(Sublattice._iwn_arr):
            der_Sigma_diwn[jj,0,0] = -Sublattice._beta*self._SE_tau[-1,0,0]/iwn - 1.0/iwn*SE_2nd_up[jj] - 1.0/iwn*linear_spline_tau_SE_dtau[jj]
        # der_Sigma = cs.Cubic_spline.get_derivative_4th_order(list(map(lambda x: x.imag, Sublattice._iwn_arr)), list(map(lambda x: x.imag, SE_2nd_up)))
        # plt.figure(3)
        # plt.plot(list(map(lambda x: x.imag, self._iwn_arr)),der_Sigma,c="grey",ms=2.0,marker='o')
        
        def Luttinger_sum_rule(mu,mu0):
            SE = np.ndarray((2*Sublattice._Ntau,2,2),dtype=complex)
            # setting n0
            n0=0.0
            for i,iwn in enumerate(Sublattice._iwn_arr):
                n0 += ( 1./( iwn + mu0 - self._Hyb[i,0,0] ) - 1./iwn ).real
            n0*=1./Sublattice._beta
            n0+=0.5
            # getting the self-energy coeffs ready
            A = self._n_up*(1.0-self._n_up) / ( n0*(1.0-n0) )
            B = ( (1.0-self._n_up)*Sublattice._U + mu0 - mu ) / ( n0*(1.0-n0)*Sublattice._U**2 )

            for i in range(len(Sublattice._iwn_arr)):
                SE[i,0,0] = Sublattice._U*self._n_up + A*SE_2nd_up[i]/( 1.0 - B*SE_2nd_up[i] ) #up
            
            for i in range(self._SE.shape[0]):
                G_der_Sigma_diwn[i,0,0] = der_Sigma_diwn[i,0,0].real*( 1.0 / ( Sublattice._iwn_arr[i] + self._mu - self._Hyb[i,0,0] - SE[i,0,0] ) ).imag
            
            print("Lutt, mu: ", mu, " mu0: ", mu0, " n0: ", n0)

            return 1.0/Sublattice._beta*np.sum(G_der_Sigma_diwn[:,0,0])
        
        def density_mu(mu,mu0):
            SE = np.ndarray((2*Sublattice._Ntau,2,2),dtype=complex)
            # setting n0
            n0=0.0
            for i,iwn in enumerate(Sublattice._iwn_arr):
                n0 += ( 1./( iwn + mu0 - self._Hyb[i,0,0] ) - 1./iwn ).real
            n0*=1./Sublattice._beta
            n0+=0.5
            # getting the self-energy coeffs ready
            A = self._n_up*(1.0-self._n_up) / ( n0*(1.0-n0) )
            B = ( (1.0-self._n_up)*Sublattice._U + mu0 - mu ) / ( n0*(1.0-n0)*Sublattice._U**2 )

            for i in range(len(Sublattice._iwn_arr)):
                SE[i,0,0] = Sublattice._U*self._n_up + A*SE_2nd_up[i]/( 1.0 - B*SE_2nd_up[i] )
            
            nn=0.0
            for i,iwn in enumerate(Sublattice._iwn_arr):
                nn += ( 1./( iwn + mu - self._Hyb[i,0,0] - SE[i,0,0] ) - 1./iwn ).real
            nn*=1./Sublattice._beta
            nn+=0.5
            print("DOS, mu: ", mu, " mu0: ", mu0, " n0: ", n0, " nn: ", nn, " A: ", A, " B: ", B)
            return A, B, nn-self._n_up
        
        def fs(mu,mu0):
            Lutt = Luttinger_sum_rule(mu,mu0)
            A, B, DOS = density_mu(mu,mu0)
            return A, B, np.array([Lutt, DOS])
 
        def Js(x,y):
            return np.random.rand(2,2)

        steps_taken, self._mu, self._mu0, A, B = broyden_good(Sublattice._U/2.0, 0.0, fs, Js)
        print("steps taken: ", steps_taken)
        print("A: ", A, " B: ", B)
        
        for i in range(len(Sublattice._iwn_arr)):
            self._SE[i,0,0] = Sublattice._U*self._n_up + A*SE_2nd_up[i]/( 1.0 - B*SE_2nd_up[i] ) #up
        
        plt.figure(1)
        plt.title(r"$\frac{\partial \Sigma^{(2)}_{\sigma}(i\omega_n)}{\partial i\omega_n}$ vs $i\omega_n$")
        plt.plot(list(map(lambda x: x.imag, self._iwn_arr)),list(map(lambda x: x.imag, SE_2nd_up)),c="red",ms=2.0,marker='o',label=r"$\sigma=\uparrow$")
        plt.xlabel(r"$i\omega_n$")
        plt.figure(2)
        plt.plot(list(map(lambda x: x.imag, self._iwn_arr)),list(map(lambda x: x.real, G_der_Sigma_diwn[:,0,0])),c="coral",ms=2.0,marker='x')
        plt.show()
        # Fourier transforming back the self-energy to iwn
        for i in range(len(Sublattice._iwn_arr)):
            self._SE[i,0,0] = Sublattice._U*self._n_up + A*SE_2nd_up[i]/( 1.0 - B*SE_2nd_up[i] ) #up
        
        

    def DMFT_AFM(self,dim : int,type_spline : str) -> None:
        """
        Method implementing the core DMFT procedure for AFM scenario. Basically, having computed the self-energy in Matsubara frequencies,
        the impurity Green's function is calculated to obtain the local density for both spins. This density is used later on to compute 
        leading moments that ought to be subtracted (different depending on the spin). Then, the lattice Green's function is constructed 
        to get the local Green's function, by integrating over the former (AFM reduced Brillouin zone). Lastly, both the hybridisation 
        and Weiss Green's functions for next iteration are computed. They will be used inside the solver, and thereof it goes on until 
        convergence is reached for both spin Weiss Green's functions.

        Parameters:
            dim (int): dimension of the system.
            type_spline (str): determines the type of spline to be used in the impurity solver.

        Returns:
            self._Hyb (ndarray): Matsubara hybridisation function set for the next iteration
            self._G0 (ndarray): Matsubara Weiss Green's function set for the next iteration
        """
        # update the self-energy with the updated objects from last iteration
        self.update_self_energy_AFM(type_spline)
        # plt.figure(2)
        # plt.title(r"$\operatorname{Im}\Sigma^{tot}_{\sigma} (i\omega_n)$ vs $i\omega_n$")
        # plt.plot(list(map(lambda x: x.imag,Sublattice._iwn_arr)),list(map(lambda x: x.imag,self._SE[:,0,0])),marker='d',ms=2.5,c='red',label=r"$\sigma=\uparrow$")
        # plt.plot(list(map(lambda x: x.imag,Sublattice._iwn_arr)),list(map(lambda x: x.imag,self._SE[:,1,1])),marker='s',ms=2.5,c='green',label=r"$\sigma=\downarrow$")
        # plt.xlabel(r"$i\omega_n$")
        # plt.legend()
        # plt.show()
        # computing the physical particle density
        for i,iwn in enumerate(Sublattice._iwn_arr):
            self._Gloc[i,0,0] = 1.0/( iwn + self._mu - Sublattice._h - self._Hyb[i,0,0] - self._SE[i,0,0] ) - 1.0/iwn #- 1.0/( iwn + self._mu - Sublattice._h - Sublattice._hyb_c/iwn - Sublattice._U*(self._n_down) - (Sublattice._U**2*self._n_down*(1.0-self._n_down))/(iwn) ) # up
            self._Gloc[i,1,1] = 1.0/( iwn + self._mu + Sublattice._h - self._Hyb[i,1,1] - self._SE[i,1,1] ) - 1.0/iwn #- 1.0/( iwn + self._mu + Sublattice._h - Sublattice._hyb_c/iwn - Sublattice._U*(self._n_up) - (Sublattice._U**2*self._n_up*(1.0-self._n_up))/(iwn) ) # down
        self._Gloc_tau[:,0,0] = cs.get_iwn_to_tau(self._Gloc[:,0,0],Sublattice._beta) #get_iwn_to_tau_hyb(G_imp_iwn[:,0,0],Sublattice._beta,self._mu-Sublattice._h-Sublattice._U*(self._n_down),Sublattice._hyb_c+Sublattice._U**2*self._n_down*(1.0-self._n_down))
        self._Gloc_tau[:,1,1] = cs.get_iwn_to_tau(self._Gloc[:,1,1],Sublattice._beta) #get_iwn_to_tau_hyb(G_imp_iwn[:,1,1],Sublattice._beta,self._mu+Sublattice._h-Sublattice._U*(self._n_up),Sublattice._hyb_c+Sublattice._U**2*self._n_up*(1.0-self._n_up))
        for i in range(len(Sublattice._tau_arr)):
            self._Gloc_tau[i,0,0] += -0.5
            self._Gloc_tau[i,1,1] += -0.5
        for i,iwn in enumerate(Sublattice._iwn_arr):
            self._Gloc[i,0,0] += 1.0/iwn
            self._Gloc[i,1,1] += 1.0/iwn
        self._n_up = -1.0*self._Gloc_tau[-1,0,0]
        self._n_down = -1.0*self._Gloc_tau[-1,1,1]
        print("impurity n_up: ", self._n_up, " and n_down: ", self._n_down)
        G_alpha_beta = np.ndarray((2,2,),dtype=complex)
        G_alpha_beta_inv = np.ndarray((2*Sublattice._Ntau,2,2,),dtype=complex)
        GAA_latt_up_k = np.empty((Sublattice._Nk,),dtype=complex)
        GAA_latt_down_k = np.empty((Sublattice._Nk,),dtype=complex)
        GAB_latt_up_k = np.empty((Sublattice._Nk,),dtype=complex)
        for i,iwn in enumerate(Sublattice._iwn_arr):
            if dim==1:
                for j,k in enumerate(Sublattice._k_AFM_arr):
                    GAA_latt_up_k[j] = 1.0/( iwn + self._mu - Sublattice._h - self._SE[i,0,0] - epsilonk_1D(k)**2/( iwn + self._mu + Sublattice._h - self._SE[i,1,1] ) )
                    GAA_latt_down_k[j] = 1.0/( iwn + self._mu + Sublattice._h - self._SE[i,1,1] - epsilonk_1D(k)**2/( iwn + self._mu - Sublattice._h - self._SE[i,0,0] ) )
                    GAB_latt_up_k[j] = epsilonk_1D(k)/( ( iwn + self._mu - Sublattice._h - self._SE[i,0,0] )*( iwn + self._mu + Sublattice._h - self._SE[i,1,1] ) - epsilonk_1D(k)**2 )
                GAA_up_loc = 1.0/(np.pi)*np.trapz(GAA_latt_up_k,Sublattice._k_AFM_arr) #up
                GAA_down_loc = 1.0/(np.pi)*np.trapz(GAA_latt_down_k,Sublattice._k_AFM_arr) #down
                GAB_up_loc = 1.0/(np.pi)*np.trapz(GAB_latt_up_k,Sublattice._k_AFM_arr) # off-diagonal terms
            # solving for inverse Gloc up and down
            G_alpha_beta[0,0]=GAA_up_loc; G_alpha_beta[0,1]=GAB_up_loc
            G_alpha_beta[1,0]=GAB_up_loc; G_alpha_beta[1,1]=GAA_down_loc
            G_alpha_beta_inv[i,:,:] = np.linalg.inv(G_alpha_beta)
    
        # update the hybdridisation function and set Weiss Green's function for the next iteration
        alpha = 0.1
        for i,iwn in enumerate(Sublattice._iwn_arr):
            self._Hyb[i,0,0] = (1.0-alpha)*(iwn + self._mu - Sublattice._h - self._SE[i,0,0] - G_alpha_beta_inv[i,0,0]) + alpha*(self._Hyb[i,0,0]) # up
            self._G0[i,0,0] = 1.0/( iwn + self._mu - Sublattice._h - self._Hyb[i,0,0] ) # up Have to search for mu away from half-filling
            self._Hyb[i,1,1] = (1.0-alpha)*(iwn + self._mu + Sublattice._h - self._SE[i,1,1] - G_alpha_beta_inv[i,1,1]) + alpha*(self._Hyb[i,1,1]) # down
            self._G0[i,1,1] = 1.0/( iwn + self._mu + Sublattice._h - self._Hyb[i,1,1] ) # down

    def DMFT(self,dim : int,type_spline : str) -> None:
        """
        Method implementing the core DMFT procedure for PM scenario. Basically, having computed the self-energy in Matsubara frequencies,
        the impurity Green's function is calculated to obtain the local density. This density is used later on to compute leading moments
        that ought to be subtracted. Then, the lattice Green's function is constructed to get the local Green's function, by integrating over
        the former. Lastly, both the hybridisation and Weiss Green's functions for next iteration are computed. They will be used inside the
        solver, and thereof it goes on until convergence is reached.

        Parameters:
            dim (int): dimension of the system
            type_spline (str): determines the type of spline to be used in the impurity solver

        Returns:
             self._Hyb (ndarray): Matsubara hybridisation function set for the next iteration
             self._G0 (ndarray): Matsubara Weiss Green's function set for the next iteration
        """
        # update the self-energy with the updated objects from last iteration
        self.update_self_energy_doped(type_spline)
        # computing the physical particle density
        for i,iwn in enumerate(Sublattice._iwn_arr):
            self._Gloc[i,0,0] = 1.0/( iwn + self._mu - self._Hyb[i,0,0] - self._SE[i,0,0] ) - 1.0/iwn# 1.0/( iwn + self._mu - Sublattice._hyb_c/iwn - Sublattice._U*(self._n_up) - (Sublattice._U**2*self._n_up*(1.0-self._n_up))/(iwn) ) # 1.0/iwn
        self._Gloc_tau[:,0,0] = cs.get_iwn_to_tau(self._Gloc[:,0,0],Sublattice._beta) #get_iwn_to_tau_hyb(self._Gloc[:,0,0],Sublattice._beta,self._mu-Sublattice._U*(self._n_up),Sublattice._hyb_c+Sublattice._U**2*self._n_up*(1.0-self._n_up)) #
        for j in range(len(Sublattice._tau_arr)):
            self._Gloc_tau[j,0,0] += -0.5
        self._n_up = -1.0*self._Gloc_tau[-1,0,0]
        print("impurity n_up: ", self._n_up)
        if dim==1:
            for i,iwn in enumerate(Sublattice._iwn_arr):
                funct_integrate_re = lambda k: ( 1.0/( iwn + self._mu - self._SE[i,0,0] - epsilonk_1D(k) ) ).real
                funct_integrate_im = lambda k: ( 1.0/( iwn + self._mu - self._SE[i,0,0] - epsilonk_1D(k) ) ).imag
                self._Gloc[i,0,0] = 1.0/(2.0*np.pi)*( quad(funct_integrate_re,-np.pi,np.pi)[0] + 1.0j*quad(funct_integrate_im,-np.pi,np.pi)[0] )
        elif dim==2:
            for i,iwn in enumerate(Sublattice._iwn_arr):
                print("i: ", i)
                integrate_kx = np.empty((Sublattice._Nk,),dtype=complex)
                for m,kx in enumerate(Sublattice._k_arr):
                    integrate_ky = np.empty((Sublattice._Nk,),dtype=complex)
                    for n,ky in enumerate(Sublattice._k_arr):
                        integrate_ky[n] = 1.0/( iwn + self._mu - self._SE[i,0,0] - epsilonk_2D(kx,ky) )
                    integrate_kx[m] = simps(integrate_ky,Sublattice._k_arr)
                self._Gloc[i,0,0] = 1.0/( (2.0*np.pi)**2 ) * simps(integrate_kx,Sublattice._k_arr)

        # update the hybdridisation function and set Weiss Green's function for the next iteration
        alpha = 0.1
        for i,iwn in enumerate(Sublattice._iwn_arr):
            self._Hyb[i,0,0] = (1.0-alpha)*( iwn + self._mu - self._SE[i,0,0] - 1.0/self._Gloc[i,0,0] ) + alpha*(self._Hyb[i,0,0])
            self._G0[i,0,0] = 1.0/( iwn + self._mu0 - self._Hyb[i,0,0] )

    def dbl_occupancy(self):
        """
        Method computing the double occupancy in the PM state.
        """
        D_iwn = np.empty((2*Sublattice._Ntau,),dtype=complex)
        for j in range(len(Sublattice._iwn_arr)):
            D_iwn[j] = (self._SE[j,0,0]-Sublattice._U*(1.0-self._n0_up))*self._Gloc[j,0,0]
        D = (1.0-self._n0_up)*self._n0_up + 1.0/Sublattice._U/Sublattice._beta*np.sum(D_iwn).real
        print("double occupancy: ", D)

    @staticmethod
    def update_mu(arr_mu : np.ndarray, mu : float) -> float:
        """
        Static method that updates the impurity/Weiss chemical potential to match the target density aimed at. This function is called 
        by the false_position_method function in the PM scenario.

        Parameters:
            arr_mu (ndarray): chunck of the denominator of the impurity/Weiss Green's function defined in Matsubara frequencies.
            mu (float): impurity/Weiss chemical potential guess
        
        Returns:
            nn (float): impurity/Weiss particle density for guessed mu
        """
        nn=0.0
        for i,iwn in enumerate(Sublattice._iwn_arr):
            nn += ( 1./( arr_mu[i,0,0] + mu ) - 1./iwn ).real
        nn*=1./Sublattice._beta
        nn+=0.5

        return nn

    @staticmethod
    def update_mu_AFM(arr_mu : np.ndarray, mu : float) -> float:
        """
        Static method that updates the impurity/Weiss chemical potential to match the target density aimed at. This function is called 
        by the false_position_method function in the AFM scenario.

        Parameters:
            arr_mu (ndarray): chunck of the denominator of the impurity/Weiss Green's function defined in Matsubara frequencies.
            mu (float): impurity/Weiss chemical potential guess
        
        Returns:
            nn (float): averaged (over different spins) impurity/Weiss particle density for guessed mu
        """
        nn=0.0
        for i,iwn in enumerate(Sublattice._iwn_arr):
            nn += ( 1./( arr_mu[i,0,0] + mu ) - 1./iwn ).real
            nn += ( 1./( arr_mu[i,1,1] + mu ) - 1./iwn ).real
        nn*=1./Sublattice._beta
        nn+=1.0

        return nn/2.0
            
    def update_densities(self, it : float, n_target : float) -> None:
        """
        Method that updates the impurity and Weiss chemical potentials to match the target density aimed at. This function is called 
        in the PM scenario.

        Parameters:
            it (float): iteration number.
            n_target (float): target density of particles on impurity
        
        Returns:
            self._mu0 (float): Weiss chemical potential matching n_target within tolerance set.
            self._mu (float): impurity chemical potential matching n_target within tolerance set.
        """
        n0_mu0 = np.ndarray((2*Sublattice._Ntau,2,2,),dtype=complex)
        n_mu = np.ndarray((2*Sublattice._Ntau,2,2,),dtype=complex)
        for i,iwn in enumerate(Sublattice._iwn_arr):
            n_mu[i,0,0] = iwn - self._Hyb[i,0,0] - self._SE[i,0,0]
            n0_mu0[i,0,0] = iwn - self._Hyb[i,0,0]
        
        get_new_n0 = lambda mu0: Sublattice.update_mu(n0_mu0,mu0)-n_target
        new_mu0 = false_position_method(get_new_n0,Sublattice._U/2.0-20.0,Sublattice._U/2.0+20.0,0.001)
        get_new_n = lambda mu: Sublattice.update_mu(n_mu,mu)-n_target
        new_mu = false_position_method(get_new_n,Sublattice._U/2.0-20.0,Sublattice._U/2.0+20.0,0.001)

        print("new mu: ", new_mu, " new mu0: ", new_mu0)
        # Updating the chemical potentials according to the target density
        self._mu0 = new_mu0
        if it>0:
            self._mu = new_mu

    def update_densities_AFM(self, it : float, n_target : float) -> None:
        """
        Method that updates the impurity and Weiss chemical potentials to match the target density aimed at. This function is called 
        in the AFM scenario.

        Parameters:
            it (float): iteration number.
            n_target (float): target density of particles on impurity
        
        Returns:
            self._mu0 (float): Weiss chemical potential matching n_target within tolerance set.
            self._mu (float): impurity chemical potential matching n_target within tolerance set.
        """
        n0_mu0 = np.ndarray((2*Sublattice._Ntau,2,2,),dtype=complex)
        n_mu = np.ndarray((2*Sublattice._Ntau,2,2,),dtype=complex)
        for i,iwn in enumerate(Sublattice._iwn_arr):
            n_mu[i,0,0] = iwn - self._Hyb[i,0,0] - self._SE[i,0,0]
            n_mu[i,1,1] = iwn - self._Hyb[i,1,1] - self._SE[i,1,1]
            n0_mu0[i,0,0] = iwn - Sublattice._h - self._Hyb[i,0,0]
            n0_mu0[i,1,1] = iwn + Sublattice._h - self._Hyb[i,1,1]
        
        get_new_n0 = lambda mu0: Sublattice.update_mu_AFM(n0_mu0,mu0)-n_target
        new_mu0 = false_position_method(get_new_n0,Sublattice._U/2.0-10.0,Sublattice._U/2.0+10.0,0.001)
        get_new_n = lambda mu: Sublattice.update_mu_AFM(n_mu,mu)-n_target
        new_mu = false_position_method(get_new_n,Sublattice._U/2.0-10.0,Sublattice._U/2.0+10.0,0.001)
        print("new mu: ", new_mu, " and new mu0: ", new_mu0)
        # Updating the chemical potentials according to the target density
        self._mu0 = new_mu0
        if it>1:
            self._mu = new_mu

    
    def save_Gloc_AFM(self, it : int, dim : int, type_spline : str) -> None:
        """
        Method to save the local Green's function in the AFM scenario (as a function of iwn).

        Parameters:
            it (float): iteration number.
            dim (int): dimension of the system.
            type_spline (str): type of spline used in impurity solver.
        
        Returns:
            None
        """
        directory_container = "{0:d}D_U_{1:.5f}_beta_{2:.5f}_n_{3:.5f}_Ntau_{4:d}_{5}".format(dim,Sublattice._U,Sublattice._beta,0.5,Sublattice._Ntau,type_spline)
        if not os.path.isdir("./data_test_IPT/"+directory_container):
            os.mkdir("./data_test_IPT/"+directory_container)
        filename = "./data_test_IPT/"+directory_container+"/Green_loc_{0:d}".format(dim)+"D_"+"{0}".format(type_spline)+"_AFM_U_{0:.5f}".format(Sublattice._U)+"_beta_{0:.5f}".format(Sublattice._beta)+"_N_tau_{0:d}".format(Sublattice._Ntau)+"_Nit_{0:d}".format(it)+".dat"
        with open(filename,"w+") as f:
            for i in range(len(Sublattice._iwn_arr)):
                if i==0:
                    f.write("iwn\t\tRe Gloc up\t\tIm Gloc up\t\tRe Gloc down\t\tIm Gloc down\n")
                f.write("{0:.8f}\t\t{1:.8f}\t\t{2:.8f}\t\t{3:.8f}\t\t{4:.8f}\n".format(Sublattice._iwn_arr[i].imag,self._Gloc[i,0,0].real,self._Gloc[i,0,0].imag,self._Gloc[i,1,1].real,self._Gloc[i,1,1].imag))
        f.close()
    

    def save_SE_AFM(self, it : int, dim : int, type_spline : str) -> None:
        """
        Method to save the self-energy function in the AFM scenario (as a function of iwn).

        Parameters:
            it (float): iteration number.
            dim (int): dimension of the system.
            type_spline (str): type of spline used in impurity solver.
        
        Returns:
            None
        """
        directory_container = "{0:d}D_U_{1:.5f}_beta_{2:.5f}_n_{3:.5f}_Ntau_{4:d}_{5}".format(dim,Sublattice._U,Sublattice._beta,0.5,Sublattice._Ntau,type_spline)
        if not os.path.isdir("./data_test_IPT/"+directory_container):
            os.mkdir("./data_test_IPT/"+directory_container)
        filename = "./data_test_IPT/"+directory_container+"/Self_energy_{0:d}".format(dim)+"D_"+"{0}".format(type_spline)+"_AFM_U_{0:.5f}".format(Sublattice._U)+"_beta_{0:.5f}".format(Sublattice._beta)+"_N_tau_{0:d}".format(Sublattice._Ntau)+"_Nit_{0:d}".format(it)+".dat"
        with open(filename,"w+") as f:
            for i in range(len(Sublattice._iwn_arr)):
                if i==0:
                    f.write("iwn\t\tRe SE up\t\tIm SE up\t\tRe SE down\t\tIm SE down\n")
                f.write("{0:.8f}\t\t{1:.8f}\t\t{2:.8f}\t\t{3:.8f}\t\t{4:.8f}\n".format(Sublattice._iwn_arr[i].imag,self._SE[i,0,0].real,self._SE[i,0,0].imag,self._SE[i,1,1].real,self._SE[i,1,1].imag))
        f.close()

    def save_Gloc_tau_AFM(self, it : int, dim : int, type_spline : str) -> None:
        """
        Method to save the local Green's function in the AFM scenario (as a function of tau). Particularly useful
        when used in conjunction with the program plot_phase_boundary.py.

        Parameters:
            it (float): iteration number.
            dim (int): dimension of the system.
            type_spline (str): type of spline used in impurity solver.
        
        Returns:
            None
        """
        # Saving files at each iteration
        directory_container = "{0:d}D_U_{1:.5f}_beta_{2:.5f}_n_{3:.5f}_Ntau_{4:d}_{5}".format(dim,Sublattice._U,Sublattice._beta,0.5,Sublattice._Ntau,type_spline)
        if not os.path.isdir("./data_test_IPT/"+directory_container):
            os.mkdir("./data_test_IPT/"+directory_container)
        filename = "./data_test_IPT/"+directory_container+"/Green_loc_tau_{0:d}".format(dim)+"D_"+"{0}".format(type_spline)+"_U_{0:.5f}".format(Sublattice._U)+"_beta_{0:.5f}".format(Sublattice._beta)+"_N_tau_{0:d}".format(Sublattice._Ntau)+"_Nit_{0:d}".format(it)+".dat"
        with open(filename,"w+") as f:
            for i in range(len(Sublattice._tau_arr)):
                if i==0:
                    f.write("tau\t\tGloc up\t\tGloc down\n")
                f.write("{0:.8f}\t\t{1:.8f}\t\t{2:.8f}\n".format(Sublattice._tau_arr[i],self._Gloc_tau[i,0,0],self._Gloc_tau[i,1,1]))
        f.close()
    

    def save_Gloc(self, it : int, dim : int, type_spline : float) -> None:
        """
        Method to save the local Green's function in the PM scenario (as a function of iwn).

        Parameters:
            it (float): iteration number.
            dim (int): dimension of the system.
            type_spline (str): type of spline used in impurity solver.
        
        Returns:
            None
        """
        filename = "./Green_loc_{0:d}".format(dim)+"D_"+"{0}".format(type_spline)+"_U_{0:.5f}".format(Sublattice._U)+"_beta_{0:.5f}".format(Sublattice._beta)+"_N_tau_{0:d}".format(Sublattice._Ntau)+"_Nit_{0:d}".format(it)+".dat"
        with open(filename,"w+") as f:
            for i in range(len(Sublattice._iwn_arr)):
                if i==0:
                    f.write("iwn\t\tRe Gloc up\t\tIm Gloc up\n")
                f.write("{0:.8f}\t\t{1:.8f}\t\t{2:.8f}\n".format(Sublattice._iwn_arr[i].imag,self._Gloc[i,0,0].real,self._Gloc[i,0,0].imag))
        f.close()

    def save_SE(self, it : int, dim : int, type_spline : str) -> None:
        """
        Method to save the self-energy in the PM scenario (as a function of iwn).

        Parameters:
            it (float): iteration number.
            dim (int): dimension of the system.
            type_spline (str): type of spline used in impurity solver.
        
        Returns:
            None
        """
        filename = "./Self_energy_{0:d}".format(dim)+"D_"+"{0}".format(type_spline)+"_U_{0:.5f}".format(Sublattice._U)+"_beta_{0:.5f}".format(Sublattice._beta)+"_N_tau_{0:d}".format(Sublattice._Ntau)+"_Nit_{0:d}".format(it)+".dat"
        with open(filename,"w+") as f:
            for i in range(len(Sublattice._iwn_arr)):
                if i==0:
                    f.write("iwn\t\tRe SE up\t\tIm SE up\n")
                f.write("{0:.8f}\t\t{1:.8f}\t\t{2:.8f}\n".format(Sublattice._iwn_arr[i].imag,self._SE[i,0,0].real,self._SE[i,0,0].imag))
        f.close()
    
    
