import numpy as np
from math import isclose
import matplotlib.pyplot as plt
from copy import deepcopy

def get_iwn_to_tau(G_iwn, beta : float, type_of="Simple"):
    """
    This function computes transformation from iwn to tau for fermionic functions (type_of="Simple") or
    for its derivative (type_of=Derivative). The leading moments have been subtracted.

    Parameters:
        G_iwn (array): function defined in fermionic Matsubara frequencies to transform to imaginary time.
        beta (float): inverse temperature

    Returns:
        tau_final_G (array): function defined in imaginary time (tau)
    """
    MM = len(G_iwn) # N = M/2
    tau_final_G = np.zeros(MM+1,dtype=float)
    # FFT
    tau_resolved_G = np.fft.fft(G_iwn)
    for i in range(MM):
        tau_final_G[i] = ( np.exp( -1.j * np.pi * i * ( 1.0/(MM) - 1.0 ) )*tau_resolved_G[i] ).real

    if type_of=="Simple":
        for i in range(MM):
            tau_final_G[MM] += ( np.exp( -1.j * np.pi * (1.0-(MM)) )*G_iwn[i] ).real
        tau_final_G *= (1./beta)

    elif type_of=="Derivative":
        tau_final_G *= (1./beta)

    return tau_final_G

class Cubic_spline(object):
    """
    Class for cubic spline.
    """
    _one_instance = True
    _delta_beta = 0.0
    _funct = np.array([],dtype=float)
    # coeffs of spline
    _ma = np.array([],dtype=float)
    _mb = np.array([],dtype=float)
    _mc = np.array([],dtype=float)
    # tridiagonal matrix elements
    _subdiagonal = np.array([],dtype=float)
    _diagonal = np.array([],dtype=float)
    _superdiagonal = np.array([],dtype=float)
    _rhs = np.array([],dtype=float)

    def __init__(self, step : float, funct):
        if Cubic_spline._one_instance:
            Cubic_spline._one_instance=False
            Cubic_spline._delta_beta = step
            Cubic_spline._funct = np.asarray(funct)
    
    @classmethod
    def reset(cls) -> None:
        """
        Needed to reset the internal data to Cubic_spline class.
        """
        cls._one_instance = True
        
        return None

    @classmethod
    def building_matrix_components(cls, left_der : float, right_der : float) -> None:
        """
        It is assumed here that the beta step is constant.
        """
        size = len(cls._funct)
        cls._subdiagonal = np.empty((size-1,),dtype=float)
        cls._superdiagonal = np.empty((size-1,),dtype=float)
        cls._diagonal = np.empty((size,),dtype=float)
        cls._rhs = np.empty((size,),dtype=float)
        for i in range(1,size-1):
            cls._subdiagonal[i-1] = 1.0*cls._delta_beta
            cls._diagonal[i] = 4.0*cls._delta_beta
            cls._superdiagonal[i] = 1.0*cls._delta_beta
            cls._rhs[i] = (3.0/cls._delta_beta)*( cls._funct[i+1] - 2.0*cls._funct[i] + cls._funct[i-1] )
        
        # Boundary conditions
        # Left
        cls._superdiagonal[0] = 1.0*cls._delta_beta
        cls._diagonal[0] = 2.0*cls._delta_beta
        cls._rhs[0] = 3.0*( (cls._funct[1]-cls._funct[0])/cls._delta_beta - left_der )
        # Right
        cls._subdiagonal[size-2] = 1.0*cls._delta_beta
        cls._diagonal[size-1] = 2.0*cls._delta_beta
        cls._rhs[size-1] = 3.0*( right_der - (cls._funct[size-1] - cls._funct[size-2])/cls._delta_beta )
        
        return None
        
    @classmethod
    def tridiagonal_LU_decomposition(cls) -> None:
        assert ( len(cls._subdiagonal)==len(cls._superdiagonal) ) and ( len(cls._subdiagonal)==(len(cls._diagonal)-1) ), "Error in sizes of the tridiagonal matrices."

        size = len(cls._subdiagonal)
    
        for i in range(size):
            if cls._diagonal[i]==0.0:
                raise Exception("Cannot have zeros along the diagonal...")
            else:
                cls._subdiagonal[i] /=  cls._diagonal[i]
                cls._diagonal[i+1] -= cls._subdiagonal[i]*cls._superdiagonal[i]
            if cls._diagonal[size]==0.0:
                raise Exception("Cannot have zeros along the diagonal...see last element.")
        return None

    @classmethod
    def tridiagonal_LU_solve(cls) -> None:
        size = len(cls._diagonal)
        cls._mb = np.zeros((size,),dtype=float)
        cls._mb[0] = cls._rhs[0]
    
        for i in range(1,size):
            cls._mb[i] = cls._rhs[i] - cls._subdiagonal[i-1]*cls._mb[i-1]

        cls._mb[size-1] /= cls._diagonal[size-1]
        for i in range(size-2,-1,-1):
            cls._mb[i] -= cls._superdiagonal[i]*cls._mb[i+1]
            cls._mb[i] /= cls._diagonal[i]

        # deleting the no-longer necessary stuff
    
        del cls._subdiagonal
        del cls._diagonal
        del cls._superdiagonal
        del cls._rhs

        return None
    
    @classmethod
    def construct_coeffs(cls, beta_array) -> None:
        size = len(beta_array)
        cls._ma = np.empty((size,),dtype=float)
        cls._mc = np.empty((size,),dtype=float)
        for j in range(size-1):
            cls._ma[j] = ( cls._mb[j+1]-cls._mb[j] )/3.0/cls._delta_beta
            cls._mc[j] = ( cls._funct[j+1]-cls._funct[j] )/cls._delta_beta - 1.0/3.0*( cls._mb[j+1]+2.0*cls._mb[j] )*cls._delta_beta
        
        cls._ma[size-1] = 0.0
        cls._mc[size-1] = 3.0*cls._ma[len(beta_array)-2]*cls._delta_beta*cls._delta_beta+2.0*cls._mb[len(beta_array)-2]*cls._delta_beta+cls._mc[len(beta_array)-2]
        
        return None

    @classmethod
    def get_spline(cls, beta_array, x : float, opt="no_derivative") -> float:
        """
        Computing the cubic spline of GG(tau)
        """
        beta_array_mod = beta_array - x
        idx = (np.where(beta_array_mod >= 0.0)[0])[0] - 1
        # print("For x ", x, " idx is ", idx, " such that ", beta_array[idx])
        if opt=="no_derivative":
            interpol = cls._ma[idx]*(x-beta_array[idx])**3 + cls._mb[idx]*(x-beta_array[idx])**2 + cls._mc[idx]*(x-beta_array[idx]) + cls._funct[idx]
        elif opt=="first_derivative":
            interpol = 3.0*cls._ma[idx]*(x-beta_array[idx])**2 + 2.0*cls._mb[idx]*(x-beta_array[idx]) + cls._mc[idx]
        elif opt=="second_derivative":
            interpol = 6.0*cls._ma[idx]*(x-beta_array[idx]) + cls._mb[idx]

        return interpol

    @staticmethod
    def get_derivative(x_arr, y_arr):
        """
        Static method that returns an array of N-2 elements representing the derivative of an input array of N points. Special care has to be 
        made to treat the boudary points (the ones relevant in fact for the spline). The ideal is to use a dense beta array.
        """
        assert len(x_arr)==len(y_arr), "The lengths of x_arr and y_arr have to be the same."
        der_f = np.empty((len(x_arr)-2,),dtype=float)
        right_most_val = 0.0
        left_most_val = 0.0
        for i in range(len(x_arr)-2):
            der_f[i] = ( y_arr[i+2] - y_arr[i] ) / ( x_arr[i+2] - x_arr[i] )

        left_most_2nd_der = ( y_arr[2] - 2.0*y_arr[1] + y_arr[0] ) / ( ( x_arr[1] - x_arr[0] )**2 )
        right_most_2nd_der = ( y_arr[-1] - 2.0*y_arr[-2] + y_arr[-3] ) / ( ( x_arr[-1] - x_arr[-2] )**2 )

        left_most_val = der_f[0] - left_most_2nd_der * (x_arr[1]-x_arr[0]) ## Assuming x_arr is ordered and evenly spaced...
        right_most_val = der_f[-1] + right_most_2nd_der * (x_arr[-1]-x_arr[-2])

        # print("left_der ", left_most_val)
        # print("right_der ", right_most_val)
        
        der_f = np.insert(der_f,0,left_most_val)
        der_f = np.append(der_f,right_most_val)

        return der_f

    @staticmethod
    def get_derivative_4th_order(x_arr, y_arr):
        """
        Static method that returns an array of N-2 elements representing the derivative of an input array of N points. Special care has to be 
        made to treat the boudary points (the ones relevant in fact for the spline). The ideal is to use a dense beta array.
        """
        assert len(x_arr)==len(y_arr), "The lengths of x_arr and y_arr have to be the same."
        der_f = np.empty((len(x_arr)-4,),dtype=float)
        right_most_val = 0.0
        left_most_val = 0.0
        h = x_arr[1] - x_arr[0] # ordered array
        for i in range(2,len(x_arr)-2):
            der_f[i-2] = ( 1.0/12.0*y_arr[i-2] - 2.0/3.0*y_arr[i-1] + 2.0/3.0*y_arr[i+1] - 1.0/12.0*y_arr[i+2] ) / h

        left_most_val = ( -25.0/12.0*y_arr[0] + 4.0*y_arr[1] - 3.0*y_arr[2] + 4.0/3.0*y_arr[3] - 1.0/4.0*y_arr[4] ) / h ## Assuming x_arr is ordered and evenly spaced...
        second_left_most_val = ( -25.0/12.0*y_arr[1] + 4.0*y_arr[2] - 3.0*y_arr[3] + 4.0/3.0*y_arr[4] - 1.0/4.0*y_arr[5] ) / h
        right_most_val = ( 25.0/12.0*y_arr[-1] - 4.0*y_arr[-2] + 3.0*y_arr[-3] - 4.0/3.0*y_arr[-4] + 1.0/4.0*y_arr[-5] ) / h
        second_right_most_val = ( 25.0/12.0*y_arr[-2] - 4.0*y_arr[-3] + 3.0*y_arr[-4] - 4.0/3.0*y_arr[-5] + 1.0/4.0*y_arr[-6] ) / h

        # print("left_der ", left_most_val)
        # print("right_der ", right_most_val)
        
        der_f = np.insert(der_f,0,second_left_most_val)
        der_f = np.insert(der_f,0,left_most_val)
        der_f = np.append(der_f,second_right_most_val)
        der_f = np.append(der_f,right_most_val)

        return der_f

    @staticmethod
    def get_derivative_6th_order(x_arr, y_arr):
        """
        Static method that returns an array of N-2 elements representing the derivative of an input array of N points. Special care has to be 
        made to treat the boudary points (the ones relevant in fact for the spline). The ideal is to use a dense beta array.
        """
        assert len(x_arr)==len(y_arr), "The lengths of x_arr and y_arr have to be the same."
        der_f = np.empty((len(x_arr)-6,),dtype=float)
        right_most_val = 0.0
        left_most_val = 0.0
        h = x_arr[1] - x_arr[0] # ordered array
        for i in range(3,len(x_arr)-3):
            der_f[i-3] = ( -1.0/60.0*y_arr[i-3] + 3.0/20.0*y_arr[i-2] - 3.0/4.0*y_arr[i-1] + 3.0/4.0*y_arr[i+1] - 3.0/20.0*y_arr[i+2] + 1.0/60.0*y_arr[i+3] ) / h

        left_most_val = ( -49.0/20.0*y_arr[0] + 6.0*y_arr[1] - 15.0/2.0*y_arr[2] + 20.0/3.0*y_arr[3] - 15.0/4.0*y_arr[4] + 6.0/5.0*y_arr[5] - 1.0/6.0*y_arr[6] ) / h ## Assuming x_arr is ordered and evenly spaced...
        second_left_most_val = ( -49.0/20.0*y_arr[1] + 6.0*y_arr[2] - 15.0/2.0*y_arr[3] + 20.0/3.0*y_arr[4] - 15.0/4.0*y_arr[5] + 6.0/5.0*y_arr[6] - 1.0/6.0*y_arr[7] ) / h
        third_left_most_val = ( -49.0/20.0*y_arr[2] + 6.0*y_arr[3] - 15.0/2.0*y_arr[4] + 20.0/3.0*y_arr[5] - 15.0/4.0*y_arr[6] + 6.0/5.0*y_arr[7] - 1.0/6.0*y_arr[8] ) / h
        right_most_val = ( 49.0/20.0*y_arr[-1] - 6.0*y_arr[-2] + 15.0/2.0*y_arr[-3] - 20.0/3.0*y_arr[-4] + 15.0/4.0*y_arr[-5] - 6.0/5.0*y_arr[-6] + 1.0/6.0*y_arr[-7] ) / h
        second_right_most_val = ( 49.0/20.0*y_arr[-2] - 6.0*y_arr[-3] + 15.0/2.0*y_arr[-4] - 20.0/3.0*y_arr[-5] + 15.0/4.0*y_arr[-6] - 6.0/5.0*y_arr[-7] + 1.0/6.0*y_arr[-8] ) / h
        third_right_most_val = ( 49.0/20.0*y_arr[-3] - 6.0*y_arr[-4] + 15.0/2.0*y_arr[-5] - 20.0/3.0*y_arr[-6] + 15.0/4.0*y_arr[-7] - 6.0/5.0*y_arr[-8] + 1.0/6.0*y_arr[-9] ) / h

        # print("left_der ", left_most_val)
        # print("right_der ", right_most_val)
        
        der_f = np.insert(der_f,0,third_left_most_val)
        der_f = np.insert(der_f,0,second_left_most_val)
        der_f = np.insert(der_f,0,left_most_val)
        der_f = np.append(der_f,third_right_most_val)
        der_f = np.append(der_f,second_right_most_val)
        der_f = np.append(der_f,right_most_val)

        return der_f

    @staticmethod
    def get_derivative_FFT(G_k_iwn, funct_dispersion, iwn_arr, k_arr, U : float, beta : float, mu : float, q=0.0, opt="positive"):
        """
        This method computes the derivatives of G(tau) at boundaries for the cubic spline. Recall that only positive imaginary time
        is used, i.e 0 < tau < beta. It takes care of subtracting the leading moments for smoother results.
        
        Parameters:
            G_k_iwn (complex np.ndarray): Green's function mesh over (k,iwn)-space.
            funct_dispersion (function): dispersion relation of the tight-binding model.
            iwn_arr (complex np.array): Fermionic Matsubara frequencies.
            k_arr (float np.array): k-space array.
            U (float): Hubbard local interaction.
            beta (float): Inverse temperature.
            mu (float): chemical potential.
            q (float): incoming momentum. Defaults q=0.0.
            opt (str): positive or negative imaginary-time Green's function derivated. Takes in "positive" of "negative". 
            Defaults opt="postitive".
        
        Returns:
            dG_tau_for_k (float np.ndarray): imaginary-time Green's function derivative mesh over (k,tau)-space.

        """
        dG_tau_for_k = np.empty((len(k_arr),len(iwn_arr)+1),dtype=float)
        G_k_iwn_tmp = deepcopy(G_k_iwn)
        beta_arr = np.linspace(0.0,beta,len(iwn_arr)+1)
        # Subtracting the leading moments of the Green's function
        for j,iwn in enumerate(iwn_arr):
            for l,k in enumerate(k_arr):
                moments = 1.0/(iwn) + funct_dispersion(k+q)/(iwn*iwn) + (U**2/4.0 + funct_dispersion(k+q)**2)/(iwn**3)
                # G
                G_k_iwn_tmp[l,j] -= moments
                if opt=="negative":
                    G_k_iwn_tmp[l,j] = np.conj(G_k_iwn_tmp[l,j]) # because G(-tau)
            G_k_iwn_tmp[:,j] *= -1.0*iwn
        
        # Calculating the dG/dtau objects
        for l,k in enumerate(k_arr):
            dG_tau_for_k[l,:] = get_iwn_to_tau(G_k_iwn_tmp[l,:],beta,type_of="Derivative")
            for j,tau in enumerate(beta_arr):
                if opt=="positive":
                    dG_tau_for_k[l,j] += 0.5*funct_dispersion(k+q) + 0.5*( beta/2.0 - tau )*( U**2/4.0 + funct_dispersion(k+q)**2 )
                elif opt=="negative":
                    dG_tau_for_k[l,j] += 0.5*funct_dispersion(k+q) + 0.5*( tau - beta/2.0 )*( U**2/4.0 + funct_dispersion(k+q)**2 )
            
            dG_tau_for_k[l,-1] = funct_dispersion(k+q)-mu+U*0.5 - 1.0*dG_tau_for_k[l,0] # Assuming half-filling
        
        return dG_tau_for_k

    @staticmethod
    def get_derivative_FFT_G0(G0_iwn, iwn_arr, beta_arr, h : float, mu : float, hyb_c : float, opt="positive"):
        """
        This method computes the derivatives of G(tau) at boundaries for the cubic spline. Recall that only positive imaginary time
        is used, i.e 0 < tau < beta. It takes care of subtracting the leading moments for smoother results.
        
        Parameters:
            G0_iwn (complex np.ndarray): Weiss Green's function mesh over (iwn)-space.
            iwn_arr (complex np.array): Fermionic Matsubara frequencies.
            beta (float): Inverse temperature.
            mu (float): chemical potential.
            opt (str): positive or negative imaginary-time Green's function derivated. Takes in "positive" of "negative". 
            Defaults opt="postitive".
        
        Returns:
            dG_tau (float np.ndarray): imaginary-time Green's function derivative mesh over (tau)-space.

        """
        beta = beta_arr[-1]
        dG_tau = np.empty((len(iwn_arr)+1),dtype=float)
        G0_iwn_tmp = deepcopy(G0_iwn)
        # Subtracting the leading moments of the Green's function
        for i,iwn in enumerate(iwn_arr):
            moments = 1.0/(iwn + mu - h - hyb_c/iwn)
            # G
            G0_iwn_tmp[i] -= moments
            if opt=="negative":
                G0_iwn_tmp[i] = np.conj(G0_iwn_tmp[i]) # because G(-tau)
        G0_iwn_tmp[i] *= -1.0*iwn
        
        # Calculating the dG/dtau objects
        
        dG_tau[:] = get_iwn_to_tau(G0_iwn_tmp,beta,type_of="Derivative")
        z1=-(mu-h)*0.5+0.5*np.sqrt((mu-h)**2+4.0*hyb_c); z2=-(mu-h)*0.5-0.5*np.sqrt((mu-h)**2+4.0*hyb_c)
        for j,tau in enumerate(beta_arr):
            if opt=="positive":
                dG_tau[j] += (z1**2/(z1-z2))*1.0/(np.exp(z1*tau)+np.exp(z1*(tau-beta))) + (z2**2/(z2-z1))*1.0/(np.exp(z2*tau)+np.exp(z2*(tau-beta)))
            elif opt=="negative":
                dG_tau[j] += (z1**2/(z1-z2))*1.0/(np.exp(-z1*tau)+np.exp(z1*(beta-tau))) + (z2**2/(z2-z1))*1.0/(np.exp(-z2*tau)+np.exp(z2*(beta-tau)))
        
        dG_tau[-1] = -(mu-h) - 1.0*dG_tau[0] # Assuming half-filling
        
        return dG_tau

class FermionsvsBosons(Cubic_spline):

    _iqn_array = np.array([],dtype=complex)
    _iwn_array = np.array([],dtype=complex)
    _cubic_spline = np.array([],dtype=complex)
    _one_instance = True
    _x_array = np.array([],dtype=float)

    def __init__(self,beta,x_array):
        """
        Internal variables:
        cubic_spline_funct_size: represents the size of the tau-defined function to be interpolated over.
        """
        cubic_spline_funct_size = len(x_array)
        NN = cubic_spline_funct_size//2
        if FermionsvsBosons._one_instance:
            FermionsvsBosons._one_instance = False
            FermionsvsBosons._iqn_array = np.array([1.0j*(2.0*n)*np.pi/beta for n in range(cubic_spline_funct_size-1)],dtype=complex)
            FermionsvsBosons._iwn_array = np.array([1.0j*(2.0*n+1.0)*np.pi/beta for n in range(-(NN),NN)],dtype=complex)
            FermionsvsBosons._x_array = np.asarray(x_array)
            FermionsvsBosons._cubic_spline = np.zeros((cubic_spline_funct_size-1,),dtype=complex)
        #print("size: ", cubic_spline_funct_size)
        assert len(Cubic_spline._funct)==len(FermionsvsBosons._x_array), "Check the lengths of the arrays in FermionsvsBosons class constructor."
        assert np.mod(cubic_spline_funct_size,2)==1, "The imaginary-time length has got to be odd. (iwn has to be even for mirroring reasons.)"
    
    @classmethod
    def reset(cls) -> None:
        """
        Needed to reset the internal data to FermionvsBosons class.
        """
        cls._one_instance = True
        
        return None
    
    @classmethod
    def bosonic_corr_funct(cls):
        size_iqn = len(cls._iqn_array)
        # Relevant coefficients to enter Fourier transform
        S_0 = Cubic_spline._funct[0]; S_beta = Cubic_spline._ma[-2]*Cubic_spline._delta_beta**3 + Cubic_spline._mb[-2]*Cubic_spline._delta_beta**2 + Cubic_spline._mc[-2]*Cubic_spline._delta_beta + Cubic_spline._funct[-2] # Minus 2 here 
        Sp_0 = Cubic_spline._mc[0]; Sp_beta = 3.0*Cubic_spline._ma[-2]*Cubic_spline._delta_beta**2 + 2.0*Cubic_spline._mb[-2]*Cubic_spline._delta_beta + Cubic_spline._mc[-2]
        Spp_0 = 2.0*Cubic_spline._mb[0]; Spp_beta = 6.0*Cubic_spline._ma[-2]*Cubic_spline._delta_beta + 2.0*Cubic_spline._mb[-2]
        
        # print("S_beta: ", S_beta, " S_0 ", S_0)
        # print("Sp_beta: ", Sp_beta, " Sp_0 ", Sp_0)
        # print("Spp_beta: ", Spp_beta, " Spp_0 ", Spp_0)
        
        Sppp = np.empty((size_iqn,),dtype=complex)
        
        for i in range(size_iqn):
            Sppp[i] = 6.0*Cubic_spline._ma[i]
        
        # Fourier transformation
        Sppp_iqn = (size_iqn)*np.fft.ifft(Sppp) #
        for j,iqn in enumerate(cls._iqn_array):
            if iqn!=complex(0.0,0.0): # Need to take care of the 0 separately..
                cls._cubic_spline[j] = ( S_beta - S_0 ) / ( iqn ) - ( Sp_beta - Sp_0 ) / ( iqn**2 ) + ( Spp_beta - Spp_0 ) / ( iqn**3 ) + ( ( 1.0-np.exp(iqn*Cubic_spline._delta_beta) ) / ( iqn**4 ) )*Sppp_iqn[j]#  
        
        for j in range(1,size_iqn+1):
            #print("j: ", j, " val: ", cls._cubic_spline[0], " ma: ", Cubic_spline._ma[j-1], " mb: ", Cubic_spline._mb[j-1], " mc: ", Cubic_spline._mc[j-1], " my: ", Cubic_spline._funct[j-1])
            cls._cubic_spline[0] += Cubic_spline._ma[j-1]*( (cls._x_array[i]-cls._x_array[i-1])**4 )/4.0 + \
            Cubic_spline._mb[j-1]*( (cls._x_array[i]-cls._x_array[i-1])**3 )/3.0 + Cubic_spline._mc[j-1]*( (cls._x_array[i]-cls._x_array[i-1])**2 )/2.0 + \
            Cubic_spline._funct[j-1]*( cls._x_array[j] - cls._x_array[j-1] )


        return cls._cubic_spline, cls._iqn_array

    @classmethod
    def fermionic_propagator(cls):
        size_iwn = len(cls._iwn_array)
        NN = size_iwn//2
        # Relevant coefficients to enter Fourier transform
        S_0 = Cubic_spline._funct[0]; S_beta = Cubic_spline._ma[-2]*Cubic_spline._delta_beta**3 + Cubic_spline._mb[-2]*Cubic_spline._delta_beta**2 + Cubic_spline._mc[-2]*Cubic_spline._delta_beta + Cubic_spline._funct[-2] # Minus 2 here 
        Sp_0 = Cubic_spline._mc[0]; Sp_beta = 3.0*Cubic_spline._ma[-2]*Cubic_spline._delta_beta**2 + 2.0*Cubic_spline._mb[-2]*Cubic_spline._delta_beta + Cubic_spline._mc[-2]
        Spp_0 = 2.0*Cubic_spline._mb[0]; Spp_beta = 6.0*Cubic_spline._ma[-2]*Cubic_spline._delta_beta + 2.0*Cubic_spline._mb[-2]
        # print("S_beta: ", S_beta, " S_0 ", S_0)
        # print("Sp_beta: ", Sp_beta, " Sp_0 ", Sp_0)
        # print("Spp_beta: ", Spp_beta, " Spp_0 ", Spp_0)
        Sppp = np.zeros((size_iwn,),dtype=complex)
        
        for i in range(size_iwn):
            Sppp[i] = 6.0*Cubic_spline._ma[i]*np.exp(1.0j*np.pi*i/(size_iwn))
            #print("Sppp[{0}]: ".format(i), Sppp[i])
        
        # Fourier transformation
        Sppp_iwn = (size_iwn)*np.fft.ifft(Sppp) #
        # Mirroring
        Sppp_iwn = np.concatenate((Sppp_iwn[NN:],Sppp_iwn[:NN]))
        
        for j,iwn in enumerate(cls._iwn_array):
            cls._cubic_spline[j] = -( S_beta + S_0 ) / ( iwn ) + ( Sp_beta + Sp_0 ) / ( iwn**2 ) - ( Spp_beta + Spp_0 ) / ( iwn**3 ) + ( ( 1.0-np.exp(iwn*Cubic_spline._delta_beta) ) / ( iwn**4 ) )*Sppp_iwn[j]#  

        return cls._cubic_spline, cls._iwn_array


def compute_Sigma_iwn_cubic_spline(G0_iwn : np.ndarray,G0_tau : np.ndarray,G0_m_tau : np.ndarray,delta_tau : float,U : float,h : float,hyb_c : float,mu : float,beta : float,tau_array : np.ndarray,iwn_array : np.ndarray,SE_tau_tmp : np.ndarray):
    """
    Computes the IPT self-energy in femrionic Matsubara frequencies given the imaginary-time Weiss Green's functions G_0. 
    To be used in the AFM case scenario.

    Parameters:
        G0_iwn (ndarray): 
    """
    # Self-energy impurity
    for i in range(SE_tau_tmp.shape[0]):
        SE_tau_tmp[i,0,0] = -1.0*U*U*G0_tau[i,0,0]*G0_m_tau[i,1,1]*G0_tau[i,1,1] #up
        SE_tau_tmp[i,1,1] = -1.0*U*U*G0_tau[i,1,1]*G0_m_tau[i,0,0]*G0_tau[i,0,0] #down
    # up
    # getting the boundary conditions
    der_G0_up = Cubic_spline.get_derivative_FFT_G0(G0_iwn[:,0,0],iwn_array,tau_array,h,mu,hyb_c)
    der_G0_down = Cubic_spline.get_derivative_FFT_G0(G0_iwn[:,1,1],iwn_array,tau_array,h,mu,hyb_c)
    der_G0_m_down = Cubic_spline.get_derivative_FFT_G0(G0_iwn[:,1,1],iwn_array,tau_array,h,mu,hyb_c,opt="negative")
    left_der_up = der_G0_up[0]*G0_m_tau[0,1,1]*G0_tau[0,1,1] + G0_tau[0,0,0]*der_G0_m_down[0]*G0_tau[0,1,1] + G0_tau[0,0,0]*G0_m_tau[0,1,1]*der_G0_down[0]
    right_der_up = der_G0_up[-1]*G0_m_tau[-1,1,1]*G0_tau[-1,1,1] + G0_tau[-1,0,0]*der_G0_m_down[-1]*G0_tau[-1,1,1] + G0_tau[-1,0,0]*G0_m_tau[-1,1,1]*der_G0_down[-1]
    
    plt.figure(2)
    plt.title(r"$\frac{\mathrm{d}\Sigma^{(2)}_{\sigma}(\tau)}{\mathrm{d}\tau}$ vs $\tau$")
    plt.plot(tau_array,-1.0*U*U*np.dot(der_G0_up,np.dot(der_G0_m_down,der_G0_down)),c="red",label=r"$\sigma=\uparrow$")
    plt.xlabel(r"$\tau$")
    
    Cubic_spline(delta_tau,SE_tau_tmp[:,0,0])
    Cubic_spline.building_matrix_components(left_der_up,right_der_up)
    Cubic_spline.tridiagonal_LU_decomposition()
    Cubic_spline.tridiagonal_LU_solve()
    # Getting m_a and m_c from the b's
    Cubic_spline.construct_coeffs(tau_array)
    FermionsvsBosons(beta,tau_array)
    Sigma_up_iwn, *_ = FermionsvsBosons.fermionic_propagator()
    # cleaning
    FermionsvsBosons.reset()
    Cubic_spline.reset()

    # down
    # getting the boundary conditions
    der_G0_up = Cubic_spline.get_derivative_FFT_G0(G0_iwn[:,0,0],iwn_array,tau_array,-1.0*h,mu,hyb_c)
    der_G0_down = Cubic_spline.get_derivative_FFT_G0(G0_iwn[:,1,1],iwn_array,tau_array,-1.0*h,mu,hyb_c)
    der_G0_m_up = Cubic_spline.get_derivative_FFT_G0(G0_iwn[:,0,0],iwn_array,tau_array,-1.0*h,mu,hyb_c,opt="negative")
    left_der_down = der_G0_down[0]*G0_m_tau[0,0,0]*G0_tau[0,0,0] + G0_tau[0,1,1]*der_G0_m_up[0]*G0_tau[0,0,0] + G0_tau[0,1,1]*G0_m_tau[0,0,0]*der_G0_up[0]
    right_der_down = der_G0_down[-1]*G0_m_tau[-1,0,0]*G0_tau[-1,0,0] + G0_tau[-1,1,1]*der_G0_m_up[-1]*G0_tau[-1,0,0] + G0_tau[-1,1,1]*G0_m_tau[-1,0,0]*der_G0_up[-1]
   
    plt.plot(tau_array,-1.0*U*U*np.dot(der_G0_down,np.dot(der_G0_m_up,der_G0_up)),c="green",label=r"$\sigma=\downarrow$")
    plt.legend()

    Cubic_spline(delta_tau,SE_tau_tmp[:,1,1])
    Cubic_spline.building_matrix_components(left_der_down,right_der_down)
    Cubic_spline.tridiagonal_LU_decomposition()
    Cubic_spline.tridiagonal_LU_solve()
    # Getting m_a and m_c from the b's
    Cubic_spline.construct_coeffs(tau_array)
    FermionsvsBosons(beta,tau_array)
    Sigma_down_iwn, *_ = FermionsvsBosons.fermionic_propagator()
    # cleaning
    FermionsvsBosons.reset()
    Cubic_spline.reset()

    return Sigma_up_iwn, Sigma_down_iwn