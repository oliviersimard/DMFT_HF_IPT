import numpy as np
from math import isclose
import matplotlib.pyplot as plt

class Cubic_spline(object):
    """Class for cubic spline.
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
        """Needed to reset the internal data to Cubic_spline class.
        """
        cls._one_instance = True
        
        return None

    @classmethod
    def building_matrix_components(cls, left_der : float, right_der : float) -> None:
        """It is assumed here that the beta step is constant.
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
        #return cls._subdiagonal, cls._diagonal, cls._superdiagonal, cls._rhs 
        
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
        #return cls._subdiagonal, cls._diagonal, cls._superdiagonal

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
        """Computing the cubic spline of GG(tau)
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
        """Returns an array of N-2 elements representing the derivative of an input array of N points. Special care has to be 
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
        
        assert isclose(np.abs(right_most_val),np.abs(left_most_val),abs_tol=1e-5), "The derivative end points have to be similar, otherwise the method lacks precision."
        
        der_f = np.insert(der_f,0,left_most_val)
        der_f = np.append(der_f,right_most_val)

        return der_f

class FermionsvsBosons(Cubic_spline):

    _iqn_array = np.array([],dtype=complex)
    _iwn_array = np.array([],dtype=complex)
    _cubic_spline = np.array([],dtype=complex)
    _one_instance = True
    _x_array = np.array([],dtype=float)

    def __init__(self,beta,x_array):
        """Internal variables:
        cubic_spline_funct_size: represents the size of the tau-defined function to be interpolated over.
        """
        cubic_spline_funct_size = len(x_array)
        if FermionsvsBosons._one_instance:
            FermionsvsBosons._one_instance = False
            FermionsvsBosons._iqn_array = np.array([1.0j*(2.0*n)*np.pi/beta for n in range(cubic_spline_funct_size-1)],dtype=complex)
            FermionsvsBosons._iwn_array = np.array([1.0j*(2.0*n+1.0)*np.pi/beta for n in range(cubic_spline_funct_size-1)],dtype=complex)
            FermionsvsBosons._x_array = np.asarray(x_array)
            FermionsvsBosons._cubic_spline = np.zeros((cubic_spline_funct_size-1,),dtype=complex)
        print("size: ", cubic_spline_funct_size)
        assert len(Cubic_spline._funct)==len(FermionsvsBosons._x_array), "Check the lengths of the arrays in FermionsvsBosons class constructor."
        assert np.mod(cubic_spline_funct_size,2)==1, "The imaginary-time length has got to be odd. (iwn has to be even for mirroring reasons.)"
    
    @classmethod
    def reset(cls) -> None:
        """Needed to reset the internal data to FermionvsBosons class.
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
        
        print("S_beta: ", S_beta, " S_0 ", S_0)
        # print("Sp_beta: ", Sp_beta, " Sp_0 ", Sp_0)
        # print("Spp_beta: ", Spp_beta, " Spp_0 ", Spp_0)
        
        # print("S_beta: ", S_beta, " S_0 ", S_0)
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
        
        # Relevant coefficients to enter Fourier transform
        S_0 = Cubic_spline._funct[0]; S_beta = Cubic_spline._ma[-2]*Cubic_spline._delta_beta**3 + Cubic_spline._mb[-2]*Cubic_spline._delta_beta**2 + Cubic_spline._mc[-2]*Cubic_spline._delta_beta + Cubic_spline._funct[-2] # Minus 2 here 
        Sp_0 = Cubic_spline._mc[0]; Sp_beta = 3.0*Cubic_spline._ma[-2]*Cubic_spline._delta_beta**2 + 2.0*Cubic_spline._mb[-2]*Cubic_spline._delta_beta + Cubic_spline._mc[-2]
        Spp_0 = 2.0*Cubic_spline._mb[0]; Spp_beta = 6.0*Cubic_spline._ma[-2]*Cubic_spline._delta_beta + 2.0*Cubic_spline._mb[-2]
        print("S_beta: ", S_beta, " S_0 ", S_0)
        # print("Sp_beta: ", Sp_beta, " Sp_0 ", Sp_0)
        # print("Spp_beta: ", Spp_beta, " Spp_0 ", Spp_0)
        Sppp = np.zeros((size_iwn,),dtype=complex)
        
        for i in range(size_iwn):
            Sppp[i] = 6.0*Cubic_spline._ma[i]*np.exp(1.0j*np.pi*i/(size_iwn))
            #print("Sppp[{0}]: ".format(i), Sppp[i])
        
        # Fourier transformation
        Sppp_iwn = (size_iwn)*np.fft.ifft(Sppp) #
        
        for j,iwn in enumerate(cls._iwn_array):
            cls._cubic_spline[j] = -( S_beta + S_0 ) / ( iwn ) + ( Sp_beta + Sp_0 ) / ( iwn**2 ) - ( Spp_beta + Spp_0 ) / ( iwn**3 ) + ( ( 1.0-np.exp(iwn*Cubic_spline._delta_beta) ) / ( iwn**4 ) )*Sppp_iwn[j]#  

        return cls._cubic_spline, cls._iwn_array




