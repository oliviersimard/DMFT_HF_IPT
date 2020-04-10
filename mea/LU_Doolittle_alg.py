import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.integrate import quad

class Data:
    """ Class data container to be passed along.
    Args:
        beta(float): temperature of calculations
        omega_0(float): mid-distance in energy between the two poles (poles are centered around).
        D_omega(float): distance between the two poles.
    """
    def __init__(self, _beta, _omega_0, _D_omega):
        self.beta = _beta
        self.omega_0 = _omega_0
        self.D_omega = _D_omega
    
    def __del__(self):
        del self.beta
        del self.omega_0
        del self.D_omega

class SplineUtils(object):
    """ This class holds the main member functions used to benchmark the cubic spline used and the Padé algorithm.
    """
    # Static variables
    m_a = np.array([],dtype=float)
    m_b = np.array([],dtype=float)
    m_c = np.array([],dtype=float)
    iqn_stat = np.array([],dtype=float)
    beta_stat = np.array([],dtype=float)

    def __init__(self, G_funct=None, tail_G_funct=None, tail_G_tau=None, data : Data=None, GtG_t=None, Gtau_pos=None, Gtau_neg=None):
        """Constructor of SplineUtils.
        """
        SplineUtils._G_funct = G_funct # Function bound to instance
        SplineUtils._tail_G_funct = tail_G_funct
        SplineUtils._tail_G_tau = tail_G_tau
        self._data = data
        self._GtG_t = GtG_t
        self._Gtau_pos = Gtau_pos
        self._Gtau_neg = Gtau_neg
        # creating Matsubara arrays
        self._iwn_array = np.array([(2.*n+1.)*np.pi/self._data.beta for n in range(len(self._GtG_t)-1)],dtype=float)
        self._iqn_array = np.array([(2.*n)*np.pi/self._data.beta for n in range(len(self._GtG_t)-1)],dtype=float)
        SplineUtils.iqn_stat = self._iqn_array # For later purposes. Should raise this variable to be static
        self._beta_array = np.linspace(0.0,self._data.beta,len(self._GtG_t))
        SplineUtils.beta_stat = self._beta_array
    
    @staticmethod
    def n_F(omega_0: float, D_omega: float, beta: float) -> tuple:
        """Fermi-Dirac distribution adapted to the problem.
        Returns:
            n_f_p(float): higher energy Fermi distribution of the test Green's function (first element).
            n_f_n(float): lower energy Fermi distribution of the test Green's function (second element).
        """
        n_f_p = 1.0/( np.exp( beta * ( omega_0 + D_omega/2.0 ) ) + 1.0 )
        n_f_n = 1.0/( np.exp( beta * ( omega_0 - D_omega/2.0 ) ) + 1.0 )

        return n_f_p, n_f_n
    
    @classmethod
    def G_taus_decorator(cls,func,omega_0: float,D_omega: float,beta: float):
        """Decorator of the bubble function
        """

        def inner_funct(iqn: float):
            inner_funct = lambda x, y: func(x,y,omega_0,D_omega,beta)
            vectorized_funct = np.vectorize(inner_funct) # func should be self.G_taus

            return vectorized_funct(cls.beta_stat,iqn)

        return inner_funct

    @staticmethod
    def G_taus(tau : float, iqn : float, omega_0: float, D_omega: float, beta: float) -> complex:
        """Generates the analytical components of the Green's function.
        Returns:
            bubble(float): positive*negative imaginary-time definition of the test bubble function.
        """
        G_tau_p = -0.5*( np.exp( -(omega_0 + D_omega/2.0) * tau ) * ( 1.0 - SplineUtils.n_F(omega_0,D_omega,beta)[0] ) 
        + np.exp( -(omega_0 - D_omega/2.0) * tau ) * ( 1.0 - SplineUtils.n_F(omega_0,D_omega,beta)[1] ) )
        G_tau_n = 0.5*( np.exp( -(omega_0 + D_omega/2.0) * tau ) * ( SplineUtils.n_F(omega_0,D_omega,beta)[0] ) 
        + np.exp( -(omega_0 - D_omega/2.0) * tau ) * ( SplineUtils.n_F(omega_0,D_omega,beta)[1] ) )

        return -1.0*G_tau_p*G_tau_n*np.exp(1.0j*iqn*tau)

        
    def boundary_conditions(self) -> tuple:
        """ Sets the boundary conditions used in the cubic spline.
        Returns:
            dG_dtau_pos(array): floating-point-valued array expressing the imaginary-time derivative of the positively-defined Green's function.
            dG_dtau_neg(array): floating-point-valued array expressing the imaginary-time derivative of the negatively-defined Green's function.
        """
        NN = len(self._beta_array) # The length of the imaginary time array is larger than that of the Matsubara array.
        Fiwn_kernel_pos = [] ; Fiwn_kernel_neg = []
        dG_dtau_pos = np.empty(NN,dtype=float) ; dG_dtau_neg = np.empty(NN,dtype=float)
        for iwn in self._iwn_array:
            Fiwn_kernel_pos.append( -1.0j * iwn * ( self._G_funct(iwn,self._data.beta,self._data.omega_0,self._data.D_omega) - self._tail_G_funct(iwn,self._data.omega_0,self._data.D_omega) ) )
            Fiwn_kernel_neg.append( -1.0j * iwn * ( np.conj( self._G_funct(iwn,self._data.beta,self._data.omega_0,self._data.D_omega) - self._tail_G_funct(iwn,self._data.omega_0,self._data.D_omega) ) ) )
        
        # Performing the FFT to get the derivatives (boundary conditions for the cubic spline)
        Ftau_kernel_pos = 1.0/self._data.beta * np.fft.fft(Fiwn_kernel_pos)
        Ftau_kernel_neg = 1.0/self._data.beta * np.fft.fft(Fiwn_kernel_neg)
        
        for i in range(NN-1):
            tau = i*self._data.beta/(NN-1)
            dG_dtau_pos[i] = ( np.exp( -1.0j * np.pi * i * (1.0/(NN-1) - 1.0) ) * Ftau_kernel_pos[i] + self._tail_G_tau(tau,self._data.omega_0,self._data.D_omega)[0] ).real
            dG_dtau_neg[i] = ( np.exp( -1.0j * np.pi * i * (1.0/(NN-1) - 1.0) ) * Ftau_kernel_neg[i] + self._tail_G_tau(tau,self._data.omega_0,self._data.D_omega)[1] ).real
        
        dG_dtau_pos[NN-1] = 1.0*self._data.omega_0 - dG_dtau_pos[0]
        dG_dtau_neg[NN-1] = 1.0*self._data.omega_0 - dG_dtau_neg[0]

        return dG_dtau_pos, dG_dtau_neg

    def splitting_for_spline(self) -> tuple:
        """This function returns appropriately sparsed arrays to solve tridiagonal system of equations.
        
        Args:
            GtG_t(array): contains the result of the bubble diagram in imaginary time.
            beta_array(array): contains the array of imaginary time with the proper discretization.
        
        Returns:
            superdiag(array): contains upper diagonal of the system of equations making up the cubic spline.
            diag(array): contains the diagonal components of the system of equations.
            subdiag(array): contains the subdiagonal elements of the tridiagonal matrix.
            rhs(array): contains the right-hand side data to the system of equations.
        """
        assert len(self._beta_array)==len(self._GtG_t) and len(self._Gtau_neg)==len(self._Gtau_pos), "The lengths of the inputs in function \"splitting_for_spline\" have to be the same."
        Ntau = len(self._beta_array)
        # Setting up the arrays returned
        superdiag = np.empty(Ntau-1,dtype=float)
        diag = np.empty(Ntau,dtype=float)
        subdiag = np.empty(Ntau-1,dtype=float)
        rhs = np.empty(Ntau,dtype=float)

        for n in range(1,Ntau-1): # Boundary conditions are determined through the first derivatives
            subdiag[n-1] = 1.0*(self._beta_array[n]-self._beta_array[n-1])
            diag[n] = 2.0*(self._beta_array[n+1]-self._beta_array[n-1])
            superdiag[n] = 1.0*(self._beta_array[n+1]-self._beta_array[n])
            rhs[n] = 3.0*( (self._GtG_t[n+1]-self._GtG_t[n])/(self._beta_array[n+1]-self._beta_array[n]) - (self._GtG_t[n]-self._GtG_t[n-1])/(self._beta_array[n]-self._beta_array[n-1]) )

        # Dealing with the boundary conditions
        dG_dtau_pos, dG_dtau_neg = self.boundary_conditions()
        # Left boundary conditions
        m_left_value = -self._Gtau_pos[0] * dG_dtau_neg[0] - self._Gtau_neg[0] * dG_dtau_pos[0]
        print("m_left_val: ", m_left_value)
        # c[0] = f', needs to be re-expressed in terms of b:
        # (2b[0]+b[1])(x[1]-x[0]) = 3 ((y[1]-y[0])/(x[1]-x[0]) - f')
        diag[0] = 2.0 * ( self._beta_array[1] - self._beta_array[0] )
        superdiag[0] = 1.0 * ( self._beta_array[1] - self._beta_array[0] )
        rhs[0] = 3.0 * ( ( self._GtG_t[1] - self._GtG_t[0] ) / ( self._beta_array[1] - self._beta_array[0] ) - m_left_value )
        # Right boundary conditions
        m_right_value = -self._Gtau_pos[Ntau-1] * dG_dtau_neg[Ntau-1] - self._Gtau_neg[Ntau-1] * dG_dtau_pos[Ntau-1]
        print("m_right_val: ", m_right_value)
        # plt.figure(0)
        # plt.plot(dG_dtau_pos)
        # plt.figure(1)
        # plt.plot(self._Gtau_pos)
        # plt.show()
        # c[n-1] = f', needs to be re-expressed in terms of b:
        # (b[n-2]+2b[n-1])(x[n-1]-x[n-2]) = 3 (f' - (y[n-1]-y[n-2])/(x[n-1]-x[n-2]))
        diag[Ntau-1] = 2.0 * ( self._beta_array[Ntau-1] - self._beta_array[Ntau-2] )
        subdiag[Ntau-2] = 1.0 * ( self._beta_array[Ntau-1] - self._beta_array[Ntau-2] )
        rhs[Ntau-1] = 3.0 * ( m_right_value - ( self._GtG_t[Ntau-1] - self._GtG_t[Ntau-2] ) / ( self._beta_array[Ntau-1] - self._beta_array[Ntau-2] ) )

        return superdiag, diag, subdiag, rhs

    @staticmethod
    def tdma(superdiag : list, diag : list, subdiag : list, rhs : list) -> None:
        """Solution of a linear system of algebraic equations with a
            tri-diagonal matrix of coefficients using the Thomas-algorithm.
        Args:
            superdiag(array): an array containing lower diagonal (a[0] is not used)
            diag(array): an array containing main diagonal 
            subdiag(array): an array containing lower diagonal (c[-1] is not used)
            rhs(array): right hand side of the system
        Returns:
            m_b(array): solution array of the system to the coefficients b of the cubic spline:
                S_n(x) = a_n*(x_{n+1}-x_n)^3 + b_n*(x_{n+1}-x_n)^2 + c_n*(x_{n+1}-x_n) + y_n
        """
        assert len(superdiag)==len(subdiag) and len(diag)==(len(superdiag)+1), "The lengths of the entries are not coherent." 
        
        n = len(diag) # same as beta array
    
        x = np.zeros(n)
        # elimination:
        
        for k in range(n-1):
            if diag[k]==0.0:
                raise ValueError("Division by 0.")
            else:
                subdiag[k] /= diag[k]
                diag[k+1] -= subdiag[k]*superdiag[k]
        
        if diag[n-1]==0.0:
            raise ValueError("Division by 0.")
        # backsubstitution:
        
        x[0] = rhs[0]
        for i in range(1,n):
            x[i] = rhs[i] - subdiag[i-1] * x[i-1]

        x[n-1] /= diag[n-1]
        for i in range(n-2,-1,-1):
            x[i] -= superdiag[i] * x[i+1]
            x[i] /= diag[i]
        
        # Binding to static class member variables
        SplineUtils.m_b = x

        return None
    
    def setting_m_a_m_c(self) -> None:
        """This functions makes sense only if used in conjunction with \"tdma\" (after it). It computes the coefficients a_n and c_n of the 
        following spline:
            S_n(x) = a_n*(x_{n+1}-x_n)^3 + b_n*(x_{n+1}-x_n)^2 + c_n*(x_{n+1}-x_n) + y_n , x_n <= x <= x_{n+1}
        The terms y_n's are the values of the function to interpolate. This function returns nothing.
        """
        n = len(self._beta_array)
    
        m_a = np.empty(n,dtype=float)
        m_c = np.empty(n,dtype=float)

        for i in range(n-1):
            m_a[i]=1.0/3.0*(SplineUtils.m_b[i+1]-SplineUtils.m_b[i])/(self._beta_array[i+1]-self._beta_array[i])
            m_c[i]=(self._GtG_t[i+1]-self._GtG_t[i])/(self._beta_array[i+1]-self._beta_array[i]) - 1.0/3.0*(2.0*SplineUtils.m_b[i]+SplineUtils.m_b[i+1])*(self._beta_array[i+1]-self._beta_array[i])
        
        h = self._beta_array[n-1]-self._beta_array[n-2]
        m_a[n-1] = 0.0
        m_c[n-1] = 3.0*m_a[n-2]*h*h+2.0*SplineUtils.m_b[n-2]*h+m_c[n-2]   ## f'_{n-1}(x_{n-1}) = f'_{n-2}(x_{n-1})

        # Binding to static class member variables
        SplineUtils.m_a = m_a
        SplineUtils.m_c = m_c

        return None

    def __call__(self, x : float, case : str) -> float:
        """Gives the value of the interpolation at a given point x bracketed between 0 and beta.
        Args:
            x(float): point to interpolate with the cubic spline.
            case(str): string to determine the method of integration: \"cubic_spline\" or \"linear_interpolation\".
        Returns:
            interpol(float): value of the interpolated point.
        """
        interpol = 0.0
        if case=="cubic_spline":
            # Find the coefficients to use to interpolate the value at x
            beta_array_diff = self._beta_array - x
            idx = np.where(beta_array_diff>=0,beta_array_diff,np.inf).argmin()
            idx = idx -1 # This should be the index before that is used
            print("idx: ", idx, " and beta_array_diff value: ", beta_array_diff[idx], " and beta_array value: ", self._beta_array[idx])
            h=x-self._beta_array[idx]
            n = len(self._beta_array)
            if x<self._beta_array[0]:
                # Left extrapolation
                interpol=(SplineUtils.m_b[0]*h + SplineUtils.m_c[0])*h + self._GtG_t[0]
            elif x>self._beta_array[n-1]:
                # Right extrapolation
                interpol=(SplineUtils.m_b[n-1]*h + SplineUtils.m_c[n-1])*h + self._GtG_t[n-1]
            else:
                # Interpolation
                interpol=((SplineUtils.m_a[idx]*h + SplineUtils.m_b[idx])*h + SplineUtils.m_c[idx])*h + self._GtG_t[idx]
                #print("interpol: ", interpol)
        elif case=="linear_interpolation":
            new_G_taus = SplineUtils.G_taus_decorator(SplineUtils.G_taus,self._data.omega_0,self._data.D_omega,self._data.beta)
            vec_val_G_taus = new_G_taus(x)
            interpol_funct = lambda xx: np.interp(xx,self._beta_array,vec_val_G_taus)
            interpol = quad(interpol_funct,0.001,self._data.beta)[0]

        return interpol

    def tau_to_iwn_bosonic(self) -> list:
        """This member function computes the Matsubara Fourier transformation of the imaginary-time function inputted.
        Returns:
            sigma_iwn(array): complex-valued array of elements defining the Matsubara function to which the Fourier transform led.
        """
        NN = len(self._GtG_t)
        hlast = self._beta_array[-1] - self._beta_array[-2]
        S_0 = self._GtG_t[0] ; S_beta = SplineUtils.m_a[-1]*hlast**3 + SplineUtils.m_b[-1]*hlast**2 + SplineUtils.m_c[-1]*hlast + self._GtG_t[-1]
        Sp_0 = SplineUtils.m_c[0] ; Sp_beta = 3.0*SplineUtils.m_a[-1]*hlast**2 + 2.0*SplineUtils.m_b[-1]*hlast + SplineUtils.m_c[-1]
        Spp_0 = 2.0*SplineUtils.m_b[0] ; Spp_beta = 6.0*SplineUtils.m_a[-1]*hlast + 2.0*SplineUtils.m_b[-1]
        Sppp_array = []
        for i in range(NN):
            Sppp_array.append( 6.0 * SplineUtils.m_a[i] )
    
        IFFT_component = np.fft.ifft(Sppp_array)
        sigma_iwn = np.empty(NN-1,dtype=complex)
        print("len iqn: ", len(self._iqn_array))
        for m in range(1,len(self._iqn_array)): # One has to treat separately iqn=0
            spline = (-S_0 + S_beta)/(1.0j*self._iqn_array[m]) + (Sp_0 - Sp_beta)/((1.0j*self._iqn_array[m])**2) + (-Spp_0 + Spp_beta)/((1.0j*self._iqn_array[m])**3) + ( 1.0 - np.exp( 1.0j * self._iqn_array[m] * self._data.beta/(NN) ) ) * IFFT_component[m] / ((1.0j*self._iqn_array[m])**4)
            sigma_iwn[m] = -1.0*spline # The minus sign is to account for the definition of the bubble diagram
        # Computing component 0 to Sigma(iqn)...
        Sigma_iqn_0 = 0.0
        for n in range(1,NN):
            Sigma_iqn_0 += ( SplineUtils.m_a[n-1]/4.0 * ( self._beta_array[n]**4 - self._beta_array[n-1]**4 ) + SplineUtils.m_b[n-1]/3.0 * ( self._beta_array[n]**3 - self._beta_array[n-1]**3 ) + SplineUtils.m_c[n-1]/2.0 * ( self._beta_array[n]**2 - self._beta_array[n-1]**2 ) + self._GtG_t[n-1] * ( self._beta_array[n] - self._beta_array[n-1] ) )
        
        sigma_iwn[0] = -1.0*Sigma_iqn_0 #1.0/NN * Sigma_iqn_0 # Does it need the prefactor? <---------- Answer to this

        # Saving the Fourier-transformed data into file
        with open("GG_for_pade.dat", "w") as f:
            for i,iqn in enumerate(self._iqn_array):
                if i==0:
                    f.write("/\n")
                f.write("{0:.10f}\t\t{1:.10f}\t\t{2:.10f}\n".format(iqn,sigma_iwn[i].real,sigma_iwn[i].imag))

        return sigma_iwn
  
class FFT_Cooley_Tukey(object):
    """Custom Fourier transformation.
    """

    @staticmethod
    def DFT_slow(x):
        """Compute the discrete Fourier Transform of the 1D array x"""
        x = np.asarray(x)
        N = x.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, x)

    @staticmethod
    def FFT(x):
        """A recursive implementation of the 1D Cooley-Tukey FFT"""
        x = np.asarray(x)
        N = x.shape[0]

        if N % 2 > 0:
            raise ValueError("size of x must be a power of 2")
        elif N <= 16:  # this cutoff should be optimized
            return FFT_Cooley_Tukey.DFT_slow(x)
        else:
            X_even = FFT_Cooley_Tukey.FFT(x[::2])
            X_odd = FFT_Cooley_Tukey.FFT(x[1::2])
            factor = np.exp(-2j * np.pi * np.arange(N) / N)
            concat = np.concatenate([X_even + factor[:N // 2] * X_odd,
                                X_even + factor[N // 2:] * X_odd])
            print("x even: ", x[::2])
            print("x odd: ", x[1::2])
            return concat
