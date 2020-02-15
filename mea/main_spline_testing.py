from LU_Doolittle_alg import *


def G_iwn(self, iwn : float, beta : int, omega_0 : float, D_omega : float) -> complex:
    """Generates the Green's function in terms of Matsubara frequencies.

    Args:
        iwn(float): Fermionic Matsubara frequencies.
    
    Returns:
        g(array): Array representation of the Matsubara Green's function.
    """
    g = 0.5/( 1.j*iwn - (omega_0 + D_omega/2.0) ) + 0.5/( 1.j*iwn - (omega_0 - D_omega/2.0) )
    return g

def tail_G_iwn(self, iwn : float, omega_0 : float, D_omega : float) -> complex:
    """Generates the tail of the Green's function considered. Useful for to increase accuracy in Fourier transformation.
    """
    tail_g = 1.0/(1.0j*iwn) + omega_0/( (1.0j*iwn)**2 ) + (omega_0**2 + (D_omega/2.0)**2)/( (1.0j*iwn)**3 )

    return tail_g

def tail_G_tau(self, tau : float, omega_0 : float, D_omega : float) -> tuple:
    tail_pos = -0.5 + omega_0/2.0 * (tau - self._data.beta/2.0) - (omega_0**2 + (D_omega/2.0)**2)/4.0 * tau * (tau-self._data.beta)
    tail_neg = 0.5 - omega_0/2.0 * (tau - self._data.beta/2.0) + (omega_0**2 + (D_omega/2.0)**2)/4.0 * tau**2

    return tail_pos, tail_neg

def bosonic_matsubara_GG(omega_pm : float, beta : float, qn : float) -> complex:
    """Bosonic Matsubara bubble function derived analytically.
    """
    prefact1 = 1.0/( 8.0*np.cosh(beta*omega_pm/2.0)**2 )
    term1_1 = ( 4.0*omega_pm*np.sinh(beta*omega_pm) ) / ( 4.0*omega_pm**2 - ( ( 1.0j*qn )**2 ) )

    prefact2 = ( 2.0*np.sinh(omega_pm*beta) )/( ( ( 1.0j*qn )**2 ) - 4.0*omega_pm**2 )
    term2_1 = np.sinh(2.0*omega_pm*beta) * (1.0j*omega_pm)
    term2_2 = ( 2.0*omega_pm ) * np.cosh(2.0*omega_pm*beta)
    overall_term2 = prefact2 * ( term2_1 - term2_2 )

    return prefact1 * ( term1_1 + overall_term2 )

if __name__ == "__main__":

    # parameters
    beta = 40
    omega_0 = 0.0
    D_omega = 0.5
    filename = "tau_greens_functions_from_pade.dat"
    data = Data(beta,omega_0,D_omega)

    # loading data
    loading_data = np.genfromtxt(filename,dtype=float,delimiter="\t\t",skip_header=1)
    beta_array = loading_data[:,0]
    Gtau_pos = loading_data[:,1]
    Gtau_neg = loading_data[:,2]
    GtG_t = loading_data[:,3]

    splineObj = SplineUtils(G_iwn,tail_G_iwn,tail_G_tau,data,GtG_t,Gtau_pos,Gtau_neg)
    
    superdiag, diag, subdiag, rhs = splineObj.splitting_for_spline()
    # Calculating the b's of the cubic spline
    SplineUtils.tdma(superdiag,diag,subdiag,rhs)
    # Working out the solutions for the a's and c's in the cubic spline
    splineObj.setting_m_a_m_c() # This is the solutions to the coefficients
    # Testing using spline function to see if it fits well
    ##### Spline #####
    interpolation_arr = []
    string_method = "linear_interpolation"
    interpol = splineObj(0.56,string_method)
    print("interpol: ", interpol)
    if string_method=="cubic_spline":
        sampl = np.random.uniform(low=0.0, high=50.0, size=100)
        for sa in sampl:
            interpolation_arr.append(splineObj(sa,string_method))
        plt.figure(1)
        plt.scatter(sampl,interpolation_arr)
    elif string_method=="linear_interpolation":
        #with open("sigma_iqn.dat","w+") as sig:
        for i,iqn in enumerate(splineObj._iqn_array):
            spl = splineObj(iqn,string_method)
            interpolation_arr.append(spl)   
                # if i==0:
                #     sig.write("/\n")
                # sig.write("{0:.10f}\t\t{1:.10f}\t\t{2:.10f}\n".format(iqn,spl.real,spl.imag))

        #sig.close()
        plt.figure(1)
        plt.scatter(splineObj._iqn_array,interpolation_arr)

    plt.figure(2)
    plt.plot(beta_array,GtG_t)
    plt.show()

    sigma_iwn = splineObj.tau_to_iwn_bosonic()

    ### Plotting the analytical solution that is expected ###
    iqn_array_anal_sol = []
    with open("sigma_iqn.dat","w+") as sig:
        for i,qn in enumerate(SplineUtils.iqn_stat):
            val = bosonic_matsubara_GG(D_omega/2.0,beta,qn)
            iqn_array_anal_sol.append( val )
            if i==0:
                sig.write("/\n")
            sig.write("{0:.10f}\t\t{1:.10f}\t\t{2:.10f}\n".format(qn,val.real,val.imag))
    sig.close()
    fig = plt.figure(5)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yscale('log')
    line, = ax.plot(SplineUtils.iqn_stat, iqn_array_anal_sol, color='blue', lw=2)
    plt.show()
    # mat = [[2, -1, -2],
    #         [-4, 6, 3],
    #         [-4, -2, 8]] 
    
    # luDecomposition(mat, 3)



def luDecomposition(mat, n): 
  
    lower = [[0 for x in range(n)]  
                for y in range(n)] 
    upper = [[0 for x in range(n)]  
                for y in range(n)] 
                  
    # Decomposing matrix into Upper  
    # and Lower triangular matrix 
    for i in range(n):

        # Upper Triangular 
        for k in range(i, n):  
            # Summation of L(i, j) * U(j, k) 
            sum = 0 
            for j in range(i): 
                sum += (lower[i][j] * upper[j][k]) 
            # Evaluating U(i, k) 
            upper[i][k] = mat[i][k] - sum

        # Lower Triangular 
        for k in range(i, n): 
            if (i == k): 
                lower[i][i] = 1 # Diagonal as 1 
            else: 
                # Summation of L(k, j) * U(j, i) 
                sum = 0 
                for j in range(i): 
                    sum += (lower[k][j] * upper[j][i]) 
  
                # Evaluating L(k, i) 
                lower[k][i] = int((mat[k][i] - sum) /
                                       upper[i][i]) 
  
    # setw is for displaying nicely 
    print("Lower Triangular\t\tUpper Triangular") 
  
    # Displaying the result : 
    for i in range(n):  
        # Lower 
        for j in range(n): 
            print(lower[i][j], end = "\t")  
        print("", end = "\t") 
  
        # Upper 
        for j in range(n): 
            print(upper[i][j], end = "\t") 
        print("") 