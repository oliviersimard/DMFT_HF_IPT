import matplotlib.pyplot as plt
import h5py
import numpy as np
from optparse import OptionParser
from re import findall
from sys import exit
from math import isclose

def get_derivative(p1 : float, p2 : float, p3 : float, p4 : float, delta_x : float) -> float:
    """p1, p2, p3 and p4 are the neighbouring points to the target points at which the derivative is looked for. delta_x is supposed to
    be constant and represents the step between two images of the function.
    """
    der = ( 1.0/12.0*p1 - 2.0/3.0*p2 + 2.0/3.0*p3 - 1.0/12.0*p4 ) / delta_x
    return der

def pade(omega_n, f_n, omega):
    """Parameters:
    - omega_n: real-valued array containing the Matsubara frequencies
    - f_n: complex-valued function to be analyticallay continued
    - omega: real-valued array containing the real frequencies
    """

    #Keeping only the positive Matsubara frequencies.
    f_n = np.array([el[1] for el in zip(omega_n,f_n) if el[0]>=0.0],dtype=complex)
    omega_n = omega_n[omega_n>=0.0] 
    
    eta = 0.001
    
    assert len(omega_n)==len(f_n), "The lengths of Matsubara array, Re F(iwn) and Im F(iwn) have to be equal."

    N = len(omega_n)-7 # Substracting the last Matsubara frequencies
    g = np.zeros((N, N), dtype = np.clongdouble)
    omega_n = omega_n[0:N]

    f_n = f_n[0:N]
    g[0,:] = f_n[:]
    g[1,:] = (f_n[0] - f_n)/((1.j*omega_n -1.j*omega_n[0])*f_n)
    for k in range(2, N):
        g[k, :] = (g[k-1, k-1] - g[k-1, :])/((1.j*omega_n - 1.j*omega_n[k-1])*g[k-1, :])

    a = np.diagonal(g)
    
    A = np.zeros((N, ), dtype = np.clongdouble)
    B = np.zeros((N, ), dtype = np.clongdouble)
    P = np.zeros((N, ), dtype = np.clongdouble)
    fw = []
    
    for k in range(len(omega)):
        z = omega[k] + 1.j*eta
        P[0] = a[0]
        A[1] = 1.
        B[1]= 1.+a[1]*(z - 1.j*omega_n[0])
        P[1] = P[0]*A[1]/B[1]
        for c in range(2, N):
            A[c] = 1. + a[c]*(z - 1.j*omega_n[c-1])/A[c-1]
            B[c] = 1. + a[c]*(z - 1.j*omega_n[c-1])/B[c-1]
            P[c] = P[c-1]*A[c]/B[c]
        fw.append(P[-1])
        
    return fw


if __name__=="__main__":

    is_ready_to_plot = True
    type_of_plot = {"plot_3D" : 0, "plot_2D_colormap" : 1}
    is_this_plot = 1
    parser = OptionParser()

    parser.add_option("--data", dest="data", default="data.in")

    if not is_ready_to_plot:
        
        parser.add_option("--wmax", dest="wmax", default=20)
        parser.add_option("--Nwr" , dest="Nwr" , default=1001)

        (options, args) = parser.parse_args()

        Nreal = int(options.Nwr)
        wmax = float(options.wmax)
        hf = h5py.File(options.data,"r")

        beta = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",options.data)[0])
        Ntau = int(findall(r"(?<=Ntau_)(\d+)",options.data)[0])
        U = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",options.data)[0])

        print("beta: ", beta, " and Ntau: ", Ntau, " and U: ", U)
        
        omega = [ -wmax + 2.0*i*(wmax)/(Nreal-1.) for i in range(Nreal) ] # Symmetric grid
        qn_array = np.array([(2.0*n)*np.pi/beta for n in range(Ntau-1)],dtype=complex)

        N_q = len(hf.keys())
        
        omega_q_chi = np.empty((N_q,Nreal),dtype=complex)
        q_array = np.empty((N_q,),dtype=float)
        plt.figure(0)
        for i,q_val in enumerate(hf.keys()):
            print(q_val)
            q_array[i] = float(q_val.split('_')[-1])
            chi_q_iqn = hf.get(q_val)
            chi_q_iqn = np.array(chi_q_iqn)
            # Useful when using the linear spline to cut off the tail
            if U==0.0:
                if isclose(q_array[i],0.0,abs_tol=1e-5) or isclose(q_array[i],2.0*np.pi,abs_tol=1e-5):
                    continue
                print("before: ",len(chi_q_iqn))
                chi_q_iqn = chi_q_iqn[chi_q_iqn.real>=1e-6]
                qn_array = qn_array[:len(chi_q_iqn)]
                print("after: ",len(chi_q_iqn))

            chi_q_w = pade(qn_array,chi_q_iqn,omega)
            omega_q_chi[i,:] = chi_q_w
            chi_q_w_im = np.array(list(map(lambda x: x.imag, chi_q_w)))
            if np.mod(i,10)==0:
                plt.plot(omega,chi_q_w_im,marker='*',c="red")
            with h5py.File(options.data+".pade_wmax_{0}".format(wmax),"a") as hfpade:
                hfpade.create_dataset(q_val, (Nreal,), dtype=np.complex, data=chi_q_w)

            hfpade.close()
            
        plt.show()
        print("finish line: ", omega_q_chi[0,:])
    
    else:
        # Data treatment before the plotting
        (options, args) = parser.parse_args()

        wmax = float(findall(r"(?<=wmax_)(\d*\.\d+|\d+)",options.data)[0])
        U = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",options.data)[0])
        beta = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",options.data)[0])

        hf = h5py.File(options.data,"r")
        N_q = len(hf.keys())
        N_omega = hf.get(list(hf.keys())[0])
        N_omega = N_omega.shape[0]

        omega = [ -wmax + 2.0*i*(wmax)/(N_omega-1.) for i in range(N_omega) ] # Symmetric grid
        # If U==0.0, it means the q=0.0 and 2.0*pi have been left out due to problems with analytical continuation using Padé (only one point in real part non zero)
        if U==0.0:
            q_arr = np.linspace(0.0,2.0,N_q+2)
            q_arr = np.delete(q_arr,0,0) # deleting the first element
            q_arr = np.delete(q_arr,-1,0) # deleting the last element
        else:
            q_arr = np.linspace(0.0,2.0,N_q)

        Omega, Q = np.meshgrid(omega,q_arr)

        mesh_Omega_Q = np.empty((N_q,N_omega),dtype=float)

        re_sigma_jj = np.empty((N_omega,),dtype=float)
        for i,q_val in enumerate(hf.keys()):
            q_tmp_val = hf.get(q_val)
            q_tmp_val = np.array(q_tmp_val)
            q_tmp_val_im = np.array(list(map(lambda x: x.imag,q_tmp_val)))
            for j,om in enumerate(omega):
                if om==0.0:
                    re_sigma_jj[j] = get_derivative(q_tmp_val_im[j-2],q_tmp_val_im[j-1],q_tmp_val_im[j+1],q_tmp_val_im[j+2],2.0*(wmax)/(N_omega-1.)) 
                else:
                    re_sigma_jj[j] = q_tmp_val_im[j] / om  
            mesh_Omega_Q[i,:] = re_sigma_jj
        
        # Different plotting options
        if is_this_plot == type_of_plot["plot_3D"]:
            from mpl_toolkits import mplot3d

            fig = plt.figure()
            ax = plt.axes(projection="3d")

            #ax.plot_wireframe(Omega, Q, mesh_Omega_Q, color='green')
            ax.plot_surface(Omega, Q, mesh_Omega_Q, rstride=1, cstride=1, edgecolor='none')#, cmap='inferno', edgecolor='none')
            ax.set_xlabel(r"$\omega$")
            ax.set_ylabel(r"$\mathbf{q}/\pi$")
            ax.set_zlabel(r"$\operatorname{Re}\sigma_{jj}(\omega,\mathbf{q})$")

            plt.show()

        elif is_this_plot == type_of_plot["plot_2D_colormap"]:
            
            plt.ylabel(r"$\omega$")
            plt.xlabel(r"$\mathbf{q}/\pi$")
            plt.title(r"$\operatorname{Re}\sigma_{jj}(\omega,\mathbf{q})$ without velocities at "+r"$U={0}t$ and $\beta={1}/t$".format(U,beta))
            #plt.title(r"$\operatorname{Re}\sigma_{jj}(\omega,\mathbf{q})$ at "+r"$U={0}t$ and $\beta={1}/t$".format(U,beta))
            plt.xticks(np.arange(min(q_arr),max(q_arr)+0.25,0.25))
            plt.yticks(np.arange(-wmax,wmax+2.0,2.0))
            im = plt.imshow(mesh_Omega_Q.transpose(), cmap=plt.cm.Blues, extent=(min(q_arr), max(q_arr), -wmax, wmax), vmax=6, vmin=0, aspect='auto', interpolation='bilinear')  
    
            plt.colorbar(im)

            plt.show()