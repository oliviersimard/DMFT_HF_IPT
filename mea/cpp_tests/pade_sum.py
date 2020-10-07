import matplotlib.pyplot as plt
import h5py
import numpy as np
from optparse import OptionParser
from re import findall
from sys import exit
from math import isclose
from scipy.integrate import simps
from operator import itemgetter

ETA = 0.001
TAILCUT = 50

def get_derivative(p1 : float, p2 : float, p3 : float, p4 : float, delta_x : float) -> float:
    """p1, p2, p3 and p4 are the neighbouring points to the target points at which the derivative is looked for. delta_x is supposed to
    be constant and represents the step between two images of the function.
    """
    der = ( 1.0/12.0*p1 - 2.0/3.0*p2 + 2.0/3.0*p3 - 1.0/12.0*p4 ) / delta_x
    return der

def lagrange_interpolation(yn_m_2 : float,yn_m_1 : float,yn_p_1 : float,yn_p_2 : float) -> float:
    return -1.0/6.0*yn_m_2 + 2.0/3.0*yn_m_1 + 2.0/3.0*yn_p_1 - 1.0/6.0*yn_p_2

def pade(omega_n, f_n, omega):
    """Parameters:
    - omega_n: real-valued array containing the Matsubara frequencies
    - f_n: complex-valued function to be analytically continued
    - omega: real-valued array containing the real frequencies
    """

    assert len(omega_n)==len(f_n), "The lengths of Matsubara array, Re F(iwn) and Im F(iwn) have to be equal."

    #Keeping only the positive Matsubara frequencies.
    f_n = np.array([el[1] for el in zip(omega_n,f_n) if el[0]>=0.0],dtype=complex)
    # setting imag part to zero
    for i in range(len(f_n)):
        f_n[i] = complex(f_n[i].real,0.0)
    omega_n = omega_n[omega_n>=0.0] 
  
    N = len(omega_n)-TAILCUT # Substracting the last Matsubara frequencies
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
        z = omega[k] + 1.j*ETA
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

    is_ready_to_plot = False
    type_of_plot = {"plot_3D" : 0, "plot_2D_colormap" : 1}
    is_this_plot = 2
    parser = OptionParser()
    AVERAGE_OR_SIMPS = "AVERAGE" ## can be "AVERAGE" or "SIMPS". The choice selects the inetagration scheme.
    DIM = 1 # can be 1 or 2
    SAVE_PADE = False # to enable the PADE analytical continuation to be saved...
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
        if "_infinite_" in options.data or "_2D_" in options.data:
            Ntau = Ntau//4
        else:
            Ntau = Ntau//2
        U = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",options.data)[0])

        print("beta: ", beta, " and Ntau: ", Ntau, " and U: ", U)
        
        omega = [ -wmax + 2.0*i*(wmax)/(Nreal-1.) for i in range(Nreal) ] # Symmetric grid
        qn_array = np.array([(2.0*n)*np.pi/beta for n in range(Ntau)],dtype=complex)

        N_q = len(hf.keys())
        len_q_array = int( np.sqrt(N_q) )
        omega_q_chi_jj = np.empty((N_q,Nreal),dtype=complex)
        omega_q_chi_szsz = np.empty((N_q,Nreal),dtype=complex)
        sum_chi_q_iqn_jj = np.zeros((Ntau,),dtype=complex)
        sum_chi_q_iqn_szsz = np.zeros((Ntau,),dtype=complex)
        k_b_t_arr_jj = np.empty((N_q,),dtype=tuple)
        k_b_t_arr_szsz = np.empty((N_q,),dtype=tuple)
        plt.figure(0)
        for i,str_tmp in enumerate(hf.keys()):
            print(str_tmp)
            if DIM == 1:
                k_bar = float(findall(r"(?<=kbar_)([-+]?\d*\.\d+)",str_tmp)[0])
                k_tilde = float(findall(r"(?<=ktilde_)([-+]?\d*\.\d+)",str_tmp)[0])
            elif DIM == 2:
                k_bar = float(findall(r"(?<=ktildex_)([-+]?\d*\.\d+)",str_tmp)[0]) # k_tilde_x
                k_tilde = float(findall(r"(?<=ktildey_)([-+]?\d*\.\d+)",str_tmp)[0]) # k_tilde_y
            # jj
            chi_q_iqn_jj_im = hf.get(str_tmp).get('jj')['IM']
            chi_q_iqn_jj_re = hf.get(str_tmp).get('jj')['RE']
            chi_q_iqn_jj = zip(chi_q_iqn_jj_re,chi_q_iqn_jj_im)
            chi_q_iqn_jj = list(map(lambda x: complex(x[0],x[1]),chi_q_iqn_jj))
            if AVERAGE_OR_SIMPS=="AVERAGE":
                if ( isclose(k_bar,np.pi,abs_tol=1e-4) or isclose(k_bar,-np.pi,abs_tol=1e-4) ) or ( isclose(k_tilde,np.pi,abs_tol=1e-4) or isclose(k_tilde,-np.pi,abs_tol=1e-4) ):
                    if ( isclose(k_bar,np.pi,abs_tol=1e-4) or isclose(k_bar,-np.pi,abs_tol=1e-4) ) and ( isclose(k_tilde,np.pi,abs_tol=1e-4) or isclose(k_tilde,-np.pi,abs_tol=1e-4) ):
                        sum_chi_q_iqn_jj += 0.25/N_q*np.array(chi_q_iqn_jj)
                    sum_chi_q_iqn_jj += 0.5/N_q*np.array(chi_q_iqn_jj)
                else:
                    sum_chi_q_iqn_jj += 1.0/N_q*np.array(chi_q_iqn_jj)
            elif AVERAGE_OR_SIMPS=="SIMPS":
                k_b_t_arr_jj[i] = (chi_q_iqn_jj,k_bar,k_tilde)
            
            # szsz
            chi_q_iqn_szsz_im = hf.get(str_tmp).get('szsz')['IM']
            chi_q_iqn_szsz_re = hf.get(str_tmp).get('szsz')['RE']
            chi_q_iqn_szsz = zip(chi_q_iqn_szsz_re,chi_q_iqn_szsz_im)
            chi_q_iqn_szsz = list(map(lambda x: complex(x[0],x[1]),chi_q_iqn_szsz))
            if AVERAGE_OR_SIMPS=="AVERAGE":
                if ( isclose(k_bar,np.pi,abs_tol=1e-4) or isclose(k_bar,-np.pi,abs_tol=1e-4) ) or ( isclose(k_tilde,np.pi,abs_tol=1e-4) or isclose(k_tilde,-np.pi,abs_tol=1e-4) ):
                    if ( isclose(k_bar,np.pi,abs_tol=1e-4) or isclose(k_bar,-np.pi,abs_tol=1e-4) ) and ( isclose(k_tilde,np.pi,abs_tol=1e-4) or isclose(k_tilde,-np.pi,abs_tol=1e-4) ):
                        sum_chi_q_iqn_szsz += 0.25/N_q*np.array(chi_q_iqn_szsz)
                    sum_chi_q_iqn_szsz += 0.5/N_q*np.array(chi_q_iqn_szsz)
                else:
                    sum_chi_q_iqn_szsz += 1.0/N_q*np.array(chi_q_iqn_szsz)
            elif AVERAGE_OR_SIMPS=="SIMPS":
                k_b_t_arr_szsz[i] = (chi_q_iqn_szsz,k_bar,k_tilde)

        if AVERAGE_OR_SIMPS=="SIMPS":
            k_b_t_arr_jj = sorted(k_b_t_arr_jj,key=itemgetter(1,2))
            k_b_t_arr_szsz = sorted(k_b_t_arr_szsz,key=itemgetter(1,2))
            k_array = [] # get k-array
            for ii in range(0,N_q,len_q_array):
                k_array.append(k_b_t_arr_szsz[ii][1])

            w_k_integrated_step_one_jj = []
            w_k_integrated_step_one_szsz = []
            for offset in range(0,N_q,len_q_array):
                omega_k_outer_integral_jj = []; omega_k_outer_integral_szsz = []
                for jj in range(Ntau):
                    omega_k_inner_integral_jj = []; omega_k_inner_integral_szsz = []
                    for ii in range(0+offset,len_q_array+offset):
                        omega_k_inner_integral_jj.append(k_b_t_arr_jj[ii][0][jj])
                        omega_k_inner_integral_szsz.append(k_b_t_arr_szsz[ii][0][jj])

                    omega_k_outer_integral_jj.append(1.0/(2.0*np.pi)*simps(omega_k_inner_integral_jj,x=k_array))
                    omega_k_outer_integral_szsz.append(1.0/(2.0*np.pi)*simps(omega_k_inner_integral_szsz,x=k_array))
                w_k_integrated_step_one_jj.append((k_array[int(offset/len_q_array)],omega_k_outer_integral_jj))
                w_k_integrated_step_one_szsz.append((k_array[int(offset/len_q_array)],omega_k_outer_integral_szsz))
            
            for jj in range(Ntau):
                kk_tmp_integral_jj = []; kk_tmp_integral_szsz = []
                for kk in range(len_q_array):
                    kk_tmp_integral_jj.append(w_k_integrated_step_one_jj[kk][1][jj])
                    kk_tmp_integral_szsz.append(w_k_integrated_step_one_szsz[kk][1][jj])
                sum_chi_q_iqn_jj[jj] = 1.0/(2.0*np.pi)*simps(kk_tmp_integral_jj,x=k_array)
                sum_chi_q_iqn_szsz[jj] = 1.0/(2.0*np.pi)*simps(kk_tmp_integral_szsz,x=k_array)

        chi_q_w_jj = pade(qn_array,sum_chi_q_iqn_jj,omega)
        chi_q_w_szsz = pade(qn_array,sum_chi_q_iqn_szsz,omega)
        chi_q_w_jj_im = np.array(list(map(lambda x: x.imag, chi_q_w_jj)))
        chi_q_w_szsz_im = np.array(list(map(lambda x: x.imag, chi_q_w_szsz)))

        norm_chi_q_w_jj = np.zeros((Nreal,),dtype=float)
        norm_chi_q_w_szsz = np.zeros((Nreal,),dtype=float)
        for j,om in enumerate(omega):
            if om==0.0:
                der_jj = get_derivative(chi_q_w_jj_im[j-2],chi_q_w_jj_im[j-1],chi_q_w_jj_im[j+1],chi_q_w_jj_im[j+2],2.0*wmax/1000)
                if np.isnan(der_jj):
                    norm_chi_q_w_jj[j] = 1e-8
                else:
                    norm_chi_q_w_jj[j] = der_jj
                print("der jj: ", norm_chi_q_w_jj[j])
                norm_chi_q_w_szsz[j] = get_derivative(chi_q_w_szsz_im[j-2],chi_q_w_szsz_im[j-1],chi_q_w_szsz_im[j+1],chi_q_w_szsz_im[j+2],2.0*wmax/1000)
                print("der szsz: ", norm_chi_q_w_szsz[j])
            else:
                norm_chi_q_w_jj[j] = chi_q_w_jj_im[j]/om
                norm_chi_q_w_szsz[j] = chi_q_w_szsz_im[j]/om

        ##
        plt.plot(qn_array,list(map(lambda x: x.real, sum_chi_q_iqn_jj)),marker='*',c="red",label="jj")
        plt.plot(qn_array,list(map(lambda x: x.real, sum_chi_q_iqn_szsz)),marker='*',c="blue",label="szsz")
        with open(options.data+"_iqn_szsz","w") as f:
            for i in range(len(qn_array)):
                f.write("{0:.8f}\t{1:.8f}\t{2:.8f}\n".format(qn_array[i].real,sum_chi_q_iqn_szsz[i].real,sum_chi_q_iqn_szsz[i].imag))
        f.close()
        with open(options.data+"_iqn_jj","w") as f:
            for i in range(len(qn_array)):
                f.write("{0:.8f}\t{1:.8f}\t{2:.8f}\n".format(qn_array[i].real,sum_chi_q_iqn_jj[i].real,sum_chi_q_iqn_jj[i].imag))
        f.close()
        if SAVE_PADE:
            with h5py.File(options.data+".pade_wmax_{0}_m_{1}_eta_{2}".format(wmax,TAILCUT,ETA),"a") as hfpade:
                grp = hfpade.create_group("susc")
                # jj dataset
                grp.create_dataset("jj", (Nreal,), dtype=float, data=norm_chi_q_w_jj)
                # szsz dataset
                grp.create_dataset("szsz", (Nreal,), dtype=float, data=norm_chi_q_w_szsz)
            hfpade.close()
        plt.legend()
        plt.show()
        plt.clf()
