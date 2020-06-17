import numpy as np
import matplotlib.pyplot as plt
import h5py
from sys import exit
from re import findall
from math import isclose
from copy import deepcopy

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage[charter]{mathdesign}\usepackage{amsmath}"]

def get_derivative(p1 : float, p2 : float, p3 : float, p4 : float, delta_x : float) -> float:
    """p1, p2, p3 and p4 are the neighbouring points to the target points at which the derivative is looked for. delta_x is supposed to
    be constant and represents the step between two images of the function.
    """
    der = ( 1.0/12.0*p1 - 2.0/3.0*p2 + 2.0/3.0*p3 - 1.0/12.0*p4 ) / delta_x
    return der

if __name__=="__main__":

    # filenames whose data we want to combine. They have to have the same U, same temperature and dimensionnality.
    filename_bare = "cpp_tests/susceptibilities/bb_1D_U_8.000000_beta_7.000000_Ntau_2048_Nk_501_isjj_0.hdf5.pade_wmax_20.0"
    filename_corr = "cpp_tests/susceptibilities/bb_1D_U_8.000000_beta_7.000000_Ntau_256_Nq_21_Nk_29_isjj_0_single_ladder_sum_1.hdf5.pade_wmax_20.0"

    wmax_bare = float(findall(r"(?<=wmax_)(\d*\.\d+|\d+)",filename_bare)[0]); wmax_corr = float(findall(r"(?<=wmax_)(\d*\.\d+|\d+)",filename_corr)[0])
    Ntau_bare = int(findall(r"(?<=Ntau_)(\d+)",filename_bare)[0]); Ntau_corr = int(findall(r"(?<=Ntau_)(\d+)",filename_corr)[0])
    beta_bare = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_bare)[0]); beta_corr = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_corr)[0])
    U_bare = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_bare)[0]); U_corr = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_corr)[0])
    is_jj_bare = bool(int(*findall(r'(?<=isjj_)(\d+)',filename_bare))); is_jj_corr = bool(int(*findall(r'(?<=isjj_)(\d+)',filename_corr)))
   
    # for plots
    str_plot_corr = ""
    if "single_ladder" in filename_corr:
        str_plot_corr = "single ladder"
    else:
        str_plot_corr = "infinite ladder"

    assert U_bare==U_corr and beta_bare==beta_corr and is_jj_bare==is_jj_corr, "Temperature, U and type of susceptibility have to correspond for bare and corrections to susceptibility."

    # taking maximum of w between two files
    wmax = max(wmax_bare,wmax_corr); beta=beta_bare
    omega = np.array([-wmax + 2.0*i*wmax/(1000) for i in range(1001)])
    tmp_k_t_k_b_omega_re_sigma_bare = np.zeros((len(omega),),dtype=float); tmp_k_t_k_b_omega_re_sigma_corr = np.zeros((len(omega),),dtype=float)
    average_k_t_k_b = np.zeros((len(omega),),dtype=float)
    # extracting the non-interacting susceptibility
    hf = h5py.File(filename_bare,"r")
    for i,q_val in enumerate(hf.keys()):
        q = float(str(q_val).split("_")[-1])
        if q==0.0:
            tmp_k_t_k_b_omega = np.array(hf.get(q_val))
            tmp_k_t_k_b_omega = np.array(list(map(lambda x: x.imag,tmp_k_t_k_b_omega)))
            for j,om in enumerate(omega):
                if om==0.0:
                    tmp_k_t_k_b_omega_re_sigma_bare[j] = get_derivative(tmp_k_t_k_b_omega[j-2],tmp_k_t_k_b_omega[j-1],tmp_k_t_k_b_omega[j+1],tmp_k_t_k_b_omega[j+2],2.0*wmax/1000)
                else:
                    tmp_k_t_k_b_omega_re_sigma_bare[j] = tmp_k_t_k_b_omega[j]/om
    hf.close()
    # extracting the correction to the susceptibility

    if "_1D_" in filename_bare and "_1D_" in filename_corr:
        hf = h5py.File(filename_corr,"r")
        list_keys = list(hf.keys())
        N_k_t_k_b = len(list_keys)
        k_b_t_arr = np.empty((N_k_t_k_b,),dtype=tuple)
        for i in range(N_k_t_k_b):
            str_k_t_k_b = str(list_keys[i])
            k_b = float(findall(r"(?<=kbar_)([-+]?\d*\.\d+)",str_k_t_k_b)[0])
            k_t = float(findall(r"(?<=ktilde_)([-+]?\d*\.\d+)",str_k_t_k_b)[0])
            diff_k = k_b-k_t
            tmp_k_t_k_b_omega = np.array(hf.get(str_k_t_k_b))
            tmp_k_t_k_b_omega = np.array(list(map(lambda x: x.imag,tmp_k_t_k_b_omega)))
            k_b_t_arr[i] = (diff_k,tmp_k_t_k_b_omega,k_b,k_t)

        k_b_t_arr = sorted(k_b_t_arr,key=lambda x: x[0])
        # mesh_Omega_Q = np.empty((N_k_t_k_b,len(omega)),dtype=float)

        for i in range(N_k_t_k_b):
            tmp_k_t_k_b_omega = k_b_t_arr[i][1]
            for j,om in enumerate(omega):
                if om==0.0:
                    tmp_k_t_k_b_omega_re_sigma_corr[j] = get_derivative(tmp_k_t_k_b_omega[j-2],tmp_k_t_k_b_omega[j-1],tmp_k_t_k_b_omega[j+1],tmp_k_t_k_b_omega[j+2],2.0*wmax/1000)
                else:
                    tmp_k_t_k_b_omega_re_sigma_corr[j] = tmp_k_t_k_b_omega[j]/om
            
            # mesh_Omega_Q[i,:] = tmp_k_t_k_b_omega_re_sigma_corr
            # Computing the average
            if not np.isnan(tmp_k_t_k_b_omega_re_sigma_corr).any():
                if isclose(np.abs(k_b_t_arr[i][2]),np.pi,abs_tol=1e-5) or isclose(np.abs(k_b_t_arr[i][3]),0.0,abs_tol=1e-5):
                    if isclose(np.abs(k_b_t_arr[i][2]),np.pi,abs_tol=1e-5) and isclose(np.abs(k_b_t_arr[i][3]),0.0,abs_tol=1e-5):
                        average_k_t_k_b += 0.25*tmp_k_t_k_b_omega_re_sigma_corr
                    average_k_t_k_b += 0.5*tmp_k_t_k_b_omega_re_sigma_corr
                else:
                    average_k_t_k_b += tmp_k_t_k_b_omega_re_sigma_corr

        average_k_t_k_b *= 1.0/np.sqrt(N_k_t_k_b)/np.sqrt(N_k_t_k_b)
        corr_susceptibility = deepcopy(average_k_t_k_b)

    for j in range(len(omega)):
        average_k_t_k_b[j] += tmp_k_t_k_b_omega_re_sigma_bare[j]

    hf.close()

    fig = plt.figure(figsize=(10, 9))
    grid = plt.GridSpec(2, 2, wspace=0.35, hspace=0.40)

    if "_1D_" in filename_bare and "_1D_" in filename_corr:
        fig.suptitle("1D")
    elif "_2D_" in filename_bare and "_2D_" in filename_corr:
        fig.suptitle("2D")

    top_panel_left = fig.add_subplot(grid[0,0])
    top_panel_left.grid()

    if (is_jj_corr):
        top_panel_left.set_title(r"k-averaged total $\langle j_xj_x\rangle$ $(U={0}, \beta={1})$".format(U_corr,beta))
    else:
        top_panel_left.set_title(r"k-averaged total $\langle S_zS_z\rangle$ $(U={0}, \beta={1})$".format(U_corr,beta))
    top_panel_left.set_xlabel(r"$\omega$",labelpad=-0.4)
    top_panel_left.set_ylabel(r"$\operatorname{Re}\sigma_{jj}(\omega,\mathbf{q}=\mathbf{0})$")

    top_panel_left.plot(omega,average_k_t_k_b,marker='o',ms=2.0,c="red")#,label=r"$\mathbf{q}$=%.2f" % (0.0))

    #top_panel_left.legend()
    # 
    top_panel_right = fig.add_subplot(grid[0,1],sharey=top_panel_left)
    top_panel_right.grid()
    
    if (is_jj_corr):
        top_panel_right.set_title(r"bare $\langle j_xj_x\rangle$ $(U={0}, \beta={1})$".format(U_corr,beta))
    else:
        top_panel_right.set_title(r"bare $\langle S_zS_z\rangle$ response $(U={0}, \beta={1})$".format(U_corr,beta))
    
    top_panel_right.set_xlabel(r"$\omega$",labelpad=-0.4)

    top_panel_right.plot(omega,tmp_k_t_k_b_omega_re_sigma_bare,marker='>',ms=2.0,c="green")#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    
    #top_panel_right.legend()
    #
    lower_panel = fig.add_subplot(grid[1,0:])
    lower_panel.grid()

    if (is_jj_corr):
        lower_panel.set_title(r"correction to bare $\langle j_xj_x\rangle$ for $U={0}, \beta={1}$ (".format(U_corr,beta)+str_plot_corr+r")")
    else:
        lower_panel.set_title(r"correction to bare $\langle S_zS_z\rangle$ for $U={0}, \beta={1}$ (".format(U_corr,beta)+str_plot_corr+r")")

    lower_panel.plot(omega,corr_susceptibility,marker='v',ms=2.0,c="black")
    lower_panel.set_ylabel(r"$\operatorname{Re}\sigma_{jj}(\omega,\mathbf{q}=\mathbf{0})$")
    lower_panel.set_xlabel(r"$\omega$")

    plt.gcf().set_size_inches(18.5/2.54,12/2.54)
    if "_1D_" in filename_corr and "_1D_" in filename_bare:
        plt.savefig("Figure_susceptibility_comparison_isjj_{0}_U_{1:.2f}_beta_{2:.2f}_1D_".format(is_jj_corr,U_corr,beta)+str_plot_corr.replace(" ","_")+".pdf")
    elif "_2D_" in filename_corr and "_2D_" in filename_bare:
        plt.savefig("Figure_susceptibility_comparison_isjj_{0}_U_{1:.2f}_beta_{2:.2f}_2D_".format(is_jj_corr,U_corr,beta)+str_plot_corr.replace(" ","_")+".pdf")
    # plt.show()
