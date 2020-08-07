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
    # beta 1
    filename_bare_1 = "cpp_tests/figs_article/bb_1D_U_14.000000_beta_6.800000_Ntau_2048_Nk_401_NCA.hdf5.pade_wmax_25.0_m_7_eta_0.001"
    filename_corr_1 = "cpp_tests/figs_article/bb_1D_U_14.000000_beta_6.800000_Ntau_256_Nq_31_Nk_27_NCA_single_ladder_sum.hdf5.pade_wmax_25.0_m_10_eta_0.001"

    wmax_bare_1 = float(findall(r"(?<=wmax_)(\d*\.\d+|\d+)",filename_bare_1)[0]); wmax_corr_1 = float(findall(r"(?<=wmax_)(\d*\.\d+|\d+)",filename_corr_1)[0])
    Ntau_bare_1 = int(findall(r"(?<=Ntau_)(\d+)",filename_bare_1)[0]); Ntau_corr_1 = int(findall(r"(?<=Ntau_)(\d+)",filename_corr_1)[0])
    beta_bare_1 = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_bare_1)[0]); beta_corr_1 = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_corr_1)[0])
    U_bare_1 = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_bare_1)[0]); U_corr_1 = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_corr_1)[0])
    # beta 2
    filename_bare_2 = "cpp_tests/figs_article/bb_1D_U_14.000000_beta_6.000000_Ntau_2048_Nk_401_NCA.hdf5.pade_wmax_25.0_m_7_eta_0.001"
    filename_corr_2 = "cpp_tests/figs_article/bb_1D_U_14.000000_beta_6.000000_Ntau_256_Nq_31_Nk_27_NCA_single_ladder_sum.hdf5.pade_wmax_25.0_m_100_eta_0.001"

    wmax_bare_2 = float(findall(r"(?<=wmax_)(\d*\.\d+|\d+)",filename_bare_2)[0]); wmax_corr_2 = float(findall(r"(?<=wmax_)(\d*\.\d+|\d+)",filename_corr_2)[0])
    Ntau_bare_2 = int(findall(r"(?<=Ntau_)(\d+)",filename_bare_2)[0]); Ntau_corr_2 = int(findall(r"(?<=Ntau_)(\d+)",filename_corr_2)[0])
    beta_bare_2 = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_bare_2)[0]); beta_corr_2 = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_corr_2)[0])
    U_bare_2 = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_bare_2)[0]); U_corr_2 = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_corr_2)[0])
   
    # for plots
    str_plot_corr = ""
    if "single_ladder" in filename_corr_1 and "single_ladder" in filename_corr_2:
        str_plot_corr = "single ladder"
    elif "infinite_ladder" in filename_corr_1 and "infinite_ladder" in filename_corr_2:
        str_plot_corr = "infinite ladder"
    else:
        raise ValueError("Error: Type of correction has to correspond.")

    assert U_bare_1==U_corr_1==U_bare_2==U_corr_2 and beta_bare_1==beta_corr_1 and beta_bare_2==beta_corr_2, "Temperature, U and type of susceptibility have to correspond for bare and corrections to susceptibility."

    # taking maximum of w between two files
    wmax = max(wmax_bare_1,wmax_corr_1)
    omega = np.array([-wmax + 2.0*i*wmax/(1000) for i in range(1001)])
    # beta 1
    tmp_k_t_k_b_omega_re_sigma_bare_jj_1 = np.zeros((len(omega),),dtype=float); tmp_k_t_k_b_omega_re_sigma_corr_jj_1 = np.zeros((len(omega),),dtype=float)
    tmp_k_t_k_b_omega_re_sigma_bare_szsz_1 = np.zeros((len(omega),),dtype=float); tmp_k_t_k_b_omega_re_sigma_corr_szsz_1 = np.zeros((len(omega),),dtype=float)
    average_k_t_k_b_jj_1 = np.zeros((len(omega),),dtype=float); average_k_t_k_b_szsz_1 = np.zeros((len(omega),),dtype=float)
    # extracting the non-interacting susceptibility
    hf = h5py.File(filename_bare_1,"r")
    for i,q_val in enumerate(hf.keys()):
        q = float(str(q_val).split("_")[-1])
        if q==0.0:
            tmp_k_t_k_b_omega_jj = np.array(hf.get(q_val).get('jj'))
            tmp_k_t_k_b_omega_jj = np.array(list(map(lambda x: x.imag,tmp_k_t_k_b_omega_jj)))
            tmp_k_t_k_b_omega_szsz = np.array(hf.get(q_val).get('szsz'))
            tmp_k_t_k_b_omega_szsz = np.array(list(map(lambda x: x.imag,tmp_k_t_k_b_omega_szsz)))
            for j,om in enumerate(omega):
                if om==0.0:
                    tmp_k_t_k_b_omega_re_sigma_bare_jj_1[j] = get_derivative(tmp_k_t_k_b_omega_jj[j-2],tmp_k_t_k_b_omega_jj[j-1],tmp_k_t_k_b_omega_jj[j+1],tmp_k_t_k_b_omega_jj[j+2],2.0*wmax/1000)
                    tmp_k_t_k_b_omega_re_sigma_bare_szsz_1[j] = get_derivative(tmp_k_t_k_b_omega_szsz[j-2],tmp_k_t_k_b_omega_szsz[j-1],tmp_k_t_k_b_omega_szsz[j+1],tmp_k_t_k_b_omega_szsz[j+2],2.0*wmax/1000)
                else:
                    tmp_k_t_k_b_omega_re_sigma_bare_jj_1[j] = tmp_k_t_k_b_omega_jj[j]/om
                    tmp_k_t_k_b_omega_re_sigma_bare_szsz_1[j] = tmp_k_t_k_b_omega_szsz[j]/om
    hf.close()
    # beta 2
    tmp_k_t_k_b_omega_re_sigma_bare_jj_2 = np.zeros((len(omega),),dtype=float); tmp_k_t_k_b_omega_re_sigma_corr_jj_2 = np.zeros((len(omega),),dtype=float)
    tmp_k_t_k_b_omega_re_sigma_bare_szsz_2 = np.zeros((len(omega),),dtype=float); tmp_k_t_k_b_omega_re_sigma_corr_szsz_2 = np.zeros((len(omega),),dtype=float)
    average_k_t_k_b_jj_2 = np.zeros((len(omega),),dtype=float); average_k_t_k_b_szsz_2 = np.zeros((len(omega),),dtype=float)
    # extracting the non-interacting susceptibility
    hf = h5py.File(filename_bare_2,"r")
    for i,q_val in enumerate(hf.keys()):
        q = float(str(q_val).split("_")[-1])
        if q==0.0:
            tmp_k_t_k_b_omega_jj = np.array(hf.get(q_val).get('jj'))
            tmp_k_t_k_b_omega_jj = np.array(list(map(lambda x: x.imag,tmp_k_t_k_b_omega_jj)))
            tmp_k_t_k_b_omega_szsz = np.array(hf.get(q_val).get('szsz'))
            tmp_k_t_k_b_omega_szsz = np.array(list(map(lambda x: x.imag,tmp_k_t_k_b_omega_szsz)))
            for j,om in enumerate(omega):
                if om==0.0:
                    tmp_k_t_k_b_omega_re_sigma_bare_jj_2[j] = get_derivative(tmp_k_t_k_b_omega_jj[j-2],tmp_k_t_k_b_omega_jj[j-1],tmp_k_t_k_b_omega_jj[j+1],tmp_k_t_k_b_omega_jj[j+2],2.0*wmax/1000)
                    tmp_k_t_k_b_omega_re_sigma_bare_szsz_2[j] = get_derivative(tmp_k_t_k_b_omega_szsz[j-2],tmp_k_t_k_b_omega_szsz[j-1],tmp_k_t_k_b_omega_szsz[j+1],tmp_k_t_k_b_omega_szsz[j+2],2.0*wmax/1000)
                else:
                    tmp_k_t_k_b_omega_re_sigma_bare_jj_2[j] = tmp_k_t_k_b_omega_jj[j]/om
                    tmp_k_t_k_b_omega_re_sigma_bare_szsz_2[j] = tmp_k_t_k_b_omega_szsz[j]/om
    hf.close()

    # extracting the correction to the susceptibility
    # beta 1
    if "_1D_" in filename_bare_1 and "_1D_" in filename_corr_1:
        hf = h5py.File(filename_corr_1,"r")
        list_keys = list(hf.keys())
        N_k_t_k_b = len(list_keys)
        k_b_t_arr_jj = np.empty((N_k_t_k_b,),dtype=tuple)
        k_b_t_arr_szsz = np.empty((N_k_t_k_b,),dtype=tuple)
        for i in range(N_k_t_k_b):
            str_k_t_k_b = str(list_keys[i])
            k_b = float(findall(r"(?<=kbar_)([-+]?\d*\.\d+)",str_k_t_k_b)[0])
            k_t = float(findall(r"(?<=ktilde_)([-+]?\d*\.\d+)",str_k_t_k_b)[0])
            diff_k = k_b-k_t
            # jj
            tmp_k_t_k_b_omega_jj = np.array(hf.get(str_k_t_k_b).get('jj'))
            tmp_k_t_k_b_omega_jj = np.array(list(map(lambda x: x.imag,tmp_k_t_k_b_omega_jj)))
            k_b_t_arr_jj[i] = (diff_k,tmp_k_t_k_b_omega_jj,k_b,k_t)
            # szsz
            tmp_k_t_k_b_omega_szsz = np.array(hf.get(str_k_t_k_b).get('szsz'))
            tmp_k_t_k_b_omega_szsz = np.array(list(map(lambda x: x.imag,tmp_k_t_k_b_omega_szsz)))
            k_b_t_arr_szsz[i] = (diff_k,tmp_k_t_k_b_omega_szsz,k_b,k_t)


        k_b_t_arr_jj = sorted(k_b_t_arr_jj,key=lambda x: x[0])
        k_b_t_arr_szsz = sorted(k_b_t_arr_szsz,key=lambda x: x[0])
        # mesh_Omega_Q = np.empty((N_k_t_k_b,len(omega)),dtype=float)

        for i in range(N_k_t_k_b):
            tmp_k_t_k_b_omega_jj = k_b_t_arr_jj[i][1]
            tmp_k_t_k_b_omega_szsz = k_b_t_arr_szsz[i][1]
            for j,om in enumerate(omega):
                if om==0.0:
                    tmp_k_t_k_b_omega_re_sigma_corr_jj_1[j] = get_derivative(tmp_k_t_k_b_omega_jj[j-2],tmp_k_t_k_b_omega_jj[j-1],tmp_k_t_k_b_omega_jj[j+1],tmp_k_t_k_b_omega_jj[j+2],2.0*wmax/1000)
                    tmp_k_t_k_b_omega_re_sigma_corr_szsz_1[j] = get_derivative(tmp_k_t_k_b_omega_szsz[j-2],tmp_k_t_k_b_omega_szsz[j-1],tmp_k_t_k_b_omega_szsz[j+1],tmp_k_t_k_b_omega_szsz[j+2],2.0*wmax/1000)
                else:
                    tmp_k_t_k_b_omega_re_sigma_corr_jj_1[j] = tmp_k_t_k_b_omega_jj[j]/om
                    tmp_k_t_k_b_omega_re_sigma_corr_szsz_1[j] = tmp_k_t_k_b_omega_szsz[j]/om
            
            # mesh_Omega_Q[i,:] = tmp_k_t_k_b_omega_re_sigma_corr
            # Computing the average
            # jj
            if not np.isnan(tmp_k_t_k_b_omega_re_sigma_corr_jj_1).any():
                if isclose(np.abs(k_b_t_arr_jj[i][2]),np.pi,abs_tol=1e-5) or isclose(np.abs(k_b_t_arr_jj[i][3]),0.0,abs_tol=1e-5):
                    if isclose(np.abs(k_b_t_arr_jj[i][2]),np.pi,abs_tol=1e-5) and isclose(np.abs(k_b_t_arr_jj[i][3]),0.0,abs_tol=1e-5):
                        average_k_t_k_b_jj_1 += 0.25*tmp_k_t_k_b_omega_re_sigma_corr_jj_1
                    average_k_t_k_b_jj_1 += 0.5*tmp_k_t_k_b_omega_re_sigma_corr_jj_1
                else:
                    average_k_t_k_b_jj_1 += tmp_k_t_k_b_omega_re_sigma_corr_jj_1
            # szsz
            if not np.isnan(tmp_k_t_k_b_omega_re_sigma_corr_szsz_1).any():
                if isclose(np.abs(k_b_t_arr_szsz[i][2]),np.pi,abs_tol=1e-5) or isclose(np.abs(k_b_t_arr_szsz[i][3]),0.0,abs_tol=1e-5):
                    if isclose(np.abs(k_b_t_arr_szsz[i][2]),np.pi,abs_tol=1e-5) and isclose(np.abs(k_b_t_arr_szsz[i][3]),0.0,abs_tol=1e-5):
                        average_k_t_k_b_szsz_1 += 0.25*tmp_k_t_k_b_omega_re_sigma_corr_szsz_1
                    average_k_t_k_b_szsz_1 += 0.5*tmp_k_t_k_b_omega_re_sigma_corr_szsz_1
                else:
                    average_k_t_k_b_szsz_1 += tmp_k_t_k_b_omega_re_sigma_corr_szsz_1

        average_k_t_k_b_jj_1 *= 1.0/np.sqrt(N_k_t_k_b)/np.sqrt(N_k_t_k_b)
        corr_susceptibility_jj_1 = deepcopy(average_k_t_k_b_jj_1)

        average_k_t_k_b_szsz_1 *= 1.0/np.sqrt(N_k_t_k_b)/np.sqrt(N_k_t_k_b)
        corr_susceptibility_szsz_1 = deepcopy(average_k_t_k_b_szsz_1)


    for j in range(len(omega)):
        average_k_t_k_b_jj_1[j] += tmp_k_t_k_b_omega_re_sigma_bare_jj_1[j]
        average_k_t_k_b_szsz_1[j] += tmp_k_t_k_b_omega_re_sigma_bare_szsz_1[j]

    hf.close()
    # beta 2
    if "_1D_" in filename_bare_2 and "_1D_" in filename_corr_2:
        hf = h5py.File(filename_corr_2,"r")
        list_keys = list(hf.keys())
        N_k_t_k_b = len(list_keys)
        k_b_t_arr_jj = np.empty((N_k_t_k_b,),dtype=tuple)
        k_b_t_arr_szsz = np.empty((N_k_t_k_b,),dtype=tuple)
        for i in range(N_k_t_k_b):
            str_k_t_k_b = str(list_keys[i])
            k_b = float(findall(r"(?<=kbar_)([-+]?\d*\.\d+)",str_k_t_k_b)[0])
            k_t = float(findall(r"(?<=ktilde_)([-+]?\d*\.\d+)",str_k_t_k_b)[0])
            diff_k = k_b-k_t
            # jj
            tmp_k_t_k_b_omega_jj = np.array(hf.get(str_k_t_k_b).get('jj'))
            tmp_k_t_k_b_omega_jj = np.array(list(map(lambda x: x.imag,tmp_k_t_k_b_omega_jj)))
            k_b_t_arr_jj[i] = (diff_k,tmp_k_t_k_b_omega_jj,k_b,k_t)
            # szsz
            tmp_k_t_k_b_omega_szsz = np.array(hf.get(str_k_t_k_b).get('szsz'))
            tmp_k_t_k_b_omega_szsz = np.array(list(map(lambda x: x.imag,tmp_k_t_k_b_omega_szsz)))
            k_b_t_arr_szsz[i] = (diff_k,tmp_k_t_k_b_omega_szsz,k_b,k_t)


        k_b_t_arr_jj = sorted(k_b_t_arr_jj,key=lambda x: x[0])
        k_b_t_arr_szsz = sorted(k_b_t_arr_szsz,key=lambda x: x[0])
        # mesh_Omega_Q = np.empty((N_k_t_k_b,len(omega)),dtype=float)

        for i in range(N_k_t_k_b):
            tmp_k_t_k_b_omega_jj = k_b_t_arr_jj[i][1]
            tmp_k_t_k_b_omega_szsz = k_b_t_arr_szsz[i][1]
            for j,om in enumerate(omega):
                if om==0.0:
                    tmp_k_t_k_b_omega_re_sigma_corr_jj_2[j] = get_derivative(tmp_k_t_k_b_omega_jj[j-2],tmp_k_t_k_b_omega_jj[j-1],tmp_k_t_k_b_omega_jj[j+1],tmp_k_t_k_b_omega_jj[j+2],2.0*wmax/1000)
                    tmp_k_t_k_b_omega_re_sigma_corr_szsz_2[j] = get_derivative(tmp_k_t_k_b_omega_szsz[j-2],tmp_k_t_k_b_omega_szsz[j-1],tmp_k_t_k_b_omega_szsz[j+1],tmp_k_t_k_b_omega_szsz[j+2],2.0*wmax/1000)
                else:
                    tmp_k_t_k_b_omega_re_sigma_corr_jj_2[j] = tmp_k_t_k_b_omega_jj[j]/om
                    tmp_k_t_k_b_omega_re_sigma_corr_szsz_2[j] = tmp_k_t_k_b_omega_szsz[j]/om
            
            # mesh_Omega_Q[i,:] = tmp_k_t_k_b_omega_re_sigma_corr
            # Computing the average
            # jj
            if not np.isnan(tmp_k_t_k_b_omega_re_sigma_corr_jj_2).any():
                if isclose(np.abs(k_b_t_arr_jj[i][2]),np.pi,abs_tol=1e-5) or isclose(np.abs(k_b_t_arr_jj[i][3]),0.0,abs_tol=1e-5):
                    if isclose(np.abs(k_b_t_arr_jj[i][2]),np.pi,abs_tol=1e-5) and isclose(np.abs(k_b_t_arr_jj[i][3]),0.0,abs_tol=1e-5):
                        average_k_t_k_b_jj_2 += 0.25*tmp_k_t_k_b_omega_re_sigma_corr_jj_2
                    average_k_t_k_b_jj_2 += 0.5*tmp_k_t_k_b_omega_re_sigma_corr_jj_2
                else:
                    average_k_t_k_b_jj_2 += tmp_k_t_k_b_omega_re_sigma_corr_jj_2
            # szsz
            if not np.isnan(tmp_k_t_k_b_omega_re_sigma_corr_szsz_2).any():
                if isclose(np.abs(k_b_t_arr_szsz[i][2]),np.pi,abs_tol=1e-5) or isclose(np.abs(k_b_t_arr_szsz[i][3]),0.0,abs_tol=1e-5):
                    if isclose(np.abs(k_b_t_arr_szsz[i][2]),np.pi,abs_tol=1e-5) and isclose(np.abs(k_b_t_arr_szsz[i][3]),0.0,abs_tol=1e-5):
                        average_k_t_k_b_szsz_2 += 0.25*tmp_k_t_k_b_omega_re_sigma_corr_szsz_2
                    average_k_t_k_b_szsz_2 += 0.5*tmp_k_t_k_b_omega_re_sigma_corr_szsz_2
                else:
                    average_k_t_k_b_szsz_2 += tmp_k_t_k_b_omega_re_sigma_corr_szsz_2

        average_k_t_k_b_jj_2 *= 1.0/np.sqrt(N_k_t_k_b)/np.sqrt(N_k_t_k_b)
        corr_susceptibility_jj_2 = deepcopy(average_k_t_k_b_jj_2)

        average_k_t_k_b_szsz_2 *= 1.0/np.sqrt(N_k_t_k_b)/np.sqrt(N_k_t_k_b)
        corr_susceptibility_szsz_2 = deepcopy(average_k_t_k_b_szsz_2)


    for j in range(len(omega)):
        average_k_t_k_b_jj_2[j] += tmp_k_t_k_b_omega_re_sigma_bare_jj_2[j]
        average_k_t_k_b_szsz_2[j] += tmp_k_t_k_b_omega_re_sigma_bare_szsz_2[j]

    hf.close()
    ########################################################### Plotting ###########################################################
    # jj
    fig_jj = plt.figure(figsize=(10, 9))
    grid = plt.GridSpec(2, 2, wspace=0.35, hspace=0.40)

    if "_1D_" in filename_bare_1 and "_1D_" in filename_corr_1:
        fig_jj.suptitle(r"1D IPT for $U={0}$ ({1})".format(U_corr_1,str_plot_corr))
    elif "_2D_" in filename_bare_1 and "_2D_" in filename_corr_1:
        fig_jj.suptitle(r"2D IPT for $U={0}$ ({1})".format(U_corr_1,str_plot_corr))

    top_panel_left = fig_jj.add_subplot(grid[0,0])
    top_panel_left.grid()
    top_panel_left.set_title(r"Total $\langle j_xj_x\rangle$")
    
    top_panel_left.set_xlabel(r"$\omega$",labelpad=-0.4)
    top_panel_left.set_ylabel(r"$\operatorname{Re}\sigma_{jj}(\omega,\mathbf{q}=\mathbf{0})$")
    top_panel_left.set_xlim(left=-15.0,right=15.0)
    top_panel_left.text(-0.15,1.15,"a)",transform=top_panel_left.transAxes,size=20,weight='bold')

    top_panel_left.plot(omega,average_k_t_k_b_jj_1,marker='o',ms=2.0,c="red",label=r"$\beta={0:.2f}$".format(beta_corr_1))#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    top_panel_left.plot(omega,average_k_t_k_b_jj_2,marker='o',ms=2.0,c="blue",label=r"$\beta={0:.2f}$".format(beta_corr_2))
    handles, labels = plt.gca().get_legend_handles_labels()
    #top_panel_left.legend()
    # 
    top_panel_right = fig_jj.add_subplot(grid[0,1],sharey=top_panel_left)
    top_panel_right.grid()
    top_panel_right.set_title(r"Bare $\langle j_xj_x\rangle$")
    
    top_panel_right.set_xlabel(r"$\omega$",labelpad=-0.4)
    top_panel_right.set_xlim(left=-15.0,right=15.0)

    top_panel_right.plot(omega,tmp_k_t_k_b_omega_re_sigma_bare_jj_1,marker='>',ms=2.0,c="red",label="_nolegend_")#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    top_panel_right.plot(omega,tmp_k_t_k_b_omega_re_sigma_bare_jj_2,marker='>',ms=2.0,c="blue",label="_nolegend_")
    #top_panel_right.legend()
    #
    lower_panel = fig_jj.add_subplot(grid[1,0:])
    lower_panel.grid()
    lower_panel.set_title(r"Correction to $\langle j_xj_x\rangle$")

    lower_panel.plot(omega,corr_susceptibility_jj_1,marker='v',ms=2.0,c="red",label="_nolegend_")
    lower_panel.plot(omega,corr_susceptibility_jj_2,marker='v',ms=2.0,c="blue",label="_nolegend_")
    lower_panel.set_ylabel(r"$\operatorname{Re}\sigma_{jj}(\omega,\mathbf{q}=\mathbf{0})$")
    lower_panel.set_xlabel(r"$\omega$")
    lower_panel.set_xlim(left=-15.0,right=15.0)

    fig_jj.legend(handles,labels,loc='upper right')

    plt.gcf().set_size_inches(18.5/2.54,12/2.54)
    if "_1D_" in filename_corr_1 and "_1D_" in filename_bare_1:
        plt.savefig("Figure_susceptibility_comparison_jj_U_{0:.2f}_beta_{1:.2f}_{2:.2f}_1D_".format(U_corr_1,beta_corr_1,beta_corr_2)+str_plot_corr.replace(" ","_")+".pdf")
    elif "_2D_" in filename_corr_1 and "_2D_" in filename_bare_1:
        plt.savefig("Figure_susceptibility_comparison_jj_U_{0:.2f}_beta_{1:.2f}_{2:.2f}_2D_".format(U_corr_1,beta_corr_1,beta_corr_2)+str_plot_corr.replace(" ","_")+".pdf")
    
    # szsz
    fig_szsz = plt.figure(figsize=(10, 9))
    grid = plt.GridSpec(2, 2, wspace=0.35, hspace=0.40)

    if "_1D_" in filename_bare_1 and "_1D_" in filename_corr_1:
        fig_szsz.suptitle(r"1D IPT for $U={0}$ ({1})".format(U_corr_1,str_plot_corr))
    elif "_2D_" in filename_bare_1 and "_2D_" in filename_corr_1:
        fig_szsz.suptitle(r"2D IPT for $U={0}$ ({1})".format(U_corr_1,str_plot_corr))

    top_panel_left = fig_szsz.add_subplot(grid[0,0])
    top_panel_left.grid()

    
    top_panel_left.set_title(r"Total $\langle S_zS_z\rangle$")
    top_panel_left.set_xlabel(r"$\omega$",labelpad=-0.4)
    top_panel_left.set_ylabel(r"$\operatorname{Im}\chi_{S_zS_z}(\omega,\mathbf{q}=\mathbf{0})*\omega^{-1}$")
    top_panel_left.set_xlim(left=-15.0,right=15.0)
    top_panel_left.text(-0.15,1.15,"a)",transform=top_panel_left.transAxes,size=20,weight='bold')

    top_panel_left.plot(omega,average_k_t_k_b_szsz_1,marker='o',ms=2.0,c="red",label=r"$\beta={0:.2f}$".format(beta_corr_1))#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    top_panel_left.plot(omega,average_k_t_k_b_szsz_2,marker='o',ms=2.0,c="blue",label=r"$\beta={0:.2f}$".format(beta_corr_2))
    handles, labels = plt.gca().get_legend_handles_labels()
    #top_panel_left.legend()
    # 
    top_panel_right = fig_szsz.add_subplot(grid[0,1],sharey=top_panel_left)
    top_panel_right.grid()
    top_panel_right.set_title(r"Bare $\langle S_zS_z\rangle$")
    
    top_panel_right.set_xlabel(r"$\omega$",labelpad=-0.4)
    top_panel_right.set_xlim(left=-15.0,right=15.0)
    top_panel_right.plot(omega,tmp_k_t_k_b_omega_re_sigma_bare_szsz_1,marker='>',ms=2.0,c="red",label="_nolegend_")#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    top_panel_right.plot(omega,tmp_k_t_k_b_omega_re_sigma_bare_szsz_2,marker='>',ms=2.0,c="blue",label="_nolegend_")
    #top_panel_right.legend()
    #
    lower_panel = fig_szsz.add_subplot(grid[1,0:])
    lower_panel.grid()

    lower_panel.set_title(r"Correction to $\langle S_zS_z\rangle$")

    lower_panel.plot(omega,corr_susceptibility_szsz_1,marker='v',ms=2.0,c="red",label="_nolegend_")
    lower_panel.plot(omega,corr_susceptibility_szsz_2,marker='v',ms=2.0,c="blue",label="_nolegend_")
    lower_panel.set_ylabel(r"$\operatorname{Im}\chi_{S_zS_z}(\omega,\mathbf{q}=\mathbf{0})*\omega^{-1}$")
    lower_panel.set_xlabel(r"$\omega$")
    lower_panel.set_xlim(left=-15.0,right=15.0)

    fig_szsz.legend(handles,labels,loc='upper right')

    plt.gcf().set_size_inches(18.5/2.54,12/2.54)
    if "_1D_" in filename_corr_1 and "_1D_" in filename_bare_1:
        plt.savefig("Figure_susceptibility_comparison_szsz_U_{0:.2f}_beta_{1:.2f}_{2:.2f}_1D_".format(U_corr_1,beta_corr_1,beta_corr_2)+str_plot_corr.replace(" ","_")+".pdf")
    elif "_2D_" in filename_corr_1 and "_2D_" in filename_bare_1:
        plt.savefig("Figure_susceptibility_comparison_szsz_U_{0:.2f}_beta_{1:.2f}_{2:.2f}_2D_".format(U_corr_1,beta_corr_1,beta_corr_2)+str_plot_corr.replace(" ","_")+".pdf")
    
    # plt.show()
