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
    filename_bare_1 = "cpp_tests/figs_article/bb_1D_U_8.000000_beta_3.800000_Ntau_2048_Nk_301.hdf5.pade_wmax_15.0_m_30_eta_0.001"
    filename_corr_1 = "cpp_tests/figs_article/bb_1D_U_8.000000_beta_3.800000_Ntau_256_Nq_31_Nk_27_NCA_single_ladder_sum.hdf5.pade_wmax_15.0_m_30_eta_0.001"
    filename_corr_inf_1 = "cpp_tests/figs_article/bb_1D_U_8.000000_beta_3.800000_Ntau_256_Nq_31_Nk_15_NCA_infinite_ladder_sum.hdf5.pade_wmax_20.0_m_40_eta_0.001"

    wmax_bare_1 = float(findall(r"(?<=wmax_)(\d*\.\d+|\d+)",filename_bare_1)[0]); wmax_corr_1 = float(findall(r"(?<=wmax_)(\d*\.\d+|\d+)",filename_corr_1)[0])
    Ntau_bare_1 = int(findall(r"(?<=Ntau_)(\d+)",filename_bare_1)[0]); Ntau_corr_1 = int(findall(r"(?<=Ntau_)(\d+)",filename_corr_1)[0])
    beta_bare_1 = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_bare_1)[0]); beta_corr_1 = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_corr_1)[0])
    U_bare_1 = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_bare_1)[0]); U_corr_1 = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_corr_1)[0])
    # beta 2
    filename_bare_2 = "cpp_tests/figs_article/bb_1D_U_8.000000_beta_3.000000_Ntau_2048_Nk_301.hdf5.pade_wmax_15.0_m_30_eta_0.001"
    filename_corr_2 = "cpp_tests/figs_article/bb_1D_U_8.000000_beta_3.000000_Ntau_256_Nq_31_Nk_27_NCA_single_ladder_sum.hdf5.pade_wmax_15.0_m_30_eta_0.001"
    filename_corr_inf_2 = "cpp_tests/figs_article/bb_1D_U_8.000000_beta_3.000000_Ntau_256_Nq_31_Nk_15_NCA_infinite_ladder_sum.hdf5.pade_wmax_20.0_m_35_eta_0.001"

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
    # extracting the non-interacting and infinite-ladder susceptibilities
    # beta 1
    k_t_k_b_omega_re_bare_jj_1 = np.zeros((len(omega),),dtype=float); k_t_k_b_omega_re_bare_szsz_1 = np.zeros((len(omega),),dtype=float)
    infinite_corr_susceptibility_jj_1 = np.zeros((len(omega),),dtype=float); infinite_corr_susceptibility_szsz_1 = np.zeros((len(omega),),dtype=float)
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
                    k_t_k_b_omega_re_bare_jj_1[j] = get_derivative(tmp_k_t_k_b_omega_jj[j-2],tmp_k_t_k_b_omega_jj[j-1],tmp_k_t_k_b_omega_jj[j+1],tmp_k_t_k_b_omega_jj[j+2],2.0*wmax/1000)
                    k_t_k_b_omega_re_bare_szsz_1[j] = get_derivative(tmp_k_t_k_b_omega_szsz[j-2],tmp_k_t_k_b_omega_szsz[j-1],tmp_k_t_k_b_omega_szsz[j+1],tmp_k_t_k_b_omega_szsz[j+2],2.0*wmax/1000)
                else:
                    k_t_k_b_omega_re_bare_jj_1[j] = tmp_k_t_k_b_omega_jj[j]/om
                    k_t_k_b_omega_re_bare_szsz_1[j] = tmp_k_t_k_b_omega_szsz[j]/om
    hf.close()
    # beta 2
    k_t_k_b_omega_re_bare_jj_2 = np.zeros((len(omega),),dtype=float); k_t_k_b_omega_re_bare_szsz_2 = np.zeros((len(omega),),dtype=float)
    infinite_corr_susceptibility_jj_2 = np.zeros((len(omega),),dtype=float); infinite_corr_susceptibility_szsz_2 = np.zeros((len(omega),),dtype=float)
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
                    k_t_k_b_omega_re_bare_jj_2[j] = get_derivative(tmp_k_t_k_b_omega_jj[j-2],tmp_k_t_k_b_omega_jj[j-1],tmp_k_t_k_b_omega_jj[j+1],tmp_k_t_k_b_omega_jj[j+2],2.0*wmax/1000)
                    k_t_k_b_omega_re_bare_szsz_2[j] = get_derivative(tmp_k_t_k_b_omega_szsz[j-2],tmp_k_t_k_b_omega_szsz[j-1],tmp_k_t_k_b_omega_szsz[j+1],tmp_k_t_k_b_omega_szsz[j+2],2.0*wmax/1000)
                else:
                    k_t_k_b_omega_re_bare_jj_2[j] = tmp_k_t_k_b_omega_jj[j]/om
                    k_t_k_b_omega_re_bare_szsz_2[j] = tmp_k_t_k_b_omega_szsz[j]/om
    hf.close()
    # inf corr
    if "_1D_" in filename_corr_inf_1:
        hf = h5py.File(filename_corr_inf_1,"r")
        
        infinite_corr_susceptibility_jj_1 = np.array(hf.get("susc").get('jj'))
        infinite_corr_susceptibility_szsz_1 = np.array(hf.get("susc").get('szsz'))

        hf.close()
    # beta 2
    if "_1D_" in filename_corr_inf_2:
        hf = h5py.File(filename_corr_inf_2,"r")

        infinite_corr_susceptibility_jj_2 = np.array(hf.get("susc").get('jj'))
        infinite_corr_susceptibility_szsz_2 = np.array(hf.get("susc").get('szsz'))

        hf.close()
    # extracting the single-ladder correction to the susceptibility
    # beta 1
    if "_1D_" in filename_corr_1:
        hf = h5py.File(filename_corr_1,"r")
        
        single_corr_susceptibility_jj_1 = np.array(hf.get("susc").get('jj'))
        single_corr_susceptibility_szsz_1 = np.array(hf.get("susc").get('szsz'))

        hf.close()
    # beta 2
    if "_1D_" in filename_corr_2:
        hf = h5py.File(filename_corr_2,"r")

        single_corr_susceptibility_jj_2 = np.array(hf.get("susc").get('jj'))
        single_corr_susceptibility_szsz_2 = np.array(hf.get("susc").get('szsz'))

        hf.close()
    ########################################################### Plotting ###########################################################
    # jj
    fig_jj = plt.figure(figsize=(10, 9))
    grid = plt.GridSpec(3, 3, wspace=0.35, hspace=0.50)

    if "_1D_" in filename_bare_1 and "_1D_" in filename_corr_1:
        fig_jj.suptitle(r"1D NCA ($U={0}$)".format(U_corr_1))
    elif "_2D_" in filename_bare_1 and "_2D_" in filename_corr_1:
        fig_jj.suptitle(r"2D NCA ($U={0}$)".format(U_corr_1))

    # 
    top_panel = fig_jj.add_subplot(grid[0,0:])
    top_panel.grid()
    top_panel.set_title(r"Bare $\langle j_xj_x\rangle$")
    #top_panel.text(-0.15,1.15,"a)",transform=top_panel.transAxes,size=20,weight='bold')
    #top_panel.set_xlabel(r"$\omega$",labelpad=-0.4)
    top_panel.set_xlim(left=-0.0,right=15.0)
    top_panel.tick_params(axis='x',bottom=False)
    top_panel.plot(omega,k_t_k_b_omega_re_bare_jj_1,marker='>',ms=2.0,c="red",label=r"$\beta={0:.2f}$".format(beta_corr_1))#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    top_panel.plot(omega,k_t_k_b_omega_re_bare_jj_2,marker='>',ms=2.0,c="blue",label=r"$\beta={0:.2f}$".format(beta_corr_2))
    handles, labels = plt.gca().get_legend_handles_labels()
    #
    middle_panel = fig_jj.add_subplot(grid[1,0:],sharex=top_panel)
    middle_panel.grid()
    middle_panel.set_title(r"Single-ladder correction to $\langle j_xj_x\rangle$")

    middle_panel.plot(omega,single_corr_susceptibility_jj_1,marker='v',ms=2.0,c="red",label="_nolegend_")
    middle_panel.plot(omega,single_corr_susceptibility_jj_2,marker='v',ms=2.0,c="blue",label="_nolegend_")
    middle_panel.set_ylabel(r"$\operatorname{Re}\sigma_{jj}(\omega,\mathbf{q}=\mathbf{0})$")
    middle_panel.tick_params(axis='x',bottom=False)
    #middle_panel.set_xlabel(r"$\omega$")
    #
    bottom_panel = fig_jj.add_subplot(grid[2,0:],sharex=top_panel)
    bottom_panel.grid()
    bottom_panel.set_title(r"Infinite-ladder correction to $\langle j_xj_x\rangle$")
    
    bottom_panel.set_xlabel(r"$\omega$",labelpad=-0.4)
    #bottom_panel.set_ylabel(r"$\operatorname{Re}\sigma_{jj}(\omega,\mathbf{q}=\mathbf{0})$")

    bottom_panel.plot(omega,infinite_corr_susceptibility_jj_1,marker='o',ms=2.0,c="red",label="_nolegend_")#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    bottom_panel.plot(omega,infinite_corr_susceptibility_jj_2,marker='o',ms=2.0,c="blue",label="_nolegend_")

    fig_jj.legend(handles,labels,loc='upper right')
    #plt.show()
    plt.gcf().set_size_inches(18.5/2.54,12/2.54)
    if "_1D_" in filename_corr_1 and "_1D_" in filename_bare_1:
        plt.savefig("Figure_susceptibility_comparison_jj_U_{0:.2f}_beta_{1:.2f}_{2:.2f}_1D_".format(U_corr_1,beta_corr_1,beta_corr_2)+str_plot_corr.replace(" ","_")+".pdf")
    elif "_2D_" in filename_corr_1 and "_2D_" in filename_bare_1:
        plt.savefig("Figure_susceptibility_comparison_jj_U_{0:.2f}_beta_{1:.2f}_{2:.2f}_2D_".format(U_corr_1,beta_corr_1,beta_corr_2)+str_plot_corr.replace(" ","_")+".pdf")
    
    # szsz
    fig_szsz = plt.figure(figsize=(10, 9))
    grid = plt.GridSpec(3, 3, wspace=0.35, hspace=0.5)

    if "_1D_" in filename_bare_1 and "_1D_" in filename_corr_1:
        fig_szsz.suptitle(r"1D NCA ($U={0}$)".format(U_corr_1))
    elif "_2D_" in filename_bare_1 and "_2D_" in filename_corr_1:
        fig_szsz.suptitle(r"2D NCA ($U={0}$)".format(U_corr_1))

    # 
    top_panel = fig_szsz.add_subplot(grid[0,0:])
    top_panel.grid()
    top_panel.set_title(r"Bare $\langle S_zS_z\rangle$")
    #top_panel.text(-0.15,1.15,"a)",transform=top_panel.transAxes,size=20,weight='bold')
    #top_panel.set_xlabel(r"$\omega$",labelpad=-0.4)
    top_panel.set_xlim(left=-0.0,right=15.0)
    top_panel.tick_params(axis='x',bottom=False)
    top_panel.plot(omega,k_t_k_b_omega_re_bare_szsz_1,marker='>',ms=2.0,c="red",label=r"$\beta={0:.2f}$".format(beta_corr_1))#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    top_panel.plot(omega,k_t_k_b_omega_re_bare_szsz_2,marker='>',ms=2.0,c="blue",label=r"$\beta={0:.2f}$".format(beta_corr_2))
    handles, labels = plt.gca().get_legend_handles_labels()
    #
    middle_panel = fig_szsz.add_subplot(grid[1,0:],sharex=top_panel)
    middle_panel.grid()

    middle_panel.set_title(r"Single-ladder correction to $\langle S_zS_z\rangle$")
    middle_panel.tick_params(axis='x',bottom=False)
    middle_panel.plot(omega,single_corr_susceptibility_szsz_1,marker='v',ms=2.0,c="red",label="_nolegend_")
    middle_panel.plot(omega,single_corr_susceptibility_szsz_2,marker='v',ms=2.0,c="blue",label="_nolegend_")
    middle_panel.set_ylabel(r"$\operatorname{Im}\chi_{S_zS_z}(\omega,\mathbf{q}=\mathbf{0})*\omega^{-1}$")
    #middle_panel.set_xlabel(r"$\omega$")
    #middle_panel.set_xlim(left=-0.0,right=4.0)
    #
    bottom_panel = fig_szsz.add_subplot(grid[2,0:],sharex=top_panel)
    bottom_panel.grid()

    bottom_panel.set_title(r"Infinite-ladder correction to $\langle S_zS_z\rangle$")
    bottom_panel.set_xlabel(r"$\omega$",labelpad=-0.4)
    #bottom_panel.set_ylabel(r"$\operatorname{Im}\chi_{S_zS_z}(\omega,\mathbf{q}=\mathbf{0})*\omega^{-1}$")
    #bottom_panel.set_xlim(left=-0.0,right=4.0)

    bottom_panel.plot(omega,infinite_corr_susceptibility_szsz_1,marker='o',ms=2.0,c="red",label="_nolegend_")#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    bottom_panel.plot(omega,infinite_corr_susceptibility_szsz_2,marker='o',ms=2.0,c="blue",label="_nolegend_")

    fig_szsz.legend(handles,labels,loc='upper right')
    #plt.show()
    plt.gcf().set_size_inches(18.5/2.54,12/2.54)
    if "_1D_" in filename_corr_1 and "_1D_" in filename_bare_1:
        plt.savefig("Figure_susceptibility_comparison_szsz_U_{0:.2f}_beta_{1:.2f}_{2:.2f}_1D_".format(U_corr_1,beta_corr_1,beta_corr_2)+str_plot_corr.replace(" ","_")+".pdf")
    elif "_2D_" in filename_corr_1 and "_2D_" in filename_bare_1:
        plt.savefig("Figure_susceptibility_comparison_szsz_U_{0:.2f}_beta_{1:.2f}_{2:.2f}_2D_".format(U_corr_1,beta_corr_1,beta_corr_2)+str_plot_corr.replace(" ","_")+".pdf")
