import numpy as np
import matplotlib.pyplot as plt
import h5py
from sys import exit
from re import findall
from math import isclose
from copy import deepcopy
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, FormatStrFormatter)

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage[charter]{mathdesign}\usepackage{amsmath}"]
plt.rcParams["legend.handlelength"] = 1.0
plt.rcParams["legend.labelspacing"] = 0.4
plt.rcParams["legend.handletextpad"] = 0.6

spin_spin_factor = 0.25 # factor coming from the definition of the spin operator!

if __name__=="__main__":
    figure_directory_added = "/Users/simardo/install/maxent/adding_corr_to_bare/"#"/Users/simardo/install/maxent/"
    figure_directory = "/Users/simardo/install/maxent/"
    figure_directory_pade = "/Users/simardo/Documents/PhD/DMFT_IPT_HF/mea/cpp_tests/susceptibilities/"
    # filenames whose data we want to combine. They have to have the same U, same temperature and dimensionnality.
    # beta 1
    filename_bare_1_jj = figure_directory+"bb_1D_U_1.000000_beta_25.000000_Ntau_4096_Nk_401.hdf5_to_philipp_jj_dos.dat"
    filename_corr_1_jj = figure_directory_added+"bb_1D_U_1.000000_beta_25.000000_Ntau_256_Nq_31_Nk_27_single_ladder_sum_jj_dos.dat"#"bb_1D_U_1.000000_beta_25.000000_Ntau_256_Nq_31_Nk_27_single_ladder_sum.hdf5_iqn_jj_dos.dat"
    filename_corr_inf_1_jj = figure_directory_added+"bb_1D_U_1.000000_beta_25.000000_Ntau_256_Nq_27_Nk_27_infinite_ladder_sum_iqn_div_4_MAX_DEPTH_7_Uren_1.000000_jj_dos.dat"#"cpp_tests/figs_article/bb_1D_U_3.000000_beta_8.200000_Ntau_256_Nq_31_Nk_15_NCA_infinite_ladder_sum.hdf5.pade_wmax_10.0_m_60_eta_0.001.int_SIMPS"
    filename_bare_1_szsz = figure_directory+"bb_1D_U_1.000000_beta_25.000000_Ntau_4096_Nk_401.hdf5_to_philipp_cut_dos.dat"
    filename_corr_1_szsz = figure_directory_added+"bb_1D_U_1.000000_beta_25.000000_Ntau_256_Nq_31_Nk_27_single_ladder_sum_szsz_dos.dat"
    filename_corr_inf_1_szsz = figure_directory_added+"bb_1D_U_1.000000_beta_25.000000_Ntau_256_Nq_27_Nk_27_infinite_ladder_sum_iqn_div_4_MAX_DEPTH_7_Uren_1.000000_szsz_dos.dat"#"cpp_tests/figs_article/bb_1D_U_3.000000_beta_8.200000_Ntau_256_Nq_31_Nk_15_NCA_infinite_ladder_sum.hdf5.pade_wmax_10.0_m_60_eta_0.001.int_SIMPS"

    Ntau_bare_1 = int(findall(r"(?<=Ntau_)(\d+)",filename_bare_1_jj)[0]); Ntau_corr_1 = int(findall(r"(?<=Ntau_)(\d+)",filename_corr_1_jj)[0])
    beta_bare_1 = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_bare_1_jj)[0]); beta_corr_1 = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_corr_1_jj)[0])
    U_bare_1 = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_bare_1_jj)[0]); U_corr_1 = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_corr_1_jj)[0])

    filename_bare_1_jj_2 = figure_directory+"bb_1D_U_2.000000_beta_9.000000_Ntau_4096_Nk_301.hdf5_to_philipp_jj_dos.dat"
    filename_corr_1_jj_2 = figure_directory_added+"bb_1D_U_2.000000_beta_9.000000_Ntau_256_Nq_33_Nk_27_single_ladder_sum_jj_dos.dat"
    filename_corr_inf_1_jj_2 = figure_directory_added+"bb_1D_U_2.000000_beta_9.000000_Ntau_256_Nq_27_Nk_27_infinite_ladder_sum_iqn_div_4_MAX_DEPTH_7_Uren_1.335000_jj_dos.dat"
    filename_bare_1_szsz_2 = figure_directory+"bb_1D_U_2.000000_beta_9.000000_Ntau_4096_Nk_301.hdf5_to_philipp_cut_dos.dat"
    filename_corr_1_szsz_2 = figure_directory_added+"bb_1D_U_2.000000_beta_9.000000_Ntau_256_Nq_33_Nk_27_single_ladder_sum_szsz_dos.dat"
    filename_corr_inf_1_szsz_2 = figure_directory_added+"bb_1D_U_2.000000_beta_9.000000_Ntau_256_Nq_27_Nk_27_infinite_ladder_sum_iqn_div_4_MAX_DEPTH_7_Uren_1.335000_szsz_dos.dat"

    U_corr_1_2 = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_corr_1_jj_2)[0]); beta_corr_1_2 = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_corr_1_jj_2)[0])

    filename_bare_1_jj_3 = figure_directory+"bb_1D_U_3.000000_beta_5.500000_Ntau_4096_Nk_301.hdf5_to_philipp_jj_cut_dos.dat"
    filename_corr_1_jj_3 = figure_directory_added+"bb_1D_U_3.000000_beta_5.500000_Ntau_256_Nq_33_Nk_27_single_ladder_sum_jj_dos.dat"
    filename_corr_inf_1_jj_3 = figure_directory_added+"bb_1D_U_3.000000_beta_5.500000_Ntau_256_Nq_27_Nk_27_infinite_ladder_sum_iqn_div_4_MAX_DEPTH_7_Uren_1.400000_jj_dos.dat"
    filename_bare_1_szsz_3 = figure_directory+"bb_1D_U_3.000000_beta_5.500000_Ntau_4096_Nk_301.hdf5_to_philipp_cut_dos.dat"
    filename_corr_1_szsz_3 = figure_directory_added+"bb_1D_U_3.000000_beta_5.500000_Ntau_256_Nq_33_Nk_27_single_ladder_sum_szsz_dos.dat"
    filename_corr_inf_1_szsz_3 = figure_directory_added+"bb_1D_U_3.000000_beta_5.500000_Ntau_256_Nq_27_Nk_27_infinite_ladder_sum_iqn_div_4_MAX_DEPTH_7_Uren_1.400000_szsz_dos.dat"

    U_corr_1_3 = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_corr_1_jj_3)[0]); beta_corr_1_3 = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_corr_1_jj_3)[0])

    # beta 2
    filename_bare_2_jj = figure_directory+"bb_1D_U_1.000000_beta_50.000000_Ntau_4096_Nk_301.hdf5_to_philipp_jj_dos.dat"
    filename_corr_2_jj = figure_directory_added+"bb_1D_U_1.000000_beta_50.000000_Ntau_256_Nq_31_Nk_27_single_ladder_sum_jj_dos.dat"
    filename_corr_inf_2_jj = figure_directory_added+"bb_1D_U_1.000000_beta_50.000000_Ntau_256_Nq_27_Nk_27_infinite_ladder_sum_iqn_div_4_MAX_DEPTH_7_Uren_1.000000_jj_dos.dat"#"cpp_tests/figs_article/bb_1D_U_3.000000_beta_7.600000_Ntau_256_Nq_31_Nk_15_NCA_infinite_ladder_sum.hdf5.pade_wmax_10.0_m_60_eta_0.001.int_SIMPS"
    filename_bare_2_szsz = figure_directory+"bb_1D_U_1.000000_beta_50.000000_Ntau_4096_Nk_301.hdf5_to_philipp_cut_dos.dat"
    filename_corr_2_szsz = figure_directory_added+"bb_1D_U_1.000000_beta_50.000000_Ntau_256_Nq_31_Nk_27_single_ladder_sum_szsz_dos.dat"
    filename_corr_inf_2_szsz = figure_directory_added+"bb_1D_U_1.000000_beta_50.000000_Ntau_256_Nq_27_Nk_27_infinite_ladder_sum_iqn_div_4_MAX_DEPTH_7_Uren_1.000000_szsz_dos.dat"#"cpp_tests/figs_article/bb_1D_U_3.000000_beta_7.600000_Ntau_256_Nq_31_Nk_15_NCA_infinite_ladder_sum.hdf5.pade_wmax_10.0_m_60_eta_0.001.int_SIMPS"

    Ntau_bare_2 = int(findall(r"(?<=Ntau_)(\d+)",filename_bare_2_jj)[0]); Ntau_corr_2 = int(findall(r"(?<=Ntau_)(\d+)",filename_corr_2_jj)[0])
    beta_bare_2 = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_bare_2_jj)[0]); beta_corr_2 = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_corr_2_jj)[0])
    U_bare_2 = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_bare_2_jj)[0]); U_corr_2 = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_corr_2_jj)[0])

    filename_bare_2_jj_2 = figure_directory+"bb_1D_U_2.000000_beta_12.000000_Ntau_4096_Nk_301.hdf5_to_philipp_jj_dos.dat"
    filename_corr_2_jj_2 = figure_directory_added+"bb_1D_U_2.000000_beta_12.000000_Ntau_256_Nq_33_Nk_27_single_ladder_sum_jj_dos.dat"
    filename_corr_inf_2_jj_2 = figure_directory_added+"bb_1D_U_2.000000_beta_12.000000_Ntau_256_Nq_27_Nk_27_infinite_ladder_sum_iqn_div_2_MAX_DEPTH_7_Uren_1.335000_jj_dos.dat"
    filename_bare_2_szsz_2 = figure_directory+"bb_1D_U_2.000000_beta_12.000000_Ntau_4096_Nk_301.hdf5_to_philipp_cut_dos.dat"
    filename_corr_2_szsz_2 = figure_directory_added+"bb_1D_U_2.000000_beta_12.000000_Ntau_256_Nq_33_Nk_27_single_ladder_sum_szsz_dos.dat"
    filename_corr_inf_2_szsz_2 = figure_directory_added+"bb_1D_U_2.000000_beta_12.000000_Ntau_256_Nq_27_Nk_27_infinite_ladder_sum_iqn_div_2_MAX_DEPTH_7_Uren_1.335000_szsz_dos.dat"

    U_corr_2_2 = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_corr_2_jj_2)[0]); beta_corr_2_2 = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_corr_2_jj_2)[0])

    filename_bare_2_jj_3 = figure_directory+"bb_1D_U_3.000000_beta_6.500000_Ntau_4096_Nk_301.hdf5_to_philipp_jj_cut_dos.dat"
    filename_corr_2_jj_3 = figure_directory_added+"bb_1D_U_3.000000_beta_6.500000_Ntau_256_Nq_33_Nk_27_single_ladder_sum_jj_dos.dat"
    filename_corr_inf_2_jj_3 = figure_directory_added+"bb_1D_U_3.000000_beta_6.500000_Ntau_256_Nq_27_Nk_27_infinite_ladder_sum_iqn_div_4_MAX_DEPTH_7_Uren_1.400000_jj_dos.dat"
    filename_bare_2_szsz_3 = figure_directory+"bb_1D_U_3.000000_beta_6.500000_Ntau_4096_Nk_301.hdf5_to_philipp_cut_dos.dat"
    filename_corr_2_szsz_3 = figure_directory_added+"bb_1D_U_3.000000_beta_6.500000_Ntau_256_Nq_33_Nk_27_single_ladder_sum_szsz_dos.dat"
    filename_corr_inf_2_szsz_3 = figure_directory_added+"bb_1D_U_3.000000_beta_6.500000_Ntau_256_Nq_27_Nk_27_infinite_ladder_sum_iqn_div_4_MAX_DEPTH_7_Uren_1.400000_szsz_dos.dat"

    U_corr_2_3 = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_corr_2_jj_3)[0]); beta_corr_2_3 = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_corr_2_jj_3)[0])
    
    # beta 3
    filename_bare_3_jj = figure_directory+"bb_1D_U_1.000000_beta_74.000000_Ntau_4096_Nk_401.hdf5_to_philipp_jj_dos.dat"
    filename_corr_3_jj = figure_directory_added+"bb_1D_U_1.000000_beta_74.000000_Ntau_256_Nq_31_Nk_27_single_ladder_sum_jj_dos.dat"
    filename_corr_inf_3_jj = figure_directory_added+"bb_1D_U_1.000000_beta_74.000000_Ntau_256_Nq_27_Nk_27_infinite_ladder_sum_iqn_div_4_MAX_DEPTH_7_Uren_1.000000_jj_dos.dat"#"cpp_tests/figs_article/bb_1D_U_3.000000_beta_7.600000_Ntau_256_Nq_31_Nk_15_NCA_infinite_ladder_sum.hdf5.pade_wmax_10.0_m_60_eta_0.001.int_SIMPS"
    filename_bare_3_szsz = figure_directory+"bb_1D_U_1.000000_beta_74.000000_Ntau_4096_Nk_401.hdf5_to_philipp_cut_dos.dat"
    filename_corr_3_szsz = figure_directory_added+"bb_1D_U_1.000000_beta_74.000000_Ntau_256_Nq_31_Nk_27_single_ladder_sum_szsz_dos.dat"
    filename_corr_inf_3_szsz = figure_directory_added+"bb_1D_U_1.000000_beta_74.000000_Ntau_256_Nq_27_Nk_27_infinite_ladder_sum_iqn_div_4_MAX_DEPTH_7_Uren_1.000000_szsz_dos.dat"#"cpp_tests/figs_article/bb_1D_U_3.000000_beta_7.600000_Ntau_256_Nq_31_Nk_15_NCA_infinite_ladder_sum.hdf5.pade_wmax_10.0_m_60_eta_0.001.int_SIMPS"

    Ntau_bare_3 = int(findall(r"(?<=Ntau_)(\d+)",filename_bare_3_jj)[0]); Ntau_corr_3 = int(findall(r"(?<=Ntau_)(\d+)",filename_corr_3_jj)[0])
    beta_bare_3 = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_bare_3_jj)[0]); beta_corr_3 = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_corr_3_jj)[0])
    U_bare_3 = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_bare_3_jj)[0]); U_corr_3 = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_corr_3_jj)[0])

    filename_bare_3_jj_2 = figure_directory+"bb_1D_U_2.000000_beta_18.000000_Ntau_4096_Nk_401.hdf5_to_philipp_jj_dos.dat"
    filename_corr_3_jj_2 = figure_directory_added+"bb_1D_U_2.000000_beta_18.000000_Ntau_256_Nq_33_Nk_27_single_ladder_sum_jj_dos.dat"
    filename_corr_inf_3_jj_2 = figure_directory_added+"bb_1D_U_2.000000_beta_18.000000_Ntau_256_Nq_27_Nk_27_infinite_ladder_sum_iqn_div_2_MAX_DEPTH_7_Uren_1.335000_jj_dos.dat"
    filename_bare_3_szsz_2 = figure_directory+"bb_1D_U_2.000000_beta_18.000000_Ntau_4096_Nk_401.hdf5_to_philipp_cut_dos.dat"
    filename_corr_3_szsz_2 = figure_directory_added+"bb_1D_U_2.000000_beta_18.000000_Ntau_256_Nq_33_Nk_27_single_ladder_sum_szsz_dos.dat"
    filename_corr_inf_3_szsz_2 = figure_directory_added+"bb_1D_U_2.000000_beta_18.000000_Ntau_256_Nq_27_Nk_27_infinite_ladder_sum_iqn_div_2_MAX_DEPTH_7_Uren_1.335000_szsz_dos.dat"

    U_corr_3_2 = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_corr_3_jj_2)[0]); beta_corr_3_2 = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_corr_3_jj_2)[0])

    filename_bare_3_jj_3 = figure_directory+"bb_1D_U_3.000000_beta_7.500000_Ntau_4096_Nk_301.hdf5_to_philipp_jj_cut_dos.dat"
    filename_corr_3_jj_3 = figure_directory_added+"bb_1D_U_3.000000_beta_7.500000_Ntau_256_Nq_33_Nk_27_single_ladder_sum_jj_dos.dat"
    filename_corr_inf_3_jj_3 = figure_directory_added+"bb_1D_U_3.000000_beta_7.500000_Ntau_256_Nq_27_Nk_27_infinite_ladder_sum_iqn_div_4_MAX_DEPTH_7_Uren_1.400000_jj_dos.dat"
    filename_bare_3_szsz_3 = figure_directory+"bb_1D_U_3.000000_beta_7.500000_Ntau_4096_Nk_301.hdf5_to_philipp_cut_dos.dat"
    filename_corr_3_szsz_3 = figure_directory_added+"bb_1D_U_3.000000_beta_7.500000_Ntau_256_Nq_33_Nk_27_single_ladder_sum_szsz_dos.dat"
    filename_corr_inf_3_szsz_3 = figure_directory_added+"bb_1D_U_3.000000_beta_7.500000_Ntau_256_Nq_27_Nk_27_infinite_ladder_sum_iqn_div_4_MAX_DEPTH_7_Uren_1.400000_szsz_dos.dat"

    U_corr_3_3 = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",filename_corr_3_jj_3)[0]); beta_corr_3_3 = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename_corr_3_jj_3)[0])

    assert U_bare_1==U_corr_1==U_bare_2==U_corr_2==U_bare_3==U_corr_3 and beta_bare_1==beta_corr_1 and beta_bare_2==beta_corr_2 and beta_bare_3==beta_corr_3, "Temperature, U and type of susceptibility have to correspond for bare and corrections to susceptibility."
    # bare 
    omega_jj_1, bare_susceptibility_jj_1 = np.loadtxt(filename_bare_1_jj,comments='#',unpack=True)
    omega_jj_2, bare_susceptibility_jj_2 = np.loadtxt(filename_bare_2_jj,comments='#',unpack=True)
    omega_jj_3, bare_susceptibility_jj_3 = np.loadtxt(filename_bare_3_jj,comments='#',unpack=True)

    omega_jj_1_2, bare_susceptibility_jj_1_2 = np.loadtxt(filename_bare_1_jj_2,comments='#',unpack=True)
    omega_jj_2_2, bare_susceptibility_jj_2_2 = np.loadtxt(filename_bare_2_jj_2,comments='#',unpack=True)
    omega_jj_3_2, bare_susceptibility_jj_3_2 = np.loadtxt(filename_bare_3_jj_2,comments='#',unpack=True)

    omega_jj_1_3, bare_susceptibility_jj_1_3 = np.loadtxt(filename_bare_1_jj_3,comments='#',unpack=True)
    omega_jj_2_3, bare_susceptibility_jj_2_3 = np.loadtxt(filename_bare_2_jj_3,comments='#',unpack=True)
    omega_jj_3_3, bare_susceptibility_jj_3_3 = np.loadtxt(filename_bare_3_jj_3,comments='#',unpack=True)

    omega_szsz_1, bare_susceptibility_szsz_1 = np.loadtxt(filename_bare_1_szsz,comments='#',unpack=True)
    omega_szsz_2, bare_susceptibility_szsz_2 = np.loadtxt(filename_bare_2_szsz,comments='#',unpack=True)
    omega_szsz_3, bare_susceptibility_szsz_3 = np.loadtxt(filename_bare_3_szsz,comments='#',unpack=True)

    omega_szsz_1_2, bare_susceptibility_szsz_1_2 = np.loadtxt(filename_bare_1_szsz_2,comments='#',unpack=True)
    omega_szsz_2_2, bare_susceptibility_szsz_2_2 = np.loadtxt(filename_bare_2_szsz_2,comments='#',unpack=True)
    omega_szsz_3_2, bare_susceptibility_szsz_3_2 = np.loadtxt(filename_bare_3_szsz_2,comments='#',unpack=True)

    omega_szsz_1_3, bare_susceptibility_szsz_1_3 = np.loadtxt(filename_bare_1_szsz_3,comments='#',unpack=True)
    omega_szsz_2_3, bare_susceptibility_szsz_2_3 = np.loadtxt(filename_bare_2_szsz_3,comments='#',unpack=True)
    omega_szsz_3_3, bare_susceptibility_szsz_3_3 = np.loadtxt(filename_bare_3_szsz_3,comments='#',unpack=True)

    # sl corr
    omega_single_corr_jj_1, single_corr_susceptibility_jj_1 = np.loadtxt(filename_corr_1_jj,unpack=True)
    omega_single_corr_jj_2, single_corr_susceptibility_jj_2 = np.loadtxt(filename_corr_2_jj,unpack=True)
    omega_single_corr_jj_3, single_corr_susceptibility_jj_3 = np.loadtxt(filename_corr_3_jj,unpack=True)

    omega_single_corr_jj_1_2, single_corr_susceptibility_jj_1_2 = np.loadtxt(filename_corr_1_jj_2,unpack=True)
    omega_single_corr_jj_2_2, single_corr_susceptibility_jj_2_2 = np.loadtxt(filename_corr_2_jj_2,unpack=True)
    omega_single_corr_jj_3_2, single_corr_susceptibility_jj_3_2 = np.loadtxt(filename_corr_3_jj_2,unpack=True)

    omega_single_corr_jj_1_3, single_corr_susceptibility_jj_1_3 = np.loadtxt(filename_corr_1_jj_3,unpack=True)
    omega_single_corr_jj_2_3, single_corr_susceptibility_jj_2_3 = np.loadtxt(filename_corr_2_jj_3,unpack=True)
    omega_single_corr_jj_3_3, single_corr_susceptibility_jj_3_3 = np.loadtxt(filename_corr_3_jj_3,unpack=True)

    omega_single_corr_szsz_1, single_corr_susceptibility_szsz_1 = np.loadtxt(filename_corr_1_szsz,unpack=True)
    omega_single_corr_szsz_2, single_corr_susceptibility_szsz_2 = np.loadtxt(filename_corr_2_szsz,unpack=True)
    omega_single_corr_szsz_3, single_corr_susceptibility_szsz_3 = np.loadtxt(filename_corr_3_szsz,unpack=True)

    omega_single_corr_szsz_1_2, single_corr_susceptibility_szsz_1_2 = np.loadtxt(filename_corr_1_szsz_2,unpack=True)
    omega_single_corr_szsz_2_2, single_corr_susceptibility_szsz_2_2 = np.loadtxt(filename_corr_2_szsz_2,unpack=True)
    omega_single_corr_szsz_3_2, single_corr_susceptibility_szsz_3_2 = np.loadtxt(filename_corr_3_szsz_2,unpack=True)

    omega_single_corr_szsz_1_3, single_corr_susceptibility_szsz_1_3 = np.loadtxt(filename_corr_1_szsz_3,unpack=True)
    omega_single_corr_szsz_2_3, single_corr_susceptibility_szsz_2_3 = np.loadtxt(filename_corr_2_szsz_3,unpack=True)
    omega_single_corr_szsz_3_3, single_corr_susceptibility_szsz_3_3 = np.loadtxt(filename_corr_3_szsz_3,unpack=True)

    # inf corr
    omega_infinite_corr_jj_1, infinite_corr_susceptibility_jj_1 = np.loadtxt(filename_corr_inf_1_jj,unpack=True)
    omega_infinite_corr_jj_2, infinite_corr_susceptibility_jj_2 = np.loadtxt(filename_corr_inf_2_jj,unpack=True)
    omega_infinite_corr_jj_3, infinite_corr_susceptibility_jj_3 = np.loadtxt(filename_corr_inf_3_jj,unpack=True)

    omega_infinite_corr_jj_1_2, infinite_corr_susceptibility_jj_1_2 = np.loadtxt(filename_corr_inf_1_jj_2,unpack=True)
    omega_infinite_corr_jj_2_2, infinite_corr_susceptibility_jj_2_2 = np.loadtxt(filename_corr_inf_2_jj_2,unpack=True)
    omega_infinite_corr_jj_3_2, infinite_corr_susceptibility_jj_3_2 = np.loadtxt(filename_corr_inf_3_jj_2,unpack=True)

    omega_infinite_corr_jj_1_3, infinite_corr_susceptibility_jj_1_3 = np.loadtxt(filename_corr_inf_1_jj_3,unpack=True)
    omega_infinite_corr_jj_2_3, infinite_corr_susceptibility_jj_2_3 = np.loadtxt(filename_corr_inf_2_jj_3,unpack=True)
    omega_infinite_corr_jj_3_3, infinite_corr_susceptibility_jj_3_3 = np.loadtxt(filename_corr_inf_3_jj_3,unpack=True)

    omega_infinite_corr_szsz_1, infinite_corr_susceptibility_szsz_1 = np.loadtxt(filename_corr_inf_1_szsz,unpack=True)
    omega_infinite_corr_szsz_2, infinite_corr_susceptibility_szsz_2 = np.loadtxt(filename_corr_inf_2_szsz,unpack=True)
    omega_infinite_corr_szsz_3, infinite_corr_susceptibility_szsz_3 = np.loadtxt(filename_corr_inf_3_szsz,unpack=True)

    omega_infinite_corr_szsz_1_2, infinite_corr_susceptibility_szsz_1_2 = np.loadtxt(filename_corr_inf_1_szsz_2,unpack=True)
    omega_infinite_corr_szsz_2_2, infinite_corr_susceptibility_szsz_2_2 = np.loadtxt(filename_corr_inf_2_szsz_2,unpack=True)
    omega_infinite_corr_szsz_3_2, infinite_corr_susceptibility_szsz_3_2 = np.loadtxt(filename_corr_inf_3_szsz_2,unpack=True)

    omega_infinite_corr_szsz_1_3, infinite_corr_susceptibility_szsz_1_3 = np.loadtxt(filename_corr_inf_1_szsz_3,unpack=True)
    omega_infinite_corr_szsz_2_3, infinite_corr_susceptibility_szsz_2_3 = np.loadtxt(filename_corr_inf_2_szsz_3,unpack=True)
    omega_infinite_corr_szsz_3_3, infinite_corr_susceptibility_szsz_3_3 = np.loadtxt(filename_corr_inf_3_szsz_3,unpack=True)
    
    # ########################################################### Plotting ###########################################################
    # jj
    fig_jj = plt.figure(figsize=(15, 15))
    grid = plt.GridSpec(3, 3, wspace=0.23, hspace=0.1)

    if "_1D_" in filename_bare_1_jj and "_1D_" in filename_corr_1_jj:
        fig_jj.suptitle(r"$\langle j_xj_x\rangle$")
    elif "_2D_" in filename_bare_1_jj and "_2D_" in filename_corr_1_jj:
        fig_jj.suptitle(r"2D IPT ($U={0}$)".format(U_corr_1))

    # U=1.0
    top_panel_left = fig_jj.add_subplot(grid[0,0])
    top_panel_left.grid()
    top_panel_left.set_title(r"$U={0}$".format(U_corr_1))
    top_panel_left.text(0.20,0.60,"bare",transform=top_panel_left.transAxes,size=10,weight='bold')
    #top_panel_left.set_xlabel(r"$\omega$",labelpad=-0.4)
    top_panel_left.set_xlim(left=0.0,right=6.0)
    top_panel_left.set_ylim(bottom=0.0,top=0.7)
    top_panel_left.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    top_panel_left.yaxis.grid(True, which='minor',linestyle='--')
    top_panel_left.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    top_panel_left.xaxis.grid(True, which='minor',linestyle='--')
    top_panel_left.tick_params(axis='x',which="both",bottom=False,labelbottom=False)
    # top_panel_left.tick_params(axis='y',which="minor",direction="out")
    top_panel_left.plot(omega_jj_1,-1.0*bare_susceptibility_jj_1,marker='>',ms=2.0,c="coral",label=r"$T={0:.3f}$".format(1.0/beta_corr_1))#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    top_panel_left.plot(omega_jj_2,-1.0*bare_susceptibility_jj_2,marker='>',ms=2.0,c="lightblue",label=r"$T={0:.3f}$".format(1.0/beta_corr_2))
    top_panel_left.plot(omega_jj_3,-1.0*bare_susceptibility_jj_3,marker='>',ms=2.0,c="lightgreen",label=r"$T={0:.3f}$".format(1.0/beta_corr_3))
    handles, labels = plt.gca().get_legend_handles_labels()
    top_panel_left.legend(handles,labels,loc="upper right")
    #
    middle_panel_left = fig_jj.add_subplot(grid[1,0],sharex=top_panel_left)
    middle_panel_left.grid()
    middle_panel_left.set_ylim(bottom=0.0,top=0.7)
    # middle_panel_left.set_title(r"Single-ladder correction to $\langle j_xj_x\rangle$")
    middle_panel_left.text(0.20,0.60,"sl correction",transform=middle_panel_left.transAxes,size=10,weight='bold')
    middle_panel_left.plot(omega_single_corr_jj_1,-1.0*single_corr_susceptibility_jj_1,marker='v',ms=2.0,c="coral",label="_nolegend_")
    middle_panel_left.plot(omega_single_corr_jj_2,-1.0*single_corr_susceptibility_jj_2,marker='v',ms=2.0,c="lightblue",label="_nolegend_")
    middle_panel_left.plot(omega_single_corr_jj_3,-1.0*single_corr_susceptibility_jj_3,marker='v',ms=2.0,c="lightgreen",label="_nolegend_")
    middle_panel_left.set_ylabel(r"$\operatorname{Re}\sigma_{jj}(\omega,\mathbf{q}=\mathbf{0})\ast \omega$")
    middle_panel_left.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    middle_panel_left.yaxis.grid(True, which='minor',linestyle='--')
    middle_panel_left.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    middle_panel_left.xaxis.grid(True, which='minor',linestyle='--')
    middle_panel_left.tick_params(axis='x',which="both",bottom=False,labelbottom=False)
    # middle_panel_left.tick_params(axis='y',which="minor",direction="out")
    #middle_panel_left.set_xlabel(r"$\omega$")
    #
    bottom_panel_left = fig_jj.add_subplot(grid[2,0],sharex=top_panel_left)
    bottom_panel_left.grid()
    bottom_panel_left.set_ylim(bottom=0.0,top=0.7)
    # bottom_panel_left.set_title(r"Infinite-ladder correction to $\langle j_xj_x\rangle$")
    bottom_panel_left.text(0.20,0.60,"dl correction",transform=bottom_panel_left.transAxes,size=10,weight='bold')
    bottom_panel_left.set_xlabel(r"$\omega$",labelpad=-0.4)
    bottom_panel_left.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    bottom_panel_left.yaxis.grid(True, which='minor',linestyle='--')
    bottom_panel_left.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    bottom_panel_left.xaxis.grid(True, which='minor',linestyle='--')
    #bottom_panel_left.set_ylabel(r"$\operatorname{Re}\sigma_{jj}(\omega,\mathbf{q}=\mathbf{0})$")

    bottom_panel_left.plot(omega_infinite_corr_jj_1,-1.0*infinite_corr_susceptibility_jj_1,marker='o',ms=2.0,c="coral",label="_nolegend_")#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    bottom_panel_left.plot(omega_infinite_corr_jj_2,-1.0*infinite_corr_susceptibility_jj_2,marker='o',ms=2.0,c="lightblue",label="_nolegend_")
    bottom_panel_left.plot(omega_infinite_corr_jj_3,-1.0*infinite_corr_susceptibility_jj_3,marker='o',ms=2.0,c="lightgreen",label="_nolegend_")


    # U=2.0
    top_panel_middle = fig_jj.add_subplot(grid[0,1])
    top_panel_middle.grid()
    top_panel_middle.set_title(r"$U={0}$".format(U_corr_1_2))
    top_panel_middle.text(0.20,0.60,"bare",transform=top_panel_middle.transAxes,size=10,weight='bold')
    #top_panel_left.set_xlabel(r"$\omega$",labelpad=-0.4)
    top_panel_middle.set_xlim(left=0.0,right=8.0)
    top_panel_middle.set_ylim(bottom=0.0,top=1.1)
    top_panel_middle.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    top_panel_middle.yaxis.grid(True, which='minor',linestyle='--')
    top_panel_middle.xaxis.set_major_locator(MultipleLocator(2))
    top_panel_middle.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    top_panel_middle.xaxis.grid(True, which='minor',linestyle='--')
    top_panel_middle.tick_params(axis='x',which="both",bottom=False,labelbottom=False)
    # top_panel_left.tick_params(axis='y',which="minor",direction="out")
    top_panel_middle.plot(omega_jj_1_2,-1.0*bare_susceptibility_jj_1_2,marker='>',ms=2.0,c="red",label=r"$T={0:.3f}$".format(1.0/beta_corr_1_2))#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    top_panel_middle.plot(omega_jj_2_2,-1.0*bare_susceptibility_jj_2_2,marker='>',ms=2.0,c="blue",label=r"$T={0:.3f}$".format(1.0/beta_corr_2_2))
    top_panel_middle.plot(omega_jj_3_2,-1.0*bare_susceptibility_jj_3_2,marker='>',ms=2.0,c="green",label=r"$T={0:.3f}$".format(1.0/beta_corr_3_2))
    handles, labels = plt.gca().get_legend_handles_labels()
    top_panel_middle.legend(handles,labels,loc="upper right")
    #
    middle_panel_middle = fig_jj.add_subplot(grid[1,1],sharex=top_panel_middle)
    middle_panel_middle.grid()
    middle_panel_middle.set_ylim(bottom=0.0,top=1.1)
    # middle_panel_left.set_title(r"Single-ladder correction to $\langle j_xj_x\rangle$")
    middle_panel_middle.text(0.20,0.60,"sl correction",transform=middle_panel_middle.transAxes,size=10,weight='bold')
    middle_panel_middle.plot(omega_single_corr_jj_1_2,-1.0*single_corr_susceptibility_jj_1_2,marker='v',ms=2.0,c="red",label="_nolegend_")
    middle_panel_middle.plot(omega_single_corr_jj_2_2,-1.0*single_corr_susceptibility_jj_2_2,marker='v',ms=2.0,c="blue",label="_nolegend_")
    middle_panel_middle.plot(omega_single_corr_jj_3_2,-1.0*single_corr_susceptibility_jj_3_2,marker='v',ms=2.0,c="green",label="_nolegend_")
    middle_panel_middle.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    middle_panel_middle.yaxis.grid(True, which='minor',linestyle='--')
    middle_panel_middle.xaxis.set_major_locator(MultipleLocator(2))
    middle_panel_middle.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    middle_panel_middle.xaxis.grid(True, which='minor',linestyle='--')
    middle_panel_middle.tick_params(axis='x',which="both",bottom=False,labelbottom=False)
    # middle_panel_left.tick_params(axis='y',which="minor",direction="out")
    #middle_panel_left.set_xlabel(r"$\omega$")
    #
    bottom_panel_middle = fig_jj.add_subplot(grid[2,1],sharex=top_panel_middle)
    bottom_panel_middle.grid()
    bottom_panel_middle.set_ylim(bottom=0.0,top=1.1)
    # bottom_panel_left.set_title(r"Infinite-ladder correction to $\langle j_xj_x\rangle$")
    bottom_panel_middle.text(0.20,0.60,"dl correction",transform=bottom_panel_middle.transAxes,size=10,weight='bold')
    bottom_panel_middle.set_xlabel(r"$\omega$",labelpad=-0.4)
    bottom_panel_middle.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    bottom_panel_middle.yaxis.grid(True, which='minor',linestyle='--')
    bottom_panel_middle.xaxis.set_major_locator(MultipleLocator(2))
    bottom_panel_middle.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    bottom_panel_middle.xaxis.grid(True, which='minor',linestyle='--')
    #bottom_panel_middle.set_ylabel(r"$\operatorname{Re}\sigma_{jj}(\omega,\mathbf{q}=\mathbf{0})$")

    bottom_panel_middle.plot(omega_infinite_corr_jj_1_2,-1.0*infinite_corr_susceptibility_jj_1_2,marker='o',ms=2.0,c="red",label="_nolegend_")#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    bottom_panel_middle.plot(omega_infinite_corr_jj_2_2,-1.0*infinite_corr_susceptibility_jj_2_2,marker='o',ms=2.0,c="blue",label="_nolegend_")
    bottom_panel_middle.plot(omega_infinite_corr_jj_3_2,-1.0*infinite_corr_susceptibility_jj_3_2,marker='o',ms=2.0,c="green",label="_nolegend_")

    # U=3.0
    top_panel_right = fig_jj.add_subplot(grid[0,2])
    top_panel_right.grid()
    top_panel_right.set_title(r"$U={0}$".format(U_corr_1_3))
    top_panel_right.text(0.20,0.60,"bare",transform=top_panel_right.transAxes,size=10,weight='bold')
    top_panel_right.set_xlim(left=0.0,right=8.0)
    top_panel_right.set_ylim(bottom=0.0,top=1.2)
    top_panel_right.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    top_panel_right.yaxis.grid(True, which='minor',linestyle='--')
    top_panel_right.xaxis.set_major_locator(MultipleLocator(2))
    top_panel_right.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    top_panel_right.xaxis.grid(True, which='minor',linestyle='--')
    top_panel_right.tick_params(axis='x',which="both",bottom=False,labelbottom=False)
    top_panel_right.plot(omega_jj_1_3,-1.0*bare_susceptibility_jj_1_3,marker='>',ms=2.0,c="maroon",label=r"$T={0:.3f}$".format(1.0/beta_corr_1_3))#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    top_panel_right.plot(omega_jj_2_3,-1.0*bare_susceptibility_jj_2_3,marker='>',ms=2.0,c="darkblue",label=r"$T={0:.3f}$".format(1.0/beta_corr_2_3))
    top_panel_right.plot(omega_jj_3_3,-1.0*bare_susceptibility_jj_3_3,marker='>',ms=2.0,c="darkgreen",label=r"$T={0:.3f}$".format(1.0/beta_corr_3_3))
    handles, labels = plt.gca().get_legend_handles_labels()
    top_panel_right.legend(handles,labels,loc="upper right")
    #
    middle_panel_right = fig_jj.add_subplot(grid[1,2],sharex=top_panel_right)
    middle_panel_right.grid()
    middle_panel_right.set_ylim(bottom=0.0,top=5.2)
    # middle_panel_left.set_title(r"Single-ladder correction to $\langle j_xj_x\rangle$")
    middle_panel_right.text(0.20,0.60,"sl correction",transform=middle_panel_right.transAxes,size=10,weight='bold')
    middle_panel_right.plot(omega_single_corr_jj_1_3,-1.0*single_corr_susceptibility_jj_1_3,marker='v',ms=2.0,c="maroon",label="_nolegend_")
    middle_panel_right.plot(omega_single_corr_jj_2_3,-1.0*single_corr_susceptibility_jj_2_3,marker='v',ms=2.0,c="darkblue",label="_nolegend_")
    middle_panel_right.plot(omega_single_corr_jj_3_3,-1.0*single_corr_susceptibility_jj_3_3,marker='v',ms=2.0,c="darkgreen",label="_nolegend_")
    middle_panel_right.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    middle_panel_right.yaxis.grid(True, which='minor',linestyle='--')
    middle_panel_right.xaxis.set_major_locator(MultipleLocator(2))
    middle_panel_right.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    middle_panel_right.xaxis.grid(True, which='minor',linestyle='--')
    middle_panel_right.tick_params(axis='x',which="both",bottom=False,labelbottom=False)
    # middle_panel_left.tick_params(axis='y',which="minor",direction="out")
    #middle_panel_left.set_xlabel(r"$\omega$")
    #
    bottom_panel_right = fig_jj.add_subplot(grid[2,2],sharex=top_panel_right)
    bottom_panel_right.grid()
    bottom_panel_right.set_ylim(bottom=0.0,top=5.2)
    # bottom_panel_left.set_title(r"Infinite-ladder correction to $\langle j_xj_x\rangle$")
    bottom_panel_right.text(0.20,0.70,"dl correction",transform=bottom_panel_right.transAxes,size=10,weight='bold')
    bottom_panel_right.set_xlabel(r"$\omega$",labelpad=-0.4)
    bottom_panel_right.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    bottom_panel_right.yaxis.grid(True, which='minor',linestyle='--')
    bottom_panel_right.xaxis.set_major_locator(MultipleLocator(2))
    bottom_panel_right.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    bottom_panel_right.xaxis.grid(True, which='minor',linestyle='--')

    bottom_panel_right.plot(omega_infinite_corr_jj_1_3,-1.0*infinite_corr_susceptibility_jj_1_3,marker='o',ms=2.0,c="maroon",label="_nolegend_")#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    bottom_panel_right.plot(omega_infinite_corr_jj_2_3,-1.0*infinite_corr_susceptibility_jj_2_3,marker='o',ms=2.0,c="darkblue",label="_nolegend_")
    bottom_panel_right.plot(omega_infinite_corr_jj_3_3,-1.0*infinite_corr_susceptibility_jj_3_3,marker='o',ms=2.0,c="darkgreen",label="_nolegend_")

    #plt.show()
    plt.gcf().set_size_inches(18.5/2.54,12/2.54)
    if "_1D_" in filename_corr_1_jj and "_1D_" in filename_bare_1_jj:
        plt.savefig("Figure_susceptibility_comparison_jj_1D.pdf")
    elif "_2D_" in filename_corr_1_jj and "_2D_" in filename_bare_1_jj:
        plt.savefig("Figure_susceptibility_comparison_jj_2D.pdf")
    
    #####################################################################################################################################
    #####################################################################################################################################
    
    # szsz
    fig_szsz = plt.figure(figsize=(15, 15))
    grid = plt.GridSpec(3, 3, wspace=0.23, hspace=0.1)

    if "_1D_" in filename_bare_1_jj and "_1D_" in filename_corr_1_jj:
        fig_szsz.suptitle(r"$\langle S_zS_z\rangle$")
    elif "_2D_" in filename_bare_1_jj and "_2D_" in filename_corr_1_jj:
        fig_szsz.suptitle(r"2D IPT ($U={0}$)".format(U_corr_1))

    # U=1.0
    top_panel_left = fig_szsz.add_subplot(grid[0,0])
    top_panel_left.grid()
    top_panel_left.set_title(r"$U={0}$".format(U_corr_1))
    top_panel_left.text(0.20,0.60,"bare",transform=top_panel_left.transAxes,size=10,weight='bold')
    # top_panel_left.ticklabel_format(axis='y',style='sci',scilimits=(-1,-1))
    # top_panel_left.set_xlabel(r"$\omega$",labelpad=-0.4)
    top_panel_left.set_xlim(left=0.0,right=6.0)
    top_panel_left.set_ylim(bottom=0.0,top=0.3*spin_spin_factor)
    top_panel_left.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    top_panel_left.yaxis.grid(True, which='minor',linestyle='--')
    top_panel_left.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    top_panel_left.xaxis.grid(True, which='minor',linestyle='--')
    top_panel_left.tick_params(axis='x',which="both",bottom=False,labelbottom=False)
    top_panel_left.plot(omega_szsz_1,-1.0*bare_susceptibility_szsz_1*spin_spin_factor,marker='>',ms=2.0,c="coral",label=r"$T={0:.3f}$".format(1.0/beta_corr_1))#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    top_panel_left.plot(omega_szsz_2,-1.0*bare_susceptibility_szsz_2*spin_spin_factor,marker='>',ms=2.0,c="lightblue",label=r"$T={0:.3f}$".format(1.0/beta_corr_2))
    top_panel_left.plot(omega_szsz_3,-1.0*bare_susceptibility_szsz_3*spin_spin_factor,marker='>',ms=2.0,c="lightgreen",label=r"$T={0:.3f}$".format(1.0/beta_corr_3))
    handles, labels = plt.gca().get_legend_handles_labels()
    top_panel_left.legend(handles,labels,loc="upper right")
    #
    middle_panel_left = fig_szsz.add_subplot(grid[1,0],sharex=top_panel_left)
    middle_panel_left.grid()
    middle_panel_left.set_ylim(bottom=0.0,top=0.2*spin_spin_factor)
    # middle_panel_left.set_title(r"Single-ladder correction to $\langle j_xj_x\rangle$")
    middle_panel_left.text(0.20,0.60,"sl correction",transform=middle_panel_left.transAxes,size=10,weight='bold')
    middle_panel_left.plot(omega_single_corr_szsz_1,-1.0*single_corr_susceptibility_szsz_1*spin_spin_factor,marker='v',ms=2.0,c="coral",label="_nolegend_")
    middle_panel_left.plot(omega_single_corr_szsz_2,-1.0*single_corr_susceptibility_szsz_2*spin_spin_factor,marker='v',ms=2.0,c="lightblue",label="_nolegend_")
    middle_panel_left.plot(omega_single_corr_szsz_3,-1.0*single_corr_susceptibility_szsz_3*spin_spin_factor,marker='v',ms=2.0,c="lightgreen",label="_nolegend_")
    middle_panel_left.set_ylabel(r"$\operatorname{Im}\chi_{S_zS_z}(\omega,\mathbf{q}=\mathbf{0})$")
    middle_panel_left.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    middle_panel_left.yaxis.grid(True, which='minor',linestyle='--')
    middle_panel_left.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    middle_panel_left.xaxis.grid(True, which='minor',linestyle='--')
    middle_panel_left.tick_params(axis='x',which="both",bottom=False,labelbottom=False)
    # middle_panel_left.tick_params(axis='y',which="minor",direction="out")
    #middle_panel_left.set_xlabel(r"$\omega$")
    #
    bottom_panel_left = fig_szsz.add_subplot(grid[2,0],sharex=top_panel_left)
    bottom_panel_left.grid()
    bottom_panel_left.set_ylim(bottom=0.0,top=0.2*spin_spin_factor)
    # bottom_panel_left.set_title(r"Infinite-ladder correction to $\langle j_xj_x\rangle$")
    bottom_panel_left.text(0.20,0.60,"dl correction",transform=bottom_panel_left.transAxes,size=10,weight='bold')
    # bottom_panel_left.ticklabel_format(axis='y',style='sci',scilimits=(-1,-1))
    bottom_panel_left.set_xlabel(r"$\omega$",labelpad=-0.4)
    bottom_panel_left.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    bottom_panel_left.yaxis.grid(True, which='minor',linestyle='--')
    bottom_panel_left.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    bottom_panel_left.xaxis.grid(True, which='minor',linestyle='--')
    #bottom_panel_left.set_ylabel(r"$\operatorname{Re}\sigma_{jj}(\omega,\mathbf{q}=\mathbf{0})$")

    bottom_panel_left.plot(omega_infinite_corr_szsz_1,-1.0*infinite_corr_susceptibility_szsz_1*spin_spin_factor,marker='o',ms=2.0,c="coral",label="_nolegend_")#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    bottom_panel_left.plot(omega_infinite_corr_szsz_2,-1.0*infinite_corr_susceptibility_szsz_2*spin_spin_factor,marker='o',ms=2.0,c="lightblue",label="_nolegend_")
    bottom_panel_left.plot(omega_infinite_corr_szsz_3,-1.0*infinite_corr_susceptibility_szsz_3*spin_spin_factor,marker='o',ms=2.0,c="lightgreen",label="_nolegend_")

    # U=2.0
    top_panel_middle = fig_szsz.add_subplot(grid[0,1])
    top_panel_middle.grid()
    top_panel_middle.set_title(r"$U={0}$".format(U_corr_1_2))
    top_panel_middle.text(0.20,0.60,"bare",transform=top_panel_middle.transAxes,size=10,weight='bold')
    #top_panel_left.set_xlabel(r"$\omega$",labelpad=-0.4)
    top_panel_middle.set_xlim(left=0.0,right=8.0)
    top_panel_middle.set_ylim(bottom=0.0,top=0.5*spin_spin_factor)
    top_panel_middle.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    top_panel_middle.yaxis.grid(True, which='minor',linestyle='--')
    top_panel_middle.xaxis.set_major_locator(MultipleLocator(2))
    top_panel_middle.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    top_panel_middle.xaxis.grid(True, which='minor',linestyle='--')
    top_panel_middle.tick_params(axis='x',which="both",bottom=False,labelbottom=False)
    # top_panel_left.tick_params(axis='y',which="minor",direction="out")
    top_panel_middle.plot(omega_szsz_1_2,-1.0*bare_susceptibility_szsz_1_2*spin_spin_factor,marker='>',ms=2.0,c="red",label=r"$T={0:.3f}$".format(1.0/beta_corr_1_2))#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    top_panel_middle.plot(omega_szsz_2_2,-1.0*bare_susceptibility_szsz_2_2*spin_spin_factor,marker='>',ms=2.0,c="blue",label=r"$T={0:.3f}$".format(1.0/beta_corr_2_2))
    top_panel_middle.plot(omega_szsz_3_2,-1.0*bare_susceptibility_szsz_3_2*spin_spin_factor,marker='>',ms=2.0,c="green",label=r"$T={0:.3f}$".format(1.0/beta_corr_3_2))
    handles, labels = plt.gca().get_legend_handles_labels()
    top_panel_middle.legend(handles,labels,loc="upper right")
    #
    middle_panel_middle = fig_szsz.add_subplot(grid[1,1],sharex=top_panel_middle)
    middle_panel_middle.grid()
    middle_panel_middle.set_ylim(bottom=0.0,top=0.5*spin_spin_factor)
    # middle_panel_left.set_title(r"Single-ladder correction to $\langle j_xj_x\rangle$")
    middle_panel_middle.text(0.20,0.60,"sl correction",transform=middle_panel_middle.transAxes,size=10,weight='bold')
    middle_panel_middle.plot(omega_single_corr_szsz_1_2,-1.0*single_corr_susceptibility_szsz_1_2*spin_spin_factor,marker='v',ms=2.0,c="red",label="_nolegend_")
    middle_panel_middle.plot(omega_single_corr_szsz_2_2,-1.0*single_corr_susceptibility_szsz_2_2*spin_spin_factor,marker='v',ms=2.0,c="blue",label="_nolegend_")
    middle_panel_middle.plot(omega_single_corr_szsz_3_2,-1.0*single_corr_susceptibility_szsz_3_2*spin_spin_factor,marker='v',ms=2.0,c="green",label="_nolegend_")
    middle_panel_middle.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    middle_panel_middle.yaxis.grid(True, which='minor',linestyle='--')
    middle_panel_middle.xaxis.set_major_locator(MultipleLocator(2))
    middle_panel_middle.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    middle_panel_middle.xaxis.grid(True, which='minor',linestyle='--')
    middle_panel_middle.tick_params(axis='x',which="both",bottom=False,labelbottom=False)
    # middle_panel_left.tick_params(axis='y',which="minor",direction="out")
    #middle_panel_left.set_xlabel(r"$\omega$")
    #
    bottom_panel_middle = fig_szsz.add_subplot(grid[2,1],sharex=top_panel_middle)
    bottom_panel_middle.grid()
    bottom_panel_middle.set_ylim(bottom=0.0,top=0.5*spin_spin_factor)
    # bottom_panel_left.set_title(r"Infinite-ladder correction to $\langle j_xj_x\rangle$")
    bottom_panel_middle.text(0.20,0.60,"dl correction",transform=bottom_panel_middle.transAxes,size=10,weight='bold')
    bottom_panel_middle.set_xlabel(r"$\omega$",labelpad=-0.4)
    bottom_panel_middle.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    bottom_panel_middle.yaxis.grid(True, which='minor',linestyle='--')
    bottom_panel_middle.xaxis.set_major_locator(MultipleLocator(2))
    bottom_panel_middle.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    bottom_panel_middle.xaxis.grid(True, which='minor',linestyle='--')
    #bottom_panel_middle.set_ylabel(r"$\operatorname{Re}\sigma_{jj}(\omega,\mathbf{q}=\mathbf{0})$")

    bottom_panel_middle.plot(omega_infinite_corr_szsz_1_2,-1.0*infinite_corr_susceptibility_szsz_1_2*spin_spin_factor,marker='o',ms=2.0,c="red",label="_nolegend_")#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    bottom_panel_middle.plot(omega_infinite_corr_szsz_2_2,-1.0*infinite_corr_susceptibility_szsz_2_2*spin_spin_factor,marker='o',ms=2.0,c="blue",label="_nolegend_")
    bottom_panel_middle.plot(omega_infinite_corr_szsz_3_2,-1.0*infinite_corr_susceptibility_szsz_3_2*spin_spin_factor,marker='o',ms=2.0,c="green",label="_nolegend_")

    # U=3.0
    top_panel_right = fig_szsz.add_subplot(grid[0,2])
    top_panel_right.grid()
    top_panel_right.set_title(r"$U={0}$".format(U_corr_1_3))
    top_panel_right.text(0.20,0.60,"bare",transform=top_panel_right.transAxes,size=10,weight='bold')
    top_panel_right.set_xlim(left=0.0,right=8.0)
    top_panel_right.set_ylim(bottom=0.0,top=0.28*spin_spin_factor)
    top_panel_right.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    top_panel_right.yaxis.grid(True, which='minor',linestyle='--')
    top_panel_right.xaxis.set_major_locator(MultipleLocator(2))
    top_panel_right.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    top_panel_right.xaxis.grid(True, which='minor',linestyle='--')
    top_panel_right.tick_params(axis='x',which="both",bottom=False,labelbottom=False)
    top_panel_right.plot(omega_szsz_1_3,-1.0*bare_susceptibility_szsz_1_3*spin_spin_factor,marker='>',ms=2.0,c="maroon",label=r"$T={0:.3f}$".format(1.0/beta_corr_1_3))#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    top_panel_right.plot(omega_szsz_2_3,-1.0*bare_susceptibility_szsz_2_3*spin_spin_factor,marker='>',ms=2.0,c="darkblue",label=r"$T={0:.3f}$".format(1.0/beta_corr_2_3))
    top_panel_right.plot(omega_szsz_3_3,-1.0*bare_susceptibility_szsz_3_3*spin_spin_factor,marker='>',ms=2.0,c="darkgreen",label=r"$T={0:.3f}$".format(1.0/beta_corr_3_3))
    handles, labels = plt.gca().get_legend_handles_labels()
    top_panel_right.legend(handles,labels,loc="upper right")
    #
    middle_panel_right = fig_szsz.add_subplot(grid[1,2],sharex=top_panel_right)
    middle_panel_right.grid()
    middle_panel_right.set_ylim(bottom=0.0,top=1.0*spin_spin_factor)
    # middle_panel_left.set_title(r"Single-ladder correction to $\langle j_xj_x\rangle$")
    middle_panel_right.text(0.20,0.60,"sl correction",transform=middle_panel_right.transAxes,size=10,weight='bold')
    middle_panel_right.plot(omega_single_corr_szsz_1_3,-1.0*single_corr_susceptibility_szsz_1_3*spin_spin_factor,marker='v',ms=2.0,c="maroon",label="_nolegend_")
    middle_panel_right.plot(omega_single_corr_szsz_2_3,-1.0*single_corr_susceptibility_szsz_2_3*spin_spin_factor,marker='v',ms=2.0,c="darkblue",label="_nolegend_")
    middle_panel_right.plot(omega_single_corr_szsz_3_3,-1.0*single_corr_susceptibility_szsz_3_3*spin_spin_factor,marker='v',ms=2.0,c="darkgreen",label="_nolegend_")
    middle_panel_right.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    middle_panel_right.yaxis.grid(True, which='minor',linestyle='--')
    middle_panel_right.xaxis.set_major_locator(MultipleLocator(2))
    middle_panel_right.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    middle_panel_right.xaxis.grid(True, which='minor',linestyle='--')
    middle_panel_right.tick_params(axis='x',which="both",bottom=False,labelbottom=False)
    # middle_panel_left.tick_params(axis='y',which="minor",direction="out")
    #middle_panel_left.set_xlabel(r"$\omega$")
    #
    bottom_panel_right = fig_szsz.add_subplot(grid[2,2],sharex=top_panel_right)
    bottom_panel_right.grid()
    bottom_panel_right.set_ylim(bottom=0.0,top=1.0*spin_spin_factor)
    # bottom_panel_left.set_title(r"Infinite-ladder correction to $\langle j_xj_x\rangle$")
    bottom_panel_right.text(0.20,0.60,"dl correction",transform=bottom_panel_right.transAxes,size=10,weight='bold')
    bottom_panel_right.set_xlabel(r"$\omega$",labelpad=-0.4)
    bottom_panel_right.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    bottom_panel_right.yaxis.grid(True, which='minor',linestyle='--')
    bottom_panel_right.xaxis.set_major_locator(MultipleLocator(2))
    bottom_panel_right.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    bottom_panel_right.xaxis.grid(True, which='minor',linestyle='--')

    bottom_panel_right.plot(omega_infinite_corr_szsz_1_3,-1.0*infinite_corr_susceptibility_szsz_1_3*spin_spin_factor,marker='o',ms=2.0,c="maroon",label="_nolegend_")#,label=r"$\mathbf{q}$=%.2f" % (0.0))
    bottom_panel_right.plot(omega_infinite_corr_szsz_2_3,-1.0*infinite_corr_susceptibility_szsz_2_3*spin_spin_factor,marker='o',ms=2.0,c="darkblue",label="_nolegend_")
    bottom_panel_right.plot(omega_infinite_corr_szsz_3_3,-1.0*infinite_corr_susceptibility_szsz_3_3*spin_spin_factor,marker='o',ms=2.0,c="darkgreen",label="_nolegend_")


    #plt.show()
    plt.gcf().set_size_inches(18.5/2.54,12/2.54)
    if "_1D_" in filename_corr_1_jj and "_1D_" in filename_bare_1_jj:
        plt.savefig("Figure_susceptibility_comparison_szsz_1D.pdf")
    elif "_2D_" in filename_corr_1_jj and "_2D_" in filename_bare_1_jj:
        plt.savefig("Figure_susceptibility_comparison_szsz_2D.pdf")
