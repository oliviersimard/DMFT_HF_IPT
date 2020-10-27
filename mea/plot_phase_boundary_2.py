import os
import numpy as np
import matplotlib.pyplot as plt
from sys import exit
from re import findall
from shutil import copy2
from glob import glob
from time import sleep
from optparse import OptionParser
import operator
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import h5py

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage[charter]{mathdesign}\usepackage{amsmath}"]

# maximum number of DMFT iterations
MAXVAL = 150-1

class Files_for_params(object):
    def __init__(self, largest_num_iteration : int, root : str, largest_int_file_for_params : str):
        self._largest_num_iter = largest_num_iteration
        self._root = root
        self._largest_int_file_for_params = largest_int_file_for_params
    def __str__(self):
        return "{"+"largest num: {0:d}".format(self._largest_num_iter)+" root: {0}".format(self._root)+" file with largest int: {0}".format(self._largest_int_file_for_params)+"}"

def put_element_in_list_if_different(list_elements : list, element_to_compare) -> list:
    """
        Returns a list containing all the different elements.
    """
    if element_to_compare not in list_elements:
        list_elements.append(element_to_compare)
    
    return list_elements

def get_params_with_many_Ntau(Us : list, betas : list, dict_params : dict) -> dict:
    """
        This function selects the parameter set with the highest Ntau, when U and beta are the same.
    """
    tmp_dict = {}
    for U in Us:
        for beta in betas:
            tmp_dict[(U,beta)] = [0]
            for key in dict_params.keys():
                if beta == key[1] and U == key[2]:
                    tmp_dict[(U,beta)][0] += 1
                    tmp_dict[(U,beta)].append(key[0])
            if tmp_dict[(U,beta)][0] > 1:
                del dict_params[(min(tmp_dict[(U,beta)][1:]),beta,U)]
    return dict_params


def extract_largest_int(list_str : list, cond_to_select_file : str = "") -> tuple:
    """
        This function extracts the highest number associated with the iteration number from a set of files.
        It does so for a subset of the files matching the condition imposed to files (cond_to_select_file).
    """
    integ_num = -1
    file_largest_int=""
    for el in list_str:
        if el.count(cond_to_select_file) > 1:
            if any(char.isdigit() for char in el):
                el_tmp=el.replace('.dat','')
                tail_int = int(el_tmp.split("_")[-1])
                if tail_int > integ_num and tail_int < MAXVAL:
                    integ_num=tail_int
                    file_largest_int=el
    return integ_num, file_largest_int

def rearrange_files_into_dir(path_to_root : str) -> None:
    """
        Rearrange files in a given directory assuming no subdirectories already exist within path_to_root: if not the 
        case, the function if skipped, meaning that the files have been sorted into their respective folders. The files
        are separated off based on the parameter values they hold as markers. You need to remove the scattered files by
        hand after the operation.
    """
    param_tupl_list = []
    # getting rid of the content before first sep /
    path_to_root_modif = '/'.join(path_to_root.split('/')[1:])
    current_parent_path = os.getcwd()+"/"+path_to_root_modif
    for root, dirs, files in os.walk(path_to_root):
        # ensuring it is all sparse with no subdirectories
        if dirs != []:
            break # means is has been ordered
        for f in files:
            Ntau = int(findall(r"(?<=N_tau_)(\d+)",f)[0])
            beta = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",f)[0])
            U = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",f)[0])
            if (Ntau,beta,U) not in param_tupl_list:
                param_tupl_list.append((Ntau,beta,U))

    # making the directories and moving the files into their respective placeholder
    n=0.5 #<----------------------------- this is the default value for now
    for param_set in param_tupl_list:
        Ntau, beta, U = param_set
        # path is the directory's finger print
        path = current_parent_path+"/"+"1D_U_{0:.5f}_beta_{1:.5f}_n_{2:.5f}_Ntau_{3:d}".format(U,beta,n,Ntau)
        os.mkdir(path,mode=0o777)
        list_files_corresponding_to_param = glob(current_parent_path+"/"+"*U_{0:.5f}*beta_{1:.5f}*N_tau_{2:d}*".format(U,beta,Ntau))
        list_files_corresponding_to_param = sorted( list_files_corresponding_to_param, key=lambda x: int( findall( r"(?<=Nit_)(\d+)", x)[0] ) )
        for f in list_files_corresponding_to_param:
            copy2(f,path)
    
    return None

def AFM_phase_boundary_condition(filename : str, delimiter='\t\t', col_num=(1,2), tol=2e-2) -> bool:
    bool_cond = False
    col_num_data_1 = []; col_num_data_2 = []
    with open(filename, "r") as f:
        for line in f.readlines():
            col_num_data_1.append(line.split(delimiter)[col_num[0]])
            col_num_data_2.append(line.split(delimiter)[col_num[1]].rstrip('\n'))
    f.close()
    magnetization = np.abs(float(col_num_data_1[-1])-float(col_num_data_2[-1]))
    print("magnetization: ", magnetization)
    if np.isclose(magnetization,0.0,rtol=tol,atol=tol):
        bool_cond = True

    return bool_cond, magnetization


def get_according_to_condition(path : str, funct) -> tuple:
    """
        The function funct passed as parameter has to return a boolean value to determine if it has passed. It has to 
        have filename as parameter only as well.
    """
    dict_to_store_passing_cond_files = {}
    Us = []
    betas = []
    dim_int = 0
    # navigating through the directory containing the data
    for root, dirs, files in os.walk(path):
        path = root.split(os.sep) # os.sep is equivalent to "/"
        print( (len(path)-1)*'---', os.path.basename(root) )
        if any(char.isdigit() for char in os.path.basename(root)):
            dim_int = int(*findall(r'^\d+',os.path.basename(root))) # All files have dimension mentionned in their name, so any file fits.
        if files != []:
            largest_int, file_largest_int = extract_largest_int(files,"tau") # The file read for magnetization has to contain "tau" in its name...
            Ntau = int(findall(r"(?<=N_tau_)(\d+)",file_largest_int)[0])
            beta = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",file_largest_int)[0])
            U = float(findall(r"(?<=U_)(\d*\.\d+|\d+)",file_largest_int)[0])
            # fetching last iteration
            container = Files_for_params(largest_int,root,file_largest_int)
            dict_to_store_passing_cond_files[(Ntau,beta,U)] = container
            put_element_in_list_if_different(Us,U)
            put_element_in_list_if_different(betas,beta)
    # Keeping the data with the highest Ntau for parameter sets having smae beta and U
    get_params_with_many_Ntau(Us,betas,dict_to_store_passing_cond_files)
    params_respecting_condiction = []
    dict_to_store_magnetization_for_Us_as_function_beta = {} # container for magnetization as function of betas and Us
    for U in Us:
        print("U:", U)
        max_beta = 0.0
        for beta in betas:
            print("beta: ", beta)
            for key in dict_to_store_passing_cond_files.keys():
                if beta == key[1] and U == key[2]: # Should be regardless of the tau grid
                    f = dict_to_store_passing_cond_files[key]._largest_int_file_for_params
                    print("largest int file: ", f)
                    #sleep(2)
                    fconcat = os.path.join(dict_to_store_passing_cond_files[key]._root,f)
                    cond,*rest = funct(fconcat) # packing rest of outputs, in this case the magnetization, into an array
                    dict_to_store_magnetization_for_Us_as_function_beta[(U,beta)] = rest # rest holds magnetization in this case
                    if cond and dict_to_store_passing_cond_files[key]._largest_num_iter<MAXVAL: # condition to consider phase delimitation. Needs to have converged in AFM.
                        if beta>max_beta:
                            max_beta=beta
                    elif rest[0]>0.10 and dict_to_store_passing_cond_files[key]._largest_num_iter>=MAXVAL: # rest[0] might change threshold, depending on hybridisation mixing
                        if beta>max_beta:
                            max_beta=beta
            print("max_beta: ", max_beta)
        put_element_in_list_if_different(params_respecting_condiction,(max_beta,U))
    
    return params_respecting_condiction, dim_int, dict_to_store_magnetization_for_Us_as_function_beta, Us, betas

def get_pts_close_to_num_old(arr, num):
    idx_val_tuple = min(enumerate(arr), key=lambda x: abs(x[1]-num))
    return idx_val_tuple

def get_pts_close_to_num(arr, num):
    idx_val_tuple = min(enumerate(arr), key=lambda x: abs(x[1][1]-num))[0]
    return idx_val_tuple, arr[idx_val_tuple]

if __name__=="__main__":

    parser = OptionParser()

    parser.add_option("--dataIPT", dest="pathIPT", default=u"./data_test_IPT")
    # parser.add_option("--dataNCA", dest="pathNCA", default=u"./data_test_IPT")
    (options, args) = parser.parse_args()

    path = str(options.pathIPT)
    # pathNCA = str(options.pathNCA)
    # if the files are all scattered in the target directory, the files are rearranged. Otherwise, it is passed.
    rearrange_files_into_dir(path)
    # rearrange_files_into_dir(pathNCA)

    params_respecting_condiction, dim_int, dict_to_store_magnetization_for_Us_as_function_beta, Us, betas = get_according_to_condition(path,AFM_phase_boundary_condition)
    # params_respecting_condictionNCA, dim_intNCA, dict_to_store_magnetization_for_Us_as_function_betaNCA, UsNCA, betasNCA = get_according_to_condition(pathNCA,AFM_phase_boundary_condition)

    # sorting the dictionary according to U and beta (ascending order first)
    list_magnetization_for_Us_as_function_beta = sorted(dict_to_store_magnetization_for_Us_as_function_beta.items(),key=operator.itemgetter(0,1))
    # list_magnetization_for_Us_as_function_betaNCA = sorted(dict_to_store_magnetization_for_Us_as_function_betaNCA.items(),key=operator.itemgetter(0,1))
    
    ######################################################################################### <------------------------ delimitation calc
    # getting the delimitations
    # sl
    if dim_int == 1:
        filename_div_sl = "./cpp_tests/div/div_1D_U_1.000000_0.500000_9.000000_beta_1.000000_0.250000_300.000000_Ntau_2048_Nk_51_single_ladder_sum.hdf5"
    elif dim_int == 2:
        filename_div_sl = "./cpp_tests/div/div_2D_U_1.000000_0.500000_10.000000_beta_1.000000_0.200000_100.000000_Ntau_1024_Nk_51_DIV_1.000000_single_ladder_sum.hdf5"
    betas = findall(r"beta_(\d*\.\d+|\d+)_(\d*\.\d+|\d+)_(\d*\.\d+|\d+)",filename_div_sl)[0]
    beta_init = float(betas[0]); beta_step = float(betas[1]); beta_max = float(betas[2])
    beta_arr = np.arange(beta_init,beta_max+beta_step,step=beta_step)

    hf = h5py.File(filename_div_sl,"r")
    list_keys = list(hf.keys())
    # extracting the Us from the keys
    Uss_sl = []
    dict_U_div_T = {}
    for i,key in enumerate(list_keys):
        Uss_sl.append(float(findall(r"(?<=U_)(\d*\.\d+|\d+)",key)[0]))
        U_slice_for_betas = list(hf.get(key))
        idx_closest, val_closest = get_pts_close_to_num(U_slice_for_betas,-1.0)
        
        other_idx=0
        if val_closest[1]>-1.0 and idx_closest!=(len(U_slice_for_betas)-1):
            other_idx=idx_closest+1
        else:
            other_idx=idx_closest-1

        print("sl: ",Uss_sl[-1]," idx closest: ", idx_closest, " val closest: ", val_closest, " other idx: ", other_idx, " other val: ", U_slice_for_betas[other_idx])
        if np.isclose(-1.0,val_closest[1],atol=2e-1): # making sure it actually crosses -1.0
            dict_U_div_T[Uss_sl[i]] = (val_closest[0]+U_slice_for_betas[other_idx][0])/2.0
    
    Uss_sl = np.array(list(dict_U_div_T.keys()))
    Tss_sl = np.array(list(dict_U_div_T.values()))
    hf.close()

    # Patching region where enhanced precision is needed. Only relevant for 1D for now.
    if dim_int == 1:
        filename_div_sl_2 = "./cpp_tests/div/div_1D_U_3.000000_0.250000_5.000000_beta_1.000000_0.050000_5.000000_Ntau_2048_Nk_51_DIV_1.000000_single_ladder_sum.hdf5"
    
        betas_2 = findall(r"beta_(\d*\.\d+|\d+)_(\d*\.\d+|\d+)_(\d*\.\d+|\d+)",filename_div_sl_2)[0]
        Us_2 = findall(r"U_(\d*\.\d+|\d+)_(\d*\.\d+|\d+)_(\d*\.\d+|\d+)",filename_div_sl_2)[0]
        beta_init_2 = float(betas_2[0]); beta_step_2 = float(betas_2[1]); beta_max_2 = float(betas_2[2])
        beta_arr_2 = np.arange(beta_init_2,beta_max_2+beta_step_2,step=beta_step_2)
        U_init_2 = float(Us_2[0]); U_max_2 = float(Us_2[2])

        hf = h5py.File(filename_div_sl_2,"r")
        list_keys = list(hf.keys())
        # extracting the Us from the keys
        Uss_sl_2 = []
        dict_U_div_T_2 = {}
        for i,key in enumerate(list_keys):
            Uss_sl_2.append(float(findall(r"(?<=U_)(\d*\.\d+|\d+)",key)[0]))
            U_slice_for_betas = list(hf.get(key))
            idx_closest, val_closest = get_pts_close_to_num(U_slice_for_betas,-1.0)
            
            other_idx=0
            if val_closest[1]>-1.0 and idx_closest!=(len(U_slice_for_betas)-1):
                other_idx=idx_closest+1
            else:
                other_idx=idx_closest-1

            print("sl: ",Uss_sl_2[-1]," idx closest: ", idx_closest, " val closest: ", val_closest, " other idx: ", other_idx, " other val: ", U_slice_for_betas[other_idx])
            if np.isclose(-1.0,val_closest[1],atol=2e-1): # making sure it actually crosses -1.0
                dict_U_div_T_2[Uss_sl_2[i]] = (val_closest[0]+U_slice_for_betas[other_idx][0])/2.0
        
        
        idx_U_min_val_to_replace = np.where(Uss_sl==U_init_2)[0][0]
        idx_U_max_val_to_replace = np.where(Uss_sl==U_max_2)[0][0]

        print("INDEXES: ", idx_U_min_val_to_replace, " ", idx_U_max_val_to_replace)

        Uss_sl_2 = np.array(list(dict_U_div_T_2.keys()))
        Tss_sl_2 = np.array(list(dict_U_div_T_2.values()))
        # to include a region where enhanced precision in needed
        Uss_sl = list(Uss_sl[:idx_U_min_val_to_replace])+list(Uss_sl_2)+list(Uss_sl[idx_U_max_val_to_replace+1:])
        Tss_sl = list(Tss_sl[:idx_U_min_val_to_replace])+list(Tss_sl_2)+list(Tss_sl[idx_U_max_val_to_replace+1:])
        hf.close()
    # NCA sl
    # filename_div_sl_NCA = "./cpp_tests/div/div_1D_U_2.000000_1.000000_7.000000_beta_1.000000_0.250000_50.000000_Ntau_4096_Nk_51_NCA_single_ladder_sum.hdf5"
    # arr_Ts = np.arange(1.0,20.0,0.25)
    # hf = h5py.File(filename_div_sl_NCA,"r")
    # list_keys = list(hf.keys())
    # # extracting the Us from the keys
    # Us_NCA_sl = []
    # dict_U_div_T_NCA_sl = {}
    # for i,key in enumerate(list_keys):
    #     Us_NCA_sl.append(float(findall(r"(?<=U_)(\d*\.\d+|\d+)",key)[0]))
    #     U_slice_for_betas = list(hf.get(key))
    #     idx_closest, val_closest = get_pts_close_to_num(U_slice_for_betas,-1.0)
    #     dict_U_div_T_NCA_sl[Us_NCA_sl[i]] = 1.0/(arr_Ts[idx_closest])
    
    # Uss_sl_NCA = list(dict_U_div_T_NCA_sl.keys())
    # Tss_sl_NCA = list(dict_U_div_T_NCA_sl.values())
    # hf.close()
    # il
    filename_div_il = "./cpp_tests/div/div_1D_U_1.000000_1.000000_10.000000_beta_1.000000_1.000000_200.000000_Ntau_256_Nk_51_infinite_ladder_sum.hdf5"
    betas = findall(r"beta_(\d*\.\d+|\d+)_(\d*\.\d+|\d+)_(\d*\.\d+|\d+)",filename_div_il)[0]
    beta_init = float(betas[0]); beta_step = float(betas[1]); beta_max = float(betas[2])
    beta_arr = np.arange(beta_init,beta_max+beta_step,step=beta_step)

    hf = h5py.File(filename_div_il,"r")
    list_keys = list(hf.keys())
    # extracting the Us from the keys
    Uss_il = []
    dict_U_div_T = {}
    for i,key in enumerate(list_keys):
        Uss_il.append(float(findall(r"(?<=U_)(\d*\.\d+|\d+)",key)[0]))
        U_slice_for_betas = list(hf.get(key))
        idx_closest, val_closest = get_pts_close_to_num(U_slice_for_betas,-1.0)
        
        other_idx=0
        if val_closest[1]>-1.0 and idx_closest!=(len(U_slice_for_betas)-1):
            other_idx=idx_closest+1
        else:
            other_idx=idx_closest-1

        print("il: ",Uss_il[-1]," idx closest: ", idx_closest, " val closest: ", val_closest, " other idx: ", other_idx, " other val: ", U_slice_for_betas[other_idx])
        if np.isclose(-1.0,val_closest[1],atol=3e-1): # making sure it actually crosses -1.0
            dict_U_div_T[Uss_il[i]] = (val_closest[0]+U_slice_for_betas[other_idx][0])/2.0
    
    Uss_il = list(dict_U_div_T.keys())
    Tss_il = list(dict_U_div_T.values())
    hf.close()
    ######################################################################################### <------------------------ delimitation calc

    ######################### plotting ##########################
    ms = 3.5
    # The top panel plot shows the phase boundary
    fig, axs = plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True)
    fig.subplots_adjust(hspace=0.05)
    axs.grid()
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    # ordering to plot with lines
    params_respecting_condiction = sorted(params_respecting_condiction,key=operator.itemgetter(1))
    T_arr = list( map( lambda x: 1.0/x[0], params_respecting_condiction ) )
    U_arr = list( map( lambda x: x[1], params_respecting_condiction ) )

    # if dim_int==0:
    #     axs.set_title(r"AFM-PM phase boundary in $\infty$ dimension")
    # else:
    #     axs.set_title(r"AFM-PM phase boundary in {0:d}D".format(dim_int))
    # axs[0].scatter(U_arr,T_arr,marker='o',c="red",s=10.0)
    axs.plot(U_arr[::],T_arr[::],marker='o',ms=ms,c="red",linewidth=0.5,label="DMFT+IPT")
    with open("DMFT_IPT_PHASE_BOUNDARY.dat","w") as f:
        for i in range(len(U_arr)):
            f.write("{0:.5f}\t\t{1:.5f}\n".format(1.0/T_arr[i],U_arr[i]))
    f.close()
    ################################################################################################# <------------------------ delimitation calc
    axs.plot(Uss_sl,Tss_sl,marker='o',ms=ms,c="black",linewidth=0.5,label="single-ladder")
    with open("IPT_SL_BOUNDARY.dat","w") as f:
        for i in range(len(Uss_sl)):
            f.write("{0:.5f}\t\t{1:.5f}\n".format(1.0/Tss_sl[i],Uss_sl[i]))
    f.close()
    # axs.plot(Uss_il,Tss_il,marker='<',ms=ms,c="blue",linewidth=0.5,label="infinite-ladder")
    with open("IPT_IL_BOUNDARY.dat","w") as f:
        for i in range(len(Uss_il)):
            f.write("{0:.5f}\t\t{1:.5f}\n".format(1.0/Tss_il[i],Uss_il[i]))
    f.close()
    ################################################################################################# <------------------------ delimitation calc
    # axs[0].set_ylabel(r"$T (1/\beta)$")
    axs.set_xlim(left=0.0,right=10.0+1.0)
    # axs[0].set_ylim(bottom=min(T_arr)-0.02,top=max(T_arr)+0.02)
    ################################################################################################# <------------------------ delimitation calc
    axs.set_ylim(bottom=min(Tss_sl)-0.02,top=0.4+0.02)
    ################################################################################################# <------------------------ delimitation calc

    axs.text(0.80,0.05,"AFM",transform=axs.transAxes,size=15,weight=15)
    axs.text(0.60,0.75,"PM",transform=axs.transAxes,size=15,weight=15)
    # Arrows that show the U values probed in vicinity of phase boundary (1D NCA)
    len_arrow_x = 0.015
    if dim_int==1 and "NCA" not in path:
        # axs[0].annotate(r"",xy=(3.0,0.116),xytext=(3.0,0.116+len_arrow_x),arrowprops=dict(arrowstyle="->"))
        # axs[0].annotate(r"",xy=(7.0,0.164),xytext=(7.0,0.164+len_arrow_x),arrowprops=dict(arrowstyle="->"))
        axs.text(1.0, 0.040, r'$\bullet$',c='green', fontsize=8, horizontalalignment='center')
        axs.text(1.0, 0.020, r'$\bullet$',c='green', fontsize=8, horizontalalignment='center')
        axs.text(1.0, 0.0135, r'$\bullet$',c='green', fontsize=8, horizontalalignment='center')
        axs.axvspan(0.9, 1.1, color='grey', alpha=0.5, lw=0)

        axs.text(2.0, 0.055, r'$\bullet$',c='green', fontsize=8, horizontalalignment='center')
        axs.text(2.0, 0.083, r'$\bullet$',c='green', fontsize=8, horizontalalignment='center')
        axs.text(2.0, 0.111, r'$\bullet$',c='green', fontsize=8, horizontalalignment='center')
        axs.axvspan(1.9, 2.1, color='grey', alpha=0.5, lw=0)

        axs.text(3.0, 0.1333, r'$\bullet$',c='green', fontsize=8, horizontalalignment='center')
        axs.text(3.0, 0.1538, r'$\bullet$',c='green', fontsize=8, horizontalalignment='center')
        axs.text(3.0, 0.1818, r'$\bullet$',c='green', fontsize=8, horizontalalignment='center')
        axs.axvspan(2.9, 3.1, color='grey', alpha=0.5, lw=0)
        #axs[0].annotate(r"",xy=(15.0,0.068),xytext=(15.0,0.068+len_arrow_x),arrowprops=dict(arrowstyle="->"))
    elif dim_int==2 and "NCA" not in path:
        axs.annotate(r"",xy=(3.0,0.157),xytext=(3.0,0.157+len_arrow_x),arrowprops=dict(arrowstyle="->"))
        axs.annotate(r"",xy=(7.0,0.295),xytext=(7.0,0.295+len_arrow_x),arrowprops=dict(arrowstyle="->"))
        #axs[0].annotate(r"",xy=(14.0,0.148),xytext=(14.0,0.148+len_arrow_x),arrowprops=dict(arrowstyle="->"))
    elif dim_int==1 and "NCA" in path:
        #axs[0].annotate(r"",xy=(4.0,0.295),xytext=(4.0,0.295+len_arrow_x),arrowprops=dict(arrowstyle="->"))
        axs.annotate(r"",xy=(8.0,0.228),xytext=(8.0,0.228+len_arrow_x),arrowprops=dict(arrowstyle="->"))
        axs.annotate(r"",xy=(14.0,0.143),xytext=(14.0,0.143+len_arrow_x),arrowprops=dict(arrowstyle="->"))
    
    axs.set_xlabel(r"$U$")
    axs.set_ylabel(r"$T (1/\beta)$")
    axs.legend()
    # The lower panel shows the evolution of the magntization as a function of temperature for different values of U
    # axs[1].grid()
    # #axs[1].tick_params(axis='y',left=False)
    # axs[1].set_xlabel(r"$U$")

    # axs[1].yaxis.set_minor_locator(AutoMinorLocator())
    # axs[1].yaxis.set_major_locator(MultipleLocator(0.1))
    # axs[1].xaxis.set_minor_locator(AutoMinorLocator())
    # #axs[1].xaxis.set_major_locator(MultipleLocator(0.025))
    # axs[1].tick_params(axis='both', which='major', direction='out', length=3.5, width=1.0)
    # axs[1].tick_params(axis='y', which='minor', direction='in', length=2.0, width=1.0)

    # # ordering to plot with lines
    # params_respecting_condictionNCA = sorted(params_respecting_condictionNCA,key=operator.itemgetter(1))
    # T_arr_NCA = list( map( lambda x: 1.0/x[0], params_respecting_condictionNCA ) )
    # U_arr_NCA = list( map( lambda x: x[1], params_respecting_condictionNCA ) )
    # axs[1].plot(U_arr_NCA[::],T_arr_NCA[::],marker='d',ms=ms,c="red",linewidth=0.5,label="DMFT+NCA")
    # # axs[1].plot(Uss_sl_NCA,Tss_sl_NCA,marker='o',ms=ms,c="black",linewidth=0.5,label="single-ladder")
    # axs[1].legend()

    fig.set_size_inches(17/2.54,12/2.54)
    if dim_int==0:
        fig.savefig("Figure_phase_diagram_infinite_dimension.pdf")
    else:
        fig.savefig("Figure_phase_diagram_{0:d}".format(dim_int)+"D.pdf")
    
    plt.show()
    
    plt.clf()

    ####################### 2nd fig #######################

    fig2, ax = plt.subplots()
    ax.grid()

    if dim_int==0:
        ax.set_title(r"Magnetization vs T in $\infty$ dimension")
    else:
        ax.set_title(r"Magnetization vs T in {0:d}D".format(dim_int))

    ax.set_ylabel(r"$n_{\uparrow}-n_{\downarrow}$")
    ax.set_xlabel(r"$T (1/\beta)$")
    # ax.set_xlim(0.0,0.3)
    ax.set_ylim(0.0,1.0)

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    
    ax.tick_params(axis='both', which='major', direction='out', length=3.5, width=1.0)
    ax.tick_params(axis='y', which='minor', direction='in', length=2.0, width=1.0)

    U_thres = max(Us)
    color = iter(plt.cm.rainbow(np.linspace(0,1,len(Us)+1)))#
    # extracting U arrays before plotting
    for U in sorted(Us):
        if U<=U_thres:
            magnetization_arr = np.array([m[1][0] for m in list_magnetization_for_Us_as_function_beta if m[0][0]==U],dtype=float)
            T_arr = np.array([1.0/m[0][1] for m in list_magnetization_for_Us_as_function_beta if m[0][0]==U],dtype=float)
            ax.plot(T_arr,magnetization_arr,ms=ms,marker='v',c=next(color),label=r"${0:.1f}$".format(U))
    
    ax.legend()
    my_dpi=128
    fig2.set_size_inches(848/my_dpi,495/my_dpi)
    if dim_int==0:
        fig2.savefig("Figure_magnetization_infinite_dimension.pdf",dpi=my_dpi)
    else:
        fig2.savefig("Figure_magnetization_{0:d}".format(dim_int)+"D.pdf",dpi=my_dpi)
    
    #plt.show()

    


