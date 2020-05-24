import os
import numpy as np
import matplotlib.pyplot as plt
from sys import exit
from re import findall
from shutil import copy2
from glob import glob
from time import sleep
from optparse import OptionParser

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage[charter]{mathdesign}\usepackage{amsmath}"]

# maximum number of DMFT iterations
MAXVAL = 100-1

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

def AFM_phase_boundary_condition(filename : str, delimiter='\t\t', col_num=(1,2), tol=1e-2) -> bool:
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

    return bool_cond


def get_according_to_condition(path : str, funct) -> list:
    """
        The function funct passed as parameter has to return a boolean value to determine if it has passed. It has to 
        have filename as parameter only as well.
    """
    dict_to_store_passing_cond_files = {}
    Us = []
    betas = []
    # navigating through the directory containing the data
    for root, dirs, files in os.walk(path):
        path = root.split(os.sep) # os.sep is equivalent to "/"
        print( (len(path)-1)*'---', os.path.basename(root) )
        if files != []:
            largest_int, file_largest_int = extract_largest_int(files,"tau")
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
                    cond = funct(fconcat)
                    if cond and dict_to_store_passing_cond_files[key]._largest_num_iter<MAXVAL:
                        if beta>max_beta:
                            max_beta=beta
            print("max_beta: ", max_beta)
        put_element_in_list_if_different(params_respecting_condiction,(max_beta,U))

    return params_respecting_condiction


if __name__=="__main__":

    parser = OptionParser()

    parser.add_option("--data", dest="path", default=u"./data_test_IPT")
    (options, args) = parser.parse_args()

    path = str(options.path)

    # if the files are all scattered in the target directory, the files are rearranged. Otherwise, it is passed.
    rearrange_files_into_dir(path)

    params_respecting_condiction = get_according_to_condition(path,AFM_phase_boundary_condition)

    fig, ax = plt.subplots(nrows=1,ncols=1)
    ax.grid()

    T_arr = list( map( lambda x: 1.0/x[0], params_respecting_condiction ) )
    U_arr = list( map( lambda x: x[1], params_respecting_condiction ) )

    ax.set_title(r"AFM-PM phase boundary in 1D")
    ax.scatter(U_arr,T_arr,marker='v',c="red",s=10.0)
    ax.set_xlabel(r"$U$")
    ax.set_ylabel(r"$T (1/\beta)$")

    ax.text(0.25,0.15,"AFM",transform=ax.transAxes,size=20,weight=20)
    ax.text(0.55,0.75,"PM",transform=ax.transAxes,size=20,weight=20)

    plt.show()

    


