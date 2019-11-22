import numpy as np
import getopt, sys
import matplotlib.pyplot as plt
import os
import re

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

try:
    opts, args = getopt.getopt(sys.argv[1:], "hfc:v", ["help", "file=", "c="])
except getopt.GetoptError as err:
    print(err) 
    sys.exit(2)

output = None
verbose = False
print(opts)
for o, a in opts:
    if o == "-v":
        verbose = True
    elif o in ("-h", "--help"):
        sys.exit()
    elif o in ("-f", "--file"):
        assert type(a)==str, "Wrong type of input command line!"
        filename = a
    elif o in ("-c"):
        i_or_r=int(a)
    else:
        assert False, "unhandled option"

getSizeSquare = int(os.popen('head -n1 {0} | grep -o " " | wc -l'.format(filename)).read())
mesh = np.empty(shape=(getSizeSquare,getSizeSquare))
ImOrRe=""
file_ImOrRe=""
if i_or_r==0:
    ImOrRe=r"$\operatorname{Re}$"
    file_ImOrRe="_Re_"
elif i_or_r==1:
    ImOrRe=r"$\operatorname{Im}$"
    file_ImOrRe="_Im_"

with open(filename) as f:
    for k1,line in enumerate(f):
        if k1<getSizeSquare:
            for k2,el in enumerate(line.split(" ")):
                if "(" in el.strip("\n"):
                    print("el: ", el)
                    tup = tuple(float(i) for i in el.strip('()').split(','))
                    mesh[k1,k2] = tup[i_or_r] ## Imaginary part of the susceptibility if 1.
        else:
            break

assert mesh.shape[0]==mesh.shape[1], "The data doesn't have \"square\" dimension"
max_val = mesh.shape[0]-1
imageDir="/Users/simardo/Documents/PhD/HF_cpp/Latex_docs/images/" # Path to save figures produced.

def main():

    opt_val=""
    beg_sv_figs="" # Input to describe saved figures (in case of).
    if "full" not in filename:
        if "Weights" in filename:
            opt_val=ImOrRe+r"$\mathcal{G}^{\sigma}(\tilde{k})\mathcal{G}^{\sigma}(\tilde{k}-q)\mathcal{G}^{\bar{\sigma}}(\bar{k}+q)\mathcal{G}^{\bar{\sigma}}(\bar{k})$"
            beg_sv_figs="Weights"
        elif "Bubble" in filename:
            opt_val=ImOrRe+r"$U\mathcal{G}^{\sigma}(\tilde{k}+q)\mathcal{G}^{\sigma}(\bar{k}+q)$"
            beg_sv_figs="Bubble"
        else:
            opt_val=ImOrRe+r"$\Gamma^{\sigma\bar{\sigma}}(\tilde{k},\bar{k},q)$"
            beg_sv_figs="Gamma"
    else:
        if "Weights" in filename:
            opt_val=ImOrRe+r"$\mathcal{G}^{\sigma}(\tilde{k})\mathcal{G}^{\sigma}(\tilde{k}-q)\mathcal{G}^{\bar{\sigma}}(\bar{k}+q)\mathcal{G}^{\bar{\sigma}}(\bar{k})$"
            beg_sv_figs="Weights"
        elif "Bubble" in filename:
            opt_val=ImOrRe+r"$U\chi_f^{\sigma\bar{\sigma}}(\tilde{k},\bar{k},q)$"
            beg_sv_figs="Bubble"
        else:
            opt_val=ImOrRe+r"$\frac{U}{1-U\chi_f^{\sigma\bar{\sigma}}(\tilde{k},\bar{k},q)}$"
            beg_sv_figs="Gamma"

    dim_val="" # Input to describe saved figures (in case of).
    if "_1D_" in filename:
        dim_val="1D"
    elif "_2D_" in filename:
        dim_val="2D"

    index_beta = filename.find("beta")
    index_Nk = filename.find("Nk")
    index_Ntau = filename.find("N_tau")
    index_U = [m.start() for m in re.finditer("U",filename)][-1]

    u = float(filename[index_U:].split("_")[1]) 
    print("u: ", u, "\n")
    beta = float(filename[index_beta:].split("_")[1])
    print("beta: ", beta, "\n")

    end_of_file = filename[index_Nk:] # Input to describe saved figures (in case of).
    if index_Nk<index_Ntau:
        end_of_file = filename[index_Nk:].rstrip(".dat")
    else:
        end_of_file = filename[index_Ntau:].rstrip(".dat")

    if "_1D_" in filename:
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        im = ax.imshow(mesh, aspect="auto", origin='lower', cmap=plt.get_cmap('magma'))
        if "full" in filename:
            ax.set_title(opt_val+r" (full ladder diagrams) for $U$={0:2.2f}, $\beta=${1:2.1f} (1D)".format(u,beta))
        else:
            ax.set_title(opt_val+r" (ladder diagrams) for $U$={0:2.2f}, $\beta=${1:2.1f} (1D)".format(u,beta))
        ax.set_xlabel(r"$\bar{k}$", fontsize=25, labelpad=10.0)
        ax.set_ylabel(r"$\tilde{k}$", fontsize=25, labelpad=10.0)
        ax.set_xticks(np.linspace(0,max_val,10),minor=False)
        ax.set_yticks(np.linspace(0,max_val,10),minor=False)
        ax.xaxis.set_tick_params(which='major',direction='inout',length=6)
        ax.yaxis.set_tick_params(which='major',direction='inout',length=6)
        ax.xaxis.set_ticklabels([round(x,2) for x in np.linspace(-np.pi,np.pi,10)], color='black', fontsize=10) ## Mapping the data onto Brillouin zone
        ax.yaxis.set_ticklabels([round(x,2) for x in np.linspace(-np.pi,np.pi,10)], color='black', fontsize=10)
        ax.set_xlim(left=0,right=max_val)
        ax.set_ylim(bottom=0,top=max_val)
        fig.colorbar(im, ax=ax)
    elif "_2D_" in filename:
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        im = ax.imshow(mesh, aspect="auto", origin='lower', cmap=plt.get_cmap('magma'))
        ax.set_title(opt_val+r" (ladder diagrams) for $U$={0:2.2f}, $\beta=${1:2.1f} (2D)".format(u,beta))
        ax.set_xlabel(r"$\bar{k}_x-\tilde{k}_x$", fontsize=25, labelpad=10.0)
        ax.set_ylabel(r"$\bar{k}_y-\tilde{k}_y$", fontsize=25, labelpad=10.0)
        ax.set_xticks(np.linspace(0,max_val,10),minor=False)
        ax.set_yticks(np.linspace(0,max_val,10),minor=False)
        ax.xaxis.set_tick_params(which='major',direction='inout',length=6)
        ax.yaxis.set_tick_params(which='major',direction='inout',length=6)
        ax.xaxis.set_ticklabels([round(x,2) for x in np.linspace(-np.pi,np.pi,10)], color='black', fontsize=10) ## Mapping the data onto Brillouin zone
        ax.yaxis.set_ticklabels([round(x,2) for x in np.linspace(-np.pi,np.pi,10)], color='black', fontsize=10)
        ax.set_xlim(left=0,right=max_val)
        ax.set_ylim(bottom=0,top=max_val)
        fig.colorbar(im, ax=ax)
    else:
        raise(ValueError("Check the input filename: problem with the dimension...only 1D or 2D possible."))

    #plt.gcf().set_size_inches(12,12)
    #plt.savefig(imageDir+beg_sv_figs+"_U_{0:3.2f}_beta_{1:3.2f}_".format(u,beta)+end_of_file+file_ImOrRe+dim_val+".pdf")
    plt.show()

    return None

if __name__ == "__main__":
    main()
