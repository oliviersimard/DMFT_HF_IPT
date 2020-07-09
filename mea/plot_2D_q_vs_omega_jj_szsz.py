import numpy as np
import matplotlib.pyplot as plt
import h5py
from sys import exit
from re import findall

def get_derivative(p1 : float, p2 : float, p3 : float, p4 : float, delta_x : float) -> float:
    """p1, p2, p3 and p4 are the neighbouring points to the target points at which the derivative is looked for. delta_x is supposed to
    be constant and represents the step between two images of the function.
    """
    der = ( 1.0/12.0*p1 - 2.0/3.0*p2 + 2.0/3.0*p3 - 1.0/12.0*p4 ) / delta_x
    return der

if __name__=="__main__":

    range_plot = 1.05*np.pi # Range of q values to be plotted
    center_plot = 0.0 # Center around which the q values are plotted

    filename = "cpp_tests/bb_1D_U_8.000000_beta_4.000000_Ntau_4096_Nk_401_NCA.hdf5.pade_wmax_20.0"

    wmax = float(findall(r"(?<=wmax_)(\d*\.\d+|\d+)",filename)[0])
    Ntau = int(findall(r"(?<=Ntau_)(\d+)",filename)[0])
    beta = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename)[0])
    omega = np.array([-wmax + 2.0*i*wmax/(1000) for i in range(1001)])
    qn_arr = np.array([2.0*n*np.pi/beta for n in range(Ntau-1)])
    tmp_q_omega_re_sigma_jj = np.empty((len(omega),),dtype=float)
    tmp_q_omega_re_sigma_szsz = np.empty((len(omega),),dtype=float)

    fig, axs = plt.subplots(nrows=1,ncols=2,sharey=False)
    fig.subplots_adjust(wspace=0.25)
    hf = h5py.File(filename,"r")
    colors = iter(plt.cm.rainbow(np.linspace(0,1,len(hf.keys())+1)))
    for i,q_val in enumerate(hf.keys()):
        q = float(str(q_val).split("_")[-1])
        print("q: ", q)
        if "_1D_" in filename:
            # jj
            tmp_q_omega_jj = np.array(hf.get(q_val).get('jj'))
            tmp_q_omega_jj = np.array(list(map(lambda x: x.imag,tmp_q_omega_jj)))
            for j,om in enumerate(omega):
                if om==0.0:
                    tmp_q_omega_re_sigma_jj[j] = get_derivative(tmp_q_omega_jj[j-2],tmp_q_omega_jj[j-1],tmp_q_omega_jj[j+1],tmp_q_omega_jj[j+2],2.0*wmax/1000)
                    print("der: ", tmp_q_omega_re_sigma_jj[j])
                else:
                    tmp_q_omega_re_sigma_jj[j] = tmp_q_omega_jj[j]/om
            
            # szsz
            tmp_q_omega_szsz = np.array(hf.get(q_val).get('szsz'))
            tmp_q_omega_szsz = np.array(list(map(lambda x: x.imag,tmp_q_omega_szsz)))
            for j,om in enumerate(omega):
                if om==0.0:
                    tmp_q_omega_re_sigma_szsz[j] = get_derivative(tmp_q_omega_szsz[j-2],tmp_q_omega_szsz[j-1],tmp_q_omega_szsz[j+1],tmp_q_omega_szsz[j+2],2.0*wmax/1000)
                    print("der: ", tmp_q_omega_re_sigma_szsz[j])
                else:
                    tmp_q_omega_re_sigma_szsz[j] = tmp_q_omega_szsz[j]/om
            ## plotting
            axs[0].grid(); axs[1].grid() 
            axs[0].set_title(r"Paramagnetic $\langle jj\rangle$ susceptibility bubble $1D$ dimension")
            axs[1].set_title(r"Paramagnetic $\langle S_zS_z\rangle$ susceptibility bubble $1D$ dimension")
            axs[0].set_xlabel(r"$\omega$"); axs[1].set_xlabel(r"$\omega$")
            axs[0].set_ylabel(r"$\langle j_xj_x\rangle$"); axs[1].set_ylabel(r"$\langle S_zS_z\rangle$")
            cc = next(colors)
            if (center_plot-range_plot) <= q <= (center_plot+range_plot):
                axs[0].plot(omega,tmp_q_omega_re_sigma_jj,marker="*",c=cc,label="%s" % (str(q_val)))
                axs[1].plot(omega,tmp_q_omega_re_sigma_szsz,marker="*",c=cc)
        elif "_2D_" in filename:
            qx = float(str(q_val).split("_")[1])
            qy = float(str(q_val).split("_")[3])
            print("qx: ", qx, " qy: ", qy)
            # jj
            tmp_q_omega_jj = np.array(hf.get(q_val).get('jj'))
            tmp_q_omega_jj = np.array(list(map(lambda x: x.imag,tmp_q_omega_jj)))
            for j,om in enumerate(omega):
                if om==0.0:
                    tmp_q_omega_re_sigma_jj[j] = get_derivative(tmp_q_omega_jj[j-2],tmp_q_omega_jj[j-1],tmp_q_omega_jj[j+1],tmp_q_omega_jj[j+2],2.0*wmax/1000)
                    print("der jj: ", tmp_q_omega_re_sigma_jj[j])
                else:
                    tmp_q_omega_re_sigma_jj[j] = tmp_q_omega_jj[j]/om
            
            # szsz
            tmp_q_omega_szsz = np.array(hf.get(q_val).get('szsz'))
            tmp_q_omega_szsz = np.array(list(map(lambda x: x.imag,tmp_q_omega_szsz)))
            for j,om in enumerate(omega):
                if om==0.0:
                    tmp_q_omega_re_sigma_szsz[j] = get_derivative(tmp_q_omega_szsz[j-2],tmp_q_omega_szsz[j-1],tmp_q_omega_szsz[j+1],tmp_q_omega_szsz[j+2],2.0*wmax/1000)
                    print("der szsz: ", tmp_q_omega_re_sigma_szsz[j])
                else:
                    tmp_q_omega_re_sigma_szsz[j] = tmp_q_omega_szsz[j]/om
            
            ## plotting
            axs[0].grid(); axs[1].grid() 
            axs[0].set_title(r"Paramagnetic $\langle jj\rangle$ susceptibility bubble $2D$ dimension")
            axs[1].set_title(r"Paramagnetic $\langle S_zS_z\rangle$ susceptibility bubble $2D$ dimension")
            axs[0].set_xlabel(r"$\omega$"); axs[1].set_xlabel(r"$\omega$")
            axs[0].set_ylabel(r"$\langle j_xj_x\rangle$"); axs[1].set_ylabel(r"$\langle S_zS_z\rangle$")
            cc = next(colors)
            if (center_plot-range_plot) <= q <= (center_plot+range_plot):
                axs[0].plot(omega,tmp_q_omega_re_sigma_jj,marker="*",c=cc,label="%s" % (str(q_val)))
                axs[1].plot(omega,tmp_q_omega_re_sigma_szsz,marker="*",c=cc)
    
    hf.close()
    
    fig.legend()
    plt.show()
    fig.clf()
