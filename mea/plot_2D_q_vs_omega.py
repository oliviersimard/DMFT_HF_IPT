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

    range_plot = 0.4 # Range of q values to be plotted
    center_plot = np.pi # Center around which the q values are plotted

    filename = "cpp_tests/bb_1D_U_2.000000_beta_50.000000_Ntau_8192_Nk_500_isjj_1.hdf5.pade_wmax_10.0"

    wmax = float(findall(r"(?<=wmax_)(\d*\.\d+|\d+)",filename)[0])
    Ntau = int(findall(r"(?<=Ntau_)(\d+)",filename)[0])
    beta = float(findall(r"(?<=beta_)(\d*\.\d+|\d+)",filename)[0])
    omega = np.array([-wmax + 2.0*i*wmax/(1000) for i in range(1001)])
    qn_arr = np.array([2.0*n*np.pi/beta for n in range(Ntau-1)])
    tmp_q_omega_re_sigma = np.empty((len(omega),),dtype=float)

    plt.figure(0)
    colors = iter(plt.cm.rainbow(np.linspace(0,1,25)))
    hf = h5py.File(filename,"r")
    for i,q_val in enumerate(hf.keys()):
        q = float(str(q_val).split("_")[-1])
        print("q: ", q)
        tmp_q_omega = np.array(hf.get(q_val))
        tmp_q_omega = np.array(list(map(lambda x: x.imag,tmp_q_omega)))
        for j,om in enumerate(omega):
            if om==0.0:
                tmp_q_omega_re_sigma[j] = get_derivative(tmp_q_omega[j-2],tmp_q_omega[j-1],tmp_q_omega[j+1],tmp_q_omega[j+2],2.0*wmax/1000)
                print("der: ", tmp_q_omega_re_sigma[j])
            else:
                tmp_q_omega_re_sigma[j] = tmp_q_omega[j]/om
        if (center_plot-range_plot) <= q <= (center_plot+range_plot):
            plt.plot(omega,tmp_q_omega_re_sigma,marker="*",c=next(colors),label="%s" % (str(q_val)))
    
    hf.close()
    
    plt.legend()
    plt.show()