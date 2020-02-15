import numpy as np

import math as m

from optparse import OptionParser



def pade(w, iw, G):

    """Calculates the Pade approximation of the analytical continuation of the

    Matsubara Green's function in frequency space on the real axis.


    :omega: Real frequencies on which to calculate G(omega)

    :iomega: The imaginary frequencies on which G(i omega) is given.

    :G: The values of G(i omega).

    :returns: The values of G(omega).


    """

    N = len(iw)

    M = len(w)


    g = np.array(G, dtype=np.clongdouble)

    a = np.array(G, dtype=np.clongdouble)

    A0 = np.zeros(M, dtype=np.clongdouble)  # A_0 = 0

    A1 = a[0]*np.ones(M, dtype=np.clongdouble)  # A_1 = a_1

    B0 = np.ones(M, dtype=np.clongdouble)  # B_0 = 1

    B1 = np.ones(M, dtype=np.clongdouble)  # B_1 = 1


    for j in range(1, N):

        g[j:] = (a[j-1] - g[j:]) / ((iw[j:]-iw[j-1])*g[j:])

        a[j:] = g[j:]

        A = A1 + (w - iw[j-1])*a[j]*A0

        B = B1 + (w - iw[j-1])*a[j]*B0

        A1, A0 = A, A1

        B1, B0 = B, B1


    return A/B



parser = OptionParser()


parser.add_option("--data", dest="data", default='data.in')

parser.add_option("--Nwm" , dest="Nwm" , default="None")

parser.add_option("--wmax", dest="wmax", default=10)

parser.add_option("--Nwr" , dest="Nwr" , default=1000)


(options, args) = parser.parse_args()



Nreal = int(options.Nwr)

wmax = float(options.wmax)


w = np.linspace(0, wmax, Nreal)


iw, ReW  = np.genfromtxt(options.data, dtype=np.float64, usecols=(0,1), unpack=True)

if options.Nwm != "None":

    Nmats = int(options.Nwm)

    ReW=ReW[:Nmats].copy()

    iw=iw[:Nmats].copy()

print(len(ReW))

ImW = np.zeros(len(ReW))


Sp = pade(w, 1j*iw, ReW+1j*ImW)


np.savetxt(options.data+'.pade',np.c_[w, Sp.real, Sp.imag])