import numpy as np
import sys, os
import getopt
import scipy.integrate as sci
import matplotlib.pyplot as plt
from glob import glob
import re

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage[charter]{mathdesign}\usepackage{amsmath}"]

ms=3.5 # Marker size uniformized

def get_option():
	try:
		name = ''
		wmin = -6
		wmax = 6
		nbrw = -1
		option, test = getopt.getopt(sys.argv[1:], 'f:l:m:n:')
	except getopt.GetoptError:
		print('options for analytical continuation:\n-f <inputFile>\n-l <lowest real frequency>\n-m <largest real frequency>\n-n <number of frequencies between wmin and wmax')
		sys.exit()
	for opt, arg in option:
		if opt == '-f':
			name = arg
		elif opt == '-l':
			wmin = float(arg)
		elif opt == '-m':
			wmax = float(arg)
		elif opt == '-n':
			nbrw = int(arg)
	if nbrw <0:
		print('number of real frequency not found. Standard value taken: nbr=100')
		number =100	
	return name, wmin, wmax, nbrw

def get_filenames() -> tuple:
	name, wmin, wmax, nbr_w = get_option()
	N_k=int(re.findall(r"(?<=Nk_)(\d+)",name)[0])
	path_to_name_arr = name.split("/")[0:-1] 
	name = name.split("/")[-1] # If the files are not in the current directory.
	if path_to_name_arr != []:
		path_to_name = "/".join(path_to_name_arr)
		path_to_name=path_to_name+"/"
	U=float(re.findall(r"(?<=U_)(\d*\.\d+|\d+)",name)[0])
	beta=float(re.findall(r"(?<=beta_)(\d*\.\d+|\d+)",name)[0])
	N_tau=int(re.findall(r"(?<=N_tau_)(\d+)",name)[0])
	# iqn=float(re.findall(r"(?<=iqn_)(\d*\.\d+|\d+)",name)[-1])
	m=re.findall(r"\d*\.\d+\.dat$",name) # Need to remove last digits before to perform glob to gather all the files.
	name=name.replace(m[0],'')
	filenames=glob(path_to_name+'*'+name+'*')
	filenames=[filename for filename in filenames if "analytic" not in filename]
	iqns=[]
	for filename in filenames:
		m=float(*re.findall(r"\d*\.\d+(?=\.dat$)",filename))
		iqns.append(m)
	file_vs_iqn = list(sorted(zip(filenames,iqns),key=lambda x: x[1]))

	return file_vs_iqn, wmin, wmax, nbr_w, U, beta, N_tau, N_k, path_to_name_arr

def file_generator(list_name: list):
	for el in list_name:
		yield el

def get_corresponding_index(list_name: list, element1: float, element2: float) -> tuple:
	""" Finds in a list the index of the number which is closest to element input """
	diff_of_vals1 = list( map( lambda x: np.abs(x - element1), list_name ) )
	diff_of_vals2 = list( map( lambda x: np.abs(x - element2), list_name ) )
	index1 = diff_of_vals1.index(min(diff_of_vals1))
	index2 = diff_of_vals2.index(min(diff_of_vals2))
	return (index1,index2)

def checkEqual(iterator):
	""" Checks if all the elements in an iterable object are equal. """
	iterator = iter(iterator)
	try:
		first = next(iterator)
	except StopIteration:
		return True
	return all(first == rest for rest in iterator)

file_vs_iqn, wmin, wmax, nbr_w, U, beta, N_tau, N_k, path_name = get_filenames()
path_name="/".join(path_name[:-1])+"/"

print(U, beta, N_tau, path_name)

# Next, we have to load the files for each bosonic Matsubara frequency. We switch over to a dictionary.
dict_iqn_content_file={}
# Getting size of square matrix
getSizeSquare = int(os.popen('head -n1 {0} | grep -o " " | wc -l'.format(file_vs_iqn[0][0])).read())
list_ind_where_max_forall_iqns=[]
for filename, iqn in file_vs_iqn:
	mesh = np.empty(shape=(getSizeSquare,getSizeSquare),dtype=complex)
	# Transforming the complex numbers in files into tuples.
	max_val=0.0
	ind_max_val=tuple()
	with open(filename) as f:
		for k1,line in enumerate(f):
			if k1<getSizeSquare:
				for k2,el in enumerate(line.split(" ")):
					if "(" in el.strip("\n"):
						tup = tuple(float(i) for i in el.strip('()').split(','))
						val = tup[0]+1.0j*tup[1]
						mesh[k1,k2] = val ## Imaginary part of the susceptibility if 1.
						if np.abs(val)>max_val:
							max_val=val
							ind_max_val=(k1,k2)
			else:
				break
	list_ind_where_max_forall_iqns.append(ind_max_val)
	dict_iqn_content_file[iqn]=mesh

arr_k = (np.linspace(-np.pi,np.pi,getSizeSquare,dtype=float)).tolist()
print("The k-space array is ", arr_k, "\n")
print("The indexes where the response is largest at iqn=0.0: ", list_ind_where_max_forall_iqns[0], 
					" and same for all iqn: ", checkEqual(list_ind_where_max_forall_iqns),"\n")
#tup_indices = get_corresponding_index(arr_k,np.pi/1.92,-np.pi/1.92) # Might just need the last tuple element in fact...
tup_indices = list_ind_where_max_forall_iqns[0]
sus_values = []
iqn_list = []
# Keeping values of susceptibilities corresponding to k-space point(s) we aim for.
for item in dict_iqn_content_file.items():
	iqn_list.append(item[0])
	sus_values.append( item[1][ tup_indices[1], tup_indices[1] ] )

sus_values=np.array(sus_values,dtype=complex)
assert len(sus_values)==len(iqn_list), "There has to be as many susceptibility values as there are bosonic Matsubara frequencies."
# This array contains all the tuples of indices we want to plot the susceptibility for.
tup_indices_arr = [tup_indices]
omega = [ wmin+i*(wmax-wmin)/(nbr_w-1.) for i in range(nbr_w) ]
dict_names={}
for tup_ind in tup_indices_arr: 
	omega_n = np.asarray(iqn_list)
	Rf_n = sus_values[:].real
	If_n = sus_values[:].imag
	
	eta = 0.001

	assert len(omega_n)==len(Rf_n)==len(If_n), "The lengths of Matsubara array, Re F(iwn) and Im F(iwn) have to be equal."

	f_n = Rf_n + 1j*If_n
	N = len(omega_n)-7 # Can tweek this number and see its effect
	g = np.zeros((N, N), dtype = complex)
	omega_n = omega_n[0:N]

	f_n = f_n[0:N]
	g[0,:] = f_n[:]
	g[1,:] = (f_n[0] - f_n)/((1j*omega_n -1j*omega_n[0])*f_n)
	for k in range(2, N):
		g[k, :] = (g[k-1, k-1] - g[k-1, :])/((1j*omega_n - 1j*omega_n[k-1])*g[k-1, :])

	a = np.diagonal(g)
	
	A = np.zeros((N, ), dtype = complex)
	B = np.zeros((N, ), dtype = complex)
	P = np.zeros((N, ), dtype = complex)
	fw = []

	for k  in range (len(omega)):
		z = omega[k] +1j*eta
		P[0] = a[0]
		A[1] = 1.
		B[1]= 1.+a[1]*(z - 1j*omega_n[0])
		P[1] = P[0]*A[1]/B[1]
		for c in range(2, N):
			A[c] = 1. + a[c]*(z -1j*omega_n[c-1])/A[c-1]
			B[c] = 1. + a[c]*(z - 1j*omega_n[c-1])/B[c-1]
			P[c] = P[c-1]*A[c]/B[c]
		fw.append(P[-1])

	dict_names[tup_ind]=fw
	try:
		f = open(path_name+"analytic_continuations/"+"analytic_continuation_test1.dat", "w") ## Give it a relevant name 
		for i in range(len(fw)):
			f.write(str(omega[i])+"	"+str(fw[i].real)+"	"+str(fw[i].imag)+"\n")

		f.close()
	except:
		print("Please create directory \"{0}analytic_continuations\"...".format(path_name))

# Plotting
list_tup_indices = list(dict_names.keys())
tot_axs=2
fig, axs = plt.subplots(tot_axs, 1, sharex=True)
fig.subplots_adjust(hspace=0.03)
i=0
while i<tot_axs:
	axs[i].grid()
	axs[i].set_ylabel(r"a. u.")
	if i == 0:
		#axs[i].set_title(r"$2^{\text{nd}}$-order IPT-D results for "+r"$\beta={0}$, $U={1}$".format(beta,U))
		axs[i].set_title(r"Optical conductivity for "+r"$\beta={0}$, $U={1}$, ".format(beta,U)+r"$N_{\tau}$"+r"$={0}$, and $N_k={1}$".format(N_tau,N_k)+r" (HF)")
	if i != tot_axs-1:
		axs[i].tick_params(axis='x',bottom=False)
	else:
		axs[i].set_xlabel(r"Energy $\omega$")
	
	if i == 0:
		real_values=list( map( lambda x: x.real, dict_names[list_tup_indices[0]] ) )
		label_real=r"$\operatorname{Re}\chi_{spsp}(\omega)$"
		axs[i].plot(omega,real_values,marker='s',ms=ms,color='green',label=label_real)
	elif i == 1:
		imag_values=list( map( lambda x: x.imag, dict_names[list_tup_indices[0]] ) )
		label_imag=r"$\operatorname{Im}\chi_{spsp}(\omega)$"
		axs[i].plot(omega,imag_values,marker='*',ms=ms,color='blue',label=label_imag)

	axs[i].legend()
	i+=1

plt.show()

