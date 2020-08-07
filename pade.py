import numpy as np
#import pylab as plt
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
		is_Green_component = True # The default value means that the Green's function components are analytically continued by default
		option, test = getopt.getopt(sys.argv[1:], 'f:l:m:n:i:')
	except getopt.GetoptError:
		print('options for analytical continuation:\n-f <inputFile>\n-l <lowest real frequency>\n-m <largest real frequency>\n-n <number of frequencies between wmin and wmax')
		sys.exit()
	for opt, arg in option:
		if opt =='-f':
			name = arg
		elif opt =='-l':
			wmin = float(arg)
		elif opt == '-m':
			wmax = float(arg)
		elif opt == '-n':
			nbrw = int(arg)
		elif opt == '-i':
			is_Green_component = bool(int(arg))
	if nbrw <0:
		print('number of real frequency not found. Standard value taken: nbr=100')
		number =100	
	return name, wmin, wmax, nbrw, is_Green_component

def get_filenames() -> tuple:
	name, wmin, wmax, nbr_w, is_Green_component = get_option()
	path_to_name_arr = name.split("/")[0:-1] 
	name = name.split("/")[-1] # If the files are not in the current directory.
	if path_to_name_arr != []:
		path_to_name = "/".join(path_to_name_arr)
		path_to_name = path_to_name+"/"
	
	good_filenames=[] # Can't remove filenames while looping over the same list.
	if is_Green_component:
		U=float(re.findall(r"(?<=U_)(\d*\.\d+|\d+)",name)[0])
		beta=float(re.findall(r"(?<=beta_)(\d*\.\d+|\d+)",name)[0])
		#nn=float(re.findall(r"(?<=n_)(\d*\.\d+|\d+)",name)[-1])
		nn=0.5
		m=re.findall(r"^(?!analytic)(^[a-zA-Z]+_[a-zA-Z]+)",name)
		name=name.replace(m[0],'')
		m=re.findall(r"\d+\.dat$",name)
		name=name.replace(m[0],'')
		filenames=glob(path_to_name+'*'+name+'*')
		filenames=[filename for filename in filenames if "analytic" not in filename]
		largest_int=0
		for filename in filenames:
			m=re.findall(r"\d+(?=\.dat$)",filename)
			if largest_int==0:
				largest_int=int(m[0])
			elif int(m[0])>largest_int:
				largest_int=int(m[0])
		for filename in filenames:
			if str(largest_int)+".dat" in filename:
				good_filenames.append(filename)
	else:
		nn=0.0 # dummy variables
		beta=0.0; U=0.0
		U=float(re.findall(r"(?<=U_)(\d*\.\d+|\d+)",name)[0])
		beta=float(re.findall(r"(?<=beta_)(\d*\.\d+|\d+)",name)[0])
		good_filenames.append(path_to_name+name)
		print(good_filenames)

	return good_filenames, wmin, wmax, nbr_w, U, beta, nn, is_Green_component

def file_generator(list_name: list):
	for el in list_name:
		yield el

filenames, wmin, wmax, nbr_w, U, beta, nn, is_Green_component = get_filenames()
print(U, beta, nn)
omega = [ wmin+i*(wmax-wmin)/(nbr_w-1.) for i in range(nbr_w) ]
dict_names={}
for name in filenames:
	#print("name: ", name)
	R = np.genfromtxt(name,skip_header=1,dtype=float,usecols=(0,1,2),delimiter='\t\t') # Watch out for the nan at the end of the file. Have to add the proper delimiter '\t\t' at the end.
	omega_n = R[:,0]
	Rf_n = R[:,1]
	If_n = R[:,2]
	#Keeping only the positive Matsubara frequencies.
	Rf_n = np.array([el[1] for el in zip(omega_n,Rf_n) if el[0]>=0.0],dtype=float)
	If_n = np.array([el[1] for el in zip(omega_n,If_n) if el[0]>=0.0],dtype=float)
	omega_n = omega_n[omega_n>=0.0] ##<------------------------------------------------------------- WATCH OUT because important to keep
	eta = 0.001

	assert len(omega_n)==len(Rf_n)==len(If_n), "The lengths of Matsubara array, Re F(iwn) and Im F(iwn) have to be equal."

	f_n = Rf_n + 1j*If_n
	N = len(omega_n)-70 # Substracting the last Matsubara frequencies
	g = np.zeros((N, N), dtype = np.clongdouble)
	omega_n = omega_n[0:N]

	f_n = f_n[0:N]
	g[0,:] = f_n[:]
	g[1,:] = (f_n[0] - f_n)/((1j*omega_n -1j*omega_n[0])*f_n)
	for k in range(2, N):
		g[k, :] = (g[k-1, k-1] - g[k-1, :])/((1j*omega_n - 1j*omega_n[k-1])*g[k-1, :])

	a = np.diagonal(g)

	A = np.zeros((N, ), dtype = np.clongdouble)
	B = np.zeros((N, ), dtype = np.clongdouble)
	P = np.zeros((N, ), dtype = np.clongdouble)
	fw = []
	# kern_ikn_w = [] ; kern_tau_pos_w = []
	# kern_ikn = [] ; kern_tau_pos = []

	for k in range(len(omega)):
		z = omega[k] + 1j*eta
		P[0] = a[0]
		A[1] = 1.
		B[1]= 1.+a[1]*(z - 1j*omega_n[0])
		P[1] = P[0]*A[1]/B[1]
		for c in range(2, N):
			A[c] = 1. + a[c]*(z - 1j*omega_n[c-1])/A[c-1]
			B[c] = 1. + a[c]*(z - 1j*omega_n[c-1])/B[c-1]
			P[c] = P[c-1]*A[c]/B[c]
		#print P[-1][0]
		fw.append(P[-1])
	dict_names[name]=fw

	try:
		path_to_name=""
		if is_Green_component:
			path_to_name_arr = name.split("/")[0:-1]
		else:
			path_to_name_arr = name.split("/")[0:-2]
		name = name.split("/")[-1] # If the files are not in the current directory.
		if path_to_name_arr != []:
			path_to_name = "/".join(path_to_name_arr)
			path_to_name=path_to_name+"/"

		print(path_to_name+"analytic_continuations/"+"analytic_continuation_"+name)
		f = open(path_to_name+"analytic_continuations/"+"analytic_continuation_"+name, "w")
		for i in range(len(fw)):
			f.write(str(omega[i])+"\t"+str(fw[i].real)+"\t"+str(fw[i].imag)+"\n")

		f.close()
	except:
		#print("Please create directory \"{0}analytic_continuations\"...".format(path_to_name)) <----- Uncomment after testing
		pass

	if "Green_loc" in name:
		# Spectral weight sum rule in order to respect anticommutation relations. Should be normalized to one.
		Spec_func = [fw[i].imag for i in range(len(fw))]
		beta_array = np.linspace(0,beta,201)
		print("norm of the spectral function = ", -(1./np.pi)*sci.simps(Spec_func, omega))
		
# Plotting
keys_list=list(dict_names.keys()) # Getting the keys of the dictionnary to properly plot the components.
list_gen=file_generator(keys_list)

fig, axs = plt.subplots(len(keys_list), 1, sharex=True)
fig.subplots_adjust(hspace=0.03)
i=0
while i<len(keys_list):
	str_filename=next(list_gen)
	if len(keys_list)==1: # This part is meant for the non-interacting susceptibility for now
		axs.grid()
		axs.set_xlabel(r"Energy $\omega$")
		axs.set_ylabel(r"a. u.")
		if is_Green_component:
			axs.set_title(r"$2^{\text{nd}}$-order IPT-D results for "+r"$\beta={0}$, $U={1}$".format(beta,U))
		else:
			axs.set_title(r"$\mathcal{A}_{\text{loc}}(\omega)$ for "+r"$\beta={0}, U={1}$".format(beta,U)) # Have to change beta by hand 
	else:
		axs[i].grid()
		axs[i].set_ylabel(r"a. u.")
		if i == 0:
			if is_Green_component:
				axs[i].set_title(r"$2^{\text{nd}}$-order IPT-D results for "+r"$\beta={0}$, $U={1}$ and $n={2}$".format(beta,U,nn))
			else:
				axs[i].set_title(r"Test results") # Have to change beta by hand
		if i != len(keys_list)-1:
			axs[i].tick_params(axis='x',bottom=False)
		else:
			axs[i].set_xlabel(r"Energy $\omega$")
	
	if "Green_loc" in str_filename:
		real_values=list( map( lambda x: x.real, dict_names[str_filename] ) )
		if is_Green_component:
			label_imag=r"$-\frac{1}{\pi}\operatorname{Im}G_{\text{loc}}$"
			label_real=r"$\operatorname{Re}G_{\text{loc}}$"
			imag_values=list( map( lambda x: -1.0/np.pi*x.imag, dict_names[str_filename] ) )
		else:
			imag_values=list( map( lambda x: -1.0/np.pi*x.imag, dict_names[str_filename] ) )
			label_imag=r"$-\frac{1}{\pi}\operatorname{Im}G_{\text{loc}}$"
			label_real=r"$\operatorname{Re}G_{\text{loc}}$"
		
	elif "Weiss" in str_filename:
		real_values=list( map( lambda x: x.real, dict_names[str_filename] ) )
		imag_values=list( map( lambda x: -1./np.pi*x.imag, dict_names[str_filename] ) )
		label_imag=r"$-\frac{1}{\pi}\operatorname{Im}G_{0}$"
		label_real=r"$\operatorname{Re}G_{0}$"
	else:
		real_values=list( map( lambda x: x.real, dict_names[str_filename] ) )
		imag_values=list( map( lambda x: x.imag, dict_names[str_filename] ) )
		if "Self" in str_filename:
			label_imag=r"$\operatorname{Im}\Sigma$"
			label_real=r"$\operatorname{Re}\Sigma$"
		else: # This case is for the non-interacting susceptibility
			label_imag=r"$\operatorname{Im}\chi^0(\omega)$"
			label_real=r"$\operatorname{Re}\chi^0(\omega)$"

	if len(keys_list)==1:
		axs.plot(omega,real_values,marker='s',ms=ms,color='green',label=label_real)
		axs.plot(omega,imag_values,marker='*',ms=ms,color='blue',label=label_imag)
		axs.legend()
	else:
		axs[i].plot(omega,real_values,marker='s',ms=ms,color='green',label=label_real)
		axs[i].plot(omega,imag_values,marker='*',ms=ms,color='blue',label=label_imag)
		axs[i].legend()
	i+=1

plt.savefig("Figure_A_w_U_{0:.2f}_beta_{1:.2f}_wmax_{2:.2f}.pdf".format(U,beta,wmax))
plt.show()

