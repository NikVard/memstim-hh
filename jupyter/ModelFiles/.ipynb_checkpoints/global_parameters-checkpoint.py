from brian2.units import *
from brian2.core.clocks import defaultclock

""" Fixed parameters required by the equations """
tau_Cl = 100.*ms                # removal rate of chloride ions in excitatory cells; healthy 0.1s | epilepsy 0.5s or 1.0s, page 88, Aussel
Ek = -100.*mV                   # resting potential of potassium channels in excitatory cells; healthy -100mV | epilepsy -90mV or -80mV, page 88, Aussel

# gains and global conductances
gM = 90*usiemens*cmetre**-2     # conductance for M-current
gCAN = 25*usiemens*cmeter**-2   # wakefulness -> table 3.6, page 59, Aussel

""" Model fixed parameters """
areas = ['EC', 'DG', 'CA3', 'CA1']
types = ['exc', 'inh']

G = 3                           # G for wakefulness // var coeff
g_max_e = 60*psiemens
g_max_i = 600*psiemens

gains_all =  [[1./G, 1.], [G, G], [1./G, 1.], [1., G]]

V_th = -20.*mvolt               # spiking threshold
refractory_time = 3*ms          # refractory time after a spike

cell_size_py = 29.e3*umetre**2  # single cell type
cell_size_inh = 14.e3*umetre**2
scale = round(1000/818,6)*umetre

sigma_noise_inh = 200.*uvolt
sigma_noise_exc = 2000.*uvolt

tstep = defaultclock.dt
integ_method = 'exponential_euler'  # integration method


# Kuramoto settings (default parameters)
N_Kur = 200
f0 = 6 # Hz
sigma = 0.5 # std of Gaussian for phase/ang.vel. initialization
kN_frac = 55. # synchronization parameter (k/N factor)
k_gain = 4. # phase reset gain
r_gain = 0.22*nA # output sin rhythm gain (scaling, in nA)
