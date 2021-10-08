from brian2.units import *
from brian2 import defaultclock

""" Fixed parameters """
# types = [1,1]                   # number of cell types per [E, I] population

tau_Cl = 0.1*second             # removal rate of chloride ions in excitatory cells; healthy 0.1s | epilepsy 0.5s or 1.0s, page 88, Aussel
Ek = -100*mV                    # resting potential of potassium channels in excitatory cells; healthy -100mV | epilepsy -90mV or -80mV, page 88, Aussel

# gains and global conductances
gM = 90.*usiemens*cmetre**-2    # conductance for M-current
gCAN = 0.25*usiemens*cmeter**-2 # wakefulness -> table 3.6, page 59, Aussel
G = 3.                          # G for wakefulness
g_max_e = 60.*psiemens
g_max_i = 600.*psiemens


V_th = -20.*mvolt               # spiking threshold
refractory_time = 3*ms          # refractory time after a spike

cell_size_py = 29e3*umetre**2   # single cell type
cell_size_inh = 14e3*umetre**2

sigma_noise_inh = 1.*uvolt
sigma_noise_exc = 10.*uvolt

tstep = defaultclock.dt

''' JSON PARAMETERS HERE (+DEFAULTS) '''
# Simulation
duration = 1*second # simulation duration
integ_method = 'exponential_euler'  # integration method

# population sizes per area | [E, I]
N_EC = [] # def: [10e3, 1e3]
N_DG = [] # def: [10e3, 0.1e3]
N_CA3 = [] # def: [1e3, 0.1e3]
N_CA1 = [] # def: [10e3, 1e3]

# intra-area conn. probabilities per area | [[E-E, E-I], [I-E, I-I]]
p_EC_all = [[],[]] # def:[[0., 0.37], [0.54, 0.]]
p_DG_all = [[],[]] # def: [[0., 0.06], [0.14, 0.]]
p_CA3_all = [[],[]] # def: [[0.56, 0.75], [0.75, 0.]]
p_CA1_all = [[],[]] # def: [[0., 0.28], [0.3, 0.7]]

# inter-area conn. probabilities
p_mono = None # def: 0.2 # monosynaptic pathway connectivity
p_tri = None # def: 0.45 # trisynaptic pathway connectivity

# topology settings
#topo  = 'real' # realistic topology
#topo_file = '' # imported image file name
