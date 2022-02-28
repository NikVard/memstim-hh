from brian2.units import *
from brian2 import defaultclock

""" Fixed parameters required by the equations """
tau_Cl = 0.1*second             # removal rate of chloride ions in excitatory cells; healthy 0.1s | epilepsy 0.5s or 1.0s, page 88, Aussel
Ek = -100.*mV                   # resting potential of potassium channels in excitatory cells; healthy -100mV | epilepsy -90mV or -80mV, page 88, Aussel

# gains and global conductances
gM = 90.*usiemens*cmetre**-2    # conductance for M-current
gCAN = 0.25*usiemens*cmeter**-2 # wakefulness -> table 3.6, page 59, Aussel

""" Model fixed parameters """
areas = ['EC', 'DG', 'CA3', 'CA1']
types = ['exc', 'inh']

G = 3.                          # G for wakefulness
g_max_e = 60.*psiemens
g_max_i = 600.*psiemens

V_th = -20.*mvolt               # spiking threshold
refractory_time = 3.*ms         # refractory time after a spike

cell_size_py = 29.e3*umetre**2  # single cell type
cell_size_inh = 14.e3*umetre**2
# scale = 150*umetre              # model scale | 150*umetre
# scale = round(1000/379,6)*umetre
scale = round(1000/818,6)*umetre

sigma_noise_inh = 1.*uvolt
sigma_noise_exc = 100.*uvolt

tstep = defaultclock.dt
integ_method = 'exponential_euler'  # integration method

# Spikes-2-Rates filter
filter_params = {
    'tauFR' : 1*ms
}


# ANSI Colored text output!
class bcolors:
    # TEXT COLOR
    BLACK = '\u001b[30m'
    RED = '\u001b[31m'
    GREEN = '\u001b[32m'
    YELLOW = '\u001b[33m'
    BLUE = '\u001b[34m'
    MAGENTA = '\u001b[35m'
    CYAN = '\u001b[36m'
    WHITE = '\u001b[37m'

    # BACKGROUND COLOR
    BGBLACK = '\u001b[40m'
    BGRED = '\u001b[41m'
    BGGREEN = '\u001b[42m'
    BGYELLOW = '\u001b[43m'
    BGBLUE = '\u001b[44m'
    BGMAGENTA = '\u001b[45m'
    BGCYAN = '\u001b[46m'
    BGWHITE = '\u001b[47m'

    # STYLES
    BOLD = '\u001b[1m'
    UND = '\u001b[4m'
    REV = '\u001b[7m'


    # RESET
    ENDC = '\u001b[0m'
