from brian2.units import *

""" Fixed parameters """

gM = 90*usiemens*cmetre**-2     # conductance for M-current
gCAN = 0.25*usiemens*cmeter**-2 # wakefulness -> table 3.6, page 59, Aussel

V_th = -20*mvolt                # spiking threshold
refractory_time = 3*ms          # refractory time after a spike

cell_size_py = 29e3*umetre**2
cell_size_inh = 14e3*umetre**2

sigma_noise_inh = 1*uvolt
sigma_noise_exc = 10*uvolt

integ_method = 'exponential_euler'  # integration method
