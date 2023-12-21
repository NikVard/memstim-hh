#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from brian2 import *
from scipy.spatial import distance as dst
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from random import sample as smplf

import os
import time
import sys
from shutil import copyfile

import argparse
import parameters
import re # regular expressions

from model.globals import *
from model.HH_equations import *
from model.kuramoto_equations import *
from model.filter_equations import *
from model.fixed_input_equations import *
from model.Vm_avg_eqs import *
from model.I_SynE_I_sum_eqs import *
from model import settings
from model import setup

from src.annex_funcs import make_flat
from src.myplot import *
from src import stimulation


# Reading Amelie's locations
def parse_positions(fname):
    """ Opens file and parses coordinates """
    pattern = r'[\[\],\n]' # to remove from read lines

    x = []
    y = []
    z = []
    with open(fname, 'r') as fp:
        for line in fp:
            line_tmp = re.sub('[\[\] \,\n]+', ' ', line)
            tok = re.sub(pattern, '', line).split()
            x.append(float(tok[0]))
            y.append(float(tok[1]))
            z.append(float(tok[2]))

    return np.array([x, y, z]).T


# Configuration
# -------------------------------------------------------------#
# Use C++ standalone code generation TODO: Experimental!
#set_device('cpp_standalone')

# Parallel w/ Cython - independent caches
cache_dir = os.path.expanduser(f'~/.cython/brian-pid-{os.getpid()}')
prefs.codegen.runtime.cython.cache_dir = cache_dir
prefs.codegen.runtime.cython.multiprocess_safe = False

# Parse arguments
parser = argparse.ArgumentParser(description='MemStim using HH neurons')

parser.add_argument('-p', '--parameters',
                    nargs='?',
                    type=str,
                    default=os.path.join('configs', 'default.json'),
                    help='Parameters file (json format)')

parser.add_argument('-sd', '--save_dir',
                    nargs='?',
                    type=str,
                    default='results',
                    help='Destination directory to save the results')

parser.add_argument('--cuda', action='store_true')
parser.add_argument('--no-cuda', dest='cuda', action='store_false')
parser.set_defaults(cuda=True)


args = parser.parse_args()
filename = args.parameters
resdir = args.save_dir

# This part controls for CUDA runs
if args.cuda:
    import brian2cuda
    set_device("cuda_standalone")
    print("[!] Running in CUDA mode!")


try:
    data = parameters.load(filename)
    print('Using "{0}"'.format(filename))
except Exception as e:
    print(bcolors.RED + '[!]' + "Error code " + str(e.errno) + ": " + e.strerror + ' | Using "default.json"' + bcolors.ENDC)
    data = parameters._data
    filename = os.path.join('configs', 'default.json')
parameters.dump(data)
print()
# locals().update(data)

# Settings initialization
settings.init(data)


# Create the necessary directories
print('\n[00] Making directories...')
print('-'*32)
dirs = {}
# dirs['results'] = 'results_opt_DG'
dirs['results'] = resdir
if not os.path.isdir(dirs['results']):
    print('[+] Creating directory', dirs['results'])
    os.makedirs(dirs['results'])

if settings.I_stim[0]:
    dirs['stim'] = os.path.join(dirs['results'], '{stimamp:.1f}_nA'.format(stimamp=settings.I_stim[0]))
    if not os.path.isdir(dirs['stim']):
        print('[+] Creating directory', dirs['stim'])
        os.makedirs(dirs['stim'])

    dirs['offset'] = os.path.join(dirs['stim'], '{phase:.2f}_{stim_on:.1f}_ms'.format(phase=settings.offset, stim_on=settings.stim_onset*1e3))
    if not os.path.isdir(dirs['offset']):
        print('[+] Creating directory', dirs['offset'])
        os.makedirs(dirs['offset'])

    dirs['base'] = dirs['offset']

else:
    dirs['stim'] = os.path.join(dirs['results'], 'None')
    if not os.path.exists(dirs['stim']):
        print('[+] Creating directory', dirs['stim'])
        os.makedirs(dirs['stim'])

    dirs['base'] = dirs['stim']

dtime = datetime.datetime.now().strftime("%d-%m-%Y %HH%MM%SS") # imported from brian2
dirs['base'] = os.path.join(dirs['base'], dtime)
if not os.path.exists(dirs['base']):
    print('[+] Creating directory', dirs['base'])
    os.makedirs(dirs['base'])

dirs['figures'] = os.path.join(dirs['base'], 'figures')
if not os.path.isdir(dirs['figures']):
    print('[+] Creating directory', dirs['figures'])
    os.makedirs(dirs['figures'])

dirs['data'] = os.path.join(dirs['base'], 'data')
if not os.path.isdir(dirs['data']):
    print('[+] Creating directory', dirs['data'])
    os.makedirs(dirs['data'])

dirs['positions'] = os.path.join(dirs['data'], 'positions')
if not os.path.isdir(dirs['positions']):
    print('[+] Creating directory', dirs['positions'])
    os.makedirs(dirs['positions'])

dirs['spikes'] = os.path.join(dirs['data'], 'spikes')
if not os.path.isdir(dirs['spikes']):
    print('[+] Creating directory', dirs['spikes'])
    os.makedirs(dirs['spikes'])

dirs['currents'] = os.path.join(dirs['data'], 'currents')
if not os.path.isdir(dirs['currents']):
    print('[+] Creating directory', dirs['currents'])
    os.makedirs(dirs['currents'])


# Copy the configuration file on the results directory for safekeeping
copyfile(filename, os.path.join(dirs['base'], 'parameters_bak.json'))


# Debugging?
# -------------------------------------------------------------#
#if settings.debugging:
if settings.debugging:
    prefs.codegen.target = 'numpy' # Use Python code generation instead of Cython
    prefs.codegen.loop_invariant_optimisations = False # Switch off some optimization that makes the link between code and equations less obvious
    np.seterr(all='raise', under='ignore') # Make numpy raise errors for all kind of floating point problems, including division by 0, but ignoring underflows
    print('###########################')
    print(' [!]  DEBUGGING MODE ON')
    print('###########################')


# Make the neuron groups
# -------------------------------------------------------------#
print('\n[10] Making the neuron groups...')
print('-'*32)

# rotation matrix for fixing rotated y-positions from stippling program
r = R.from_euler('x', 180, degrees=True)

def parse_coords(fname, NG):
    """ Opens text file and parses coordinates """
    pattern = r'[\[\],\n]' # to remove from read lines

    with open(fname, 'r') as fin:
        idx = 0
        for line in fin:
            tok = re.sub(pattern, '', line).split()
            NG.x_soma[idx] = float(tok[0])*scale
            NG.y_soma[idx] = float(tok[1])*scale
            NG.z_soma[idx] = float(tok[2])*scale
            idx += 1

print('[+] Groups:')
# CA1
# E
pos = np.load(os.path.join('model', 'neuron_positions', 'full', 'CA1_E-stipple-10000.npy'))
pos = hstack((pos, zeros((len(pos), 1)))) # add z-axis
pos = r.apply(pos)
pos *= scale
pos[:,2] += 15*mm*rand(len(pos))
pos = pos[smplf(range(10000), settings.N_CA1[0]),:]
# pos = parse_positions(os.path.join('positions', 'CA1_exc.txt'))
idx = np.argsort(pos[:,2]) # sort neurons by increasing z-coordinate
pos = pos[idx]
G_E = NeuronGroup(N=settings.N_CA1[0],
    model=py_CAN_inp_eqs,
    threshold='v>V_th',
    reset=reset_eqs,
    refractory=refractory_time,
    method=integ_method,
    name='CA1_pyCAN')
G_E.size = cell_size_py
G_E.glu = 1
# G_E.sigma = settings.sigma_CA1[0]*uvolt
G_E.x_soma = pos[:,0]*metre
G_E.y_soma = pos[:,1]*metre
G_E.z_soma = pos[:,2]*metre
# G_E.x_soma = pos[:,0]*scale_aussel
# G_E.y_soma = pos[:,1]*scale_aussel
# G_E.z_soma = pos[:,2]*scale_aussel

# I
pos = np.load(os.path.join('model', 'neuron_positions', 'full', 'CA1_I-stipple-1000.npy'))
pos = hstack((pos, zeros((len(pos), 1)))) # add z-axis
pos = r.apply(pos)
pos *= scale
pos[:,2] += 15*mm*rand(len(pos))
pos = pos[smplf(range(1000), settings.N_CA1[1]),:]
# pos = parse_positions(os.path.join('positions', 'CA1_inh.txt'))
idx = np.argsort(pos[:,2]) # sort neurons by increasing z-coordinate
pos = pos[idx]
G_I = NeuronGroup(N=settings.N_CA1[1],
    model=inh_inp_eqs,
    threshold='v>V_th',
    refractory=refractory_time,
    method=integ_method,
    name='CA1_inh')
G_I.size = cell_size_inh
# G_I.sigma = settings.sigma_CA1[1]*uvolt
G_I.x_soma = pos[:,0]*metre
G_I.y_soma = pos[:,1]*metre
G_I.z_soma = pos[:,2]*metre
# G_I.x_soma = pos[:,0]*scale_aussel
# G_I.y_soma = pos[:,1]*scale_aussel
# G_I.z_soma = pos[:,2]*scale_aussel

# Add to list
G_E_v0 = -60.*mvolt - 10.*mvolt*rand(settings.N_CA1[0])
G_I_v0 = -60.*mvolt - 10.*mvolt*rand(settings.N_CA1[1])
G_E.v = G_E_v0
G_I.v = G_I_v0
print('[\u2022]\tCA1: done')


print("[!] Stimulation applied @", G_E.name)
G_E.r = 1 # 1 means on

print("[!] Stimulation applied @", G_I.name)
G_I.r = 1 # 1 means on


# Make the synapses
# -------------------------------------------------------------#
print('\n[12] Making the synapses...')

# gains
gains_CA1 = [1., G]
print("[!] Gains:", gains_CA1)

# intra
print('[+] Intra-region')

syn_CA1_all = setup.connect_intra([G_E], [G_I], settings.p_CA1_all, gains_CA1)
print('[\u2022]\tCA1-to-CA1: done')


# # Add groups for monitoring the avg Vm and I_SynE / I
# # -------------------------------------------------------------#
# # Average Vm per group per area
# print('\n[20] Vm average groups...')
# print('-'*32)

# G_E_avg = NeuronGroup(1, eq_record_neurons, name='Vm_avg_CA1_E')
# G_I_avg = NeuronGroup(1, eq_record_neurons, name='Vm_avg_CA1_I')
# G_E_avg.sum_v = mean(G_E_v0)
# G_I_avg.sum_v = mean(G_I_v0)

# Syn_E_avg = Synapses(G_E, G_E_avg, model=eq_record_synapses)
# Syn_E_avg.connect()
# Syn_I_avg = Synapses(G_I, G_I_avg, model=eq_record_synapses)
# Syn_I_avg.connect()
# print('[\u2022]\tSynapses: done')


# Make the spikes-to-rates group
# -------------------------------------------------------------#
print('\n[30] Spikes-to-rates group...')
print('-'*32)

G_S2R = NeuronGroup(1,
    model=firing_rate_filter_eqs,
    method='exact',
    #method='integ_method',
    name='S2R_filter',
    namespace=filter_params)
G_S2R.Y = 0 # initial conditions
print('[\u2022]\tGroup: done')

print('\n[31] Making the synapses...')
syn_CA1_2_rates = Synapses(G_E, G_S2R, on_pre='Y_post += (1/tauFR)/N_incoming', namespace=filter_params)
syn_CA1_2_rates.connect()
print('[\u2022]\tConnecting CA1-to-S2R: done')


# Inputs
# -------------------------------------------------------------#
print('\n[40] Inputs...')
print('-'*32)

print('[+] Time-vector')
tv = linspace(0, settings.duration/second, int(settings.duration/(settings.dt))+1)

# Kuramoto Oscillators
print('\n[41] Making the Kuramoto oscillators group...')
print(bcolors.YELLOW + '[!]' + bcolors.ENDC + ' Using dynamic input; Kuramoto oscillators of size N=%d w/ f0 = %.2f Hz | rhythm gain: %.2f nA | reset gain: %.2f' % (settings.N_Kur, settings.f0, settings.r_gain/nA, settings.k_gain))

# Make the necessary groups
# f0 = settings.f0 # settings.f0 does not work inside equations
# sigma = settings.sigma
G_K = NeuronGroup(settings.N_Kur,
    model=kuramoto_eqs_stim,
    threshold='True',
    method='euler',
    name='Kuramoto_oscillators_N_%d' % settings.N_Kur)
#G_K.Theta = '2*pi*rand()' # uniform U~[0,2π]
#G_K.omega = '2*pi*(f0+sigma*randn())' # normal N~(f0,σ)
theta0 = 2*pi*rand(settings.N_Kur) # uniform U~[0,2π]
omega0 = 2*pi*(settings.f0 + settings.sigma*randn(settings.N_Kur)) # ~N(2πf0,σ)
G_K.Theta = theta0
G_K.omega = omega0
G_K.kN = settings.kN_frac
G_K.G_in = settings.k_gain
G_K.offset = settings.offset
print('[\u2022]\tKuramoto oscillators group: done')

syn_kuramoto =  Synapses(G_K, G_K, on_pre=syn_kuramoto_eqs, method='euler', name='Kuramoto_intra')
syn_kuramoto.connect(condition='i!=j')
print('[\u2022]\tSynapses (Kuramoto): done')

# Kuramoto order parameter group
G_pop_avg = NeuronGroup(1,
    model=pop_avg_eqs,
    #method='euler',
    name='Kuramoto_averaging')
r0 = 1/settings.N_Kur * sum(exp(1j*G_K.Theta))
G_pop_avg.x = real(r0)  # avoid division by zero
G_pop_avg.y = imag(r0)
G_pop_avg.G_out = settings.r_gain
print('[\u2022]\tOrder parameter group: done')

syn_avg = Synapses(G_K, G_pop_avg, syn_avg_eqs, name='Kuramoto_avg')
syn_avg.connect()
print('[\u2022]\tSynapses (OP): done')

# Connections
print('\n[42] Connecting oscillators and filters...')
print('-'*32)

# connect the S2R group to the Kuramoto oscillators by linking input X to firing rates (drive)
G_K.X = linked_var(G_S2R, 'drive')
print('[\u2022]\tLinking S2R to Kuramoto oscillators: done')

# Monitors
# Kuramoto monitors
print('\n[43] Kuramoto and Filter Monitors...')
print('-'*32)

state_mon_kuramoto = StateMonitor(G_K, ['Theta'], record=True)
state_mon_order_param = StateMonitor(G_pop_avg, ['coherence', 'phase', 'rhythm', 'rhythm_rect'], record=True)
print('[\u2022]\tState monitor [Theta]: done')

# connect the fixed input to the I_exc variable in EC_E and EC_I
print('[+]\tLinking input (theta rhythm) to group ', G_E.name)
G_E.I_exc = linked_var(G_pop_avg, 'rhythm')
G_I.I_exc = linked_var(G_pop_avg, 'rhythm')


print('[\u2022]\tLinking input rhythm: done')
print('[\u2022]\tInputs: done')


# Stimulation and other inputs
# -------------------------------------------------------------#
print('\n[50] Stimulation...')
print('-'*32)

# generate empty stim signal
xstim_zero = zeros(tv.shape)
tv_stim_zero = tv

# generate stimulation signal
if settings.I_stim[0]:
    print(bcolors.GREEN + '[+]' + bcolors.ENDC + ' Stimulation ON')
    xstim, tv_stim = stimulation.generate_stim(duration=settings.stim_duration,
                                      dt=settings.stim_dt,
                                      I_stim=settings.I_stim,
                                      stim_on=settings.stim_onset,
                                      nr_of_trains=settings.nr_of_trains,
                                      nr_of_pulses=settings.nr_of_pulses,
                                      stim_freq=settings.stim_freq,
                                      pulse_width=settings.pulse_width,
                                      pulse_freq=settings.pulse_freq,
                                      ipi=settings.stim_ipi)
else:
    print(bcolors.RED + '[-]' + bcolors.ENDC + ' No stimulation defined; using empty TimedArray')
    xstim = xstim_zero
    tv_stim = tv_stim_zero

inputs_stim = TimedArray(values=xstim*nA, dt=settings.stim_dt*second, name='Input_stim')


# Tonic inputs -- Added on 19/06/2023
val = 0.0;
inputs_tonic = TimedArray(np.array([0] + [val]*5 + [0])*nA, dt=1*second)
stim_MS_gain = 0
stim_MS = TimedArray(values=stim_MS_gain*xstim_zero, dt=settings.stim_dt*second, name='MS_stim')




# Add any extra monitors here
# -------------------------------------------------------------#
print('\n[60] Adding extra monitors...')


# Create the Network
# -------------------------------------------------------------#
print('\n[70] Connecting the network...')
print('-'*32)

net = Network()
net.add(G_E, G_I)
net.add(G_S2R)
net.add(G_K, G_pop_avg)

for syn_intra_curr in make_flat(syn_CA1_all): # add synapses (intra)
    if syn_intra_curr != 0:
        net.add(syn_intra_curr)

net.add(syn_avg, syn_CA1_2_rates, syn_kuramoto)


# Add monitors
net.add(state_mon_kuramoto, state_mon_order_param)
print('[\u2022]\tNetwork groups: done')

spike_mon_CA1_E = SpikeMonitor(G_E, name=G_E.name+'_spikemon')
spike_mon_CA1_I = SpikeMonitor(G_I, name=G_I.name+'_spikemon')
net.add(spike_mon_CA1_E, spike_mon_CA1_I)
print('[\u2022]\tSpike monitors: done')

rate_mon_CA1_E = PopulationRateMonitor(G_E, name=G_E.name+'_ratemon')
rate_mon_CA1_I = PopulationRateMonitor(G_I, name=G_I.name+'_ratemon')
net.add(rate_mon_CA1_E, rate_mon_CA1_I)
print('[\u2022]\tRate monitors: done')


# spikes2rates monitor (vout)
state_mon_s2r = StateMonitor(G_S2R, ['drive'], record=True)
net.add(state_mon_s2r)
print('[\u2022]\tState monitor [drive]: done')


# Run the simulation
# -------------------------------------------------------------#
defaultclock.dt = settings.dt
tstep = defaultclock.dt

# Preparation for simulations
t_step = 1*second
t_run = settings.duration
run_sim = True

print('\n[80] Starting simulation...')
print('-'*32)

start = time.time()
net.run(1000*ms, report='text', report_period=5*second, profile=True)
end = time.time()

print(bcolors.GREEN + '[+]' + ' All simulations ended' + bcolors.ENDC)
print(bcolors.YELLOW + '[!]' + bcolors.ENDC + ' Simulation ran for '+str((end-start)/60)+' minutes')
print(profiling_summary(net=net, show=4)) # show the top 10 objects that took the longest

print('###########################')
print(' [!]  NUMBER OF SYNAPSES')
print('E->E: ', 0 if not syn_CA1_all[0][0] else int64(syn_CA1_all[0][0].N))
print('E->I: ', 0 if not syn_CA1_all[0][1] else int64(syn_CA1_all[0][1].N))
print('I->E: ', 0 if not syn_CA1_all[1][0] else int64(syn_CA1_all[1][0].N))
print('I->I: ', 0 if not syn_CA1_all[1][1] else int64(syn_CA1_all[1][1].N))
print('###########################')


# Plot the results
# -------------------------------------------------------------#
print('\n[90] Post-simulation actions')
print('-'*32)

print('\n[91] Plotting results...')
tight_layout()

# raster plot of all regions
raster_fig = plt.figure(figsize=(10,5))
ax0 = plt.subplot(3,1,1)
ax0.plot(spike_mon_CA1_E.t/ms, spike_mon_CA1_E.i, 'b.')
ax1 = plt.subplot(3,1,2)
ax1.plot(spike_mon_CA1_I.t/ms, spike_mon_CA1_I.i, 'r.')
ax2 = plt.subplot(3,1,3)
ax2.plot(state_mon_order_param.t/ms, state_mon_order_param.rhythm[0], 'k')
ax2.set_xlabel('Time [ms]')

print("[+] Saving figure 'figures/rasters_CA1.png'")
plot_watermark(raster_fig, os.path.basename(__file__), filename, settings.git_branch, settings.git_short_hash)
raster_fig.savefig(os.path.join(dirs['figures'], 'rasters_CA1.png'))


# Save the results as .txt files (rows: time | cols: data)
# -------------------------------------------------------------#
print('\n[92] Saving results...')


# Write data to disk
# Kuramoto monitors
print("[+] Saving Kuramoto monitor data")
np.savetxt(os.path.join(dirs['data'], 'order_param_mon_phase.txt'), state_mon_order_param.phase.T, fmt='%.8f')
np.savetxt(os.path.join(dirs['data'], 'order_param_mon_rhythm.txt'), state_mon_order_param.rhythm.T/nA, fmt='%.8f')
np.savetxt(os.path.join(dirs['data'], 'order_param_mon_coherence.txt'), state_mon_order_param.coherence.T, fmt='%.8f')

# CA1 firing rate
print("[+] Saving CA1 firing rate")
np.savetxt(os.path.join(dirs['data'], 'rate_mon_E_CA1.txt'), rate_mon_CA1_E.smooth_rate(window='gaussian', width=50*ms).T/Hz, fmt='%.8f')
np.savetxt(os.path.join(dirs['data'], 'rate_mon_E_CA1.txt'), rate_mon_CA1_I.smooth_rate(window='gaussian', width=50*ms).T/Hz, fmt='%.8f')
np.savetxt(os.path.join(dirs['data'], 's2r_mon_drive.txt'), state_mon_s2r.drive_.T, fmt='%.8f')

# External stimulus
print("[+] Saving external stimulus")
np.savetxt(os.path.join(dirs['data'], 'stim_input.txt'), xstim, fmt='%.2f')


# Save the spikes and their times
# -------------------------------------------------------------#
print("\n[93] Saving spikes in time....")
SM_i = []
SM_t = []
for SM in make_flat([spike_mon_CA1_E, spike_mon_CA1_I]):
    for i_val in SM.i:
        SM_i.append(i_val)

    for t_val in SM.t:
        SM_t.append(t_val/msecond)

    print("[+] Saving spikes from", SM.source.name)
    fname = SM.name
    np.savetxt(os.path.join(dirs['spikes'], fname + '_i.txt'), np.array(SM_i).astype(np.int16), fmt='%d')
    np.savetxt(os.path.join(dirs['spikes'], fname + '_t.txt'), np.array(SM_t).astype(np.float32), fmt='%.1f')

    SM_i.clear()
    SM_t.clear()


sys.exit(0)