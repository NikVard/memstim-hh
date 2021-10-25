#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from brian2 import *
from tqdm import tqdm

import time

import argparse
import parameters

from model.globals import *
from model.HH_equations import *
from model.kuramoto_equations import *
from model.filter_equations import *
from model import setup

from src.annex_funcs import make_flat



# Configuration
# -------------------------------------------------------------_#
# Use C++ standalone code generation TODO: Experimental!
#set_device('cpp_standalone')
parser = argparse.ArgumentParser(description='MemStim using HH neurons')
parser.add_argument('parameters_file',
                    nargs='?',
                    metavar='m',
                    type=str,
                    default='configs/default.json',
                    help='Parameters file (json format)')
args = parser.parse_args()
filename = args.parameters_file
print('Using "{0}"'.format(filename))

try:
    data = parameters.load(filename)
except:
    data = parameters._data
parameters.dump(data)
print()
locals().update(data)



# Neuronal population sizes > [E, I]
global N_EC, N_DG, N_CA3, NCA1
N_EC = [data['areas']['EC']['E']['N'], data['areas']['EC']['I']['N']]
N_DG = [data['areas']['DG']['E']['N'], data['areas']['DG']['I']['N']]
N_CA3 = [data['areas']['CA3']['E']['N'], data['areas']['CA3']['I']['N']]
N_CA1 = [data['areas']['CA1']['E']['N'], data['areas']['CA1']['I']['N']]
N_all = [N_EC, N_DG, N_CA3, N_CA1]

# Intra-conn. probabilities | [[E-E, E-I], [I-E, I-I]]
global p_EC_all, p_DG_all, p_CA3_all, p_CA1_all
p_EC_all = data['connectivity']['intra']['EC']
p_DG_all = data['connectivity']['intra']['DG']
p_CA3_all = data['connectivity']['intra']['CA3']
p_CA1_all = data['connectivity']['intra']['CA1']
p_intra_all = [p_EC_all, p_DG_all, p_CA3_all, p_CA1_all]

# Inter-conn. probabilities | p_mono / p_tri
global p_mono, p_tri
p_mono = data['connectivity']['inter']['p_mono'] # monosynaptic pathway connectivity
p_tri = data['connectivity']['inter']['p_tri'] # trisynaptic pathway connectivity

global duration
duration = data['simulation']['duration']*ms

global N_Kur, f0, sigma, kN_frac
N_Kur = data['Kuramoto']['N']
f0 = data['Kuramoto']['f0']
sigma = data['Kuramoto']['sigma']
kN_frac = data['Kuramoto']['kN']
k_gain = data['Kuramoto']['gain']

seed(data['seed_val'])



# Make the neuron groups
# -------------------------------------------------------------_#
print('\n >  Making the neuron groups...')

fig, axs = subplots(nrows=1, ncols=1)

G_all = [[[] for pops in range(2)] for areas in range(4)]

# EC -> theta input from MS
G_E = NeuronGroup(N=N_EC[0],
    model=py_CAN_inp_eqs,
    threshold='v>V_th',
    reset=reset_eqs,
    refractory=refractory_time,
    method=integ_method,
    name='EC_pyCAN')
G_E.size = cell_size_py
G_E.glu = 1
pos = np.load('./neuron_positions/full/EC_E-stipple-10000.npy')
G_E.x_soma = pos[:,0]*scale
G_E.y_soma = pos[:,1]*scale
axs.plot(G_E.x_soma, G_E.y_soma, 'b.', markersize=.5, alpha=0.5, label='EC-PyCAN')

G_I = NeuronGroup(N=N_EC[1],
    model=inh_inp_eqs,
    threshold='v>V_th',
    refractory=refractory_time,
    method=integ_method,
    name='EC_inh')
G_I.size = cell_size_inh
print('EC: done')
pos = np.load('./neuron_positions/full/EC_I-stipple-1000.npy')
G_I.x_soma = pos[:,0]*scale
G_I.y_soma = pos[:,1]*scale
axs.plot(G_I.x_soma, G_I.y_soma, 'r.', markersize=.5, alpha=0.5, label='EC-Inh')

G_all[0][0].append(G_E)
G_all[0][1].append(G_I)


# DG
G_E = NeuronGroup(N=N_DG[0],
    model=py_eqs,
    threshold='v>V_th',
    reset=reset_eqs,
    refractory=refractory_time,
    method=integ_method,
    name='DG_py')
G_E.size = cell_size_py
G_E.glu = 1
pos = np.load('./neuron_positions/full/DG_E-stipple-10000.npy')
G_E.x_soma = pos[:,0]*scale
G_E.y_soma = pos[:,1]*scale
axs.plot(G_E.x_soma, G_E.y_soma, 'g.', markersize=.5, alpha=0.5, label='DG-Py')

G_I = NeuronGroup(N=N_DG[1],
    model=inh_eqs,
    threshold='v>V_th',
    refractory=refractory_time,
    method=integ_method,
    name='DG_inh')
G_I.size = cell_size_inh
print('DG: done')
pos = np.load('./neuron_positions/full/DG_I-stipple-100.npy')
G_I.x_soma = pos[:,0]*scale
G_I.y_soma = pos[:,1]*scale
axs.plot(G_I.x_soma, G_I.y_soma, 'r.', markersize=.5, alpha=0.5, label='DG-Inh')

G_all[1][0].append(G_E)
G_all[1][1].append(G_I)


# CA3
G_E = NeuronGroup(N=N_CA3[0],
    model=py_CAN_eqs,
    threshold='v>V_th',
    reset=reset_eqs,
    refractory=refractory_time,
    method=integ_method,
    name='CA3_pyCAN')
G_E.size = cell_size_py
G_E.glu = 1
pos = np.load('./neuron_positions/full/CA3_E-stipple-1000.npy')
G_E.x_soma = pos[:,0]*scale
G_E.y_soma = pos[:,1]*scale
axs.plot(G_E.x_soma, G_E.y_soma, 'b.', markersize=.5, alpha=0.5, label='CA3-PyCAN')

G_I = NeuronGroup(N=N_CA3[1],
    model=inh_eqs,
    threshold='v>V_th',
    refractory=refractory_time,
    method=integ_method,
    name='CA3_inh')
G_I.size = cell_size_inh
print('CA3: done')
pos = np.load('./neuron_positions/full/CA3_I-stipple-100.npy')
G_I.x_soma = pos[:,0]*scale
G_I.y_soma = pos[:,1]*scale
axs.plot(G_I.x_soma, G_I.y_soma, 'r.', markersize=.5, alpha=0.5, label='CA3-Inh')

G_all[2][0].append(G_E)
G_all[2][1].append(G_I)


# CA1
G_E = NeuronGroup(N=N_CA1[0],
    model=py_CAN_eqs,
    threshold='v>V_th',
    reset=reset_eqs,
    refractory=refractory_time,
    method=integ_method,
    name='CA1_pyCAN')
G_E.size = cell_size_py
G_E.glu = 1
pos = np.load('./neuron_positions/full/CA1_E-stipple-10000.npy')
G_E.x_soma = pos[:,0]*scale
G_E.y_soma = pos[:,1]*scale
axs.plot(G_E.x_soma, G_E.y_soma, 'b.', markersize=.5, alpha=0.5, label='CA1-PyCAN')

G_I = NeuronGroup(N=N_CA1[1],
    model=inh_eqs,
    threshold='v>V_th',
    refractory=refractory_time,
    method=integ_method,
    name='CA1_inh')
G_I.size = cell_size_inh
print('CA1: done')
pos = np.load('./neuron_positions/full/CA1_I-stipple-1000.npy')
G_I.x_soma = pos[:,0]*scale
G_I.y_soma = pos[:,1]*scale
axs.plot(G_I.x_soma, G_I.y_soma, 'r.', markersize=.5, alpha=0.5, label='CA1-Inh')

G_all[3][0].append(G_E)
G_all[3][1].append(G_I)

G_flat = make_flat(G_all)

# initialize the group variables
for ngroup in G_flat:
    ngroup.v = '-60*mvolt-rand()*10*mvolt' # str -> individual init. val. per neuron

    # EC populations get stimulated
    if ngroup.name=='EC_pyCAN':# or ngroup.name=='EC_inh':
        ngroup.r = 1 #1
    else:
        ngroup.r = 0 # int -> same init. val. for all neurons

#show()
#exit()



# Make the synapses
# -------------------------------------------------------------_#
print('\n >  Making the synapses...')

# gains
gains_all =  [[1./G, 1.], [G, G], [1./G, 1.], [1., G]]

# intra
print('     * intra-region')

syn_EC_all = setup.connect_intra(G_all[0][0], G_all[0][1], p_EC_all, gains_all[0])
print('EC-to-EC: done')
syn_DG_all = setup.connect_intra(G_all[1][0], G_all[1][1], p_DG_all, gains_all[1])
print('DG-to-DG: done')
syn_CA3_all = setup.connect_intra(G_all[2][0], G_all[2][1], p_CA3_all, gains_all[2])
print('CA3-to-CA3: done')
syn_CA1_all = setup.connect_intra(G_all[3][0], G_all[3][1], p_CA1_all, gains_all[3])
print('CA1-to-CA1: done')
syn_intra_all = [syn_EC_all, syn_DG_all, syn_CA3_all, syn_CA1_all]

# inter
print('     * inter-region')

# p_inter_all = [[[[],[]],[[],[]]],[[[],[]],[[],[]]],[[[],[]],[[],[]]],[[[],[]],[[],[]]]]
p_inter_all = [[[[0,0] for ii in range(2)] for jj in range(4)] for kk in range(4)]
p_inter_all[0][1][0] = [p_tri for ii in range(2)] # EC_E to DG_E | DG_I
p_inter_all[0][2][0] = [p_mono for ii in range(2)] # EC_E to CA3_E | CA3_I
p_inter_all[0][3][0] = [p_mono for ii in range(2)] # EC_E to CA1_E | CA1_I
p_inter_all[1][2][0] = [p_tri for ii in range(2)] # DG_E to CA3_E | CA3_I
p_inter_all[2][3][0] = [p_tri for ii in range(2)] # CA3_E to CA1_E | CA1_I
p_inter_all[3][0][0] = [p_tri for ii in range(2)] # CA1_E to EC_E | EC_I

syn_EC_DG_all = setup.connect_inter(G_all[0][0], G_all[1][0], G_all[1][1], p_inter_all[0][1], gains_all[0])
syn_EC_CA3_all = setup.connect_inter(G_all[0][0], G_all[2][0], G_all[2][1], p_inter_all[0][2], gains_all[0])
syn_EC_CA1_all = setup.connect_inter(G_all[0][0], G_all[3][0], G_all[3][1], p_inter_all[0][3], gains_all[0])
print('EC-to-all: done')

syn_DG_CA3_all = setup.connect_inter(G_all[1][0], G_all[2][0], G_all[2][1], p_inter_all[1][2], gains_all[1])
print('DG-to-CA3: done')

syn_CA3_CA1_all = setup.connect_inter(G_all[2][0], G_all[3][0], G_all[3][1], p_inter_all[2][3], gains_all[2])
print('CA3-to-CA1: done')

syn_CA1_EC_all = setup.connect_inter(G_all[3][0], G_all[0][0], G_all[0][1], p_inter_all[3][0], gains_all[3])
print('CA1-to-EC: done')
syn_inter_all = [syn_EC_DG_all, syn_EC_CA3_all, syn_EC_CA1_all, syn_DG_CA3_all, syn_CA3_CA1_all, syn_CA1_EC_all]



# Add the monitors (spikes/rates)
# -------------------------------------------------------------_#
print('\n >  Monitors...')
state_mon_all = []
spike_mon_all = []
rate_mon_all = []

state_mon_E_all = [[StateMonitor(G_py, 'v', record=True) for G_py in G_all[i][0] if G_py] for i in range(4)]
state_mon_I_all = [[StateMonitor(G_inh, 'v', record=True) for G_inh in G_all[i][1] if G_inh] for i in range(4)]

spike_mon_E_all = [[SpikeMonitor(G_py) for G_py in G_all[i][0] if G_py] for i in range(4)]
spike_mon_I_all = [[SpikeMonitor(G_inh) for G_inh in G_all[i][1] if G_inh] for i in range(4)]

rate_mon_E_all = [[PopulationRateMonitor(G_py) for G_py in G_all[i][0] if G_py] for i in range(4)]
rate_mon_I_all = [[PopulationRateMonitor(G_inh) for G_inh in G_all[i][1] if G_inh] for i in range(4)]



# Stimulation and other inputs
# -------------------------------------------------------------_#
print('\n >  Inputs and Stimulation...')
dt_stim = 1*ms
I_stim = 1*namp
tv = linspace(0, duration/second, int(duration/(dt_stim))+1)
xstim = I_stim * logical_and(tv>0.75, tv<0.76)
inputs_stim = TimedArray(xstim, dt=dt_stim)



# Kuramoto Oscillators (MS)
# -------------------------------------------------------------_#
# Make the necessary groups
G_K = NeuronGroup(N_Kur,
    model=kuramoto_eqs_stim,
    threshold='True',
    method='euler',
    name='Kuramoto_oscillators_N_%d' % N_Kur)
G_K.Theta = '2*pi*rand()' # uniform U~[0,2π]
G_K.omega = '2*pi*(f0+sigma*randn())' # normal N~(f0,σ)
G_K.kN = kN_frac
G_K.kG = k_gain
G_flat.append(G_K) # append to the group list!
syn_kuramoto =  Synapses(G_K, G_K, on_pre=syn_kuramoto_eqs, method='euler', name='Kuramoto_intra')
syn_kuramoto.connect(condition='i!=j')

# Kuramoto order parameter group
G_pop_avg = NeuronGroup(1, pop_avg_eqs)
r0 = 1/N_Kur * sum(exp(1j*G_K.Theta))
G_pop_avg.x = real(r0)  # avoid division by zero
G_pop_avg.y = imag(r0)
G_flat.append(G_pop_avg) # append to the group list!
syn_avg = Synapses(G_K, G_pop_avg, syn_avg_eqs, name='Kuramoto_avg')
syn_avg.connect()
print('Kuramoto oscillators: done')



# Firing Rate Filter Population
# -------------------------------------------------------------_#
# Make the spikes-to-rates group

G_S2R = NeuronGroup(1,
    model=firing_rate_filter_eqs,
    method='exact',
    namespace=filter_params)
G_S2R.Y = 0 # initial conditions
G_flat.append(G_S2R) # append to the group list!
print('Spikes-to-rates LPF: done')



# Connections
# -------------------------------------------------------------_#
# CA1 spikes-to-rates synapses
# find the CA1-E group
G_CA1_E = None
for g in G_flat:
    if g.name=='CA1_pyCAN':
        G_CA1_E = g
        break

# connect the CA1-E group to the low-pass-filter spikes-2-rates (S2R) group
if G_CA1_E:
    syn_CA1_2_rates = Synapses(G_CA1_E, G_S2R, on_pre='Y_post += (1/tauFR)/%d'%G_CA1_E.N, namespace=filter_params)
    syn_CA1_2_rates.connect()
print('CA1-to-S2R: done')


# connect the S2R group to the Kuramoto by linking input X to firing rates (drive)
G_K.X = linked_var(G_S2R, 'drive')
print('Linking S2R to Kuramoto oscillators: done')

# connect the Kuramoto ensemble rhythm to the I_exc variable in EC_E and EC_I (Kuramoto output as input to EC_E/I pop.)
'''for g in G_flat:
    if g.name=='EC_pyCAN' or g.name=='EC_inh':
        print('>> Setting input rhythm for group ', g.name)
        g.I_exc = linked_var(G_pop_avg, 'rhythm_rect')
'''
G_flat[0].I_exc = linked_var(G_pop_avg, 'rhythm_zero')
G_flat[1].I_exc = linked_var(G_pop_avg, 'rhythm_zero')


# Monitors
# -------------------------------------------------------------_#
# Kuramoto monitors
print("Preparing monitors")
kuramoto_mon = StateMonitor(G_K, ['Theta'], record=True)
order_param_mon = StateMonitor(G_pop_avg, ['coherence', 'phase', 'rhythm', 'rhythm_rect'], record=True)

# spikes2rates monitor (vout)
s2r_mon = StateMonitor(G_S2R, ['drive'], record=True)

'''
G_CA1_E, G_CA1_I = None, None
for g in G_flat:
    if g.name=='CA1_pyCAN':
        G_CA1_E = g
    if g.name=='CA1_inh':
        G_CA1_I = g
    if G_CA1_E and G_CA1_I:
        break

mon_tmp_E = StateMonitor(G_CA1_E, [], record=True)
mon_tmp_I = StateMonitor(G_CA1_I, [], record=True)
'''

# Create the Network
# -------------------------------------------------------------_#
print('\n >  Connecting the network...')
net = Network()
net.add(G_all) # add groups
net.add(G_K)
net.add(G_pop_avg)
net.add(G_S2R)

for syn_intra_curr in make_flat(syn_intra_all): # add synapses (intra)
    if syn_intra_curr!=0:
        net.add(syn_intra_curr)

for syn_inter_curr in make_flat(syn_inter_all): # add synapses (inter)
    if syn_inter_curr!=0:
        net.add(syn_inter_curr)

net.add(syn_kuramoto) # kuramoto intra-synapses
net.add(syn_avg) # kuramoto population average (order parameter) synapses
net.add(syn_CA1_2_rates) # CA1 spikes2rates

net.add(state_mon_E_all) # monitors
net.add(state_mon_I_all)
net.add(spike_mon_E_all)
net.add(spike_mon_I_all)
net.add(rate_mon_E_all)
net.add(rate_mon_I_all)
net.add(kuramoto_mon)
net.add(order_param_mon)
net.add(s2r_mon)



# Run the simulation
# -------------------------------------------------------------_#
global tstep
tstep = defaultclock.dt


print('\n >  Starting simulation...')
start = time.time()
net.run(duration, report='text', report_period=10*second, profile=True)
end = time.time()
print('Simulation ended')
print('Simulation ran for '+str((end-start)/60)+' minutes')

profiling_summary(net=net, show=10)  # show the top 10 objects that took the longest



# Plot the results (dumb plot, not a function yet)
# -------------------------------------------------------------_#
zones = ['EC', 'DG', 'CA3', 'CA1']
types = ['exc', 'inh']

# raster plot of all regions
raster_fig = figure()
raster_fig.set_figheight(12)
raster_fig.set_figwidth(12)

for ii in range(4):
    subplot(4,2,ii*2+1)
    title('raster '+zones[ii]+' exc')
    plot(spike_mon_E_all[ii][0].t/msecond, spike_mon_E_all[ii][0].i, '.g',
                markersize=.5,alpha=0.5)
    xlim(0, duration/msecond)
    ylim(0, N_all[ii][0])
    xlabel('Time (ms)')
    ylabel('Neuron index')

    subplot(4,2,ii*2+2)
    title('raster '+zones[ii]+' inh')
    plot(spike_mon_I_all[ii][0].t/msecond, spike_mon_I_all[ii][0].i, '.r',
                markersize=.5,alpha=0.5)
    xlim(0, duration/msecond)
    ylim(0, N_all[ii][1])
    xlabel('Time (ms)')
    ylabel('Neuron index')
raster_fig.savefig('figures/Raster_stim_EC_E_I_only.png')

# kuramoto order parameter plots
fig, axs = subplots(3,1)
fig.set_figheight(8)
fig.set_figwidth(16)

'''
# calculate order parameter in the end
samples = len(kuramoto_mon.Theta[0])
r = np.zeros(samples, dtype='complex')
for s in range(samples):
    r[s] = 1/N_Kur * sum(exp(1j*kuramoto_mon.Theta[:,s])) # order parameter r(t)
'''

#axs[0].plot(kuramoto_mon.t/second, mean(sin(kuramoto_mon.Theta), axis=0), label='Mean theta')
#axs[0].plot(kuramoto_mon.t/second, imag(r), '--', label='sin(Im(r))')
axs[0].plot(order_param_mon.t/second, order_param_mon.rhythm_rect[0], '-', label='Order Param Monitor')
axs[1].plot(order_param_mon.t/second, order_param_mon.phase[0], '-', label='Avg. phase of N=%d oscillators'%N_Kur)
axs[2].plot(order_param_mon.t/second, order_param_mon.coherence[0], '-', label='Coherence (|r|)')

axs[0].set_ylabel("Ensemble Theta Rhythm")
axs[1].set_ylabel("Average Phase")
axs[2].set_ylabel("Phase Coherence")
axs[2].set_xlabel("Time [s]")
#axs[0].set_ylim([-1,1])
axs[1].set_ylim([-pi,pi])
axs[2].set_ylim([0,1])
axs[0].axvline(x=0.75, ymin=-1, ymax=1, c="red", linewidth=2, zorder=0, clip_on=False)
axs[1].axvline(x=0.75, ymin=-1, ymax=1, c="red", linewidth=2, zorder=0, clip_on=False)
axs[2].axvline(x=0.75, ymin=-1, ymax=1, c="red", linewidth=2, zorder=0, clip_on=True)

# make things pretty
axs[0].legend()
axs[0].grid()
axs[1].legend()
axs[1].grid()
axs[2].legend()
axs[2].grid()
fig.savefig('figures/Kuramoto_rhythms_stim_EC_E_I_only.png')






# Plot more stuff
fig_extra, raster_extra = subplots(nrows=5, ncols=1)
fig_extra.set_figheight(8)
fig_extra.set_figwidth(16)

title('raster plot')

raster_extra[0].plot(spike_mon_E_all[0][0].t/msecond, spike_mon_E_all[0][0].i, '.g',
            markersize=.5,alpha=0.5)
raster_extra[0].set_xlim(0, duration/msecond)
raster_extra[0].set_ylim(0, N_all[0][0])
raster_extra[0].set_ylabel('Neuron index')

raster_extra[1].plot(tv, xstim)
raster_extra[1].set_ylabel('Stim. Input')
raster_extra[1].grid()

raster_extra[2].plot(s2r_mon.t/second, s2r_mon.drive[0], label='LPF Output')
raster_extra[2].set_ylabel('Population Firing Rates')
raster_extra[2].legend()
raster_extra[2].grid()

raster_extra[3].plot(order_param_mon.t/second, order_param_mon.coherence[0], '-', label='Order Param')
raster_extra[3].set_ylabel('Order Parameter')
raster_extra[3].legend()
raster_extra[3].grid()

raster_extra[4].plot(order_param_mon.t/second, order_param_mon.rhythm_rect[0], '-', label='Theta Rhythm (Ensemble)')
raster_extra[4].set_ylabel('Theta Rhythm (corr)')
raster_extra[4].set_xlabel('Time (ms)')
raster_extra[4].legend()
raster_extra[4].grid()
fig_extra.savefig('figures/Kuramoto_extra_stim_EC_E_I_only.png')



tight_layout()
show()
