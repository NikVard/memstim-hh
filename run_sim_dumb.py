#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from brian2 import *

import time

import argparse
import parameters

from model.globals import *
from model.HH_equations import *
from model import setup

from src.annex_funcs import make_flat


# Configuration
# -------------------------------------------------------------_#
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

# Make the neuron groups
# -------------------------------------------------------------_#
print('\n >  Making the neuron groups...')

G_all = [[[] for pops in range(2)] for areas in range(4)]

# EC
G_E = NeuronGroup(N=N_EC[0],
    model=py_CAN_eqs,
    threshold='v>V_th',
    reset=reset_eqs,
    refractory=refractory_time,
    method=integ_method,
    name='EC_pyCAN')
G_E.size = cell_size_py
G_E.glu = 1

G_I = NeuronGroup(N=N_EC[1],
    model=inh_eqs,
    threshold='v>V_th',
    refractory=refractory_time,
    method=integ_method,
    name='EC_inh')
G_I.size = cell_size_inh
print('EC: done')

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

G_I = NeuronGroup(N=N_DG[1],
    model=inh_eqs,
    threshold='v>V_th',
    refractory=refractory_time,
    method=integ_method,
    name='DG_inh')
G_I.size = cell_size_inh
print('DG: done')

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

G_I = NeuronGroup(N=N_CA3[1],
    model=inh_eqs,
    threshold='v>V_th',
    refractory=refractory_time,
    method=integ_method,
    name='CA3_inh')
G_I.size = cell_size_inh
print('CA3: done')

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

G_I = NeuronGroup(N=N_CA1[1],
    model=inh_eqs,
    threshold='v>V_th',
    refractory=refractory_time,
    method=integ_method,
    name='CA1_inh')
G_I.size = cell_size_inh
print('CA1: done')

G_all[3][0].append(G_E)
G_all[3][1].append(G_I)

G_flat = make_flat(G_all)

for ngroup in G_flat:
    ngroup.v = '-60*mvolt-rand()*10*mvolt' # str -> individual init. val. per neuron
    ngroup.r = 0 # int -> same init. val. for all neurons

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


# Stimulation and other inputs
# -------------------------------------------------------------_#
print('\n >  Inputs and Stimulation...')
dt_stim = 1*ms
I_stim = 1*namp
tv = linspace(0, duration/second, int(duration/(dt_stim))+1)
xstim = I_stim * logical_and(tv>0.1, tv<0.11)
inputs_stim = TimedArray(xstim, dt=dt_stim)

G_all[0][0][0].r = 1 # stim aplpied @ EC-py_CAN
'''for group in [G_all[jj][0][0] for jj in range(4)]:
    group.r = 0
'''
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

'''
for group in G_flat:
    state_mon_all.append(StateMonitor(group, 'v', record=True))
    spike_mon_all.append(SpikeMonitor(group))
    rate_mon_all.append(PopulationRateMonitor(group))

'''

# Create the Network
# -------------------------------------------------------------_#
print('\n >  Connecting the network...')
net = Network()
net.add(G_all) # add groups

for syn_intra_curr in make_flat(syn_intra_all): # add synapses (intra)
    if syn_intra_curr!=0:
        net.add(syn_intra_curr)

for syn_inter_curr in make_flat(syn_inter_all): # add synapses (inter)
    if syn_inter_curr!=0:
        net.add(syn_inter_curr)

net.add(state_mon_E_all) # monitors
net.add(state_mon_I_all)
net.add(spike_mon_E_all)
net.add(spike_mon_I_all)
net.add(rate_mon_E_all)
net.add(rate_mon_I_all)


# run simulation
# -------------------------------------------------------------_#
global tstep
tstep = defaultclock.dt


print('\n >  Starting simulation...')
start = time.time()
net.run(duration, report='text', report_period=60*second, profile=True)
end = time.time()
print('Simulation ended')
print('Simulation ran for '+str((end-start)/60)+' minutes')


# dumb plot, not a function yet
# -------------------------------------------------------------_#
zones = ['EC', 'DG', 'CA3', 'CA1']
types = ['exc', 'inh']

raster_fig = figure()
for ii in range(4):
    subplot(4,2,ii*2+1)
    title('raster '+zones[ii]+' exc')
    plot(spike_mon_E_all[ii][0].t/msecond, spike_mon_E_all[ii][0].i, '.b',
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

tight_layout()
show()


profiling_summary(net=net, show=10)  # show the top 10 objects that took the longest
