#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from brian2 import *

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
print('   >  Making the neuron groups...')

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
'''
gains_all = [[1 for i in range(2)] for j in range(4)]
gains_all[0][:1] = [1/G]
gains_all[1] = [G for ii in range(2)]
gains_all[2][:1] = [1/G]
gains_all[3][1:] = [G]
'''
gains_all =  [[1/G, 1], [G, G], [1/G, 1], [1, G]]

# intra
'''# EC intra
p_EC_e = [0, 0.37]
p_EC_i = [0.54, 0]
p_EC_all = [p_EC_e, p_EC_i]

# GD intra
p_DG_e = [0, 0.06]
p_DG_i = [0.14, 0]
p_DG_all = [p_DG_e, p_DG_i]

# CA3 intra
p_CA3_e = [0.56, 0.75]
p_CA3_i = [0.75, 0]
p_CA3_all = [p_CA3_e, p_CA3_i]

# CA1 intra
p_CA1_e = [0, 0.28]
p_CA1_i = [0.3, 0.7]
p_CA1_all = [p_CA1_e, p_CA1_i]
'''

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


# Stimulation and other inputs
# -------------------------------------------------------------_#
print('\n >  Inputs and Stimulation...')


# Add the monitors (spikes/rates)
# -------------------------------------------------------------_#
print('\n >  Monitors...')
state_mon_all = []
spike_mon_all = []
rate_mon_all = []

for group in G_flat:
    state_mon_all.append(StateMonitor(group, 'v', record=True))
    spike_mon_all.append(SpikeMonitor(group))
    rate_mon_all.append(PopulationRateMonitor(group))


# Create the Network
# -------------------------------------------------------------_#
print('\n >  Connecting the network...')
net = Network()
net.add(G_flat)
net.add(state_mon_all)
net.add(spike_mon_all)
net.add(rate_mon_all)


# run simulation
# -------------------------------------------------------------_#
global tstep
tstep = defaultclock.dt
inputs_stim = TimedArray([0*mA], dt=0.1*ms)


print('\n >  Starting simulation...')
net.run(duration, report='text', report_period=60*second)
print('Simulation ended')
