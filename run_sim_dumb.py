#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from brian2 import *
from scipy.spatial import distance as dst
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import os
import time
from shutil import copyfile

import argparse
import parameters
import re # regular expressions

from model.globals import *
from model.HH_equations import *
from model.kuramoto_equations import *
from model.filter_equations import *
from model import settings
from model import setup

from src.annex_funcs import make_flat
from src.myplot import *
from src import stimulation


# Configuration
# -------------------------------------------------------------#
# Use C++ standalone code generation TODO: Experimental!
#set_device('cpp_standalone')
parser = argparse.ArgumentParser(description='MemStim using HH neurons')
parser.add_argument('-p', '--parameters',
                    nargs='?',
                    metavar='-p',
                    type=str,
                    default=os.path.join('configs', 'default.json'),
                    help='Parameters file (json format)')
args = parser.parse_args()
filename = args.parameters

try:
    data = parameters.load(filename)
    print('Using "{0}"'.format(filename))
except Exception as e:
    print(e)
    print('Using "default.json"')
    data = parameters._data
parameters.dump(data)
print()
#locals().update(data)

# settings initialization
settings.init(data)

# Create the necessary directories
dirs = {}

dirs['results'] = 'results'
if not os.path.isdir(dirs['results']):
    print('Making directory', dirs['results'])
    os.makedirs(dirs['results'])

if settings.I_stim[0]:
    dirs['stim'] = os.path.join(dirs['results'], '{stimamp:d}_nA'.format(stimamp=int(settings.I_stim[0])))
    if not os.path.isdir(dirs['stim']):
        print('Making directory', dirs['stim'])
        os.makedirs(dirs['stim'])

    dirs['offset'] = os.path.join(dirs['stim'], '{phase:.2f}_{stim_on:.1f}_ms'.format(phase=settings.offset, stim_on=settings.stim_onset*1e3))
    if not os.path.isdir(dirs['offset']):
        print('Making directory', dirs['offset'])
        os.makedirs(dirs['offset'])

    dirs['base'] = dirs['offset']

else:
    dirs['stim'] = os.path.join(dirs['results'], 'None')
    if not os.path.exists(dirs['stim']):
        print('Making directory', dirs['stim'])
        os.makedirs(dirs['stim'])

    dirs['base'] = dirs['stim']

dtime = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S") # imported from brian2
dirs['base'] = os.path.join(dirs['base'], dtime)
if not os.path.isdir(dirs['base']):
    print('Making directory', dirs['base'])
    os.makedirs(dirs['base'])

dirs['figures'] = os.path.join(dirs['base'], 'figures')
if not os.path.isdir(dirs['figures']):
    print('Making directory', dirs['figures'])
    os.makedirs(dirs['figures'])

dirs['data'] = os.path.join(dirs['base'], 'data')
if not os.path.isdir(dirs['data']):
    print('Making directory', dirs['data'])
    os.makedirs(dirs['data'])

dirs['positions'] = os.path.join(dirs['data'], 'positions')
if not os.path.isdir(dirs['positions']):
    print('Making directory', dirs['positions'])
    os.makedirs(dirs['positions'])

dirs['spikes'] = os.path.join(dirs['data'], 'spikes')
if not os.path.isdir(dirs['spikes']):
    print('Making directory', dirs['spikes'])
    os.makedirs(dirs['spikes'])

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
    print(' >  DEBUGGING MODE: ON')
    print('###########################')



# Make the neuron groups
# -------------------------------------------------------------#
print('\n >  Making the neuron groups...')

G_all = [[[] for pops in range(2)] for areas in range(4)]

#fig, axs = subplots(nrows=1, ncols=1)
fig_anat = figure()
ax_anat = fig_anat.add_subplot(111, projection='3d')

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

# EC -> receives theta input from MS
pos = np.load(os.path.join('neuron_positions', 'full', 'EC_E-stipple-10000.npy'))
pos = hstack((pos, zeros((settings.N_EC[0], 1))))
pos = r.apply(pos)
pos *= scale
pos[:,2] += 15*mm*rand(settings.N_EC[0])
idx = np.argsort(pos[:,2]) # sort neurons by increasing z-coordinate
pos = pos[idx]
G_E = NeuronGroup(N=settings.N_EC[0],
    model=py_CAN_inp_eqs,
    threshold='v>V_th',
    reset=reset_eqs,
    refractory=refractory_time,
    method=integ_method,
    name='EC_pyCAN')
G_E.size = cell_size_py
G_E.glu = 1
G_E.x_soma = pos[:,0]*metre
G_E.y_soma = pos[:,1]*metre
G_E.z_soma = pos[:,2]*metre

# Plot X,Y,Z
ax_anat.scatter(G_E.x_soma, G_E.y_soma, G_E.z_soma, c='blue')

pos = np.load(os.path.join('neuron_positions', 'full', 'EC_I-stipple-1000.npy'))
pos = hstack((pos, zeros((settings.N_EC[1], 1)))) # add z-axis
pos = r.apply(pos)
pos *= scale
pos[:,2] += 15*mm*rand(settings.N_EC[1])
idx = np.argsort(pos[:,2]) # sort neurons by increasing z-coordinate
pos = pos[idx]
G_I = NeuronGroup(N=settings.N_EC[1],
    model=inh_inp_eqs,
    threshold='v>V_th',
    refractory=refractory_time,
    method=integ_method,
    name='EC_inh')
G_I.size = cell_size_inh
G_I.x_soma = pos[:,0]*metre
G_I.y_soma = pos[:,1]*metre
G_I.z_soma = pos[:,2]*metre

# Plot X,Y,Z
ax_anat.scatter(G_I.x_soma, G_I.y_soma, G_I.z_soma, c='red')

G_all[0][0].append(G_E)
G_all[0][1].append(G_I)
print('EC: done')


# DG
pos = np.load(os.path.join('neuron_positions', 'full', 'DG_E-stipple-10000.npy'))
pos = hstack((pos, zeros((settings.N_DG[0], 1)))) # add z-axis
pos = r.apply(pos)
pos *= scale
pos[:,2] += 15*mm*rand(settings.N_DG[0])
idx = np.argsort(pos[:,2]) # sort neurons by increasing z-coordinate
pos = pos[idx]
G_E = NeuronGroup(N=settings.N_DG[0],
    model=py_eqs,
    threshold='v>V_th',
    reset=reset_eqs,
    refractory=refractory_time,
    method=integ_method,
    name='DG_py')
G_E.size = cell_size_py
G_E.glu = 1
G_E.x_soma = pos[:,0]*metre
G_E.y_soma = pos[:,1]*metre
G_E.z_soma = pos[:,2]*metre

# Plot X,Y,Z
ax_anat.scatter(G_E.x_soma, G_E.y_soma, G_E.z_soma, c='green')

pos = np.load(os.path.join('neuron_positions', 'full', 'DG_I-stipple-100.npy'))
pos = hstack((pos, zeros((settings.N_DG[1], 1)))) # add z-axis
pos = r.apply(pos)
pos *= scale
pos[:,2] += 15*mm*rand(settings.N_DG[1])
idx = np.argsort(pos[:,2]) # sort neurons by increasing z-coordinate
pos = pos[idx]
G_I = NeuronGroup(N=settings.N_DG[1],
    model=inh_eqs,
    threshold='v>V_th',
    refractory=refractory_time,
    method=integ_method,
    name='DG_inh')
G_I.size = cell_size_inh
G_I.x_soma = pos[:,0]*metre
G_I.y_soma = pos[:,1]*metre
G_I.z_soma = pos[:,2]*metre

# Plot X,Y,Z
ax_anat.scatter(G_I.x_soma, G_I.y_soma, G_I.z_soma, c='red')

G_all[1][0].append(G_E)
G_all[1][1].append(G_I)
print('DG: done')


# CA3
pos = np.load(os.path.join('neuron_positions', 'full', 'CA3_E-stipple-1000.npy'))
pos = hstack((pos, zeros((settings.N_CA3[0], 1)))) # add z-axis
pos = r.apply(pos)
pos *= scale
pos[:,2] += 15*mm*rand(settings.N_CA3[0])
idx = np.argsort(pos[:,2]) # sort neurons by increasing z-coordinate
pos = pos[idx]
G_E = NeuronGroup(N=settings.N_CA3[0],
    model=py_CAN_eqs,
    threshold='v>V_th',
    reset=reset_eqs,
    refractory=refractory_time,
    method=integ_method,
    name='CA3_pyCAN')
G_E.size = cell_size_py
G_E.glu = 1
G_E.x_soma = pos[:,0]*metre
G_E.y_soma = pos[:,1]*metre
G_E.z_soma = pos[:,2]*metre

# Plot X,Y,Z
ax_anat.scatter(G_E.x_soma, G_E.y_soma, G_E.z_soma, c='blue')

pos = np.load(os.path.join('neuron_positions', 'full', 'CA3_I-stipple-100.npy'))
pos = hstack((pos, zeros((settings.N_CA3[1], 1)))) # add z-axis
pos = r.apply(pos)
pos *= scale
pos[:,2] += 15*mm*rand(settings.N_CA3[1])
idx = np.argsort(pos[:,2]) # sort neurons by increasing z-coordinate
pos = pos[idx]
G_I = NeuronGroup(N=settings.N_CA3[1],
    model=inh_eqs,
    threshold='v>V_th',
    refractory=refractory_time,
    method=integ_method,
    name='CA3_inh')
G_I.size = cell_size_inh
G_I.x_soma = pos[:,0]*metre
G_I.y_soma = pos[:,1]*metre
G_I.z_soma = pos[:,2]*metre

# Plot X,Y,Z
ax_anat.scatter(G_I.x_soma, G_I.y_soma, G_I.z_soma, c='red')

G_all[2][0].append(G_E)
G_all[2][1].append(G_I)
print('CA3: done')


# CA1
pos = np.load(os.path.join('neuron_positions', 'full', 'CA1_E-stipple-10000.npy'))
pos = hstack((pos, zeros((settings.N_CA1[0], 1)))) # add z-axis
pos = r.apply(pos)
pos *= scale
pos[:,2] += 15*mm*rand(settings.N_CA1[0])
idx = np.argsort(pos[:,2]) # sort neurons by increasing z-coordinate
pos = pos[idx]
G_E = NeuronGroup(N=settings.N_CA1[0],
    model=py_CAN_eqs,
    threshold='v>V_th',
    reset=reset_eqs,
    refractory=refractory_time,
    method=integ_method,
    name='CA1_pyCAN')
G_E.size = cell_size_py
G_E.glu = 1
G_E.x_soma = pos[:,0]*metre
G_E.y_soma = pos[:,1]*metre
G_E.z_soma = pos[:,2]*metre

# Plot X,Y,Z
ax_anat.scatter(G_E.x_soma, G_E.y_soma, G_E.z_soma, c='blue')

pos = np.load(os.path.join('neuron_positions', 'full', 'CA1_I-stipple-1000.npy'))
pos = hstack((pos, zeros((settings.N_CA1[1], 1)))) # add z-axis
pos = r.apply(pos)
pos *= scale
pos[:,2] += 15*mm*rand(settings.N_CA1[1])
idx = np.argsort(pos[:,2]) # sort neurons by increasing z-coordinate
pos = pos[idx]
G_I = NeuronGroup(N=settings.N_CA1[1],
    model=inh_eqs,
    threshold='v>V_th',
    refractory=refractory_time,
    method=integ_method,
    name='CA1_inh')
G_I.size = cell_size_inh
G_I.x_soma = pos[:,0]*metre
G_I.y_soma = pos[:,1]*metre
G_I.z_soma = pos[:,2]*metre

# Plot X,Y,Z
ax_anat.scatter(G_I.x_soma, G_I.y_soma, G_I.z_soma, c='red')

G_all[3][0].append(G_E)
G_all[3][1].append(G_I)
print('CA1: done')

G_flat = make_flat(G_all)

# initialize the groups, set initial conditions
for ngroup in G_flat:
    #ngroup.z_soma = '15*mm*rand()' # add a third dimension to the structures
    ngroup.v = '-60.*mvolt-rand()*10*mvolt' # str -> individual init. val. per neuron
    #ngroup.v = -60.*mV

    # CA1 populations get stimulated
    if (ngroup.name=='{group}_pyCAN'.format(group=settings.stim_target) or ngroup.name=='{group}_py'.format(group=settings.stim_target)) or ngroup.name=='{group}_inh'.format(group=settings.stim_target):
        print("Stimulation applied @", ngroup.name)

        # calculate the distance
        # ngroup.r = 1 # 1 means on
        # d1 = '{rho}*mm/(4*pi*sqrt((x_soma-{x0}*mm)**2 + (y_soma-{y0}*mm)**2 + (z_soma-{z0}*mm)**2))'.format(rho=settings.stim_rho, x0=settings.stim_coordinates[0], y0=settings.stim_coordinates[1], z0=settings.stim_coordinates[2])

        # alternatively, calculate distances like so:
        neuron_pos = column_stack((ngroup.x_soma/mm, ngroup.y_soma/mm, ngroup.z_soma/mm))
        elec_pos = array(settings.stim_coordinates)[np.newaxis,...]
        d0 = dst.cdist(elec_pos, neuron_pos)
        ngroup.r = 100/(4*pi*d0)
    else:
        ngroup.r = 0 # int -> same init. val. for all neurons


# DEBUGGING DISTANCES
print('Intra-region distances:')
for group in G_flat:
    # organize positions for this specific group
    neuron_pos = column_stack((group.x_soma, group.y_soma, group.z_soma))

    # calculate pair-wise distances using pdist
    dist_res = dst.pdist(neuron_pos, 'euclidean')

    # if using cdist: generate a mask to skip the diagonal for minimum calculation
    #mask = ones(dist_res.shape, dtype=bool)
    #fill_diagonal(mask, False)

    # calculate intra-area distance boundaries
    min_dist, max_dist = (dist_res.min(), dist_res.max())

    print('{:10} pdist: ({:20} , {:20})\n\tx ---> ({:28}, {:28})\n\ty ---> ({:28}, {:28})\n\tz ---> ({:28}, {:28})\n'.format(
    '{}'.format(group.name),
    '{}'.format(min_dist),
    '{}'.format(max_dist),
    '{}'.format(group.x_soma[:].min()),
    '{}'.format(group.x_soma[:].max()),
    '{}'.format(group.y_soma[:].min()),
    '{}'.format(group.y_soma[:].max()),
    '{}'.format(group.z_soma[:].min()),
    '{}'.format(group.z_soma[:].max())))



# Make the synapses
# -------------------------------------------------------------#
print('\n >  Making the synapses...')

# gains
gains_all =  [[1./G, 1.], [G, G], [1./G, 1.], [1., G]]

# intra
print('     * intra-region')

syn_EC_all = setup.connect_intra(G_all[0][0], G_all[0][1], settings.p_EC_all, gains_all[0])
print('EC-to-EC: done')
syn_DG_all = setup.connect_intra(G_all[1][0], G_all[1][1], settings.p_DG_all, gains_all[1])
print('DG-to-DG: done')
syn_CA3_all = setup.connect_intra(G_all[2][0], G_all[2][1], settings.p_CA3_all, gains_all[2])
print('CA3-to-CA3: done')
syn_CA1_all = setup.connect_intra(G_all[3][0], G_all[3][1], settings.p_CA1_all, gains_all[3])
print('CA1-to-CA1: done')
syn_intra_all = [syn_EC_all, syn_DG_all, syn_CA3_all, syn_CA1_all]

# inter
print('     * inter-region')

# p_inter_all = [[[[],[]],[[],[]]],[[[],[]],[[],[]]],[[[],[]],[[],[]]],[[[],[]],[[],[]]]]
p_inter_all = [[[[0,0] for ii in range(2)] for jj in range(4)] for kk in range(4)]
p_inter_all[0][1][0] = [settings.p_tri for ii in range(2)] # EC_E to DG_E | DG_I
p_inter_all[0][2][0] = [settings.p_mono for ii in range(2)] # EC_E to CA3_E | CA3_I
p_inter_all[0][3][0] = [settings.p_mono for ii in range(2)] # EC_E to CA1_E | CA1_I
p_inter_all[1][2][0] = [settings.p_tri for ii in range(2)] # DG_E to CA3_E | CA3_I
p_inter_all[2][3][0] = [settings.p_tri for ii in range(2)] # CA3_E to CA1_E | CA1_I
p_inter_all[3][0][0] = [settings.p_tri for ii in range(2)] # CA1_E to EC_E | EC_I

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
# -------------------------------------------------------------#
print('\n >  Monitors...')
state_mon_all = []
spike_mon_all = []
rate_mon_all = []

state_mon_E_all = [[StateMonitor(G_py, ['v'], record=True) for G_py in G_all[i][0] if G_py] for i in range(4)]
state_mon_I_all = [[StateMonitor(G_inh, ['v'], record=True) for G_inh in G_all[i][1] if G_inh] for i in range(4)]
print('State monitors [v]: done')

spike_mon_E_all = [[SpikeMonitor(G_py, name=G_py.name+'_spikemon') for G_py in G_all[i][0] if G_py] for i in range(4)]
spike_mon_I_all = [[SpikeMonitor(G_inh, name=G_inh.name+'_spikemon') for G_inh in G_all[i][1] if G_inh] for i in range(4)]
print('Spike monitors: done')

rate_mon_E_all = [[PopulationRateMonitor(G_py) for G_py in G_all[i][0] if G_py] for i in range(4)]
rate_mon_I_all = [[PopulationRateMonitor(G_inh) for G_inh in G_all[i][1] if G_inh] for i in range(4)]
print('Rate monitors: done')

state_mon_noise_all = [StateMonitor(G, ['noise'], record=True) for G in G_flat]



# Stimulation and other inputs
# -------------------------------------------------------------#
print('\n >  Inputs and Stimulation...')
tv = linspace(0, settings.duration/second, int(settings.duration/(settings.stim_dt))+1)
# xstim = settings.I_stim * logical_and(tv>settings.t_stim/second, tv<settings.t_stim/second+0.01)
# inputs_stim = TimedArray(xstim, dt=settings.dt_stim)

# generate stimulation signal
if settings.I_stim[0]:
    print("Stimulation ON")
    xstim = stimulation.generate_stim(duration=settings.stim_duration,
                                      dt=settings.stim_dt,
                                      I_stim=settings.I_stim,
                                      stim_on=settings.stim_onset,
                                      nr_of_trains=settings.nr_of_trains,
                                      nr_of_pulses=settings.nr_of_pulses,
                                      stim_freq=settings.stim_freq,
                                      pulse_width=settings.pulse_width,
                                      pulse_freq=settings.pulse_freq,
                                      ipi=settings.stim_ipi)
    # inputs_stim = TimedArray(values=xstim*nA, dt=settings.stim_dt*second, name='Input_stim')
else:
    print("No stimulation defined; using empty TimedArray")
    xstim = zeros(int(settings.stim_duration/settings.stim_dt))

inputs_stim = TimedArray(values=xstim*nA, dt=settings.stim_dt*second, name='Input_stim')


# inp_theta_sin = 1*sin(2*pi*4*tv)
# inp_theta_rect = (-cos(2*pi*4*tv)+1)/2
# trail_zeros = zeros(int(250*ms/(settings.stim_dt*second)))
# inp_theta_slow = concatenate((trail_zeros, inp_theta_rect))
# inp_theta = TimedArray(inp_theta_slow*nA, dt=settings.stim_dt*second) # external theta (TESTING)



# Kuramoto Oscillators (MS)
# -------------------------------------------------------------#
print('\n >  Kuramoto Oscillators...')

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
G_K.gain = settings.k_gain
G_K.offset = settings.offset
G_flat.append(G_K) # append to the group list!
print('Group: done')

syn_kuramoto =  Synapses(G_K, G_K, on_pre=syn_kuramoto_eqs, method='euler', name='Kuramoto_intra')
syn_kuramoto.connect(condition='i!=j')
print('Synapses: done')

# Kuramoto order parameter group
G_pop_avg = NeuronGroup(1,
    model=pop_avg_eqs,
    #method='euler',
    name='Kuramoto_averaging')
r0 = 1/settings.N_Kur * sum(exp(1j*G_K.Theta))
G_pop_avg.x = real(r0)  # avoid division by zero
G_pop_avg.y = imag(r0)
G_flat.append(G_pop_avg) # append to the group list!
syn_avg = Synapses(G_K, G_pop_avg, syn_avg_eqs, name='Kuramoto_avg')
syn_avg.connect()
print('Order parameter group: done')



# Firing Rate Filter Population
# -------------------------------------------------------------#
# Make the spikes-to-rates group
print('\n >  Spikes-to-Rates Filter...')

G_S2R = NeuronGroup(1,
    model=firing_rate_filter_eqs,
    method='exact',
    #method='integ_method',
    name='S2R_filter',
    namespace=filter_params)
G_S2R.Y = 0 # initial conditions
G_flat.append(G_S2R) # append to the group list!
print('Group: done')



# Connections
# -------------------------------------------------------------#
print('\n >  Connections...')

# CA1 spikes-to-rates synapses
# find the CA1-E group
G_CA1_E = None
for g in G_flat:
    if g.name=='CA1_pyCAN':
        G_CA1_E = g
        break

# connect the CA1-E group to the low-pass-filter spikes-2-rates (S2R) group
if G_CA1_E:
    syn_CA1_2_rates = Synapses(G_CA1_E, G_S2R, on_pre='Y_post += (1/tauFR)/N_incoming', namespace=filter_params)
    syn_CA1_2_rates.connect()
print('CA1-to-S2R: done')


# connect the S2R group to the Kuramoto oscillators by linking input X to firing rates (drive)
G_K.X = linked_var(G_S2R, 'drive')
print('Linking S2R to Kuramoto oscillators: done')

# connect the Kuramoto ensemble rhythm to the I_exc variable in EC_E and EC_I (Kuramoto output as input to EC_E/I pop.)
'''for g in G_flat:
    if g.name=='EC_pyCAN' or g.name=='EC_inh':
        print('>> Setting input rhythm for group ', g.name)
        g.I_exc = linked_var(G_pop_avg, 'rhythm_rect')
'''
# avoid linking when using a fixed theta input sin : TESTING
# G_flat[0].I_exc = linked_var(G_pop_avg, 'rhythm_zero')
# G_flat[1].I_exc = linked_var(G_pop_avg, 'rhythm_zero')
G_flat[0].I_exc = linked_var(G_pop_avg, 'rhythm_rect')
G_flat[1].I_exc = linked_var(G_pop_avg, 'rhythm_rect')



# Monitors
# -------------------------------------------------------------#
# Kuramoto monitors
print('\n >  Kuramoto and Filter Monitors...')

kuramoto_mon = StateMonitor(G_K, ['Theta'], record=True)
order_param_mon = StateMonitor(G_pop_avg, ['coherence', 'phase', 'rhythm', 'rhythm_rect'], record=True)
print('State monitor [Theta]: done')

# spikes2rates monitor (vout)
s2r_mon = StateMonitor(G_S2R, ['drive'], record=True)
print('State monitor [drive]: done')

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
# -------------------------------------------------------------#
print('\n >  Connecting the network...')
net = Network()
net.add(G_all) # add groups
net.add(G_K)
net.add(G_pop_avg)
net.add(G_S2R)
print('Network groups: done')

for syn_intra_curr in make_flat(syn_intra_all): # add synapses (intra)
    if syn_intra_curr!=0:
        net.add(syn_intra_curr)

for syn_inter_curr in make_flat(syn_inter_all): # add synapses (inter)
    if syn_inter_curr!=0:
        net.add(syn_inter_curr)

net.add(syn_kuramoto) # kuramoto intra-synapses
net.add(syn_avg) # kuramoto population average (order parameter) synapses
net.add(syn_CA1_2_rates) # CA1 spikes2rates
print('Network connections: done')

net.add(state_mon_E_all) # monitors
net.add(state_mon_I_all)
net.add(spike_mon_E_all)
net.add(spike_mon_I_all)
net.add(rate_mon_E_all)
net.add(rate_mon_I_all)
net.add(kuramoto_mon)
net.add(order_param_mon)
net.add(s2r_mon)
net.add(state_mon_noise_all)
print('Network monitors: done')



# Run the simulation
# -------------------------------------------------------------#
#defaultclock.dt = 0.01*ms
tstep = defaultclock.dt

print('\n >  Starting simulation...')
start = time.time()
net.run(settings.duration, report='text', report_period=10*second, profile=True)
end = time.time()
print('Simulation ended')
print('Simulation ran for '+str((end-start)/60)+' minutes')
print(profiling_summary(net=net, show=4)) # show the top 10 objects that took the longest



# Plot the results
# -------------------------------------------------------------#
# raster plot of all regions
raster_fig, raster_axs, fig_name = plot_raster_all(spike_mon_E_all, spike_mon_I_all)
print("Saving figure 'figures/%s'" %fig_name)
plot_watermark(raster_fig, os.path.basename(__file__), filename, settings.git_branch, settings.git_short_hash)
raster_fig.savefig(os.path.join(dirs['figures'], fig_name))

'''
# calculate order parameter in the end
samples = len(kuramoto_mon.Theta[0])
r = np.zeros(samples, dtype='complex')
for s in range(samples):
    r[s] = 1/N_Kur * sum(exp(1j*kuramoto_mon.Theta[:,s])) # order parameter r(t)
'''

# kuramoto order parameter plots
kuramoto_fig, kuramoto_axs, fig_name = plot_kuramoto(order_param_mon)
plot_watermark(kuramoto_fig, os.path.basename(__file__), filename, settings.git_branch, settings.git_short_hash)
print("Saving figure 'figures/%s'" %fig_name)
kuramoto_fig.savefig(dirs['figures'] + fig_name)

# Plot more stuff
fig_extra, extra_axs, fig_name = plot_network_output(spike_mon_E_all[-1][0], spike_mon_I_all[-1][0], s2r_mon, order_param_mon, tv, xstim)
plot_watermark(fig_extra, os.path.basename(__file__), filename, settings.git_branch, settings.git_short_hash)
print("Saving figure 'figures/%s'" %fig_name)
fig_extra.savefig(os.path.join(dirs['figures'], fig_name))

tight_layout()
#show()


# Save the results as .txt files (rows: time | cols: data)
# -------------------------------------------------------------#
# Kuramoto monitors
np.savetxt(os.path.join(dirs['data'], 'kuramoto_mon_Theta.txt'), kuramoto_mon.Theta.T, fmt='%.8f')
np.savetxt(os.path.join(dirs['data'], 'order_param_mon_phase.txt'), order_param_mon.phase.T, fmt='%.8f')
np.savetxt(os.path.join(dirs['data'], 'order_param_mon_rhythm.txt'), order_param_mon.rhythm_rect.T/nA, fmt='%.8f')
np.savetxt(os.path.join(dirs['data'], 'order_param_mon_coherence.txt'), order_param_mon.coherence.T, fmt='%.8f')

# CA1 firing rate
np.savetxt(os.path.join(dirs['data'], 'rate_mon_E_CA1.txt'), rate_mon_E_all[3][0].smooth_rate(window='gaussian', width=50*ms).T/Hz, fmt='%.8f')
np.savetxt(os.path.join(dirs['data'], 's2r_mon_drive.txt'), s2r_mon.drive_.T, fmt='%.8f')

# External stimulus
np.savetxt(os.path.join(dirs['data'], 'stim_input.txt'), xstim, fmt='%.2f')



# Save the spikes and their times
# -------------------------------------------------------------#
print("\nSaving spikes in time....")
SM_i = []
SM_t = []
for SM in make_flat([spike_mon_E_all, spike_mon_I_all]):
    for i_val in SM.i:
        SM_i.append(i_val)

    for t_val in SM.t:
        SM_t.append(t_val/msecond)

    fname = SM.name
    np.savetxt(os.path.join(dirs['spikes'], fname, '_i.txt'), np.array(SM_i).astype(np.int16), fmt='%d')
    np.savetxt(os.path.join(dirs['spikes'], fname, '_t.txt'), np.array(SM_t).astype(np.float32), fmt='%.1f')

    SM_i.clear()
    SM_t.clear()


# Save the positions of the neurons in npy files
# -------------------------------------------------------------#
print("\nSaving neuron positions...")
for G in G_flat:
    try:
        print('Current group: ', G.name)
        fname = '{group}'.format(group=G.name)
        pos = np.array([G.x_soma, G.y_soma, G.z_soma]).T
        np.save(dirs['positions'] + fname, pos)

    except AttributeError:
        print('pass...')
        continue
