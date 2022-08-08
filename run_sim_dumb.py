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
from model.fixed_input_equations import *
from model.Vm_avg_eqs import *
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
                    metavar='-p',
                    type=str,
                    default=os.path.join('configs', 'default.json'),
                    help='Parameters file (json format)')

parser.add_argument('-sd', '--save_dir',
                    nargs='?',
                    metavar='-sd',
                    type=str,
                    default='results',
                    help='Destination directory to save the results')

args = parser.parse_args()
filename = args.parameters
resdir = args.save_dir

try:
    data = parameters.load(filename)
    print('Using "{0}"'.format(filename))
except Exception as e:
    print(bcolors.RED + '[!]' + e + ' Using "default.json"' + bcolors.ENDC)
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
    dirs['stim'] = os.path.join(dirs['results'], '{stimamp:d}_nA'.format(stimamp=round(settings.I_stim[0],1)))
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

print('[+] Groups:')
# EC -> receives theta input from MS
# E
pos = np.load(os.path.join('neuron_positions', 'full', 'EC_E-stipple-10000.npy'))
pos = hstack((pos, zeros((settings.N_EC[0], 1))))
pos = r.apply(pos)
pos *= scale
pos[:,2] += 15*mm*rand(settings.N_EC[0])
# pos = parse_positions(os.path.join('positions', 'EC_exc.txt'))
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
G_E.sigma = settings.sigma_EC[0]*uvolt
G_E.x_soma = pos[:,0]*metre
G_E.y_soma = pos[:,1]*metre
G_E.z_soma = pos[:,2]*metre
# G_E.x_soma = pos[:,0]*scale_aussel
# G_E.y_soma = pos[:,1]*scale_aussel
# G_E.z_soma = pos[:,2]*scale_aussel

# I
pos = np.load(os.path.join('neuron_positions', 'full', 'EC_I-stipple-1000.npy'))
pos = hstack((pos, zeros((settings.N_EC[1], 1)))) # add z-axis
pos = r.apply(pos)
pos *= scale
pos[:,2] += 15*mm*rand(settings.N_EC[1])
# pos = parse_positions(os.path.join('positions', 'EC_inh.txt'))
idx = np.argsort(pos[:,2]) # sort neurons by increasing z-coordinate
pos = pos[idx]
G_I = NeuronGroup(N=settings.N_EC[1],
    model=inh_inp_eqs,
    threshold='v>V_th',
    refractory=refractory_time,
    method=integ_method,
    name='EC_inh')
G_I.size = cell_size_inh
G_I.sigma = settings.sigma_EC[1]*uvolt
G_I.x_soma = pos[:,0]*metre
G_I.y_soma = pos[:,1]*metre
G_I.z_soma = pos[:,2]*metre
# G_I.x_soma = pos[:,0]*scale_aussel
# G_I.y_soma = pos[:,1]*scale_aussel
# G_I.z_soma = pos[:,2]*scale_aussel

# Plot X,Y,Z
ax_anat.scatter(G_E.x_soma, G_E.y_soma, G_E.z_soma, c='blue')
ax_anat.scatter(G_I.x_soma, G_I.y_soma, G_I.z_soma, c='red')

# Add to list
G_all[0][0].append(G_E)
G_all[0][1].append(G_I)
print('[\u2022]\tEC: done')


# DG
# E
pos = np.load(os.path.join('neuron_positions', 'full', 'DG_E-stipple-10000.npy'))
pos = hstack((pos, zeros((settings.N_DG[0], 1)))) # add z-axis
pos = r.apply(pos)
pos *= scale
pos[:,2] += 15*mm*rand(settings.N_DG[0])
# pos = parse_positions(os.path.join('positions', 'DG_exc.txt'))
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
G_E.sigma = settings.sigma_DG[0]*uvolt
G_E.x_soma = pos[:,0]*metre
G_E.y_soma = pos[:,1]*metre
G_E.z_soma = pos[:,2]*metre
# G_E.x_soma = pos[:,0]*scale_aussel
# G_E.y_soma = pos[:,1]*scale_aussel
# G_E.z_soma = pos[:,2]*scale_aussel

# I
pos = np.load(os.path.join('neuron_positions', 'full', 'DG_I-stipple-100.npy'))
pos = hstack((pos, zeros((settings.N_DG[1], 1)))) # add z-axis
pos = r.apply(pos)
pos *= scale
pos[:,2] += 15*mm*rand(settings.N_DG[1])
# pos = parse_positions(os.path.join('positions', 'DG_inh.txt'))
idx = np.argsort(pos[:,2]) # sort neurons by increasing z-coordinate
pos = pos[idx]
G_I = NeuronGroup(N=settings.N_DG[1],
    model=inh_eqs,
    threshold='v>V_th',
    refractory=refractory_time,
    method=integ_method,
    name='DG_inh')
G_I.size = cell_size_inh
G_I.sigma = settings.sigma_DG[1]*uvolt
G_I.x_soma = pos[:,0]*metre
G_I.y_soma = pos[:,1]*metre
G_I.z_soma = pos[:,2]*metre
# G_I.x_soma = pos[:,0]*scale_aussel
# G_I.y_soma = pos[:,1]*scale_aussel
# G_I.z_soma = pos[:,2]*scale_aussel

# Plot X,Y,Z
ax_anat.scatter(G_E.x_soma, G_E.y_soma, G_E.z_soma, c='green')
ax_anat.scatter(G_I.x_soma, G_I.y_soma, G_I.z_soma, c='red')

# Add to list
G_all[1][0].append(G_E)
G_all[1][1].append(G_I)
print('[\u2022]\tDG: done')


# CA3
# E
pos = np.load(os.path.join('neuron_positions', 'full', 'CA3_E-stipple-1000.npy'))
pos = hstack((pos, zeros((settings.N_CA3[0], 1)))) # add z-axis
pos = r.apply(pos)
pos *= scale
pos[:,2] += 15*mm*rand(settings.N_CA3[0])
# pos = parse_positions(os.path.join('positions', 'CA3_exc.txt'))
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
G_E.sigma = settings.sigma_CA3[0]*uvolt
G_E.x_soma = pos[:,0]*metre
G_E.y_soma = pos[:,1]*metre
G_E.z_soma = pos[:,2]*metre
# G_E.x_soma = pos[:,0]*scale_aussel
# G_E.y_soma = pos[:,1]*scale_aussel
# G_E.z_soma = pos[:,2]*scale_aussel

# I
pos = np.load(os.path.join('neuron_positions', 'full', 'CA3_I-stipple-100.npy'))
pos = hstack((pos, zeros((settings.N_CA3[1], 1)))) # add z-axis
pos = r.apply(pos)
pos *= scale
pos[:,2] += 15*mm*rand(settings.N_CA3[1])
# pos = parse_positions(os.path.join('positions', 'CA3_inh.txt'))
idx = np.argsort(pos[:,2]) # sort neurons by increasing z-coordinate
pos = pos[idx]
G_I = NeuronGroup(N=settings.N_CA3[1],
    model=inh_eqs,
    threshold='v>V_th',
    refractory=refractory_time,
    method=integ_method,
    name='CA3_inh')
G_I.size = cell_size_inh
G_I.sigma = settings.sigma_CA3[1]*uvolt
G_I.x_soma = pos[:,0]*metre
G_I.y_soma = pos[:,1]*metre
G_I.z_soma = pos[:,2]*metre
# G_I.x_soma = pos[:,0]*scale_aussel
# G_I.y_soma = pos[:,1]*scale_aussel
# G_I.z_soma = pos[:,2]*scale_aussel

# Plot X,Y,Z
ax_anat.scatter(G_E.x_soma, G_E.y_soma, G_E.z_soma, c='blue')
ax_anat.scatter(G_I.x_soma, G_I.y_soma, G_I.z_soma, c='red')

# Add to list
G_all[2][0].append(G_E)
G_all[2][1].append(G_I)
print('[\u2022]\tCA3: done')


# CA1
# E
pos = np.load(os.path.join('neuron_positions', 'full', 'CA1_E-stipple-10000.npy'))
pos = hstack((pos, zeros((settings.N_CA1[0], 1)))) # add z-axis
pos = r.apply(pos)
pos *= scale
pos[:,2] += 15*mm*rand(settings.N_CA1[0])
# pos = parse_positions(os.path.join('positions', 'CA1_exc.txt'))
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
G_E.sigma = settings.sigma_CA1[0]*uvolt
G_E.x_soma = pos[:,0]*metre
G_E.y_soma = pos[:,1]*metre
G_E.z_soma = pos[:,2]*metre
# G_E.x_soma = pos[:,0]*scale_aussel
# G_E.y_soma = pos[:,1]*scale_aussel
# G_E.z_soma = pos[:,2]*scale_aussel

# I
pos = np.load(os.path.join('neuron_positions', 'full', 'CA1_I-stipple-1000.npy'))
pos = hstack((pos, zeros((settings.N_CA1[1], 1)))) # add z-axis
pos = r.apply(pos)
pos *= scale
pos[:,2] += 15*mm*rand(settings.N_CA1[1])
# pos = parse_positions(os.path.join('positions', 'CA1_inh.txt'))
idx = np.argsort(pos[:,2]) # sort neurons by increasing z-coordinate
pos = pos[idx]
G_I = NeuronGroup(N=settings.N_CA1[1],
    model=inh_eqs,
    threshold='v>V_th',
    refractory=refractory_time,
    method=integ_method,
    name='CA1_inh')
G_I.size = cell_size_inh
G_I.sigma = settings.sigma_CA1[1]*uvolt
G_I.x_soma = pos[:,0]*metre
G_I.y_soma = pos[:,1]*metre
G_I.z_soma = pos[:,2]*metre
# G_I.x_soma = pos[:,0]*scale_aussel
# G_I.y_soma = pos[:,1]*scale_aussel
# G_I.z_soma = pos[:,2]*scale_aussel

# Plot X,Y,Z
ax_anat.scatter(G_E.x_soma, G_E.y_soma, G_E.z_soma, c='blue')
ax_anat.scatter(G_I.x_soma, G_I.y_soma, G_I.z_soma, c='red')

# Add to list
G_all[3][0].append(G_E)
G_all[3][1].append(G_I)
print('[\u2022]\tCA1: done')

# Flatten
G_flat = make_flat(G_all)

# initialize the groups, set initial conditions
for ngroup in G_flat:
    ngroup.v = '-60.*mvolt-rand()*10*mvolt' # str -> individual init. val. per neuron, randn is Gaussian

    # CA1 populations get stimulated
    if (ngroup.name=='{group}_pyCAN'.format(group=settings.stim_target) or ngroup.name=='{group}_py'.format(group=settings.stim_target)) or ngroup.name=='{group}_inh'.format(group=settings.stim_target):
        print("[!] Stimulation applied @", ngroup.name)

        # calculate the distance
        # ngroup.r = 1 # 1 means on
        # d0 = '1/({sigma}*(siemens/metre)*4*pi*sqrt((x_soma-{x0}*mm)**2 + (y_soma-{y0}*mm)**2 + (z_soma-{z0}*mm)**2))'.format(rho=settings.stim_sigma, x0=settings.stim_coordinates[0], y0=settings.stim_coordinates[1], z0=settings.stim_coordinates[2])

        # alternatively, calculate distances like so:
        neuron_pos = column_stack((ngroup.x_soma/mm, ngroup.y_soma/mm, ngroup.z_soma/mm))
        elec_pos = array(settings.stim_coordinates)[np.newaxis,...]
        d0 = dst.cdist(elec_pos, neuron_pos)
        # ngroup.r = clip(100/(4*pi*d0), a_min=0, amax=1) # clip the values, r is NOT a gain
        ngroup.r = 1
    else:
        ngroup.r = 0 # int -> same init. val. for all neurons

    # neuron_pos = column_stack((ngroup.x_soma/mm, ngroup.y_soma/mm, ngroup.z_soma/mm))
    # elec_pos = array(settings.stim_coordinates)[np.newaxis,...]
    # d0 = dst.cdist(elec_pos, neuron_pos)
    # ngroup.r = clip(100/(4*pi*d0), a_min=0, a_max=1) # clip the values, r is NOT a gain


# DEBUGGING DISTANCES
print('\n[11] Intra-region distances...')
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
print('\n[12] Making the synapses...')

# gains
gains_all =  [[1./G, 1.], [G, G], [1./G, 1.], [1., G]]
print("[!] Gains:", gains_all)

# intra
print('[+] Intra-region')

syn_EC_all = setup.connect_intra(G_all[0][0], G_all[0][1], settings.p_EC_all, gains_all[0])
print('[\u2022]\tEC-to-EC: done')

syn_DG_all = setup.connect_intra(G_all[1][0], G_all[1][1], settings.p_DG_all, gains_all[1])
print('[\u2022]\tDG-to-DG: done')

syn_CA3_all = setup.connect_intra(G_all[2][0], G_all[2][1], settings.p_CA3_all, gains_all[2])
print('[\u2022]\tCA3-to-CA3: done')

syn_CA1_all = setup.connect_intra(G_all[3][0], G_all[3][1], settings.p_CA1_all, gains_all[3])
print('[\u2022]\tCA1-to-CA1: done')

syn_intra_all = [syn_EC_all, syn_DG_all, syn_CA3_all, syn_CA1_all]

# inter
print('[+] Inter-region')

syn_EC_DG_all = setup.connect_inter(G_all[0][0], G_all[1][0], G_all[1][1], settings.p_inter_all[0][1], gains_all[0])
syn_EC_CA3_all = setup.connect_inter(G_all[0][0], G_all[2][0], G_all[2][1], settings.p_inter_all[0][2], gains_all[0])
syn_EC_CA1_all = setup.connect_inter(G_all[0][0], G_all[3][0], G_all[3][1], settings.p_inter_all[0][3], gains_all[0])
print('[\u2022]\tEC-to-all: done')

syn_DG_CA3_all = setup.connect_inter(G_all[1][0], G_all[2][0], G_all[2][1], settings.p_inter_all[1][2], gains_all[1])
print('[\u2022]\tDG-to-CA3: done')

syn_CA3_CA1_all = setup.connect_inter(G_all[2][0], G_all[3][0], G_all[3][1], settings.p_inter_all[2][3], gains_all[2])
print('[\u2022]\tCA3-to-CA1: done')

syn_CA1_EC_all = setup.connect_inter(G_all[3][0], G_all[0][0], G_all[0][1], settings.p_inter_all[3][0], gains_all[3])
print('[\u2022]\tCA1-to-EC: done')

syn_inter_all = [syn_EC_DG_all, syn_EC_CA3_all, syn_EC_CA1_all, syn_DG_CA3_all, syn_CA3_CA1_all, syn_CA1_EC_all]



# Add the monitors (spikes/rates)
# -------------------------------------------------------------#
print('\n[13] Adding monitors...')
state_mon_all = []
spike_mon_all = []
rate_mon_all = []

state_mon_E_all = [[StateMonitor(G_py, ['v'], record=True) for G_py in G_all[i][0] if G_py] for i in range(4)]
state_mon_I_all = [[StateMonitor(G_inh, ['v'], record=True) for G_inh in G_all[i][1] if G_inh] for i in range(4)]
print('[\u2022]\tState monitors [v]: done')

spike_mon_E_all = [[SpikeMonitor(G_py, name=G_py.name+'_spikemon') for G_py in G_all[i][0] if G_py] for i in range(4)]
spike_mon_I_all = [[SpikeMonitor(G_inh, name=G_inh.name+'_spikemon') for G_inh in G_all[i][1] if G_inh] for i in range(4)]
print('[\u2022]\tSpike monitors: done')

rate_mon_E_all = [[PopulationRateMonitor(G_py) for G_py in G_all[i][0] if G_py] for i in range(4)]
rate_mon_I_all = [[PopulationRateMonitor(G_inh) for G_inh in G_all[i][1] if G_inh] for i in range(4)]
print('[\u2022]\tRate monitors: done')

# state_mon_noise_all = [StateMonitor(G, ['noise'], record=True) for G in G_flat]
# print('[\u2022]\tNoise monitors: done')

# state_mon_Vm_EC_exc = StateMonitor(G_all[0][0][0], ['v'], record=np.concatenate((np.arange(0,10), np.arange(5000, 5010), np.arange(9000, 9010))))
# state_mon_Vm_EC_inh = StateMonitor(G_all[0][1][0], ['v'], record=np.concatenate((np.arange(0,10), np.arange(500, 510), np.arange(900, 910))))
# state_mon_Vm_CA1_exc = StateMonitor(G_all[3][0][0], ['v'], record=np.concatenate((np.arange(0,10), np.arange(5000, 5010), np.arange(9000,9010))))
# state_mon_Vm_CA1_inh = StateMonitor(G_all[3][1][0], ['v'], record=np.concatenate((np.arange(0,10), np.arange(500, 510), np.arange(900, 910))))
# print('[\u2022]\tVm monitors: done')

# state_mon_EC_ICAN = StateMonitor(G_all[0][0][0], ['I_CAN'], record=[0])
# print('[\u2022]\tEC I_CAN monitor: done')



# Add groups for monitoring the avg Vm
# -------------------------------------------------------------#
# Average Vm per group per area
print('\n[20] Vm average groups...')
print('-'*32)

G_Vm_avg = [[[] for pops in range(2)] for areas in range(4)]
syn_Vm_avg_all = [[[] for pops in range(2)] for areas in range(4)]
state_mon_Vm_avg = [[[] for pops in range(2)] for areas in range(4)]

for area_idx in range(4):
    print('[+] {0}:'.format(areas[area_idx]))

    G_E = NeuronGroup(1, eq_record_neurons, name='Vm_avg_{0}_E'.format(areas[area_idx]))
    G_I = NeuronGroup(1, eq_record_neurons, name='Vm_avg_{0}_I'.format(areas[area_idx]))
    G_Vm_avg[area_idx][0].append(G_E)
    G_Vm_avg[area_idx][1].append(G_I)
    print('[\u2022]\tNeuronGroups: done')

    Syn_E = Synapses(G_all[area_idx][0][0], G_E, model=eq_record_synapses)
    Syn_E.connect()
    Syn_I = Synapses(G_all[area_idx][1][0], G_I, model=eq_record_synapses)
    Syn_I.connect()
    syn_Vm_avg_all[area_idx][0].append(Syn_E)
    syn_Vm_avg_all[area_idx][1].append(Syn_I)
    print('[\u2022]\tSynapses: done')

    SM_Vm_avg_E = StateMonitor(G_E, ['sum_v'], record=True, name='Vm_avg_mon_{0}_E'.format(areas[area_idx]))
    SM_Vm_avg_I = StateMonitor(G_I, ['sum_v'], record=True, name='Vm_avg_mon_{0}_I'.format(areas[area_idx]))
    state_mon_Vm_avg[area_idx][0].append(SM_Vm_avg_E)
    state_mon_Vm_avg[area_idx][1].append(SM_Vm_avg_I)
    print('[\u2022]\tMonitors: done')



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
print('[\u2022]\tConnecting CA1-to-S2R: done')

# spikes2rates monitor (vout)
print('\n[32] Adding monitors...')
state_mon_s2r = StateMonitor(G_S2R, ['drive'], record=True)
print('[\u2022]\tState monitor [drive]: done')



# Inputs
# -------------------------------------------------------------#
print('\n[40] Inputs...')
print('-'*32)

G_inputs = []
syn_inputs = []
state_mon_inputs = []

print('[+] Time-vector')
tv = linspace(0, settings.duration/second, int(settings.duration/(settings.dt))+1)

if settings.fixed_input_enabled:
    # Fixed input
    print('\n[41] Making the fixed input group...')

    A0 = settings.fixed_input_low
    A1 = settings.fixed_input_high
    f_rhythm = settings.fixed_input_frequency

    print(bcolors.YELLOW + '[!]' + bcolors.ENDC + ' Using fixed input at %dHz | range [%d, %d]' % (f_rhythm, A0, A1))

    # Make the input TimedArray
    inp_theta_rect = (A1-A0)*(sin(2*pi*f_rhythm/Hz*tv-pi/2)+1)/2+A0
    trail_zeros = zeros(int(settings.fixed_input_delay/(settings.dt*second)))
    inp_theta_delayed = concatenate((trail_zeros, inp_theta_rect/nA))
    inp_theta = TimedArray(inp_theta_delayed*nA, dt=settings.dt) # external theta (TESTING)
    # THERE WAS A 0.5 GAIN HERE!!!

    # Make the input group and append it to the list
    G_input = NeuronGroup(1, model=fixed_input_TA_eqs,
            threshold='False',
            method='euler',
            name='Fixed_input_%dHz' % f_rhythm)
    G_inputs.append(G_input)
    print('[\u2022]\tFixed input group: done')

    # state monitor
    state_mon_theta_rhytm = StateMonitor(G_input, ['rhythm'], record=True)
    state_mon_inputs.append(state_mon_theta_rhytm)
    print('[\u2022]\tState monitor [rhythm]: done')

else:
    # Kuramoto Oscillators
    print('\n[41] Making the Kuramoto oscillators group...')

    print(bcolors.YELLOW + '[!]' + bcolors.ENDC + ' Using dynamic input; Kuramoto oscillators of size N=%d w/ f0 = %dHz | rhythm gain: %d' % (settings.N_Kur, settings.f0, settings.r_gain))

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
    syn_inputs.append(syn_kuramoto)
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
    G_input = G_pop_avg # G_input as alias for G_pop_avg
    G_inputs.append(G_input) # append to the group list!
    print('[\u2022]\tOrder parameter group: done')

    syn_avg = Synapses(G_K, G_pop_avg, syn_avg_eqs, name='Kuramoto_avg')
    syn_avg.connect()
    syn_inputs.append(syn_avg)
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
    state_mon_inputs.append(state_mon_kuramoto)
    state_mon_inputs.append(state_mon_order_param)
    print('[\u2022]\tState monitor [Theta]: done')


# connect the fixed input to the I_exc variable in EC_E and EC_I
for g in G_flat:
    if g.name=='EC_pyCAN' or g.name=='EC_inh':
        print('[+] Linking input (theta rhythm) to group ', g.name)
        g.I_exc = linked_var(G_input, 'rhythm')
print('[\u2022]\tLinking input rhythm: done')
print('[\u2022]\tInputs: done')



# Stimulation and other inputs
# -------------------------------------------------------------#
print('\n[50] Stimulation...')
print('-'*32)

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
    xstim = zeros(tv.shape)
    tv_stim = tv

inputs_stim = TimedArray(values=xstim*nA, dt=settings.stim_dt*second, name='Input_stim')



# Create the Network
# -------------------------------------------------------------#
print('\n[60] Connecting the network...')
print('-'*32)

net = Network()
net.add(G_all) # add groups
net.add(G_inputs)
net.add(G_Vm_avg)
print('[\u2022]\tNetwork groups: done')

for syn_intra_curr in make_flat(syn_intra_all): # add synapses (intra)
    if syn_intra_curr != 0:
        net.add(syn_intra_curr)

for syn_inter_curr in make_flat(syn_inter_all): # add synapses (inter)
    if syn_inter_curr != 0:
        net.add(syn_inter_curr)

for syn_Vm_avg in make_flat(syn_Vm_avg_all): # add synapses Vm avg
    if syn_Vm_avg !=0:
        net.add(syn_Vm_avg)

net.add(syn_inputs) # synapses avout inputs
print('[\u2022]\tNetwork connections: done')

net.add(state_mon_E_all) # monitors
net.add(state_mon_I_all)
net.add(spike_mon_E_all)
net.add(spike_mon_I_all)
net.add(rate_mon_E_all)
net.add(rate_mon_I_all)
net.add(state_mon_inputs)
net.add(state_mon_Vm_avg)
print('[\u2022]\tNetwork monitors: done')



# Run the simulation
# -------------------------------------------------------------#
defaultclock.dt = settings.dt
tstep = defaultclock.dt

print('\n[70] Starting simulation...')
print('-'*32)

start = time.time()
t_elapsed = 0*ms
net.run(settings.duration, report='text', report_period=10*second, profile=True)

# # Theta OFF
# print("[+] Theta off; nothing happening")
# G_K.kN = 0
# G_pop_avg.G_out = 0.
# t_run = 200*ms
# net.run(t_run, report='text', report_period=10*second, profile=True)
# t_elapsed += t_run
#
# # Theta ON + stim
# print("[+] Theta on + stim.")
# G_K.kN = settings.kN_frac
# G_pop_avg.G_out = settings.r_gain
# t_run = settings.duration - (500*ms + t_elapsed)
# net.run(t_run, report='text', report_period=10*second, profile=True)
# t_elapsed += t_runinp_theta
#
# # Theta OFF
# print("[+] Theta off | relaxation")
# G_pop_avg.G_out = 0.
# t_run = settings.duration - t_elapsed
# net.run(t_run, report='text', report_period=10*second, profile=True)
# t_elapsed += t_run

end = time.time()
print('-'*32)
print(bcolors.GREEN + '[+]' + ' Simulation ended' + bcolors.ENDC)
print(bcolors.YELLOW + '[!]' + bcolors.ENDC + ' Simulation ran for '+str((end-start)/60)+' minutes')
print()
print(profiling_summary(net=net, show=4)) # show the top 10 objects that took the longest


print('\n[71] Mean firing rates...')
print('-'*32)

for area in range(len(G_all)):
    # Calculate mean firing rates
    FR_exc_mean = (len(spike_mon_E_all[area][0].t)/settings.duration)/spike_mon_E_all[area][0].source.N
    FR_inh_mean = (len(spike_mon_I_all[area][0].t)/settings.duration)/spike_mon_I_all[area][0].source.N

    print(spike_mon_E_all[area][0].name.split('_')[0], 'E: ', FR_exc_mean, '\t', 'I: ', FR_inh_mean)
    print('='*16)



# Plot the results
# -------------------------------------------------------------#
print('\n[80] Post-simulation actions')
print('-'*32)

print('\n[81] Plotting results...')
tight_layout()

# Plot the 3D shape
print("[+] Saving figure 'figures/anatomy.png'")
fig_anat.savefig(os.path.join(dirs['figures'], 'anatomy.png'))

# raster plot of all regions
# raster_fig, raster_axs, fig_name = plot_raster_all(spike_mon_E_all, spike_mon_I_all)
raster_fig, raster_axs, fig_name = plot_raster_all(spike_mon_E_all, spike_mon_I_all)
print("[+] Saving figure 'figures/%s'" %fig_name)
plot_watermark(raster_fig, os.path.basename(__file__), filename, settings.git_branch, settings.git_short_hash)
raster_fig.savefig(os.path.join(dirs['figures'], fig_name))

'''
# calculate order parameter in the end
samples = len(state_mon_kuramoto.Theta[0])
r = np.zeros(samples, dtype='complex')
for s in range(samples):
    r[s] = 1/N_Kur * sum(exp(1j*state_mon_kuramoto.Theta[:,s])) # order parameter r(t)
'''

# Fig2 version
if settings.fixed_input_enabled:
    # Plot the non-Kuramoto theta drive
    fig_theta, ax_theta = subplots(1,1, figsize=(12,9))
    ax_theta.plot(tv, inp_theta_delayed[:len(tv)])
    print("[+] Saving figure 'figures/theta_inp.png'")
    fig_theta.savefig(os.path.join(dirs['figures'], 'theta_inp.png'))

    fig2, axs2, fig_name = plot_fig2(spike_mon_E_all, spike_mon_I_all, state_mon_s2r, state_mon_theta_rhytm, tv_stim, xstim)
    plot_watermark(fig2, os.path.basename(__file__), filename, settings.git_branch, settings.git_short_hash)
else:
    # kuramoto order parameter plots
    kuramoto_fig, kuramoto_axs, fig_name = plot_kuramoto(state_mon_order_param)
    plot_watermark(kuramoto_fig, os.path.basename(__file__), filename, settings.git_branch, settings.git_short_hash)
    print("[+] Saving figure 'figures/%s'" %fig_name)
    kuramoto_fig.savefig(os.path.join(dirs['figures'], fig_name))

    fig2, axs2, fig_name = plot_fig2(spike_mon_E_all, spike_mon_I_all, state_mon_s2r, state_mon_theta_rhytm, tv_stim, xstim, mode="phase")
    plot_watermark(fig2, os.path.basename(__file__), filename, settings.git_branch, settings.git_short_hash)

print("[+] Saving figure 'figures/%s'" %fig_name)
fig2.savefig(os.path.join(dirs['figures'], fig_name))


# # Plot more stuff
# fig_extra, extra_axs, fig_name = plot_network_output(spike_mon_E_all[-1][0], spike_mon_I_all[-1][0], state_mon_s2r, state_mon_order_param, tv_stim, xstim)
# plot_watermark(fig_extra, os.path.basename(__file__), filename, settings.git_branch, settings.git_short_hash)
# print("[+] Saving figure 'figures/%s'" %fig_name)
# fig_extra.savefig(os.path.join(dirs['figures'], fig_name))
#
# # newer version
# fig_extra, extra_axs, fig_name = plot_network_output2(spike_mon_E_all[-1][0], spike_mon_I_all[-1][0], state_mon_s2r, state_mon_order_param, tv_stim, xstim)
# plot_watermark(fig_extra, os.path.basename(__file__), filename, settings.git_branch, settings.git_short_hash)
# print("[+] Saving figure 'figures/%s'" %fig_name)
# fig_extra.savefig(os.path.join(dirs['figures'], fig_name))

# tight_layout()
#show()



# Save the results as .txt files (rows: time | cols: data)
# -------------------------------------------------------------#
print('\n[82] Saving results...')

# if not using fixed input
if not settings.fixed_input_enabled:
    # Kuramoto monitors
    print("[+] Saving Kuramoto monitor data")
    # np.savetxt(os.path.join(dirs['datas'], 'state_mon_kuramoto_Theta.txt'), state_mon_kuramoto.Theta.T, fmt='%.8f')
    np.savetxt(os.path.join(dirs['data'], 'order_param_mon_phase.txt'), state_mon_order_param.phase.T, fmt='%.8f')
    np.savetxt(os.path.join(dirs['data'], 'order_param_mon_rhythm.txt'), state_mon_order_param.rhythm_rect.T/nA, fmt='%.8f')
    np.savetxt(os.path.join(dirs['data'], 'order_param_mon_coherence.txt'), state_mon_order_param.coherence.T, fmt='%.8f')

# CA1 firing rate
print("[+] Saving CA1 firing rate")
np.savetxt(os.path.join(dirs['data'], 'rate_mon_E_CA1.txt'), rate_mon_E_all[3][0].smooth_rate(window='gaussian', width=50*ms).T/Hz, fmt='%.8f')
np.savetxt(os.path.join(dirs['data'], 's2r_mon_drive.txt'), state_mon_s2r.drive_.T, fmt='%.8f')

# External stimulus
print("[+] Saving external stimulus")
np.savetxt(os.path.join(dirs['data'], 'stim_input.txt'), xstim, fmt='%.2f')

# Vm avgs
print("[+] Saving Vm avgs")
for StM in make_flat(state_mon_Vm_avg):
    print("[\u2022]\tStateMon: ", StM.name)
    np.savetxt(os.path.join(dirs['data'], StM.name+'.txt'), StM.sum_v, fmt='%.8f')



# Save the spikes and their times
# -------------------------------------------------------------#
print("\n[83] Saving spikes in time....")
SM_i = []
SM_t = []
for SM in make_flat([spike_mon_E_all, spike_mon_I_all]):
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


# Save the positions of the neurons in npy files
# -------------------------------------------------------------#
print("\n[84] Saving neuron positions...")
for G in G_flat:
    try:
        print("[+] Saving group ", G.name)
        fname = '{group}'.format(group=G.name)
        pos = np.array([G.x_soma, G.y_soma, G.z_soma]).T
        np.save(os.path.join(dirs['positions'], fname), pos)

    except AttributeError:
        print(bcolors.RED + '[-]\t' + bcolors.ENDC + 'pass...')
        continue


# print("\n[85] Cleanup...")
# print('\n' + bcolors.YELLOW + '[!]' + bcolors.ENDC + ' Clearing cython cache')
# clear_cache('cython')

exit(0)
