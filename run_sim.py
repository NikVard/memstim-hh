#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from brian2 import *

from tqdm import tqdm
import argparse
import parameters
from model.globals import *
from model.HH_equations import *
from model import setup

# Configuration
# -------------------------------------------------------------_#
parser = argparse.ArgumentParser(description='MemStim using HH neurons')
parser.add_argument('parameters_file',
                    nargs='?',
                    metavar='f',
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


# Data structure parsing
#N_areas = len(data['areas'])
N_areas = 4

# Make the neuron groups -> G_all[0]: area names [str] | G_all[1]: NeuronGroup E | G_all[2]: NeuronGroup I
# -------------------------------------------------------------_#
print('   >  Making the neuron groups...')

'''
cnt = 0
G_all = [[[] for jj in range(2)] for ii in range(N_areas)] # list to save the groups
for area in tqdm(data['areas'].keys()):
    #print(area)
    for pop in data['areas'][area].keys():
        #print(data['areas'][area][pop].keys())
        #print(pop)
        params = data['areas'][area][pop]
        ptype = data['areas'][area][pop]['type']

        if ptype.lower() == 'py' or ptype.lower() == 'pycan':
            idx = 0
        else:
            idx = 1
        G_all[cnt][idx].append(setup.create_group(params, area+'_'+pop))
    cnt += 1
'''

G_all = [[[] for jj in range(2)] for ii in range(N_areas)]  # where we save the neuron groups
G_all[0][0].append(setup.create_group(data['areas']['EC']['E'], 'EC_pyCAN'))    # EC exc
G_all[0][1].append(setup.create_group(data['areas']['EC']['I'], 'EC_inh'))      # EC inh

G_all[1][0].append(setup.create_group(data['areas']['DG']['E'], 'DG_py'))       # DG exc
G_all[1][1].append(setup.create_group(data['areas']['DG']['I'], 'DG_inh'))      # DG inh

G_all[2][0].append(setup.create_group(data['areas']['CA3']['E'], 'CA3_pyCAN'))  # CA3 exc
G_all[2][1].append(setup.create_group(data['areas']['CA3']['I'], 'CA3_inh'))    # CA3 inh

G_all[3][0].append(setup.create_group(data['areas']['CA1']['E'], 'CA1_pyCAN'))  # CA1 exc
G_all[3][1].append(setup.create_group(data['areas']['CA1']['I'], 'CA1_inh'))    # CA1 inh


# Make the synapses
# -------------------------------------------------------------_#
print('\n >  Making the synapses...')

# intra-region

print('     * intra-region')

# intra-connectivity probabilities

p_EC = data['connectivity']['intra']['EC']
p_EC_e = [p_EC[0] for ii in range(types[0])] + [p_EC[1] for jj in range(types[1])] # all excitatory in EC
p_EC_i = [p_EC[2] for ii in range(types[0])] + [p_EC[3] for jj in range(types[1])] # all inhibitory in EC
all_p_EC = [p_EC_i for ii in range(types[0])] + [p_EC_i for jj in range(types[1])]

p_DG = data['connectivity']['intra']['DG']
p_DG_e = [p_DG[0] for ii in range(types[0])] + [p_DG[1] for jj in range(types[1])]
p_DG_i = [p_DG[2] for ii in range(types[0])] + [p_DG[3] for jj in range(types[1])]
all_p_DG = [p_DG_i for ii in range(types[0])] + [p_DG_i for jj in range(types[1])]

p_CA3 = data['connectivity']['intra']['CA3']
p_CA3_e = [p_CA3[0] for ii in range(types[0])] + [p_CA3[1] for jj in range(types[1])]
p_CA3_i = [p_CA3[2] for ii in range(types[0])] + [p_CA3[3] for jj in range(types[1])]
all_p_CA3 = [p_CA3_i for ii in range(types[0])] + [p_CA3_i for jj in range(types[1])]

p_CA1 = data['connectivity']['intra']['CA1']
p_CA1_e = [p_CA1[0] for ii in range(types[0])] + [p_CA1[1] for jj in range(types[1])]
p_CA1_i = [p_CA1[2] for ii in range(types[0])] + [p_CA1[3] for jj in range(types[1])]
all_p_CA1 = [p_CA1_i for ii in range(types[0])] + [p_CA1_i for jj in range(types[1])]

all_p_intra = [all_p_EC, all_p_DG, all_p_CA3, all_p_CA1]
print('done p_intra')

# all gains
all_gains = [[1 for ii in range(types[0]+types[1])] for jj in range(4)]
all_gains[0][:types[0]] = [1/G]*types[0]
all_gains[1] = [G]*(types[0]+types[1])
all_gains[2][:types[0]] = [1/G]*types[0]
all_gains[3][types[0]:] = [G]*types[1]
print('done all_gains')

all_syn_EC = setup.connect_intra(G_all[0][0], G_all[0][1], all_p_EC, all_gains[0])
print('done #1')
all_syn_DG = setup.connect_intra(G_all[1][0], G_all[1][1], all_p_DG, all_gains[1])
print('done #2')
all_syn_CA3 = setup.connect_intra(G_all[2][0], G_all[2][1], all_p_CA3, all_gains[2])
print('done #3')
all_syn_CA1 = setup.connect_intra(G_all[3][0], G_all[3][1], all_p_CA1, all_gains[3])
print('done #4')
all_syn_intra = [all_syn_EC, all_syn_DG, all_syn_CA3, all_syn_CA1]


# inter-region
print('     * inter-region')


# Stimulation and other inputs
# -------------------------------------------------------------_#
print('\n >  Inputs and Stimulation...')


# Add the monitors (spikes/rates)
# -------------------------------------------------------------_#
print('\n >  Monitors...')


# Create the Network
# -------------------------------------------------------------_#
print('\n >  Connecting the network...')
#net = Network()
#net.add


# run simulation
# -------------------------------------------------------------_#
print('\n >  Starting simulation...')
#net.run(duration)
print('Simulation ended')
