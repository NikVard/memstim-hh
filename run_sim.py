#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from brian2 import *

import argparse
import parameters
from model.globals import *
from model.HH_equations import *
from model import setup

# Simulation
duration = 3*second
integ_method='exponential_euler'


# Configuration
# -------------------------------------------------------------_#
parser = argparse.ArgumentParser(description='MemStim using HH neurons')
parser.add_argument('parameters_file',
                    nargs="?",
                    metavar='f',
                    type=str,
                    default="configs/default.json",
                    help="Parameters file (json format)")
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


# Make the neuron groups
# -------------------------------------------------------------_#
G_all = [[] for i in range(3)] # 2D list of the groups, label[0]/E[1]/I[2]
for area in data['areas'].keys():
    #print(area)
    cnt = 0
    G_all[cnt].append(area)
    for pop in data['areas'][area].keys():
        #print(data['areas'][area][pop].keys())
        #print(pop)
        cnt += 1
        params = data['areas'][area][pop]
        G_all[cnt].append(setup.create_group(params, area+'_'+pop))


# Make the synapses
# -------------------------------------------------------------_#



# Stimulation and other inputs
# -------------------------------------------------------------_#



# Add the monitors (spikes/rates)
# -------------------------------------------------------------_#


# Create the Network
# -------------------------------------------------------------_#
#net = Network()
#net.add


# run simulation
# -------------------------------------------------------------_#
#net.run(duration)
