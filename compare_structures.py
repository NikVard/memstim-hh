#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from brian2 import *
from scipy import optimize
import re # regular expressions

# Initialization
areas = ['EC', 'DG', 'CA1', 'CA3']
types1 = ['exc', 'inh']
types2 = ['E', 'I']
scale_aussel = 150*umetre


# ==============
# Optimization
# ==============
def parse_coords(fname):
    """ Opens file and parses coordinates """
    pattern = r'[\[\],\n]' # to remove from read lines

    x_tmp = []
    y_tmp = []
    with open(fname, 'r') as fin:
        for line in fin:
            tok = re.sub(pattern, '', line).split()
            x_tmp.append(float(tok[0]))
            y_tmp.append(float(tok[1]))

    return array([x_tmp,y_tmp]).T


# read coordinates from files
ref = []
adj = []


# EC
# ============
## E
tmp1 = parse_coords(fname='positions/EC_exc.txt')
tmp2 = load('./neuron_positions/full/EC_E-stipple-10000.npy')
ref.append(tmp1)
adj.append(tmp2)

## I
tmp1 = parse_coords(fname='positions/EC_inh.txt')
tmp2 = load('./neuron_positions/full/EC_I-stipple-1000.npy')
ref.append(tmp1)
adj.append(tmp2)

# DG
# ============
## E
tmp1 = parse_coords(fname='positions/DG_exc.txt')
tmp2 = load('./neuron_positions/full/DG_E-stipple-10000.npy')
ref.append(tmp1)
adj.append(tmp2)

## I
tmp1 = parse_coords(fname='positions/DG_inh.txt')
tmp2 = load('./neuron_positions/full/DG_I-stipple-100.npy')
ref.append(tmp1)
adj.append(tmp2)

# CA3
# ============
## E
tmp1 = parse_coords(fname='positions/CA3_exc.txt')
tmp2 = load('./neuron_positions/full/CA3_E-stipple-1000.npy')
ref.append(tmp1)
adj.append(tmp2)

##I
tmp1 = parse_coords(fname='positions/CA3_inh.txt')
tmp2 = load('./neuron_positions/full/CA3_I-stipple-100.npy')
ref.append(tmp1)
adj.append(tmp2)

# CA1
# ============
## E
tmp1 = parse_coords(fname='positions/CA1_exc.txt')
tmp2 = np.load('./neuron_positions/full/CA1_E-stipple-10000.npy')
ref.append(tmp1)
adj.append(tmp2)

## I
tmp1 = parse_coords(fname='positions/CA1_inh.txt')
tmp2 = load('./neuron_positions/full/CA1_I-stipple-1000.npy')
ref.append(tmp1)
adj.append(tmp2)
# ============

def f0(pc1, pc2):
    ''' Function to be minimized - difference between point clouds '''
    # we will add the range differences between point clouds
    diff = 0.

    for idx in arange(len(pc1)):
        curr_ref = pc1[idx]
        rx_ref = abs(amin(curr_ref[:,0]) - amax(curr_ref[:,0]))
        ry_ref = abs(amin(curr_ref[:,1]) - amax(curr_ref[:,1]))

        curr_adj = pc2[idx]
        rx = abs(amin(curr_adj[:,0]) - amax(curr_adj[:,0]))
        ry = abs(amin(curr_adj[:,1]) - amax(curr_adj[:,1]))

        diff += (rx_ref - rx)**2 + (ry_ref - ry)**2

    return diff


def f1(x, *params):
    reference, adjusted, scale_ref = params

    # calculate the new scale
    scale_adj = (1000./x)*umetre

    # scale the point clouds, remove units
    reference_scaled = [elem*scale_ref/mmetre for elem in reference]
    adjusted_scaled = [elem*scale_adj/mmetre for elem in adjusted]

    return f0(pc1=reference_scaled, pc2=adjusted_scaled)


# run the optimization
resbrute = optimize.brute(func=f1, args=(ref, adj, scale_aussel), ranges=((500.,1000.),), Ns=501, full_output=True, disp=True)

# update the scale
scale_adj = round(1000./resbrute[0][0],6)*umetre

print("========================================")
print("Minimized value: ", resbrute[0][0])
print("Adjusted scale: ", scale_adj)
print("========================================")

# set up a figure twice as wide as it is tall
fig = figure(figsize=plt.figaspect(0.5))


# =============
# First subplot
# =============
# set up the axes for the first plot
ax = fig.add_subplot(121, projection='3d')
ax.set_title("Reference")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")

# Function for parsing coordinates
def parse_coords_3d(fname):
    """ Opens file and parses coordinates """
    pattern = r'[\[\],\n]' # to remove from read lines
    coords = []
    with open(fname, 'r') as fin:
        idx = 0
        for line in fin:
            tok = re.sub(pattern, '', line).split()
            tmp = [float(tok[0]), float(tok[1]), float(tok[2])]
            coords.append(tmp)
            idx += 1

    return coords


# Iterate over position files from original Aussel model
for area in areas:
    for type in types1:
        fname = 'positions/' + area + '_' + type + '.txt'
        coords_aussel = asarray(parse_coords_3d(fname))*scale_aussel

        if type=='exc':
            if area=='DG':
                c = 'green'
            else:
                c = 'blue'
        else:
            c = 'red'

        ax.scatter(coords_aussel[:,0], coords_aussel[:,1], coords_aussel[:,2], c=c)



# ==============
# Second subplot
# ==============
# set up the axes for the second plot
ax = fig.add_subplot(122, projection='3d')
ax.set_title("Adjusted")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")

# Iterate over position files from adjusted model
for area in areas:
    for type in types2:
        fname = 'neuron_positions/test/' + area + '_' + type + '-stipple.npy'
        coords_adj = load(fname)

        N = max(coords_adj.shape)
        coords_adj = hstack((coords_adj, zeros((N, 1)))) # add z-axis
        coords_adj *= scale_adj
        coords_adj[:,2] += 15*mm*rand(N)

        if type=='E':
            if area=='DG':
                c = 'green'
            else:
                c = 'blue'
        else:
            c = 'red'

        ax.scatter(coords_adj[:,0], -coords_adj[:,1], coords_adj[:,2], c=c)


show()
