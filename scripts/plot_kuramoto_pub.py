# -*- coding: utf-8 -*-

# OS stuff
import os
import sys
import warnings
from pathlib import Path

# Computational stuff
import numpy as np
import matplotlib as mplb
import matplotlib.pyplot as plt

# Set the random seed
# np.random.seed(123)

N = [50, 20] # Number of oscillators
rad = 1. # unit circle

# Figure sizes [inches]
fig_width = 6.
fig_height = 3.

# Font sizes
fsize_ticks = 8
fsize_xylabels = 9
fsize_titles = 11
fsize_panels = 12

# ILLUSTRATOR STUFF
mplb.rcParams['pdf.fonttype'] = 42
mplb.rcParams['ps.fonttype'] = 42


# Plot Kuramoto oscillators
# No sync
theta0 = np.random.normal(0.5, 2*np.pi/3, N[0]) # initial phase
omega0 = np.random.uniform(0.5, 1, N[0]) # color indicates velocity

X0 = rad*np.cos(theta0)
Y0 = rad*np.sin(theta0)
S0 = 125*np.ones((1,N[0]))
V0 = np.ones((1,N[0]))

# Some sync
theta1 = np.random.normal(-np.pi/4, np.pi/6, N[1]) # initial phase
omega1 = np.random.uniform(0.1, 1, N[1]) # color indicates velocity

X1 = rad*np.cos(theta1)
Y1 = rad*np.sin(theta1)
S1 = 125*np.ones((1,N[1]))
V1 = np.ones((1,N[1]))

# order parameter
z0 = 1/N[0] * np.sum(np.exp(1j*theta0))
z1 = 1/N[1] * np.sum(np.exp(1j*theta1))

# Plot
fig, ax = plt.subplots(1, 2, figsize=(fig_width,fig_height))

# Equal aspect ratios
ax[0].set_aspect('equal', adjustable='box')
ax[1].set_aspect('equal', adjustable='box')

# Draw the circle
circ0 = plt.Circle((0, 0), radius=rad, edgecolor='k', facecolor='None', lw=0.5)
circ1 = plt.Circle((0, 0), radius=rad, edgecolor='k', facecolor='None', lw=0.5)
ax[0].add_patch(circ0)
ax[1].add_patch(circ1)

# Scatters
ax[0].scatter(X0, Y0, S0, omega0, vmin=0, vmax=1, cmap="Blues", edgecolor="white", zorder=10)
ax[1].scatter(X1, Y1, S1, omega1, vmin=0, vmax=1, cmap="Reds", edgecolor="white", zorder=10)

# order parameter
ax[0].arrow(0, 0, np.real(z0), np.imag(z0), alpha=0.5, width=0.015, edgecolor='blue', facecolor='blue', lw=2, zorder=4, length_includes_head=True, head_length=0.06, head_width=0.06, shape='full')
ax[1].arrow(0, 0, np.real(z1), np.imag(z1), alpha=0.5, width=0.015, edgecolor='red', facecolor='red', lw=2, zorder=4, length_includes_head=True, head_length=0.06, head_width=0.06, shape='full')

# Set titles
ax[0].set_title('Low synchronization', fontsize=fsize_titles)
ax[1].set_title('High synchronization', fontsize=fsize_titles)

# Set text for order parameter
ax[0].text(0.0, +0.25, 'r={:.2f}'.format(np.abs(z0)), ha='center', va='center', fontsize=10)
ax[1].text(0.0, +0.25, 'r={:.2f}'.format(np.abs(z1)), ha='center', va='center', fontsize=10)

# Remove axis
ax[0].set_axis_off()
ax[1].set_axis_off()

# Set lims
ax[0].set_xlim([-1.2,1.2])
ax[0].set_ylim([-1.2,1.2])
ax[1].set_xlim([-1.2,1.2])
ax[1].set_ylim([-1.2,1.2])
# ax[0].set_xlim([-1.5,1.5])
# ax[0].set_ylim([-1.5,1.5])
# ax[1].set_xlim([-1.5,1.5])
# ax[1].set_ylim([-1.5,1.5])


# Make the image gradient
XY_triangle = [[-0.65, -1.1], [-0.65, -0.9], [-1.2, -0.9]]
min_xy = np.array(XY_triangle).min(axis=0)
max_xy = np.array(XY_triangle).max(axis=0)

# Omega patches for colormaps
img0 = ax[0].imshow(np.flip(np.linspace(0, 1, 51)).reshape(1, -1), cmap='Blues_r', extent=[min_xy[0],max_xy[0],min_xy[1],max_xy[1]], transform=ax[0].transData, origin='lower')
img1 = ax[1].imshow(np.flip(np.linspace(0, 1, 51)).reshape(1, -1), cmap='Reds_r', extent=[min_xy[0],max_xy[0],min_xy[1],max_xy[1]], transform=ax[1].transData, origin='lower')

# Add the triangles
t0 = plt.Polygon(XY_triangle, facecolor='none', edgecolor='black', lw=0.8, alpha=1., clip_on=False)
t1 = plt.Polygon(XY_triangle, facecolor='none', edgecolor='black', lw=0.8, alpha=1., clip_on=False)

# Add patches
ax[0].add_patch(t0)
ax[1].add_patch(t1)

# Set clipping path
img0.set_clip_path(t0)
img1.set_clip_path(t1)

# Add text
ax[0].text(min_xy[0], min_xy[1], r'$\omega$ values', ha='center', va='center', fontsize=10)
ax[1].text(min_xy[0], min_xy[1], r'$\omega$ values', ha='center', va='center', fontsize=10)

# Set tight layout
fig.tight_layout()

# Save the figure
print('[+] Saving the figure...')
fig.savefig(os.path.join('figures', 'fig1', 'kuramoto_1D.png'), transparent=True, dpi=300, format='png')
fig.savefig(os.path.join('figures', 'fig1', 'kuramoto_1D.pdf'), transparent=True, dpi=300, format='pdf')

# Show the figure
plt.show()

# Exit properly
sys.exit(0)
