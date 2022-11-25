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

# Other scripts and my stuff
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = Path(script_dir).parent
sys.path.insert(0, os.path.abspath(parent_dir))

from src.figure_plots_parameters import *

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
fig, ax = plt.subplots(1, 2, figsize=(fig_width,fig_height), constrained_layout=True)

# Inset colorbars
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axin_low = inset_axes(ax[0], width='40%', height='4%', loc='lower right',
                        bbox_to_anchor=(0.26, 0.1, 1, 1),
                        bbox_transform=ax[0].transAxes,
                        borderpad=0)
axin_high = inset_axes(ax[0], width='40%', height='4%', loc='lower right',
                        bbox_to_anchor=(0.26, 0.05, 1, 1),
                        bbox_transform=ax[0].transAxes,
                        borderpad=0)

# Tick locations
axin_low.xaxis.set_ticks_position("bottom")
axin_high.xaxis.set_ticks_position("bottom")

# Equal aspect ratios
ax[0].set_aspect('equal', adjustable='box')
ax[1].set_aspect('equal', adjustable='box')

# Draw the circle
circ0 = plt.Circle((0, 0), radius=rad, edgecolor='k', facecolor='None', lw=0.5)
circ1 = plt.Circle((0, 0), radius=rad, edgecolor='k', facecolor='None', lw=0.5)
ax[0].add_patch(circ0)
ax[1].add_patch(circ1)

# Scatters
sc_low = ax[0].scatter(X0, Y0, S0, omega0, vmin=0, vmax=1, cmap="Blues", edgecolor="white", zorder=10)
sc_high = ax[1].scatter(X1, Y1, S1, omega1, vmin=0, vmax=1, cmap="Reds", edgecolor="white", zorder=10)

# order parameter
ax[0].arrow(0, 0, np.real(z0), np.imag(z0), alpha=0.5, width=0.015, edgecolor='blue', facecolor='blue', lw=2, zorder=4, length_includes_head=True, head_length=0.06, head_width=0.06, shape='full')
ax[1].arrow(0, 0, np.real(z1), np.imag(z1), alpha=0.5, width=0.015, edgecolor='red', facecolor='red', lw=2, zorder=4, length_includes_head=True, head_length=0.06, head_width=0.06, shape='full')

# Set titles
ax[0].set_title('Low synchronization', fontsize=fsize_titles)
ax[1].set_title('High synchronization', fontsize=fsize_titles)

# Remove axis
ax[0].set_axis_off()
ax[1].set_axis_off()

# Set lims
ax[0].set_xlim([-1.2,1.2])
ax[0].set_ylim([-1.2,1.2])
ax[1].set_xlim([-1.2,1.2])
ax[1].set_ylim([-1.2,1.2])

# Make the arrows
r = 0.8
ctheta = np.pi/2
dtheta = np.pi/8
xs, ys = (r*np.cos(ctheta-dtheta), r*np.sin(ctheta-dtheta))
xe, ye = (r*np.cos(ctheta+dtheta), r*np.sin(ctheta+dtheta))

arr_style = "Simple, tail_width=0.5, head_width=5, head_length=5"
kw = dict(arrowstyle=arr_style, color="k")

arr0 = mplb.patches.FancyArrowPatch((xs, ys), (xe, ye), connectionstyle="arc3,rad=0.3", **kw)
arr1 = mplb.patches.FancyArrowPatch((xs, ys), (xe, ye), connectionstyle="arc3,rad=0.3", **kw)

# Add patches
ax[0].add_patch(arr0)
ax[1].add_patch(arr1)

# Add texts
ax[0].text(0.0, +0.25, 'r = {:.2f}'.format(np.abs(z0)), ha='center', va='center', fontsize=fsize_misc)
ax[1].text(0.0, +0.25, 'r = {:.2f}'.format(np.abs(z1)), ha='center', va='center', fontsize=fsize_misc)
ax[0].text(0, 0.7, r'$\omega$', ha='center', va='center', fontsize=fsize_misc)
ax[1].text(0, 0.7, r'$\omega$', ha='center', va='center', fontsize=fsize_misc)


# ax[0].text(-1.1, -1.1, r'Angular Velocity', ha='center', va='center', fontsize=fsize_misc)
# ax[1].text(-1.1, -1.1, r'Angular Velocity', ha='center', va='center', fontsize=fsize_misc)

# Add colorbars
cbar_low = fig.colorbar(sc_low, cax=axin_low, orientation='horizontal', ticks=[])
cbar_high = fig.colorbar(sc_high, cax=axin_high, orientation='horizontal', ticks=[0, 1])

# Set cbar titles
cbar_low.ax.set_xlabel(r'Angular Velocities ($\omega$)', fontsize=fsize_xylabels)
cbar_low.ax.xaxis.set_label_position('top')
# cbar_high.ax.set_ylabel('Angular Velocity')
cbar_high.ax.set_xticklabels(['min', 'max'])  # vertically oriented colorbar
cbar_high.ax.tick_params(labelsize=fsize_ticks, length=2)

# Set tight layout
# fig.tight_layout(True)

# Save the figure
print('[+] Saving the figure...')
fig.savefig(os.path.join('figures', 'fig1', 'kuramoto_1D.png'), transparent=True, dpi=300, format='png')
fig.savefig(os.path.join('figures', 'fig1', 'kuramoto_1D.pdf'), transparent=True, dpi=300, format='pdf')

# Show the figure
plt.show()

# Exit properly
sys.exit(0)
