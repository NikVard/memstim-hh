#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TODO : add a colorbar in the top-right of the PRC instead of a legend
#        1. https://stackoverflow.com/questions/32462881/add-colorbar-to-existing-axis
#        2. https://stackoverflow.com/questions/43805821/matplotlib-add-colorbar-to-non-mappable-object

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib as mplb
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = Path(script_dir).parent
sys.path.insert(0, os.path.abspath(parent_dir))

from src.figure_plots_parameters import *

# Set font to Arial -- is this working?
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

# ILLUSTRATOR STUFF
mplb.rcParams['pdf.fonttype'] = 42
mplb.rcParams['ps.fonttype'] = 42
mplb.rcParams['axes.titlesize'] = fsize_titles
mplb.rcParams['axes.labelsize'] = fsize_xylabels

dt = .1 # [msec]
fs = 10e3
delay = 25 # [samples]

# make a time vector (200 ms, one period)
tv = np.linspace(0,250,2501)
PRC_all = []
xvals_all = []

# Plot tuning
fig_height = 2.5
fig_width = 2.6
ms = 4 # markersize
lw = 1.5 # linewidth
fsize = [9, 11] # axes, titles

# Open-loop results directory| no stim
# dir_cl = os.path.join('..', 'results', 'None', 'OL')
# phase_ol = np.loadtxt(os.path.join(dir_ol, 'data', 'order_param_mon_phase.txt'))

# Closed-loop results directory | no stim
dir_cl = os.path.join('results_PRC_new', 'None')
dir_cl = os.path.join(dir_cl, os.listdir(dir_cl)[0])
phase_cl = np.loadtxt(os.path.join(dir_cl, 'data', 'order_param_mon_phase.txt'))

# Load the CL rhythm
rhythm_cl = np.loadtxt(os.path.join(dir_cl, 'data', 'order_param_mon_rhythm.txt'))

# Results directories
dirs = []
dirs.append(os.path.join('results_PRC_new', '2_nA'))
dirs.append(os.path.join('results_PRC_new', '5_nA'))
dirs.append(os.path.join('results_PRC_new', '10_nA'))
dirs.append(os.path.join('results_PRC_new', '20_nA'))
dirs.append(os.path.join('results_PRC_new', '40_nA'))
# dirs.append(os.path.join('results_PRC_new', '10_nA'))

# set default phase to compare against
phase_def = phase_cl

# stimulation points for rhythm plotting later
t_stim_arr = []

# iterate over stimulation amplitudes
for root in dirs:
    print(root)
    # initialization of current stim amplitude lists
    PRC = []
    xvals = []

    # iterate over saved PRC results
    for item in os.listdir(root):

        # is this a simulation directory
        if os.path.isdir(os.path.join(root, item)):
            # get the stimulation onset [ms]
            tmp = item.split("_")
            t_on = float(tmp[1])
            print(t_on)
            t_stim_arr.append(t_on)

            # find index of phase reset (based on stim_on)
            idx = int(t_on/dt)

            # load the data for the current simulation
            curr_path = os.path.join(root, item)
            curr_path = os.path.join(curr_path, os.listdir(curr_path)[0])

            # frate = np.loadtxt(os.path.join(curr_path, 'data', 's2r_mon_drive.txt'))
            phase = np.loadtxt(os.path.join(curr_path, 'data', 'order_param_mon_phase.txt'))

            # calculate d_phi
            # d_phi = np.mean(phase[idx-10:idx]) - np.mean(phase[idx:idx+10])
            # d_phi = phase[idx+delay] - phase_def[idx+delay]
            d_phi = (phase[idx+delay] - phase_def[idx+delay])
            d_phi = (d_phi + np.pi) % (2*np.pi) - np.pi     # wrap between [-pi pi]

            PRC.append(d_phi)
            xvals.append(phase[idx])

    # Get sorting indices
    idxs = np.argsort(xvals)

    # Sort the data points
    xvals = [xvals[i] for i in idxs]
    PRC = [PRC[i] for i in idxs]

    # add to list
    PRC_all.append(PRC)
    xvals_all.append(xvals)

    # break

# Plotting
# ------------------------
fig, axs = plt.subplots(nrows=1, ncols=1)
fig.set_figheight(fig_height)
fig.set_figwidth(fig_width)

# Twin axes, shares x-axis
axs2 = axs.twinx()

# Set background color
cbg = '#f0f8fe'
cbg = '#fcfbf9'
axs.set_facecolor(cbg)

# Colormaps
n = 5
# colors = plt.cm.Spectral(np.linspace(0,1,n))
cvals = np.array([1., 0.8, 0.7, 0.3, 0.])
colors = plt.cm.Spectral(cvals)
cmap = mplb.cm.Spectral
cmap = mplb.colors.ListedColormap(colors)
# colors = plt.cm.hot(np.linspace(0.2,0.8,n))
ticks = np.array([2, 5, 10, 20, 40])
bins =  np.concatenate((np.zeros(1), ticks[:-1]+(ticks[1:] - ticks[:-1])/2, [ticks[-1]+10]))


# rectangle around the colorbar
# bb = mplb.transforms.Bbox([[0.15, 0.25], [0.35, 0.06]])
bwidth = 0.37
bheight = 0.22
p0 = (0.06, 0.06)
p1 = (p0[0]+bwidth, p0[1]+bheight)
# plt.plot(p0[0], p0[1], 'x', c='r', markersize=6, transform=axs.transAxes)
# plt.plot(p1[0], p1[1], 'x', c='r', markersize=6, transform=axs.transAxes)
fancy = mplb.patches.FancyBboxPatch(p0, bwidth, bheight, boxstyle="round,pad=0.05", transform=axs.transAxes)
# fc=(1, 0.8, 1, 0.5), ec=(1, 0.5, 1, 0.5)
fancy.set(edgecolor='none', facecolor='white', zorder=10, alpha=0.5)
axs2.add_patch(fancy)

# colorbar
cax = fig.add_axes([0.18, 0.27, 0.32, 0.03])
norm = mplb.colors.BoundaryNorm(boundaries=bins, ncolors=cmap.N)
# sm = mplb.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
cbar = fig.colorbar(mplb.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=ticks, orientation='horizontal')
cbar.set_ticklabels([2, 5, 10, 20, 40])
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_title('  Stim. Amp. [nA]', fontsize=fsize[0]-1, loc='center')

# 1-50nA
axs.plot(xvals_all[0], PRC_all[0], c=colors[0], ls='-', linewidth=lw, marker='o', markersize=ms, markeredgecolor="white", markeredgewidth=0.5, label=r'2nA', zorder=10)
axs.plot(xvals_all[1], PRC_all[1], c=colors[1], ls='-', linewidth=lw, marker='o', markersize=ms, markeredgecolor="white", markeredgewidth=0.5, label=r'5nA', zorder=11)
axs.plot(xvals_all[2], PRC_all[2], c=colors[2], ls='-', linewidth=lw, marker='o', markersize=ms, markeredgecolor="white", markeredgewidth=0.5, label=r'10nA', zorder=12)
axs.plot(xvals_all[3], PRC_all[3], c=colors[3], ls='-', linewidth=lw, marker='o', markersize=ms, markeredgecolor="white", markeredgewidth=0.5, label=r'20nA', zorder=13)
axs.plot(xvals_all[4], PRC_all[4], c=colors[4], ls='-', linewidth=lw, marker='o', markersize=ms, markeredgecolor="white", markeredgewidth=0.5, label=r'40nA', zorder=14)
# axs.plot(xvals_all[5], PRC_all[5], c=colors[5], ls='--', marker='o', markersize=11, label=r'25nA', zorder=15)

# Plot the theoretical PRC in the background
# phase_vec = np.linspace(-np.pi, np.pi,512)
# axs.plot(phase_vec, -np.sin(phase_vec), c='k', zorder=0, label=r'Theoretical')

# find the time indices for the peri-stim phase cycle
tmin = np.array(t_stim_arr).min()
tmax = np.array(t_stim_arr).max()
tmin_idx = int(tmin/1000*fs)
tmax_idx = int(tmax/1000*fs)

# phase / rhythm
phase_cut = phase_cl[tmin_idx:tmax_idx]
rhythm_cut = rhythm_cl[tmin_idx:tmax_idx]

# Get sorting indices
idxs = np.argsort(phase_cut)

# Plot the actual theta rhythm w.r.t. phase
# axs2.plot(phase_cut[idxs], rhythm_cut[idxs]/rhythm_cut.max(), c='k', alpha=0.4, zorder=0, label=r'Rhythm')
rphase = np.linspace(-np.pi, np.pi, 101)
rcos = (np.cos(rphase) + 1)/2
axs2.plot(rphase, rcos, c='k', alpha=0.25, zorder=0, label=r'Rhythm')

# Horizontal line @ y=0
axs.hlines(y=0., xmin=-2*np.pi, xmax=2*np.pi, linestyle='-', colors='k', linewidth=1., zorder=-2)

# Vertical line @ x=0
axs.vlines(x=0., ymin=-4., ymax=4., linestyle='-', colors='k', linewidth=1., zorder=-2)

# Fill positive (y>0) and negative (y<0)
# axs.fill_between(x=[-4,4], y1=[4., 4.], color='green', alpha=0.1)
# axs.fill_between(x=[-4,4], y1=[-4., -4.], color='red', alpha=0.1)

# Add text boxes
# boxp = dict(boxstyle='square', alpha=0.75, facecolor='white', edgecolor='none')
# axs.text(x=-3.3, y=0.28, color='black', s='ADV', fontsize=fsize[1], verticalalignment='top', horizontalalignment='left', bbox=boxp, zorder=20)
# axs.text(x=-3.3, y=-0.28, color='black', s='DEL', bbox=boxp, zorder=21)

# Limits
axs.set_xlim([-3.3, 3.3])
# axs.set_ylim([-3.5, 3.5])
axs.set_ylim([-np.pi/4-0.05, np.pi/4+0.05])
axs2.set_ylim([-0.1,1.1])

# Ticks
def format_func(x, pos):
    # find number of multiples of pi/2
    N = int(np.round(2 * x / np.pi))
    if N == -2:
        return r"-$\pi$"
    elif N == -1:
        return r"-$\pi/2$"
    elif N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)

def format_func_minors(x, pos):
    # find number of multiples of pi/8
    N = int(np.round(x/(np.pi/8)))
    if N == -8: return r"-$\pi$"
    elif N == -7: return r"-$7\pi/8$"
    elif N == -6: return r"-$3\pi/4$"
    elif N == -5: return r"-$5\pi/8$"
    elif N == -4: return r"-$\pi/2$"
    elif N == -3: return r"-$3\pi/8$"
    elif N == -2: return r"-$\pi/4$"
    elif N == -1: return r"-$\pi/8$"
    elif N == 0: return r"0"
    elif N == 1: return r"$\pi/8$"
    elif N == 2: return r"$\pi/4$"
    elif N == 3: return r"$3\pi/8$"
    elif N == 4: return r"$\pi/2$"
    elif N == 5: return r"$5\pi/8$"
    elif N == 6: return r"$3\pi/4$"
    elif N == 7: return r"$7\pi/8$"
    elif N == 8: return r"$\pi$"
    else: return r"${0}\pi$".format(N // 2)

axs.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
axs.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
axs.xaxis.set_minor_locator(plt.FixedLocator(np.linspace(-np.pi, np.pi, 17)+1e-4))
axs.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
# axs.xaxis.set_minor_formatter(plt.FuncFormatter(format_func_minors))

axs.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
axs.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 8))
axs.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
axs.yaxis.set_minor_formatter(plt.FuncFormatter(format_func_minors))

# axs2.xaxis.set_minor_locator(plt.FixedLocator(np.linspace(-np.pi, np.pi, 17)+1e-6))
axs2.yaxis.set_major_locator(plt.MultipleLocator(0.5))
axs2.yaxis.set_minor_locator(plt.FixedLocator(np.arange(0,1,0.1)))
# axs2.minorticks_off()
# axs2.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

axs.tick_params(which='major', length=6, width=0.8)
axs.tick_params(which='minor', length=3, width=0.8, labelcolor='0.25')
axs.tick_params(axis='both', which='major', labelsize=fsize[0]-1)
# axs.tick_params(axis='x', which='minor', color='red', top=True, labeltop=True, bottom=False, labelbottom=False)
axs.tick_params(axis='y', which='both', labelsize=fsize[0]-1)
axs2.tick_params(axis='both', which='major', labelsize=fsize[0]-1)

# Labels
axs.set_xlabel('Stim. Phase [rad]', fontsize=fsize[0], labelpad=0)
axs.set_ylabel(r'$\Delta\phi$ [rad]', fontsize=fsize[0], labelpad=-7.)
axs2.set_ylabel(r'Amplitude', rotation=270, va="center", fontsize=fsize[0], labelpad=5.)
# axs.set_ylabel('')
axs2.set_ylabel('')

# Title
axs.set_title("Phase Response Curve (PRC)", fontsize=fsize[1])

# Spines
# axs.spines['top'].set_visible(False)
# axs.spines['bottom'].set_visible(False)
# axs.spines['left'].set_visible(False)
# axs.spines['right'].set_visible(False)
#
# axs2.spines['top'].set_visible(False)
# axs2.spines['bottom'].set_visible(False)
# axs2.spines['left'].set_visible(False)
# axs2.spines['right'].set_visible(False)

# Grids
# axs.grid()
# axs.grid(b=True, which='minor', color='k', alpha=0.2, linestyle='--')

# Legend

# ask matplotlib for the plotted objects and their labels
lines, labels = axs.get_legend_handles_labels()
lines2, labels2 = axs2.get_legend_handles_labels()
# axs2.legend(lines + lines2, labels + labels2, loc=0, fontsize=fsize[0]-1, frameon=True)
# axs.legend()

# Set tight layout
# fig.tight_layout()
fig.subplots_adjust(left=0.16, bottom=0.17, right=0.89, top=0.9)

# Save the figures
fig.savefig('figures/fig2/fig2_D_PRC.png', transparent=True, dpi=300, format='png', pad_inches=0)
fig.savefig('figures/fig2/fig2_D_PRC.pdf', transparent=True, dpi=300, format='pdf', pad_inches=0)

# Show the figure
plt.show()

# Exit - no errors
sys.exit(0)
