# -*- coding: utf-8 -*-

# OS stuff
import os
import sys
import warnings
from pathlib import Path

import json

# Computational stuff
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import ticker

# Other scripts and my stuff
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = Path(script_dir).parent.parent
sys.path.insert(0, os.path.abspath(parent_dir))

from src.freq_analysis import *
from src.figure_plots_parameters import *

import parameters

def add_sizebar(ax, xlocs, ylocs, bcolor, text, textx, texty, fsize, rot, ha, va):
    """ Add a sizebar to the provided axis """
    ax.plot(xlocs, ylocs, ls='-', c=bcolor, linewidth=1., rasterized=False, clip_on=False)

    # add the text
    if type(text) is list:
        # bottom text
        ax.text(x=textx[0], y=texty[0], s=text[0], rotation=rot[0], fontsize=fsize, va=va, ha=ha, clip_on=False)

        # top text
        ax.text(x=textx[1], y=texty[1], s=text[1], rotation=rot[1], fontsize=fsize, va=va, ha=ha, clip_on=False)
    else:
        ax.text(x=textx, y=texty, s=text, rotation=rot, fontsize=fsize, va=va, ha=ha, clip_on=False)

# ILLUSTRATOR STUFF
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.titlesize'] = fsize_titles
plt.rcParams['axes.labelsize'] = fsize_xylabels

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": "Arial",
})

# Arial font everywhere
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

# Directories definition
dir_pointB = os.path.join('results', 'fig4', 'K_0.13', 'data', '7.0_nA', '0.00_1800.0_ms', '15-08-2023 12H34M03S')
dir_noICAN = os.path.join('results', 'fig5B_test_noICAN', '7.0_nA', '0.00_1800.0_ms', '22-09-2023 16H27M41S')
data_dirs = [os.path.join(dir_pointB, 'data'), os.path.join(dir_noICAN, 'data')]
filename = 'parameters_bak.json'

# Load the configuration files for this simulation
print('[+] Loading parameters file...')
print('[>] Point B...')
try:
    params_pointB = parameters.load(os.path.join(dir_pointB, filename))
except Exception as e:
    print('[!]' + "Error code " + str(e.errno) + ": " + e.strerror)
    sys.exit(-1)

print('[>] Without I_CAN...')
try:
    params_noICAN = parameters.load(os.path.join(dir_noICAN, filename))
except Exception as e:
    print('[!]' + "Error code " + str(e.errno) + ": " + e.strerror)
    sys.exit(-1)

# Parameters initialization
print('[+] Setting up parameters...')

# Timing
second = 1.
ms = 1e-3
duration = params_noICAN["simulation"]["duration"]
dt = params_noICAN["simulation"]["dt"]
t_stim_pointB = params_pointB["stimulation"]["onset"]
t_stim_noICAN = params_noICAN["stimulation"]["onset"]
fs = int(1/dt)
winsize_FR = 5*ms
overlap_FR = 0.9
winstep_FR = winsize_FR*round(1-overlap_FR,4)
fs_FR = int(1/winstep_FR)
binnum = int(duration/winsize_FR)
tv = np.arange(0., duration, dt)
t_lims = [1500*ms, 4000*ms] # ms : x-axs limits
t_lims_post = [2000*ms, 7000*ms]
duration_post = t_lims_post[1] - t_lims_post[0]

# Area names and sizes
areas = [['EC_pyCAN', 'EC_inh'], ['DG_py', 'DG_inh'], ['CA3_pyCAN', 'CA3_inh'], ['CA1_pyCAN', 'CA1_inh']]
area_labels = ['EC', 'DG', 'CA3', 'CA1']
N_tot = [[10000, 1000], [10000, 100], [1000, 100], [10000, 1000]]

# Raster downsampling
N_scaling = 100
N_gap = 10

# Color selection
c_inh = '#bf616a'
c_exc = '#5e81ac'

# Firing rates plotting gap
rates_gap = 25 # Hz

# Set raster limits
xlims_rasters = [t for t in t_lims]
ylims_rasters = [0, 2*N_scaling+N_gap]

# Set firing rate limits
xlims_rates = [t for t in t_lims]
ylims_rates = [-1, 400]

# Make figure outline
fig_width = 7.5
fig_height = 4.5
dpi = 300

# Panels
panel_labels = [r"A.", r"B."]
panel_titles = [r"With $I_{CAN}$", r"Without $I_{CAN}$"]

# Make a figure
fig = plt.figure(figsize=(fig_width,fig_height))

# Use gridspecs
G_outer = fig.add_gridspec(1, 2, left=0.02, right=0.9, bottom=0.05, top=0.95,
                                        # height_ratios=(0.15, 0.2125, 0.2125, 0.2125, 0.2125),
                                        wspace=0.15)

# Separate the panels
G_A = GridSpecFromSubplotSpec(2, 1, hspace=0.1, subplot_spec=G_outer[0])
G_B = GridSpecFromSubplotSpec(2, 1, hspace=0.1, subplot_spec=G_outer[1])
G_rasters = []
G_FRs = []
G_rasters.append(G_A[0])
G_rasters.append(G_B[0])
G_FRs.append(G_A[1])
G_FRs.append(G_B[1])

# Fix tight layout
G_outer.tight_layout(fig)

# Organize axes
#------------------------
axs_rasters = []
axs_FRs = []

for idx, Gs in enumerate(zip(G_rasters, G_FRs)):
    # Rasters
    # ------------------------
    print('[>] Rasters')

    # Create the axes and append them to the list
    ax_raster = fig.add_subplot(Gs[0])
    axs_rasters.append(ax_raster)

    # Stim indicator
    ax_raster.scatter(x=t_stim_pointB, y=225, s=15, linewidth=1., marker='v', c='gray', edgecolors=None, alpha=1, rasterized=False, clip_on=False)

    # Set limits
    ax_raster.set_xlim(xlims_rasters)
    ax_raster.set_ylim(ylims_rasters)

    # Titles
    # ax_raster.set_title('CA1 Activity', fontsize=fsize_titles)
    # ax_raster.set_title(panel_names[idx], fontsize=fsize_titles)
    ax_raster.set_title(panel_titles[idx], fontsize=fsize_panels)
    ax_raster.text(x=0., y=1.02, transform=ax_raster.transAxes, 
                   s=panel_labels[idx], weight='bold', fontsize=fsize_panels, 
                   ha='left', va='bottom', clip_on=False)

    # Axes
    ax_raster.xaxis.set_visible(False)
    # ax_raster.yaxis.set_visible(False)

    # Spines
    ax_raster.spines['top'].set_visible(False)
    ax_raster.spines['bottom'].set_visible(False)
    ax_raster.spines['left'].set_visible(False)
    ax_raster.spines['right'].set_visible(False)

    # Fix the ytick locators
    ax_raster.yaxis.set_major_locator(ticker.NullLocator())
    ax_raster.yaxis.set_minor_locator(ticker.NullLocator())

    # Remove tick labels
    # ax_raster.xaxis.set_ticklabels([])
    ax_raster.yaxis.set_ticklabels([])


    # Firing Rates
    # ------------------------
    print('[>] Firing Rates')
    ax_rate = fig.add_subplot(Gs[1])
    axs_FRs.append(ax_rate)

    # Stim indicator
    ax_rate.axvline(x=t_stim_pointB, ymin=0, ymax=2.1, color='gray', alpha=0.75, ls='-', linewidth=0.75, zorder=10, rasterized=False, clip_on=False)

    # Set the x-y limits
    ax_rate.set_xlim(xlims_rasters)
    ax_rate.set_ylim(ylims_rates)

    # Axes
    ax_rate.xaxis.set_visible(False)
    ax_rate.yaxis.set_visible(False)

    # Spines
    ax_rate.spines['top'].set_visible(False)
    ax_rate.spines['bottom'].set_visible(False)
    ax_rate.spines['left'].set_visible(False)
    ax_rate.spines['right'].set_visible(False)

    # Fix the ytick locators
    ax_rate.yaxis.set_major_locator(ticker.NullLocator())
    ax_rate.yaxis.set_minor_locator(ticker.NullLocator())

    # Remove tick labels
    # ax_rate.xaxis.set_ticklabels([])
    ax_rate.yaxis.set_ticklabels([])


# ==================
# Plot Rasters + FRs
# ==================
for input_dir, ax_raster, ax_FR in zip(data_dirs, axs_rasters, axs_FRs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, append=1)

        # load t-i arrays for this area
        print('[+] Loading the spikes for CA1')
        i_exc = np.loadtxt(os.path.join(input_dir, 'spikes', 'CA1_pyCAN_spikemon_i.txt'))
        t_exc = np.loadtxt(os.path.join(input_dir, 'spikes', 'CA1_pyCAN_spikemon_t.txt'))
        i_inh = np.loadtxt(os.path.join(input_dir, 'spikes', 'CA1_inh_spikemon_i.txt'))
        t_inh = np.loadtxt(os.path.join(input_dir, 'spikes', 'CA1_inh_spikemon_t.txt'))

    i_exc = i_exc.astype(int)
    t_exc = t_exc*ms
    i_inh = i_inh.astype(int)
    t_inh = t_inh*ms

    # sort based on index number (lower to higher)
    idx_sort_exc = np.argsort(i_exc)
    idx_sort_inh = np.argsort(i_inh)
    i_exc = i_exc[idx_sort_exc]
    t_exc = t_exc[idx_sort_exc]
    i_inh = i_inh[idx_sort_inh]
    t_inh = t_inh[idx_sort_inh]

    # set number of neurons
    N_exc = N_tot[3][0]
    N_inh = N_tot[3][1]

    # select some neurons randomly, subscaling
    exc_mixed = np.arange(0, N_exc+1, int(N_exc/N_scaling))
    inh_mixed = np.arange(0, N_inh+1, int(N_inh/N_scaling))

    idx_exc = np.in1d(i_exc, exc_mixed)
    idx_inh = np.in1d(i_inh, inh_mixed)

    i_exc_sub = i_exc[idx_exc]
    t_exc_sub = t_exc[idx_exc]
    i_inh_sub = i_inh[idx_inh]
    t_inh_sub = t_inh[idx_inh]

    # assign new neuron count numbers
    cnt = 0
    i_exc_sub_new = np.copy(i_exc_sub)
    for ii in exc_mixed:
        idx_tmp = np.where(i_exc_sub == ii)
        # print('changing ', ii, 'to ', cnt)
        i_exc_sub_new[idx_tmp] = cnt
        cnt += 1
    i_exc_sub = i_exc_sub_new

    # cnt = 0
    cnt += N_gap
    i_inh_sub_new = np.copy(i_inh_sub)
    for ii in inh_mixed:
        idx_tmp = np.where(i_inh_sub == ii)
        # print('changing ', ii, 'to ', cnt)
        i_inh_sub_new[idx_tmp] = cnt
        cnt += 1
    i_inh_sub = i_inh_sub_new

    # plot spikes
    print('[>] Plotting spikes...')

    # inhibitory
    ax_raster.scatter(t_inh_sub, i_inh_sub, s=1.25, linewidth=1., marker='.', c=c_inh, edgecolors='none', alpha=1., rasterized=True)

    # excitatory
    ax_raster.scatter(t_exc_sub, i_exc_sub, s=1.25, linewidth=1., marker='.', c=c_exc, edgecolors='none', alpha=1., rasterized=True)

    # mean FRs
    FR_inh_mean = (sum((t_inh>=t_lims_post[0]) & (t_inh<t_lims_post[1]))/duration_post)/N_inh
    FR_exc_mean = (sum((t_exc>=t_lims_post[0]) & (t_exc<t_lims_post[1]))/duration_post)/N_exc

    # add it as a text
    ax_raster.text(x=xlims_rates[1]+30*ms, y=1.75*N_scaling+N_gap, s=r'$\mu_I$: {0:.1f} Hz'.format(FR_inh_mean), fontsize=fsize_xylabels, ha='left', color=c_inh, clip_on=False)
    ax_raster.text(x=xlims_rates[1]+30*ms, y=N_scaling//2, s=r'$\mu_E$: {0:.1f} Hz'.format(FR_exc_mean), fontsize=fsize_xylabels, ha='left', color=c_exc, clip_on=False)

    # calculate firing rates
    print('[>] Computing firing rates...')
    tv_inh_FR, FR_inh, fs_FR2 = my_FR(spikes=t_inh, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
    tv_exc_FR, FR_exc, _ = my_FR(spikes=t_exc, duration=duration, window_size=winsize_FR, overlap=overlap_FR)

    # Normalize the FRs
    FR_inh_norm = (FR_inh/winsize_FR)/N_inh
    FR_exc_norm = (FR_exc/winsize_FR)/N_exc

    # Plot the FRs
    print('[>] Plotting rates...')
    ax_FR.plot(tv_inh_FR, FR_inh_norm+FR_exc_norm.max()+rates_gap, ls='-', linewidth=1.2, c=c_inh, label='inh', zorder=10, rasterized=False)
    ax_FR.plot(tv_exc_FR, FR_exc_norm, ls='-', linewidth=1.2, c=c_exc, label='exc', zorder=10, rasterized=False)

# Text mark inhibitory/excitatory in rasters
ax_FR.text(x=xlims_rates[1]+30*ms, y=ylims_rates[0]+75+FR_exc_norm.max()+rates_gap, s='Inhibitory', fontsize=fsize_legends, ha='left', color=c_inh, clip_on=False)
ax_FR.text(x=xlims_rates[1]+30*ms, y=ylims_rates[0]+25, s='Excitatory', fontsize=fsize_legends, ha='left', color=c_exc, clip_on=False)

# Sizebar for x-axis
xlims_sz = [t_lims[0] + 25*ms, t_lims[0]+275*ms]
ylims_sz = [-5, 25]
add_sizebar(axs_FRs[0], xlims_sz, [-10]*2, 'black', '250 ms', fsize=fsize_xylabels, rot=0, textx=np.mean(xlims_sz), texty=ylims_sz[0]-15, ha='center', va='top')

# Save and show the figure
print('[+] Saving the figure...')
fig.savefig(os.path.join(parent_dir, 'figures', 'fig5', "I_CAN_comparison" + '.png'), transparent=True, dpi=dpi, format='png', bbox_inches='tight')
fig.savefig(os.path.join(parent_dir, 'figures', 'fig5', "I_CAN_comparison" + '.pdf'), transparent=True, dpi=dpi, format='pdf', bbox_inches='tight')

plt.show()

# Exit - no errors
sys.exit(0)

sys.exit(0)