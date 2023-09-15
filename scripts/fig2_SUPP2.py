#!/usr/bin/env python3
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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import font_manager as fm
from matplotlib import ticker

# Other scripts and my stuff
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = Path(script_dir).parent
sys.path.insert(0, os.path.abspath(parent_dir))

from src.freq_analysis import *
from src.figure_plots_parameters import *

# ILLUSTRATOR STUFF
mplb.rcParams['pdf.fonttype'] = 42
mplb.rcParams['ps.fonttype'] = 42
mplb.rcParams['axes.titlesize'] = 11
mplb.rcParams['axes.labelsize'] = 9

# Arial font everywhere
# ILLUSTRATOR STUFF
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": "Arial",
})

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

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

# Main program
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate revision figure on gamma oscillations')

    parser.add_argument('-fn', '--figure-name',
                        type=str,
                        default='fig_rev_gamma',
                        help='Name of the output figure [w/o file extension]')
    args = parser.parse_args()

    """ Parameters initialization """
    print('[+] Setting up parameters...')

    # Directories
    data_dir = os.path.join(parent_dir, 'jupyter', 'data', 'fig_gamma_small_net')
    spikes_dir = os.path.join(data_dir, 'spikes')
    figures_dir = os.path.join(data_dir, 'figures')
    data_IF_dir = os.path.join(parent_dir, 'jupyter', 'data', 'I_F_curves')
    spikes_IF_dir = os.path.join(data_IF_dir, 'spikes')
    counts_IF_dir = os.path.join(data_IF_dir, 'data')

    # Timing
    second = 1
    ms = 1e-3
    duration = 5*second
    dt = 0.1*ms
    fs = int(1*second/dt)
    tv = np.linspace(0, duration, fs*duration)/ms
    winsize_FR = 5*ms
    overlap_FR = 0.9

    # Area names and sizes
    areas = ['pyCAN', 'inh']
    N_E = 1000
    N_I = 100

    # Color selection
    c_inh = '#bf616a'
    c_exc = '#5e81ac'

    # Figure sizes
    fig_width = 7.5
    fig_height = 8.0

    # Ramping input - non-repeating
    dt_ramp = 1*ms
    fs_ramp = 1/dt_ramp
    ramp_min = 0.
    ramp_max = 1.
    ramp_peak_t = 5*second
    tv_ramp = np.arange(0, duration, dt_ramp)
    xvals = np.arange(0, (ramp_peak_t/second), dt_ramp /second)
    ramp = ramp_min + (ramp_max-ramp_min)/(ramp_peak_t/second) * xvals

    # Parameters for FRs
    window_size = 1*ms
    bin_num = int(duration/window_size)
    fs_FR = 1/window_size

    # Load the spike data, case #1:
    conn_cases = ["DC", "DC"]
    noise_E = [0, 1000]
    noise_I = [0, 100]
    ramp_E = True
    ramp_I = True

    # Filenames
    fnames = []
    fnames.append(f"input_ramp_E_{bool(ramp_E)}_I_{bool(ramp_I)}_noise_E_{noise_E[0]:0.0f}uV_I_{noise_I[0]:0.0f}uV")
    fnames.append(f"input_ramp_E_{bool(ramp_E)}_I_{bool(ramp_I)}_noise_E_{noise_E[1]:0.0f}uV_I_{noise_I[1]:0.0f}uV")

    # F-I Data for panel A
    E_counts = np.loadtxt(os.path.join(counts_IF_dir, 'E_count_500.txt'))
    I_counts = np.loadtxt(os.path.join(counts_IF_dir, 'I_count_500.txt'))
    I_input = np.loadtxt(os.path.join(counts_IF_dir, 'I_input_500.txt'))

    # Plotting here
    # -------------------------------------------------
    # Figure sizes
    fig_width = 7.5
    fig_height = 8.0

    # Limits
    xlims_time = [0.5, 1.85]
    ylims_input = [0.05, 0.4]
    ylims_rasters = [0, N_E+N_I]
    ylims_FRs = [-0.1, 2.1]

    # Make a figure
    fig = plt.figure(figsize=(fig_width,fig_height))

    G_outer = GridSpec(4,1, left=0.1, right=0.925, bottom=0.05, top=0.97,
               height_ratios=(0.2, 0.1, 0.35, 0.35),
               # hspace=0.5, wspace=0.5,
               hspace=0.35,
               figure=fig)
    G_FIs = GridSpecFromSubplotSpec(1, 2, wspace=0.2, subplot_spec=G_outer[0])
    ax_FI_E = fig.add_subplot(G_FIs[0])
    ax_FI_I = fig.add_subplot(G_FIs[1])
    ax_input = fig.add_subplot(G_outer[1])
    ax_rasters = []
    ax_rasters.append(fig.add_subplot(G_outer[2], sharex=ax_input))
    ax_rasters.append(fig.add_subplot(G_outer[3], sharex=ax_input))

    # Fixed tick locators
    ax_input.set_xticks(np.arange(0, max(xlims_time)+0.1, 0.5))

    # Set the xlims/ylims
    ax_input.set_xlim(xlims_time)
    ax_rasters[0].set_ylim(ylims_rasters)
    ax_rasters[1].set_ylim(ylims_rasters)

    # Proper fontsizes
    ax_input.tick_params(axis='both', which='both', labelsize=fsize_ticks)
    ax_rasters[0].tick_params(axis='both', which='both', labelsize=fsize_ticks)
    ax_rasters[1].tick_params(axis='both', which='both', labelsize=fsize_ticks)

    # Remove the box borders
    ax_input.spines['top'].set_visible(False)
    ax_input.spines['right'].set_visible(False)
    ax_input.spines['bottom'].set_visible(False)
    ax_input.spines['left'].set_visible(False)

    ax_FI_E.spines['top'].set_visible(False)
    ax_FI_E.spines['right'].set_visible(False)
    # ax_FI_E.spines['bottom'].set_visible(False)
    # ax_FI_E.spines['left'].set_visible(False)

    ax_FI_I.spines['top'].set_visible(False)
    ax_FI_I.spines['right'].set_visible(False)
    # ax_FI_I.spines['bottom'].set_visible(False)
    # ax_FI_I.spines['left'].set_visible(False)

    ax_rasters[0].spines['top'].set_visible(False)
    ax_rasters[0].spines['right'].set_visible(False)
    ax_rasters[0].spines['bottom'].set_visible(False)
    ax_rasters[0].spines['left'].set_visible(False)

    ax_rasters[1].spines['top'].set_visible(False)
    ax_rasters[1].spines['right'].set_visible(False)
    ax_rasters[1].spines['bottom'].set_visible(False)
    ax_rasters[1].spines['left'].set_visible(False)

    # Remove ticks
    ax_input.get_xaxis().set_ticks([])
    ax_input.get_yaxis().set_ticks([])
    ax_rasters[0].get_xaxis().set_ticks([])
    ax_rasters[0].get_yaxis().set_ticks([])
    ax_rasters[1].get_xaxis().set_ticks([])
    ax_rasters[1].get_yaxis().set_ticks([])

    # Titles
    ax_input.set_title('Ramp Input', loc='center', fontsize=fsize_titles)
    ax_FI_E.set_title('Input-Frequency curve (Excitatory)', fontsize=fsize_titles)
    ax_FI_I.set_title('Input-Frequency curve (Inhibitory)', fontsize=fsize_titles)
    ax_rasters[0].set_title('Excitatory-Inhibitory Decoupled (noiseless)', loc='center', fontsize=fsize_titles)
    ax_rasters[1].set_title('Excitatory-Inhibitory Decoupled (added noise)', loc='center', fontsize=fsize_titles)

    # xy-labels
    ax_FI_E.set_ylabel('Spiking rate [Hz]', fontsize=fsize_xylabels)
    ax_FI_E.set_xlabel('Input [nA]', fontsize=fsize_xylabels)
    ax_FI_I.set_xlabel('Input [nA]', fontsize=fsize_xylabels)

    # Ticks
    ax_FI_E.tick_params(labelsize=fsize_ticks)
    ax_FI_I.tick_params(labelsize=fsize_ticks)

    # Panels text
    fig.text(0.02, 0.98, 'A.', weight='bold', fontsize=fsize_panels)
    fig.text(0.02, 0.78, 'B.', weight='bold', fontsize=fsize_panels)
    fig.text(0.02, 0.63, 'C.', weight='bold', fontsize=fsize_panels)
    fig.text(0.02, 0.3, 'D.', weight='bold', fontsize=fsize_panels)

    # Read the data
    for idx,conn in enumerate(conn_cases):

        # E
        try:
            E_i = np.loadtxt(os.path.join(spikes_dir, f"{conn}_{areas[0]}_{fnames[idx]}_i.txt"))
        except OSError:
            print('File not found for parameter set:')
            print(os.path.join(spikes_dir, f"{conn}_{areas[0]}_{fnames[idx]}_i.txt"))
            print(f'\tConnectivity: {conn}')
            print(f'\tramp_E: {ramp_E} | ramp_I {ramp_I}')
            print(f'\tnoise_E: {noise_E[idx]} | noise_I {noise_I[idx]}')
            print('Continuing...')
            print('-'*16)
            continue

        E_t = np.loadtxt(os.path.join(spikes_dir, f"{conn}_{areas[0]}_{fnames[idx]}_t.txt"))
        E_t *= ms

        # I
        I_i = np.loadtxt(os.path.join(spikes_dir, f"{conn}_{areas[1]}_{fnames[idx]}_i.txt"))
        I_t = np.loadtxt(os.path.join(spikes_dir, f"{conn}_{areas[1]}_{fnames[idx]}_t.txt"))
        I_t *= ms

        # Compute the FRs
        tv_inh_FR, FR_inh, fs_FR2 = my_FR(spikes=I_t, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
        tv_exc_FR, FR_exc, _ = my_FR(spikes=E_t, duration=duration, window_size=winsize_FR, overlap=overlap_FR)

        # Plot the rasters and FRs
        ax_rasters[idx].plot(E_t, E_i, linestyle='None', color=c_exc, marker='.', markersize=1)
        ax_rasters[idx].plot(I_t, I_i+N_E, linestyle='None', color=c_inh, marker='.', markersize=1)

        # ax_FRs.plot(tv_E, (FR_E/N_E), color=c_exc, label='Excitatory')
        # ax_FRs.plot(tv_I, (FR_I/N_I)+1, color=c_inh, label='Inhibitory')


    # Plot the common data
    ax_input.plot(tv_ramp, ramp, color='k')
    ax_input.set_ylim(ylims_input)

    # Plot the F-I Curves
    ax_FI_E.plot(I_input/1e-9, E_counts, color=c_exc, label='Excitatory')
    ax_FI_I.plot(I_input/1e-9, I_counts, color=c_inh, label='Inhibitory')


    # Add sizebars
    add_sizebar(ax_input, [xlims_time[1]+0.03, xlims_time[1]+0.03], [0, 0.25],
                'black', ['0 nA', '0.25'], fsize=fsize_misc, rot=[0, 0], 
                textx=[xlims_time[1]+0.05]*2, texty=[-0., 0.25],
                ha='left', va='center')

    xlims_sz = [xlims_time[1]-300*ms, xlims_time[1]-50*ms]
    ylims_sz = [-50]*2
    add_sizebar(ax_rasters[1], xlims_sz, ylims_sz,
                'black', '250 ms', fsize=fsize_misc, rot=0,
                textx=np.mean(xlims_sz), texty=ylims_sz[0]-50,
                ha='center', va='top')

    # add_sizebar(ax_FRs, [xlims_time[1]-0.25, xlims_time[1]-0.], [-0.1, -0.1], 'black', '250ms', fsize=fsize_misc, rot=0, 
    #             textx=np.mean([xlims_time[1]-0.25, xlims_time[1]-0.]), texty=-0.13, 
    #             ha='center', va='top')

    # Set the xlims
    ax_input.set_xlim(xlims_time)
    # ax_rasters.set_xlim(xlims_time)
    # ax_FRs.set_xlim(xlims_time)

    # Legends
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax_FI_E, ax_FI_I]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax_FI_E.legend(lines, labels, loc='upper left')

    # Show the figure
    # G_outer.tight_layout(fig)
    
    # Save the figure
    fig.savefig(os.path.join('figures', 'rev_eLife', "S2_2_rev_eLife.png"))
    fig.savefig(os.path.join('figures', 'rev_eLife', "S2_2_rev_eLife.pdf"))

    # Show the figure
    plt.show()

    # Close the figure
    # plt.close()

    # Plotting done
    print('Done with generating figures.')

    # Exit properly
    sys.exit(0)