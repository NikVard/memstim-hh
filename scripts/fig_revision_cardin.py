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

# Sig processing
from scipy.signal import welch
from scipy.integrate import simpson

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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate revision figure on optogenetic stimulation')

    parser.add_argument('-fn', '--figure-name',
                        type=str,
                        default='Pxx_Cardin_rev_eLife',
                        help='Name of the output figure [w/o file extension]')
    args = parser.parse_args()

    """ Parameters initialization """
    print('[+] Setting up parameters...')

    # Timing
    second = 1
    ms = 1e-3
    duration = 5*second
    dt = 0.1*ms
    fs = int(1*second/dt)
    tv = np.linspace(0, duration, fs*duration)/ms
    winsize_FR = 5*ms

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

    # Parameters for FRs
    window_size = 1*ms
    bin_num = int(duration/window_size)
    fs_FR = 1/window_size

    # Parameters for Pxxs
    win_len = 1*second;
    nperseg = int(win_len * fs_FR)
    overlap = 0.99
    NFFT = 2**((nperseg-1).bit_length()+1) # np.ceil(np.log2(window_width))
    NFFT = nperseg
    fband = 10

    # Define the range of stimulation frequencies
    fstims_all = np.arange(0,201,10)
    fstims_all[0] = 1 # first value cannot be zero

    # Directories
    datasets = ['Friday_v2_5nA', 'Friday_v3_5nA', 'Friday_v4_5nA']
    filestuff = [[1, 0], [0, 1], [1, 1]]
    data_dir = os.path.join(parent_dir, 'jupyter', 'data', 'StimOpto')
    figs_dir = os.path.join(parent_dir, 'jupyter', 'figures', 'StimOpto')
    figs_out_dir = os.path.join(parent_dir, 'figures', 'revision', 'figures_Cardin')

    # Make a figure
    fig,axs = plt.subplots(3,1, figsize=(fig_width,fig_height))

    for idx_data,foldername in enumerate(datasets):
        spikes_dir = os.path.join(data_dir, f'spikes_{datasets[idx_data]}')
        figures_dir = os.path.join(data_dir, f'figs_{datasets[idx_data]}')
        stim_E, stim_I = filestuff[idx_data]

        # Storage of data
        vals_E = []
        vals_I = []

        for idx,fstim in enumerate(fstims_all):
            print(f'IDX {idx}:', '-'*16)

            # Range for the relative power computation (Cardin et al. 2009)
            frange = fstim + np.round(fband/2,2) * np.array([-1,1])

            # Check only valid ranges
            if (frange < 0).any():
                print('[!] Stimulation frequency range out of bounds, skipping!')
                continue

            # Filenames
            fname_E = f'{idx}_Spikes_E_fstim_{fstim}Hz_Istim_5nA_stim_E_{bool(stim_E)}_stim_I_{bool(stim_I)}_tonic_E_False_tonic_I_False_ICAN_on'
            fname_I = f'{idx}_Spikes_I_fstim_{fstim}Hz_Istim_5nA_stim_E_{bool(stim_E)}_stim_I_{bool(stim_I)}_tonic_E_False_tonic_I_False_ICAN_on'
            
            # Load rasters
            print('[+] Loading spikes...')
            E_i = np.loadtxt(os.path.join(spikes_dir, f'{fname_E}_i.txt'))
            E_t = np.loadtxt(os.path.join(spikes_dir, f'{fname_E}_t.txt'))
            E_t *= ms

            I_i = np.loadtxt(os.path.join(spikes_dir, f'{fname_I}_i.txt'))
            I_t = np.loadtxt(os.path.join(spikes_dir, f'{fname_I}_t.txt'))
            I_t *= ms

            # Calculate FRs
            print('[+] Computing FRs...')
            FR_E, tv_E = np.histogram(E_t/second, bins=bin_num, range=[0,duration])
            FR_I, tv_I = np.histogram(I_t/second, bins=bin_num, range=[0,duration])

            # Adjust the time vectors
            tv_E = tv_E[0:-1] + window_size/2
            tv_I = tv_I[0:-1] + window_size/2

            # Compute the Pxxs here (Welch's estimate)
            # Calculate the Pxxs
            print('[+] Computing the Pxx...')
            (f_E, Pxx_E) = welch(FR_E, fs_FR, nperseg=nperseg, nfft=NFFT, noverlap=round(nperseg*overlap), scaling='spectrum')
            (f_I, Pxx_I) = welch(FR_I, fs_FR, nperseg=nperseg, nfft=NFFT, noverlap=round(nperseg*overlap), scaling='spectrum')

            # Find intersecting values in frequency vector
            idx_band = np.logical_and(f_E >= frange[0], f_E <= frange[1])

            # Relative power ratio
            print('[+] Computing relative power ratios...')
            Pxx_tot = simpson(Pxx_E, f_E)
            Pxx_band = simpson(Pxx_E[idx_band], f_E[idx_band])
            vals_E.append(Pxx_band / Pxx_tot)

            Pxx_tot = simpson(Pxx_I, f_I)
            Pxx_band = simpson(Pxx_I[idx_band], f_I[idx_band])
            vals_I.append(Pxx_band / Pxx_tot)

        # Plotting here
        # -------------------------------------------------
        # Relative power
        axs[idx_data].plot(fstims_all[1:], vals_E, color=c_exc, linestyle='-', marker='o', markersize=5, label='Excitatory')
        axs[idx_data].plot(fstims_all[1:], vals_I, color=c_inh, linestyle='-', marker='s', markersize=5, label='Inhibitory')
        axs[idx_data].set_title(f'Stim. E {bool(stim_E)} | I {bool(stim_I)})', fontsize=fsize_titles)
        axs[idx_data].set_ylabel('Relative power', fontsize=fsize_xylabels)
        axs[idx_data].set_xlim([8, 202])
        axs[idx_data].set_ylim([-0.005, 0.36])
        axs[idx_data].xaxis.set_major_locator(ticker.FixedLocator(fstims_all))

    # Last idx
    axs[idx_data].set_xlabel('Stim. frequency [Hz]')
    axs[idx_data].legend(fontsize=fsize_legends)

    # Remove xticks
    axs[0].xaxis.set_ticklabels([])
    axs[1].xaxis.set_ticklabels([])

    # Fixed locator for y-ticks
    for ax in axs:
        # ax.set_yticks(np.arange(0.0, 0.35, 0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.15))
        # ax.yaxis.set_major_formatter('{x:.0f}')
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    # Proper fontsizes
    for ax in axs:
        ax.tick_params(axis='both', which='both', labelsize=fsize_ticks)
        # ax.tick_params(axis='both', which='major', length=0)
        # ax.tick_params(axis='both', which='minor', length=2)

    # Supertitle
    fig.suptitle(f'Relative power ({fband}-Hz band over total)', fontsize=fsize_titles)

    # Show the figure
    plt.tight_layout()

    # Plotting done
    print('[+] Done with generating figures. Saving...')
    
    # Save the figure
    fig.savefig(os.path.join(figs_out_dir, f"Pxx_Cardin_rev_eLife.png"))
    fig.savefig(os.path.join(figs_out_dir, f"Pxx_Cardin_rev_eLife.pdf"))

    # Show the output
    plt.show()

    # Close the figure
    # fig.close()

    # Exit properly
    sys.exit(0)