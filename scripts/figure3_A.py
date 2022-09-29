#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib as mplb
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import font_manager as fm


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = Path(script_dir).parent
sys.path.insert(0, os.path.abspath(parent_dir))

from src.freq_analysis import *

fontprops = fm.FontProperties(size=12, family='monospace')

# ILLUSTRATOR STUFF
mplb.rcParams['pdf.fonttype'] = 42
mplb.rcParams['ps.fonttype'] = 42
mplb.rcParams['axes.titlesize'] = 11
mplb.rcParams['axes.labelsize'] = 9


# main program
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate figure 3A from paper')

    parser.add_argument('-fn', '--figure-name',
                        type=str,
                        default='fig3_A',
                        help='Name of the output figure [w/o file extension]')

    args = parser.parse_args()

    """ Parameters initialization """
    print('[+] Setting up parameters...')

    # Timing
    second = 1
    ms = 1e-3
    duration = 3*second
    dt = 0.1*ms
    fs = int(1*second/dt)
    tv = np.linspace(0, duration, fs*duration)
    winsize_FR = 5*ms
    overlap_FR = 0.9
    winstep_FR = winsize_FR*round(1-overlap_FR,4)
    fs_FR = int(1/winstep_FR)

    # Area names and sizes
    areas = [['EC_pyCAN', 'EC_inh'], ['DG_py', 'DG_inh'], ['CA3_pyCAN', 'CA3_inh'], ['CA1_pyCAN', 'CA1_inh']]
    area_labels = ['EC', 'DG', 'CA3', 'CA1']
    N_tot = [[10000, 1000], [10000, 100], [1000, 100], [10000, 1000]]

    # Directories
    fig3_dir = os.path.join(parent_dir, 'results', 'analysis', 'current', 'fig3')
    fig3_data = os.path.join(fig3_dir, 'data')
    fig3_currents = os.path.join(fig3_data, 'currents')

    # Color selection
    c_inh = '#bf616a'
    c_exc = '#5e81ac'

    # Figure sizes
    fig_width = 7.5
    fig_height = 8.5

    # Font size
    fsize = 9


    """ Plot Figure 3 of the paper - TODO: Add DOI"""
    print('[+] Generating the figure...')

    # Make a figure
    fig = plt.figure(figsize=(fig_width,fig_height))

    # Use gridspecs
    G_outer = GridSpec(4, 2, left=0.1, right=0.9, bottom=0.05, top=0.95,
                        wspace=0.5, hspace=0.3, width_ratios=(0.5, 0.5), figure=fig)
    G_panel_A = G_outer[:,0].subgridspec(4, 1, hspace=0.3)
    G_panel_B = G_outer[:2,1].subgridspec(2, 1, hspace=0.3)
    G_panel_C = G_outer[2:,1].subgridspec(1, 1, hspace=0.3)
    G_outer.tight_layout(fig)


    # Organize axes
    #------------------------
    axs = []

    # Panel A - I/E rates | I_CAN | I_M - case w/ currents
    ax_A0 = fig.add_subplot(G_panel_A[0])
    ax_A1 = fig.add_subplot(G_panel_A[1], sharex=ax_A0, sharey=ax_A0)
    ax_A2 = fig.add_subplot(G_panel_A[2], sharex=ax_A0)
    ax_A3 = fig.add_subplot(G_panel_A[3], sharex=ax_A0)
    axs.append([ax_A0, ax_A1, ax_A2, ax_A3])

    # Set titles and x-y labels
    ax_A0.text(0.5, 1.2, 'I_CAN on', fontsize=12, horizontalalignment='center', verticalalignment='center', transform=ax_A0.transAxes, clip_on=False)

    ax_A0.set_title('CA1 Inhibitory FR')
    ax_A1.set_title('CA1 Excitatory FR')
    ax_A2.set_title('CA1 Excitatory I_CAN')
    ax_A3.set_title('CA1 Excitatory I_M')

    ax_A0.set_ylabel('FR [Hz]')
    ax_A1.set_ylabel('FR [Hz]')
    ax_A2.set_ylabel(r'I_{CAN} [nA]')
    ax_A3.set_ylabel(r'I_{M} [nA]')
    ax_A3.set_xlabel('Time [ms]')

    # Hide some spines
    ax_A0.spines['top'].set_visible(False)
    # ax_A0.spines['bottom'].set_visible(False)
    # ax_A0.spines['left'].set_visible(False)
    ax_A0.spines['right'].set_visible(False)
    ax_A1.spines['top'].set_visible(False)
    # ax_A1.spines['bottom'].set_visible(False)
    # ax_A1.spines['left'].set_visible(False)
    ax_A1.spines['right'].set_visible(False)
    ax_A2.spines['top'].set_visible(False)
    # ax_A2.spines['bottom'].set_visible(False)
    # ax_A2.spines['left'].set_visible(False)
    ax_A2.spines['right'].set_visible(False)
    ax_A3.spines['top'].set_visible(False)
    # ax_A3.spines['bottom'].set_visible(False)
    # ax_A3.spines['left'].set_visible(False)
    ax_A3.spines['right'].set_visible(False)

    # Remove tick labels
    # ax_A0.xaxis.set_ticklabels([])
    # ax_A0.yaxis.set_ticklabels([])
    # ax_A1.xaxis.set_ticklabels([])
    # ax_A1.yaxis.set_ticklabels([])
    # ax_A2.xaxis.set_ticklabels([])
    # ax_A2.yaxis.set_ticklabels([])
    # ax_A3.xaxis.set_ticklabels([])
    # ax_A3.yaxis.set_ticklabels([])
    plt.setp(ax_A0.get_xticklabels(), visible=False)
    plt.setp(ax_A1.get_xticklabels(), visible=False)
    plt.setp(ax_A2.get_xticklabels(), visible=False)

    # Panel B - I/E rates - case w/o currents
    ax_B0 = fig.add_subplot(G_panel_B[0], sharex=ax_A0)
    ax_B1 = fig.add_subplot(G_panel_B[1], sharex=ax_A0, sharey=ax_B0)
    axs.append([ax_B0, ax_B1])

    # Set titles and x-y labels
    ax_B0.text(0.5, 1.2, 'I_CAN off', fontsize=12, horizontalalignment='center', verticalalignment='center', transform=ax_B0.transAxes, clip_on=False)

    ax_B0.set_title('CA1 Inhibitory FR')
    ax_B1.set_title('CA1 Excitatory FR')

    ax_B0.set_ylabel('FR [Hz]')
    ax_B1.set_ylabel('FR [Hz]')
    ax_B1.set_xlabel('Time [ms]')

    # Hide some spines
    ax_B0.spines['top'].set_visible(False)
    # ax_B0.spines['bottom'].set_visible(False)
    # ax_B0.spines['left'].set_visible(False)
    ax_B0.spines['right'].set_visible(False)
    ax_B1.spines['top'].set_visible(False)
    # ax_B1.spines['bottom'].set_visible(False)
    # ax_B1.spines['left'].set_visible(False)
    ax_B1.spines['right'].set_visible(False)

    # Remove tick labels
    plt.setp(ax_B0.get_xticklabels(), visible=False)


    # Panel C - Quantification
    ax_C0 = fig.add_subplot(G_panel_C[0])
    ax_C02 = ax_C0.twinx()
    axs.append([ax_C0])

    # Set titles and x-y labels
    ax_C0.set_xlabel('Stimulation amplitude [nA]')
    ax_C0.set_ylabel('Spike count [spk/neuron]')
    ax_C02.set_ylabel('Burst duration [ms]', color='r')


    # Fix the ytick locators
    # ax_C0.yaxis.set_major_locator(ticker.NullLocator())
    # ax_C0.yaxis.set_minor_locator(ticker.NullLocator())


    # Load the data
    #------------------------
    dir_data = [os.path.join(parent_dir, 'results_cluster', 'results_noICAN_fig3_quantify', '10.0_nA', '0.00_1333.0_ms', '27-09-2022 18H07M53S', 'data'), os.path.join(parent_dir, 'results_cluster', 'results_ICAN_fig3_quantify', '10.0_nA', '0.00_1333.0_ms', '27-09-2022 18H42M54S', 'data')]
    dir_spikes = [os.path.join(dir_data[0], 'spikes'), os.path.join(dir_data[1], 'spikes')]
    dir_currents = [os.path.join(dir_data[0], 'currents'), os.path.join(dir_data[1], 'currents')]

    # Spikes
    print('[+] Loading the rasters...')
    # 10nA w/ I_CAN
    CA1_E_ICAN_t = np.loadtxt(os.path.join(dir_spikes[1], 'CA1_pyCAN_spikemon_t.txt'))
    CA1_E_ICAN_i = np.loadtxt(os.path.join(dir_spikes[1], 'CA1_pyCAN_spikemon_i.txt'))
    CA1_I_ICAN_t = np.loadtxt(os.path.join(dir_spikes[1], 'CA1_inh_spikemon_t.txt'))
    CA1_I_ICAN_i = np.loadtxt(os.path.join(dir_spikes[1], 'CA1_inh_spikemon_i.txt'))

    # 10nA w/o I_CAN
    CA1_E_noICAN_t = np.loadtxt(os.path.join(dir_spikes[0], 'CA1_pyCAN_spikemon_t.txt'))
    CA1_E_noICAN_i = np.loadtxt(os.path.join(dir_spikes[0], 'CA1_pyCAN_spikemon_i.txt'))
    CA1_I_noICAN_t = np.loadtxt(os.path.join(dir_spikes[0], 'CA1_inh_spikemon_t.txt'))
    CA1_I_noICAN_i = np.loadtxt(os.path.join(dir_spikes[0], 'CA1_inh_spikemon_i.txt'))

    # fix data
    i_exc_ICAN = CA1_E_ICAN_i.astype(int)
    t_exc_ICAN = CA1_E_ICAN_t*ms
    i_inh_ICAN = CA1_I_ICAN_i.astype(int)
    t_inh_ICAN = CA1_I_ICAN_t*ms

    i_exc_noICAN = CA1_E_noICAN_i.astype(int)
    t_exc_noICAN = CA1_E_noICAN_t*ms
    i_inh_noICAN = CA1_I_noICAN_i.astype(int)
    t_inh_noICAN = CA1_I_noICAN_t*ms


    # Calculate the FRs
    print('[+] Calculating FRs...')
    tv_inh_ICAN_FR, FR_inh_ICAN = my_FR(spikes=t_inh_ICAN, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
    tv_exc_ICAN_FR, FR_exc_ICAN = my_FR(spikes=t_exc_ICAN, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
    tv_inh_noICAN_FR, FR_inh_noICAN = my_FR(spikes=t_inh_noICAN, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
    tv_exc_noICAN_FR, FR_exc_noICAN = my_FR(spikes=t_exc_noICAN, duration=duration, window_size=winsize_FR, overlap=overlap_FR)

    # Normalize the FRs
    print('[+] Normalizing FRs...')
    FR_inh_ICAN_norm = (FR_inh_ICAN/winsize_FR)/N_tot[3][1]
    FR_exc_ICAN_norm = (FR_exc_ICAN/winsize_FR)/N_tot[3][0]
    FR_inh_noICAN_norm = (FR_inh_noICAN/winsize_FR)/N_tot[3][1]
    FR_exc_noICAN_norm = (FR_exc_noICAN/winsize_FR)/N_tot[3][0]

    # Currents
    print('[+] Loading I_CAN / I_M currents...')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, append=1)
        CA1_E_I_CAN = np.loadtxt(os.path.join(dir_currents[1], 'CA1_E_currents_I_CAN.txt'))
        CA1_E_I_M = np.loadtxt(os.path.join(dir_currents[1], 'CA1_E_currents_I_M.txt'))


    # Plot panel A
    #------------------------
    print('[>] Plotting panel A')
    ax_A0.plot(tv_inh_ICAN_FR/ms, FR_inh_ICAN_norm, ls='-', linewidth=1.2, c=c_inh, label='inh', zorder=10, rasterized=False)
    ax_A1.plot(tv_exc_ICAN_FR/ms, FR_exc_ICAN_norm, ls='-', linewidth=1.2, c=c_exc, label='exc', zorder=10, rasterized=False)
    ax_A2.plot(tv/ms, CA1_E_I_CAN[0], ls='-', linewidth=1.2, c='k', label='I_CAN')
    ax_A3.plot(tv/ms, CA1_E_I_M[0], ls='-', linewidth=1.2, c='k', label='I_M')


    # Plot panel B
    #------------------------
    print('[>] Plotting panel B')
    ax_B0.plot(tv_inh_noICAN_FR/ms, FR_inh_noICAN_norm, ls='-', linewidth=1.2, c=c_inh, label='inh', zorder=10, rasterized=False)
    ax_B1.plot(tv_exc_noICAN_FR/ms, FR_exc_noICAN_norm, ls='-', linewidth=1.2, c=c_exc, label='exc', zorder=10, rasterized=False)


    # Panel C - Quantification
    #------------------------
    print('[>] Plotting panel C')
    dir_cluster = os.path.join(parent_dir, 'results_cluster')
    dir_ICAN = os.path.join(dir_cluster, 'results_noICAN_fig3_quantify')
    dir_noICAN = os.path.join(dir_cluster, 'results_ICAN_fig3_quantify')

    # w/ ICAN
    print('[!+] Working on +I_CAN data')
    print('-'*32)

    duration_ICAN = []
    spikes_cnt_ICAN = []
    xarr_ICAN = []

    for item in os.listdir(dir_ICAN):
        print('[>] ', item)

        # get the stim amplitude - xval
        tmp = item.split("_")
        xval = float(tmp[0])
        xarr_ICAN.append(xval)

        # go over the sub-directories
        stim_dir = os.path.join(dir_ICAN, item)
        t_stim_dir = os.listdir(stim_dir)[0]
        stim_onset_dir = os.path.join(stim_dir, t_stim_dir)
        curr_path = os.path.join(stim_onset_dir, os.listdir(stim_onset_dir)[0])

        # load the data for the current simulation
        dir_data_curr = os.path.join(curr_path, 'data')
        dir_spikes_curr = os.path.join(dir_data_curr, 'spikes')

        # rasters
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, append=1)
            CA1_E_t = np.loadtxt(os.path.join(dir_spikes_curr, 'CA1_pyCAN_spikemon_t.txt'))
            CA1_E_i = np.loadtxt(os.path.join(dir_spikes_curr, 'CA1_pyCAN_spikemon_i.txt'))
            CA1_I_t = np.loadtxt(os.path.join(dir_spikes_curr, 'CA1_inh_spikemon_t.txt'))
            CA1_I_i = np.loadtxt(os.path.join(dir_spikes_curr, 'CA1_inh_spikemon_i.txt'))

        i_exc = CA1_E_i.astype(int)
        t_exc = CA1_E_t*ms
        i_inh = CA1_I_i.astype(int)
        t_inh = CA1_I_t*ms

        # calculate FRs
        tv_inh_FR, FR_inh = my_FR(spikes=t_inh, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
        tv_exc_FR, FR_exc = my_FR(spikes=t_exc, duration=duration, window_size=winsize_FR, overlap=overlap_FR)

        FR_inh_norm = (FR_inh/winsize_FR)/N_tot[3][1]
        FR_exc_norm = (FR_exc/winsize_FR)/N_tot[3][0]

        # envelope of bursts
        FR_exc_envelope = (FR_exc_norm > 10)

        # get number of bursts
        num_bursts = (np.convolve([-10, 0, 10], FR_exc_envelope)>0).sum()//2

        # get total burst duration
        dur_burst = np.count_nonzero(FR_exc_envelope)*dt/ms

        # append both values
        duration_ICAN.append([num_bursts, dur_burst])

        # count spikes after t_onset
        t_stim = float(t_stim_dir.split("_")[1])*ms
        spkcnt_exc = np.count_nonzero(t_exc >= t_stim)
        spkcnt_inh = np.count_nonzero(t_inh >= t_stim)
        spikes_cnt_ICAN.append([spkcnt_exc, spkcnt_inh])

        print('[+]....Spikes exc: ', spkcnt_exc)
        print('[+]....Spikes inh: ', spkcnt_inh)
        print()



    # w/o ICAN
    print()
    print('[!-] Working on -I_CAN data')
    print('-'*32)

    duration_noICAN = []
    spikes_cnt_noICAN = []
    xarr_noICAN = []

    for item in os.listdir(dir_noICAN):
        print('[>] ', item)

        # get the stim amplitude - xval
        tmp = item.split("_")
        xval = float(tmp[0])
        xarr_noICAN.append(xval)

        # go over the sub-directories
        stim_dir = os.path.join(dir_noICAN, item)
        t_stim_dir = os.listdir(stim_dir)[0]
        stim_onset_dir = os.path.join(stim_dir, t_stim_dir)
        curr_path = os.path.join(stim_onset_dir, os.listdir(stim_onset_dir)[0])

        # load the data for the current simulation
        dir_data_curr = os.path.join(curr_path, 'data')
        dir_spikes_curr = os.path.join(dir_data_curr, 'spikes')

        # rasters
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, append=1)
            CA1_E_t = np.loadtxt(os.path.join(dir_spikes_curr, 'CA1_pyCAN_spikemon_t.txt'))
            CA1_E_i = np.loadtxt(os.path.join(dir_spikes_curr, 'CA1_pyCAN_spikemon_i.txt'))
            CA1_I_t = np.loadtxt(os.path.join(dir_spikes_curr, 'CA1_inh_spikemon_t.txt'))
            CA1_I_i = np.loadtxt(os.path.join(dir_spikes_curr, 'CA1_inh_spikemon_i.txt'))

        i_exc = CA1_E_i.astype(int)
        t_exc = CA1_E_t*ms
        i_inh = CA1_I_i.astype(int)
        t_inh = CA1_I_t*ms

        # calculate FRs
        tv_inh_FR, FR_inh = my_FR(spikes=t_inh, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
        tv_exc_FR, FR_exc = my_FR(spikes=t_exc, duration=duration, window_size=winsize_FR, overlap=overlap_FR)

        FR_inh_norm = (FR_inh/winsize_FR)/N_tot[3][1]
        FR_exc_norm = (FR_exc/winsize_FR)/N_tot[3][0]

        # envelope of bursts
        FR_exc_envelope = (FR_exc_norm > 10)

        # get number of bursts
        num_bursts = (np.convolve([-10, 0, 10], FR_exc_envelope)>0).sum()//2

        # get total burst duration
        dur_burst = np.count_nonzero(FR_exc_envelope)*dt/ms

        # append both values
        duration_noICAN.append([num_bursts, dur_burst])

        # count spikes after t_onset
        t_stim = float(t_stim_dir.split("_")[1])*ms
        spkcnt_exc = np.count_nonzero(t_exc >= t_stim)
        spkcnt_inh = np.count_nonzero(t_inh >= t_stim)
        spikes_cnt_noICAN.append([spkcnt_exc, spkcnt_inh])

        print('[-]....Spikes exc: ', spkcnt_exc)
        print('[-]....Spikes inh: ', spkcnt_inh)
        print()

    # sort the arrays
    idx_ICAN = np.argsort(xarr_ICAN)
    idx_noICAN = np.argsort(xarr_noICAN)

    # plot the E - w/ ICAN vs the E - w/o ICAN
    ln01 = ax_C0.plot(np.array(xarr_ICAN)[idx_ICAN], np.array(spikes_cnt_ICAN)[idx_ICAN,0]/N_tot[3][0], '-xk', label='Spike count w/ I_CAN')
    ln02 = ax_C0.plot(np.array(xarr_noICAN)[idx_noICAN], np.array(spikes_cnt_noICAN)[idx_noICAN,0]/N_tot[3][0], '-ok', label='Spike count w/ I_CAN')
    ln10 = ax_C02.plot(np.array(xarr_ICAN)[idx_ICAN], np.array(duration_ICAN)[idx_ICAN,1], '-xr', label='burst duration w/ I_CAN')
    ln20 = ax_C02.plot(np.array(xarr_noICAN)[idx_noICAN], np.array(duration_noICAN)[idx_noICAN,1], '-or', label='burst duration w/o I_CAN')

    # one legend
    lns = ln01+ln02+ln10+ln20
    labs = [l.get_label() for l in lns]
    ax_C0.legend(lns, labs, loc=0, fontsize=8)


    # Save and show the figure
    print('[+] Saving the figure...')
    fig.savefig(os.path.join(parent_dir, 'figures', 'fig3', args.figure_name + '.png'), transparent=True, dpi=300, format='png', bbox_inches='tight')
    fig.savefig(os.path.join(parent_dir, 'figures', 'fig3', args.figure_name + '.pdf'), transparent=True, dpi=300, format='pdf', bbox_inches='tight')

    plt.show()

    # Exit properly
    sys.exit(0)
