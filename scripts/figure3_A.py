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
from matplotlib import ticker


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

# main program
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate figure 3A from paper')

    parser.add_argument('-ra', '--rasters-all',
                        action='store_true',
                        default=False,
                        help='Set to plot all rasters instead of only for CA1.')

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
    td = 100*ms # distance between consecutive spikes to count as a separate burst

    # Area names and sizes
    areas = [['EC_pyCAN', 'EC_inh'], ['DG_py', 'DG_inh'], ['CA3_pyCAN', 'CA3_inh'], ['CA1_pyCAN', 'CA1_inh']]
    area_labels = ['EC', 'DG', 'CA3', 'CA1']
    N_tot = [[10000, 1000], [10000, 100], [1000, 100], [10000, 1000]]

    # Color selection
    c_inh = '#bf616a'
    c_exc = '#5e81ac'
    c_ICAN = '#40916c'
    # c_IM = '#1b4332'
    c_IM = '#b76935'
    c_w_ICAN = 'k'
    c_wo_ICAN = 'k'

    # Figure sizes
    fig_width = 5.5
    fig_height = 4.0
    if args.rasters_all:
        fig_height = 8.5

    # Rasters
    N_scaling = 200
    N_gap = 5

    # Firing rates plotting gap
    rates_gap = 15 # Hz

    # Limits
    xlims = [1230, 1840]

    # Text parameters
    fsize_ticks = fsize_legends = 8
    fsize_xylabels = 9
    fsize_titles = 10
    fsize_figtitles = 11
    sizebar_off = 50 # sizebar offset


    """ Plot Figure 3 of the paper - TODO: Add DOI """
    print('[+] Generating the figure...')

    # Make a figure - TODO: Fix constrained_layout!!!
    fig = plt.figure(figsize=(fig_width,fig_height), tight_layout=True)

    # # Use gridspecs
    # G_outer = GridSpec(1, 2, left=0.1, right=0.95, bottom=0.15, top=0.925,
    #                     wspace=0.2, width_ratios=(0.5, 0.5), figure=fig)
    # G_left = G_outer[0,0]
    # if args.rasters_all:
    #     G_panel_A = G_left.subgridspec(3, 1, hspace=0.2, height_ratios=(0.6, 0.2, 0.2))
    #     G_rasters_A = G_panel_A[0].subgridspec(4, 1, hspace=0.1)
    # else:
    #     G_panel_A = G_left.subgridspec(3, 1, hspace=0.2, height_ratios=(0.35, 0.35, 0.3))
    #     G_rasters_A = G_panel_A[0]
    # G_FRs_A = G_panel_A[1]
    # G_currents = G_panel_A[2]
    #
    # G_right = G_outer[0,1].subgridspec(2, 1, height_ratios=(0.74, 0.26))
    # G_panel_B = G_right[0].subgridspec(2, 1, hspace=0.2, height_ratios=(0.5, 0.5))
    # G_panel_C = G_right[1]
    # if args.rasters_all:
    #     G_rasters_B = G_panel_B[0].subgridspec(4, 1, hspace=0.1)
    # else:
    #     G_rasters_B = G_panel_B[0]
    # G_FRs_B = G_panel_B[1]

    # panel A]
    if args.rasters_all:
        G_outer = GridSpec(3, 2, #left=0.1, right=0.95, bottom=0.15, top=0.925,
                           #hspace=0.1,
                           height_ratios=(0.6, 0.2, 0.2),
                           width_ratios=(0.5, 0.5),
                           figure=fig)
        G_rasters_A = G_outer[0, 0].subgridspec(4, 1)
    else:
        G_outer = GridSpec(3, 2, #left=0.1, right=0.95, bottom=0.15, top=0.925,
                           #hspace=0.1,
                           height_ratios=(0.3, 0.35, 0.35),
                           width_ratios=(0.5, 0.5),
                           figure=fig)
        G_rasters_A = G_outer[0, 0]
    G_FRs_A = G_outer[1, 0]
    G_currents = G_outer[2, 0]

    # panel B
    if args.rasters_all:
        G_rasters_B = G_outer[0, 1].subgridspec(4, 1)
    else:
        G_rasters_B = G_outer[0, 1]
    G_FRs_B = G_outer[1, 1]

    # panel C
    G_panel_C = G_outer[2, 1]

    G_outer.tight_layout(fig)


    # Organize axes
    #------------------------
    axs = []

    # Panel A - I/E rates | I_CAN | I_M - case w/ currents
    if args.rasters_all:
        ax_A0_0 = fig.add_subplot(G_rasters_A[0]) # EC E/I rasters
        ax_A0_1 = fig.add_subplot(G_rasters_A[1], sharex=ax_A0_0, sharey=ax_A0_0) # DG E/I rasters
        ax_A0_2 = fig.add_subplot(G_rasters_A[2], sharex=ax_A0_0, sharey=ax_A0_0) # CA3 E/I rasters
        ax_A0_3 = fig.add_subplot(G_rasters_A[3], sharex=ax_A0_0, sharey=ax_A0_0) # CA1 E/I rasters
        axs.insert(0, [ax_A0_0, ax_A0_1, ax_A0_2, ax_A0_3])

        # set label as area name
        ax_A0_0.set_title(r'   Activity w/ $I_{CAN}$', loc='center', fontsize=fsize_titles)
        # ax_A0_0.set_title(area_labels[0])
        # ax_A0_1.set_title(area_labels[1])
        # ax_A0_2.set_title(area_labels[2])
        # ax_A0_3.set_title(area_labels[3])
        ax_A0_0.set_ylabel(areas[0][0].split('_')[0], fontsize=fsize_xylabels, rotation=0, labelpad=20.)
        ax_A0_1.set_ylabel(areas[1][0].split('_')[0], fontsize=fsize_xylabels, rotation=0, labelpad=20.)
        ax_A0_2.set_ylabel(areas[2][0].split('_')[0], fontsize=fsize_xylabels, rotation=0, labelpad=20.)
        ax_A0_3.set_ylabel(areas[3][0].split('_')[0], fontsize=fsize_xylabels, rotation=0, labelpad=20.)

        # Set x-lims
        ax_A0_0.set_xlim(xlims)

        # Set y-lims
        ax_A0_0.set_ylim([0, 2*N_scaling+N_gap+1])

        # Hide x-y axes
        ax_A0_0.xaxis.set_visible(False)
        # ax_A0_0.yaxis.set_visible(False)
        ax_A0_1.xaxis.set_visible(False)
        # ax_A0_1.yaxis.set_visible(False)
        ax_A0_2.xaxis.set_visible(False)
        # ax_A0_2.yaxis.set_visible(False)
        ax_A0_3.xaxis.set_visible(False)
        # ax_A0_3.yaxis.set_visible(False)

        # Hide some spines
        ax_A0_0.spines['top'].set_visible(False)
        ax_A0_0.spines['bottom'].set_visible(False)
        ax_A0_0.spines['left'].set_visible(False)
        ax_A0_0.spines['right'].set_visible(False)
        ax_A0_1.spines['top'].set_visible(False)
        ax_A0_1.spines['bottom'].set_visible(False)
        ax_A0_1.spines['left'].set_visible(False)
        ax_A0_1.spines['right'].set_visible(False)
        ax_A0_2.spines['top'].set_visible(False)
        ax_A0_2.spines['bottom'].set_visible(False)
        ax_A0_2.spines['left'].set_visible(False)
        ax_A0_2.spines['right'].set_visible(False)
        ax_A0_3.spines['top'].set_visible(False)
        ax_A0_3.spines['bottom'].set_visible(False)
        ax_A0_3.spines['left'].set_visible(False)
        ax_A0_3.spines['right'].set_visible(False)

        # Fix the ytick locators
        ax_A0_0.yaxis.set_major_locator(ticker.NullLocator())
        ax_A0_0.yaxis.set_minor_locator(ticker.NullLocator())
        ax_A0_1.yaxis.set_major_locator(ticker.NullLocator())
        ax_A0_1.yaxis.set_minor_locator(ticker.NullLocator())
        ax_A0_2.yaxis.set_major_locator(ticker.NullLocator())
        ax_A0_2.yaxis.set_minor_locator(ticker.NullLocator())
        ax_A0_3.yaxis.set_major_locator(ticker.NullLocator())
        ax_A0_3.yaxis.set_minor_locator(ticker.NullLocator())


    else:
        ax_A0 = fig.add_subplot(G_rasters_A)
        axs.insert(0, [ax_A0])

        # Set titles and x-y labels
        # ax_A0.text(0.5, 1.2, 'I_CAN on', fontsize=12, horizontalalignment='center', verticalalignment='center', transform=ax_A0.transAxes, clip_on=False)
        ax_A0.set_title(r'   Activity w/ $I_{CAN}$', fontsize=fsize_titles, loc='center')
        # ax_A1.set_title('CA1 Firing Rates')
        # ax_A2.set_title('CA1 Currents I_CAN / I_M')
        ax_A0.set_ylabel('Spiking Activity', fontsize=fsize_xylabels)

        # Set x-lims
        ax_A0.set_xlim(xlims)

        # Set y-lims
        ax_A0.set_ylim([0, 2*N_scaling+N_gap+1])

        # Hide x-y axes
        ax_A0.xaxis.set_visible(False)
        # ax_A0.yaxis.set_visible(False)

        # Hide some spines
        ax_A0.spines['top'].set_visible(False)
        ax_A0.spines['bottom'].set_visible(False)
        ax_A0.spines['left'].set_visible(False)
        ax_A0.spines['right'].set_visible(False)

        # Fix the tick locators
        ax_A0.xaxis.set_major_locator(ticker.NullLocator())
        ax_A0.xaxis.set_minor_locator(ticker.NullLocator())
        ax_A0.yaxis.set_major_locator(ticker.NullLocator())
        ax_A0.yaxis.set_minor_locator(ticker.NullLocator())

        # Remove tick labels
        # ax0.xaxis.set_ticklabels([])
        ax_A0.yaxis.set_ticklabels([])

    # Other axes in panel A
    ax_A1 = fig.add_subplot(G_FRs_A)
    ax_A2 = fig.add_subplot(G_currents)
    axs[0].append(ax_A1)
    axs[0].append(ax_A2)

    # Set titles and x-y labels
    ax_A1.set_ylabel('Firing Rates [Hz]', fontsize=fsize_xylabels, labelpad=20.)
    ax_A2.set_ylabel(r'$I_{CAN , M}$ [nA]', fontsize=fsize_xylabels)
    ax_A2.set_xlabel('Time [ms]', fontsize=fsize_xylabels)

    # Hide some spines
    ax_A1.spines['top'].set_visible(False)
    ax_A1.spines['bottom'].set_visible(False)
    ax_A1.spines['left'].set_visible(False)
    ax_A1.spines['right'].set_visible(False)
    ax_A2.spines['top'].set_visible(False)
    # ax_A2.spines['bottom'].set_visible(False)
    # ax_A2.spines['left'].set_visible(False)
    ax_A2.spines['right'].set_visible(False)

    # Set x-lims
    ax_A1.set_xlim(xlims)
    ax_A2.set_xlim(xlims)

    # Set y-lims
    # ax_A1.set_ylim(ylims)
    # ax_A2.set_ylim(ylims)

    # Remove tick labels
    # plt.setp(ax_A0.get_xticklabels(), visible=False)
    # plt.setp(ax_A1.get_xticklabels(), visible=False)
    # plt.setp(ax_A2.get_xticklabels(), visible=True)

    # ax_A1.xaxis.set_major_locator(ticker.FixedLocator(np.arange(1000, 2000, 100)))
    # ax_A1.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(1000, 2000, 100)+50))
    major_locs = [1233, 1333, 1583, 1833]
    minor_locs = np.concatenate(([1283], np.arange(1383, 1590, 50), np.arange(1583+50, 1833, 50)))
    # ax_A1.xaxis.set_major_locator(ticker.FixedLocator(major_locs))
    # ax_A1.xaxis.set_minor_locator(ticker.FixedLocator(minor_locs))
    ax_A1.yaxis.set_major_locator(ticker.NullLocator())
    ax_A1.yaxis.set_minor_locator(ticker.NullLocator())
    ax_A2.xaxis.set_major_locator(ticker.FixedLocator(major_locs))
    ax_A2.xaxis.set_minor_locator(ticker.FixedLocator(minor_locs))
    # ax_A1.set_xticklabels([-50, 0, 250, 500])
    ax_A2.set_xticklabels([-50, 0, 250, 500])
    ax_A1.xaxis.set_major_locator(ticker.NullLocator())
    ax_A1.xaxis.set_minor_locator(ticker.NullLocator())
    # ax_A2.xaxis.set_major_locator(ticker.NullLocator())
    # ax_A2.xaxis.set_minor_locator(ticker.NullLocator())

    # Tick font sizes
    ax_A2.tick_params(axis='both', which='major', labelsize=fsize_ticks)

    # Panel B - I/E rates - case w/o currents
    if args.rasters_all:
        ax_B0_0 = fig.add_subplot(G_rasters_B[0], sharex=ax_A0_0, sharey=ax_A0_0)
        ax_B0_1 = fig.add_subplot(G_rasters_B[1], sharex=ax_A0_0, sharey=ax_A0_0)
        ax_B0_2 = fig.add_subplot(G_rasters_B[2], sharex=ax_A0_0, sharey=ax_A0_0)
        ax_B0_3 = fig.add_subplot(G_rasters_B[3], sharex=ax_A0_0, sharey=ax_A0_0)
        axs.insert(1, [ax_B0_0, ax_B0_1, ax_B0_2, ax_B0_3])

        # Set titles and labels
        ax_B0_0.set_title(r'    Activity w/o $I_{CAN}$', loc='center')

        # Set x-lims
        ax_B0_0.set_xlim(xlims)

        # Set y-lims
        ax_B0_0.set_ylim([0, 2*N_scaling+N_gap+1])

        # Hide x-y axes
        ax_B0_0.xaxis.set_visible(False)
        # ax_B0_0.yaxis.set_visible(False)
        ax_B0_1.xaxis.set_visible(False)
        # ax_B0_1.yaxis.set_visible(False)
        ax_B0_2.xaxis.set_visible(False)
        # ax_B0_2.yaxis.set_visible(False)
        ax_B0_3.xaxis.set_visible(False)
        # ax_B0_3.yaxis.set_visible(False)

        # Hide some spines
        ax_B0_0.spines['top'].set_visible(False)
        ax_B0_0.spines['bottom'].set_visible(False)
        ax_B0_0.spines['left'].set_visible(False)
        ax_B0_0.spines['right'].set_visible(False)
        ax_B0_1.spines['top'].set_visible(False)
        ax_B0_1.spines['bottom'].set_visible(False)
        ax_B0_1.spines['left'].set_visible(False)
        ax_B0_1.spines['right'].set_visible(False)
        ax_B0_2.spines['top'].set_visible(False)
        ax_B0_2.spines['bottom'].set_visible(False)
        ax_B0_2.spines['left'].set_visible(False)
        ax_B0_2.spines['right'].set_visible(False)
        ax_B0_3.spines['top'].set_visible(False)
        ax_B0_3.spines['bottom'].set_visible(False)
        ax_B0_3.spines['left'].set_visible(False)
        ax_B0_3.spines['right'].set_visible(False)

        # Fix the ytick locators
        ax_B0_0.yaxis.set_major_locator(ticker.NullLocator())
        ax_B0_0.yaxis.set_minor_locator(ticker.NullLocator())
        ax_B0_1.yaxis.set_major_locator(ticker.NullLocator())
        ax_B0_1.yaxis.set_minor_locator(ticker.NullLocator())
        ax_B0_2.yaxis.set_major_locator(ticker.NullLocator())
        ax_B0_2.yaxis.set_minor_locator(ticker.NullLocator())
        ax_B0_3.yaxis.set_major_locator(ticker.NullLocator())
        ax_B0_3.yaxis.set_minor_locator(ticker.NullLocator())

        # Remove tick labels
        # ax_B0_0.xaxis.set_ticklabels([])
        ax_B0_0.yaxis.set_ticklabels([])
        # ax_B0_1.xaxis.set_ticklabels([])
        ax_B0_1.yaxis.set_ticklabels([])
        # ax_B0_2.xaxis.set_ticklabels([])
        ax_B0_2.yaxis.set_ticklabels([])
        # ax_B0_3.xaxis.set_ticklabels([])
        ax_B0_3.yaxis.set_ticklabels([])

    else:
        ax_B0 = fig.add_subplot(G_rasters_B, sharex=ax_A0, sharey=ax_A0)
        axs.insert(1, [ax_B0])

        # Set titles and x-y labels
        ax_B0.set_title(r'    Activity w/o $I_{CAN}$', fontsize=fsize_titles, loc='center')

        # Hide x-y axes
        ax_B0.xaxis.set_visible(False)
        # ax_B0.yaxis.set_visible(False)

        # Hide some spines
        ax_B0.spines['top'].set_visible(False)
        ax_B0.spines['bottom'].set_visible(False)
        ax_B0.spines['left'].set_visible(False)
        ax_B0.spines['right'].set_visible(False)

        # Fix the tick locators
        ax_B0.xaxis.set_major_locator(ticker.NullLocator())
        ax_B0.xaxis.set_minor_locator(ticker.NullLocator())
        ax_B0.yaxis.set_major_locator(ticker.NullLocator())
        ax_B0.yaxis.set_minor_locator(ticker.NullLocator())

        # Remove tick labels
        # ax_B0.xaxis.set_ticklabels([])
        ax_B0.yaxis.set_ticklabels([])

    ax_B1 = fig.add_subplot(G_FRs_B, sharex=ax_A1, sharey=ax_A1)
    axs[1].append(ax_B1)

    # Hide some spines
    ax_B1.spines['top'].set_visible(False)
    ax_B1.spines['bottom'].set_visible(False)
    ax_B1.spines['left'].set_visible(False)
    ax_B1.spines['right'].set_visible(False)

    # Remove tick labels
    # plt.setp(ax_B0.get_xticklabels(), visible=False)
    # ax_B1.xaxis.set_major_locator(ticker.FixedLocator(np.arange(1000, 2000, 100)))
    # ax_B1.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(1000, 2000, 100)+50))

    # Set x-lims
    ax_B1.set_xlim(xlims)


    # Panel C - Quantification
    ax_C0 = fig.add_subplot(G_panel_C)
    axs.insert(2, [ax_C0])

    # Set titles and x-y labels
    ax_C0.set_title('')
    ax_C0.set_xlabel('Stimulation amplitude [nA]', fontsize=fsize_xylabels)
    ax_C0.set_ylabel('# of Bursts', fontsize=fsize_xylabels)
    # ax_C02.set_ylabel('Burst duration [ms]', color='r')
    # ax_C1.set_ylabel('Spike count [spk/neuron]')
    # ax_C1.set_xlabel('Stimulation amplitude [nA]')

    # Fix the ytick locators
    ax_C0.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax_C0.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax_C0.yaxis.set_major_locator(ticker.FixedLocator([0, 1, 2, 3, 4]))
    ax_C0.yaxis.set_minor_locator(ticker.NullLocator())

    # Tick font sizes
    ax_C0.tick_params(axis='both', which='major', labelsize=fsize_ticks)

    # set ylims
    ax_C0.set_ylim([-0.1, 2.8])

    # Hide some spines
    ax_C0.spines['top'].set_visible(False)
    # ax_C0.spines['bottom'].set_visible(False)
    # ax_C0.spines['left'].set_visible(False)
    ax_C0.spines['right'].set_visible(False)


    # Load the data
    #------------------------
    dir_data = [os.path.join(parent_dir, 'results_cluster', 'results_noICAN_fig3_quantify', '10.0_nA', '0.00_1333.0_ms', '27-09-2022 18H07M53S', 'data'), os.path.join(parent_dir, 'results_cluster', 'results_ICAN_fig3_quantify', '10.0_nA', '0.00_1333.0_ms', '27-09-2022 18H42M54S', 'data')]
    dir_spikes = [os.path.join(dir_data[0], 'spikes'), os.path.join(dir_data[1], 'spikes')]
    dir_currents = [os.path.join(dir_data[0], 'currents'), os.path.join(dir_data[1], 'currents')]


    # Spikes
    print('[+] Loading the rasters...')

    for area_idx in range(len(areas)):
        if not args.rasters_all and area_idx<3:
            continue;

        # load t-i arrays for this area
        print('[+] Loading the spikes for area', areas[area_idx][0].split('_')[0])

        print('[+]....ICAN ON')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, append=1)

            # 10nA w/ I_CAN
            i_exc_ICAN = np.loadtxt(os.path.join(dir_spikes[1], '{0}_spikemon_i.txt'.format(areas[area_idx][0])))
            t_exc_ICAN = np.loadtxt(os.path.join(dir_spikes[1], '{0}_spikemon_t.txt'.format(areas[area_idx][0])))
            i_inh_ICAN = np.loadtxt(os.path.join(dir_spikes[1], '{0}_spikemon_i.txt'.format(areas[area_idx][1])))
            t_inh_ICAN = np.loadtxt(os.path.join(dir_spikes[1], '{0}_spikemon_t.txt'.format(areas[area_idx][1])))

        print('[+]....ICAN OFF')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, append=1)

            # 10nA w/o I_CAN
            i_exc_noICAN = np.loadtxt(os.path.join(dir_spikes[0], '{0}_spikemon_i.txt'.format(areas[area_idx][0])))
            t_exc_noICAN = np.loadtxt(os.path.join(dir_spikes[0], '{0}_spikemon_t.txt'.format(areas[area_idx][0])))
            i_inh_noICAN = np.loadtxt(os.path.join(dir_spikes[0], '{0}_spikemon_i.txt'.format(areas[area_idx][1])))
            t_inh_noICAN = np.loadtxt(os.path.join(dir_spikes[0], '{0}_spikemon_t.txt'.format(areas[area_idx][1])))

        # CA1_E_ICAN_t = np.loadtxt(os.path.join(dir_spikes[1], 'CA1_pyCAN_spikemon_t.txt'))
        # CA1_E_ICAN_i = np.loadtxt(os.path.join(dir_spikes[1], 'CA1_pyCAN_spikemon_i.txt'))
        # CA1_I_ICAN_t = np.loadtxt(os.path.join(dir_spikes[1], 'CA1_inh_spikemon_t.txt'))
        # CA1_I_ICAN_i = np.loadtxt(os.path.join(dir_spikes[1], 'CA1_inh_spikemon_i.txt'))
        #
        # # 10nA w/o I_CAN
        # CA1_E_noICAN_t = np.loadtxt(os.path.join(dir_spikes[0], 'CA1_pyCAN_spikemon_t.txt'))
        # CA1_E_noICAN_i = np.loadtxt(os.path.join(dir_spikes[0], 'CA1_pyCAN_spikemon_i.txt'))
        # CA1_I_noICAN_t = np.loadtxt(os.path.join(dir_spikes[0], 'CA1_inh_spikemon_t.txt'))
        # CA1_I_noICAN_i = np.loadtxt(os.path.join(dir_spikes[0], 'CA1_inh_spikemon_i.txt'))

        # fix data
        i_exc_ICAN = i_exc_ICAN.astype(int)
        t_exc_ICAN = t_exc_ICAN*ms
        i_inh_ICAN = i_inh_ICAN.astype(int)
        t_inh_ICAN = t_inh_ICAN*ms

        i_exc_noICAN = i_exc_noICAN.astype(int)
        t_exc_noICAN = t_exc_noICAN*ms
        i_inh_noICAN = i_inh_noICAN.astype(int)
        t_inh_noICAN = t_inh_noICAN*ms

        # downsample the rasters
        N_exc = N_tot[3][0]
        N_inh = N_tot[3][1]

        exc_mixed = np.arange(0, N_exc, int(N_exc/N_scaling))
        inh_mixed = np.arange(0, N_inh, int(N_inh/N_scaling))

        idx_exc_ICAN = np.in1d(i_exc_ICAN, exc_mixed)
        idx_inh_ICAN = np.in1d(i_inh_ICAN, inh_mixed)
        idx_exc_noICAN = np.in1d(i_exc_noICAN, exc_mixed)
        idx_inh_noICAN = np.in1d(i_inh_noICAN, inh_mixed)

        i_exc_sub_ICAN = i_exc_ICAN[idx_exc_ICAN]
        t_exc_sub_ICAN = t_exc_ICAN[idx_exc_ICAN]
        i_inh_sub_ICAN = i_inh_ICAN[idx_inh_ICAN]
        t_inh_sub_ICAN = t_inh_ICAN[idx_inh_ICAN]
        i_exc_sub_noICAN = i_exc_noICAN[idx_exc_noICAN]
        t_exc_sub_noICAN = t_exc_noICAN[idx_exc_noICAN]
        i_inh_sub_noICAN = i_inh_noICAN[idx_inh_noICAN]
        t_inh_sub_noICAN = t_inh_noICAN[idx_inh_noICAN]

        # assign new neuron count numbers
        cnt = 0
        i_exc_sub_ICAN_new = np.copy(i_exc_sub_ICAN)
        for ii in exc_mixed:
            idx_tmp = np.where(i_exc_sub_ICAN == ii)
            # print('changing ', ii, 'to ', cnt)
            i_exc_sub_ICAN_new[idx_tmp] = cnt
            cnt += 1
        i_exc_sub_ICAN = i_exc_sub_ICAN_new

        # cnt = 0
        cnt += N_gap
        i_inh_sub_ICAN_new = np.copy(i_inh_sub_ICAN)
        for ii in inh_mixed:
            idx_tmp = np.where(i_inh_sub_ICAN == ii)
            # print('changing ', ii, 'to ', cnt)
            i_inh_sub_ICAN_new[idx_tmp] = cnt
            cnt += 1
        i_inh_sub_ICAN = i_inh_sub_ICAN_new

        # assign new neuron count numbers
        cnt = 0
        i_exc_sub_noICAN_new = np.copy(i_exc_sub_noICAN)
        for ii in exc_mixed:
            idx_tmp = np.where(i_exc_sub_noICAN == ii)
            # print('changing ', ii, 'to ', cnt)
            i_exc_sub_noICAN_new[idx_tmp] = cnt
            cnt += 1
        i_exc_sub_noICAN = i_exc_sub_noICAN_new

        # cnt = 0
        cnt += N_gap
        i_inh_sub_noICAN_new = np.copy(i_inh_sub_noICAN)
        for ii in inh_mixed:
            idx_tmp = np.where(i_inh_sub_noICAN == ii)
            # print('changing ', ii, 'to ', cnt)
            i_inh_sub_noICAN_new[idx_tmp] = cnt
            cnt += 1
        i_inh_sub_noICAN = i_inh_sub_noICAN_new

        # select axis
        if args.rasters_all:
            ax_curr_ICAN = axs[0][area_idx]
            ax_curr_noICAN = axs[1][area_idx]
        else:
            ax_curr_ICAN = axs[0][0]
            ax_curr_noICAN = axs[1][0]

        print('[>] Plotting panel A - Rasters w/ ICAN')
        ax_curr_ICAN.scatter(t_inh_sub_ICAN/ms, i_inh_sub_ICAN, s=0.55, linewidth=1., marker='|', c=c_inh, edgecolors=None, alpha=1., rasterized=True)
        ax_curr_ICAN.scatter(t_exc_sub_ICAN/ms, i_exc_sub_ICAN, s=0.55, linewidth=1., marker='|', c=c_exc, edgecolors=None, alpha=1., rasterized=True)

        print('[>] Plotting panel B - Rasters w/o ICAN')
        ax_curr_noICAN.scatter(t_inh_sub_noICAN/ms, i_inh_sub_noICAN, s=0.55, linewidth=1., marker='|', c=c_inh, edgecolors=None, alpha=1., rasterized=True)
        ax_curr_noICAN.scatter(t_exc_sub_noICAN/ms, i_exc_sub_noICAN, s=0.55, linewidth=1., marker='|', c=c_exc, edgecolors=None, alpha=1., rasterized=True)

        # stimulation lines
        ax_curr_ICAN.axvline(x=1333, ymin=-0.5, ymax=1.1, color='gray', alpha=0.75, ls='-', linewidth=0.75, zorder=10, rasterized=False, clip_on=False)
        ax_curr_noICAN.axvline(x=1333, ymin=-0.5, ymax=1.1, color='gray', alpha=0.75, ls='-', linewidth=0.75, zorder=10, rasterized=False, clip_on=False)

        ax_curr_ICAN.set_xlim(xlims)
        ax_curr_noICAN.set_xlim(xlims)

    # Don't forget to mark the stimulation onset!
    # add marker for stimulation onset
    axs[0][0].scatter(x=1333, y=485, s=15, linewidth=1., marker='v', c='gray', edgecolors=None, alpha=1, rasterized=False, clip_on=False)
    axs[1][0].scatter(x=1333, y=485, s=15, linewidth=1., marker='v', c='gray', edgecolors=None, alpha=1, rasterized=False, clip_on=False)


    # Calculate the FRs
    print('[+] Calculating CA1 FRs...')
    tv_inh_ICAN_FR, FR_inh_ICAN = my_FR(spikes=t_inh_ICAN, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
    tv_exc_ICAN_FR, FR_exc_ICAN = my_FR(spikes=t_exc_ICAN, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
    tv_inh_noICAN_FR, FR_inh_noICAN = my_FR(spikes=t_inh_noICAN, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
    tv_exc_noICAN_FR, FR_exc_noICAN = my_FR(spikes=t_exc_noICAN, duration=duration, window_size=winsize_FR, overlap=overlap_FR)

    # Normalize the FRs
    print('[+] Normalizing CA1 FRs...')
    FR_inh_ICAN_norm = (FR_inh_ICAN/winsize_FR)/N_tot[3][1]
    FR_exc_ICAN_norm = (FR_exc_ICAN/winsize_FR)/N_tot[3][0]
    FR_inh_noICAN_norm = (FR_inh_noICAN/winsize_FR)/N_tot[3][1]
    FR_exc_noICAN_norm = (FR_exc_noICAN/winsize_FR)/N_tot[3][0]

    # Currents
    print('[+] Loading CA1-E I_CAN / I_M currents...')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, append=1)
        CA1_E_I_CAN = np.loadtxt(os.path.join(dir_currents[1], 'CA1_E_currents_I_CAN.txt'))
        CA1_E_I_M = np.loadtxt(os.path.join(dir_currents[1], 'CA1_E_currents_I_M.txt'))


    # Plot panel A
    #------------------------
    # print('[>] Plotting panel A - Rasters w/ ICAN')
    # ax_A0.scatter(t_inh_sub_ICAN/ms, i_inh_sub_ICAN, s=0.55, linewidth=1., marker='|', c=c_inh, edgecolors=None, alpha=1., rasterized=True)
    # ax_A0.scatter(t_exc_sub_ICAN/ms, i_exc_sub_ICAN, s=0.55, linewidth=1., marker='|', c=c_exc, edgecolors=None, alpha=1., rasterized=True)

    print('[>] Plotting panel A - FRs w/ ICAN')
    ax_A1.plot(tv_inh_ICAN_FR/ms, FR_inh_ICAN_norm+FR_exc_ICAN_norm.max()+rates_gap, ls='-', linewidth=1.2, c=c_inh, label='inh', zorder=10, rasterized=False)
    ax_A1.plot(tv_exc_ICAN_FR/ms, FR_exc_ICAN_norm, ls='-', linewidth=1.2, c=c_exc, label='exc', zorder=10, rasterized=False)
    # ax_A1.legend(loc='upper center', frameon=False, fontsize=8)

    print('[>] Plotting panel A - Currents w/ ICAN')
    ax_A2.plot(tv/ms, CA1_E_I_M[0], ls='-', linewidth=1.2, c=c_IM, label=r'$I_{M}$')
    ax_A2.plot(tv/ms, CA1_E_I_CAN[0], ls='-', linewidth=1.2, c=c_ICAN, label=r'$I_{CAN}$')
    ax_A2.legend(loc='upper right', frameon=False, fontsize=8)
    ax_A2.set_ylim([-0.35, 0.7])

    # stimulation lines
    ax_A1.axvline(x=1333, ymin=-0.5, ymax=1, color='gray', alpha=0.75, ls='-', linewidth=0.75, zorder=10, rasterized=False, clip_on=False)
    ax_A2.axvline(x=1333, ymin=0, ymax=1, color='gray', alpha=0.75, ls='-', linewidth=0.75, zorder=10, rasterized=False, clip_on=False)

    # # add a sizebar for the x-axis
    # xlims_sz = [xlims[0]-sizebar_off, xlims[0]-sizebar_off+100]
    # add_sizebar(ax_A1, xlims_sz, [-50, -50], 'black', '100ms', rot=0, textx=np.mean(xlims_sz), texty=-70, ha='center', va='top')
    #
    # # add a sizebar for the y-axis
    # ylims_sz = [-50, 50]
    # add_sizebar(ax_A1, [xlims[0]-sizebar_off, xlims[0]-sizebar_off], ylims_sz, 'black', '100Hz', rot=90, textx=xlims[0]-sizebar_off-20, texty=np.mean(ylims_sz), ha='right', va='center')


    # Plot panel B
    #------------------------
    # print('[>] Plotting panel B - Rasters w/o ICAN')
    # ax_B0.scatter(t_inh_sub_noICAN/ms, i_inh_sub_noICAN, s=0.55, linewidth=1., marker='|', c=c_inh, edgecolors=None, alpha=1., rasterized=True)
    # ax_B0.scatter(t_exc_sub_noICAN/ms, i_exc_sub_noICAN, s=0.55, linewidth=1., marker='|', c=c_exc, edgecolors=None, alpha=1., rasterized=True)

    print('[>] Plotting panel B - FRs w/ ICAN')
    ax_B1.plot(tv_inh_noICAN_FR/ms, FR_inh_noICAN_norm+FR_exc_ICAN_norm.max()+rates_gap, ls='-', linewidth=1.2, c=c_inh, label='inh', zorder=10, rasterized=False)
    ax_B1.plot(tv_exc_noICAN_FR/ms, FR_exc_noICAN_norm, ls='-', linewidth=1.2, c=c_exc, label='exc', zorder=10, rasterized=False)

    # ax_B1.legend(loc='upper right', frameon=False, fontsize=8)
    ax_B1.text(x=0.75, y=0.475, transform=ax_B1.transAxes, s='Inhibitory', fontsize=fsize_legends, ha='center', color=c_inh, clip_on=False)
    ax_B1.text(x=0.75, y=0.125, transform=ax_B1.transAxes, s='Excitatory', fontsize=fsize_legends, ha='center', color=c_exc, clip_on=False)

    # stimulation lines
    ax_B0.axvline(x=1333, ymin=-0.2, ymax=1.1, color='gray', alpha=0.75, ls='-', linewidth=0.75, zorder=10, rasterized=False, clip_on=False)
    ax_B1.axvline(x=1333, ymin=0.1, ymax=1, color='gray', alpha=0.75, ls='-', linewidth=0.75, zorder=10, rasterized=False, clip_on=False)

    # add a sizebar for the x-axis
    xlims_sz = [xlims[0]-sizebar_off, xlims[0]-sizebar_off+100]
    add_sizebar(ax_B1, xlims_sz, [+175, +175], 'black', '100 ms', fsize=fsize_legends, rot=0, textx=np.mean(xlims_sz), texty=155, ha='center', va='top')

    # add a sizebar for the y-axis
    ylims_sz = [+175, +275]
    add_sizebar(ax_B1, [xlims[0]-sizebar_off, xlims[0]-sizebar_off], ylims_sz, 'black', '100 Hz', fsize=fsize_legends, rot=90, textx=xlims[0]-sizebar_off-20, texty=np.mean(ylims_sz), ha='right', va='center')


    # Panel C - Quantification
    #------------------------
    print('[>] Plotting panel C - Number of pulses / pulse duration')
    dir_cluster = os.path.join(parent_dir, 'results_cluster')
    dir_ICAN = os.path.join(dir_cluster, 'results_ICAN_fig3_quantify')
    dir_noICAN = os.path.join(dir_cluster, 'results_noICAN_fig3_quantify')

    # w/ ICAN
    print('[!+] Working on +I_CAN data')
    print('-'*32)

    duration_ICAN = []
    spikes_cnt_ICAN = []
    xarr_ICAN = []
    bnums_ICAN = []

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
        N_exc = N_tot[3][0]
        N_inh = N_tot[3][1]
        tn_exc = int(N_exc*0.1)
        tn_inh = int(N_inh*0.1)
        bnum = 0
        spk_cnt_per_burst = 0
        flag = False
        if t_exc.size > 0:
            tval_p = t_exc[0]
            spk_cnt_per_burst += 1
            for tval_c in t_exc[1:]:
                if tval_c - tval_p <= td:   # spikes are close, is it a burst?
                    if flag:    # we already adjusted the burst counter, continue
                        continue
                    spk_cnt_per_burst += 1

                    if spk_cnt_per_burst >= tn_exc:
                        bnum += 1
                        flag = True
                else:   # we take a large step, the spikes are too spread out, new burst?
                    spk_cnt_per_burst = 0
                    flag = False
                tval_p = tval_c
        bnums_ICAN.append(bnum)
        # num_bursts = (np.convolve([-10, 0, 10], FR_exc_envelope)>0).sum()//2

        # get total burst duration
        dur_burst = np.count_nonzero(FR_exc_envelope)*dt/ms

        # append both values
        duration_ICAN.append([bnums_ICAN, dur_burst])

        # count spikes after t_onset
        t_stim = float(t_stim_dir.split("_")[1])*ms
        spkcnt_exc = np.count_nonzero(t_exc >= t_stim)
        spkcnt_inh = np.count_nonzero(t_inh >= t_stim)
        spikes_cnt_ICAN.append([spkcnt_exc, spkcnt_inh])

        print('[+]....Spikes exc: ', spkcnt_exc)
        print('[+]....Spikes inh: ', spkcnt_inh)
        if (t_exc.size>0):
            print('[+]....Last spike: ', np.round(t_exc[-1],4))
        print()


    # w/o ICAN
    print()
    print('[!-] Working on -I_CAN data')
    print('-'*32)

    duration_noICAN = []
    spikes_cnt_noICAN = []
    xarr_noICAN = []
    bnums_noICAN = []

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
        N_exc = N_tot[3][0]
        N_inh = N_tot[3][1]
        tn_exc = int(N_exc*0.1)
        tn_inh = int(N_inh*0.1)
        bnum = 0
        spk_cnt_per_burst = 0
        flag = False
        if t_exc.size > 0:
            tval_p = t_exc[0]
            spk_cnt_per_burst += 1
            for tval_c in t_exc[1:]:
                if tval_c - tval_p <= td:   # spikes are close, is it a burst?
                    if flag:    # we already adjusted the burst counter, continue
                        continue
                    spk_cnt_per_burst += 1

                    if spk_cnt_per_burst >= tn_exc:
                        bnum += 1
                        flag = True
                else:   # we take a large step, the spikes are too spread out, new burst?
                    spk_cnt_per_burst = 0
                    flag = False
                tval_p = tval_c
        bnums_noICAN.append(bnum)
        # num_bursts = (np.convolve([-10, 0, 10], FR_exc_envelope)>0).sum()//2

        # get total burst duration
        dur_burst = np.count_nonzero(FR_exc_envelope)*dt/ms

        # append both values
        duration_noICAN.append([bnums_noICAN, dur_burst])

        # count spikes after t_onset
        t_stim = float(t_stim_dir.split("_")[1])*ms
        spkcnt_exc = np.count_nonzero(t_exc >= t_stim)
        spkcnt_inh = np.count_nonzero(t_inh >= t_stim)
        spikes_cnt_noICAN.append([spkcnt_exc, spkcnt_inh])

        print('[-]....Spikes exc: ', spkcnt_exc)
        print('[-]....Spikes inh: ', spkcnt_inh)
        if (t_exc.size>0):
            print('[+]....Last spike: ', t_exc[-1].round(4), 'ms')
        print()

    # sort the arrays
    idx_ICAN = np.argsort(xarr_ICAN)
    idx_noICAN = np.argsort(xarr_noICAN)

    # plot the E - w/ ICAN vs the E - w/o ICAN
    ln01 = ax_C0.plot(np.array(xarr_ICAN)[idx_ICAN], np.array(bnums_ICAN)[idx_ICAN], linestyle='-', c=c_w_ICAN, label=r'w/ $I_{CAN}$')
    ln02 = ax_C0.plot(np.array(xarr_noICAN)[idx_noICAN], np.array(bnums_noICAN)[idx_noICAN], linestyle='--', c=c_wo_ICAN, alpha=1., label=r'w/o $I_{CAN}$')
    # ln11 = ax_B1.plot(np.array(xarr_ICAN)[idx_ICAN], np.array(spikes_cnt_ICAN)[idx_ICAN,0]/N_tot[3][0], linestyle='-', marker='x', c=c_w_ICAN)
    # ln12 = ax_B1.plot(np.array(xarr_noICAN)[idx_noICAN], np.array(spikes_cnt_noICAN)[idx_noICAN,0]/N_tot[3][0], linestyle='-', marker='o', c=c_wo_ICAN, alpha=0.6)
    # ln10 = ax_B1.plot(np.array(xarr_ICAN)[idx_ICAN], np.array(duration_ICAN)[idx_ICAN,1], '-xc', label='burst duration w/ I_CAN')
    # ln20 = ax_B1.plot(np.array(xarr_noICAN)[idx_noICAN], np.array(duration_noICAN)[idx_noICAN,1], '-og', label='burst duration w/o I_CAN')

    # one legend
    # lns = ln01+ln02+ln11+ln12
    lns = ln01 + ln02
    labs = [l.get_label() for l in lns]
    # ax_B0.legend(lns, labs, loc=0, fontsize=8)
    ax_C0.legend(loc='upper left', fontsize=8, frameon=False)


    # Save and show the figure
    print('[+] Saving the figure...')
    fig.savefig(os.path.join(parent_dir, 'figures', 'fig3', args.figure_name + '.png'), transparent=True, dpi=300, format='png', bbox_inches='tight')
    fig.savefig(os.path.join(parent_dir, 'figures', 'fig3', args.figure_name + '.pdf'), transparent=True, dpi=300, format='pdf', bbox_inches='tight')

    plt.show()

    # Exit properly
    sys.exit(0)
