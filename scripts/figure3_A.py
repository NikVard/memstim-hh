#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
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
mplb.rcParams['axes.labelsize'] = 8


# main program
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate figure 3A from paper')

    parser.add_argument('-fn', '--figure-name',
                        type=str,
                        default='fig3_A',
                        help='Name of the output figure [w/o file extension]')




    """ Parameters initialization """
    print('[+] Setting up parameters...')

    # Timing
    second = 1
    ms = 1e-3
    duration = 3*second
    dt = 0.1*ms
    fs = int(1*second/dt)

    # Area names and sizes
    areas = [['EC_pyCAN', 'EC_inh'], ['DG_py', 'DG_inh'], ['CA3_pyCAN', 'CA3_inh'], ['CA1_pyCAN', 'CA1_inh']]
    area_labels = ['EC', 'DG', 'CA3', 'CA1']
    N_tot = [[10000, 1000], [10000, 100], [1000, 100], [10000, 1000]]

    # Directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = Path(script_dir).parent
    sys.path.insert(0, os.path.abspath(parent_dir))
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


    """ Loading the FRs from disk """
    print('[+] Loading FRs from disk...')

    # Load the non-normalized FRs
    FR_inh = np.loadtxt(os.path.join(parent_dir, 'test_FRs', 'FR_inh.txt'))
    tv_FR_inh = np.loadtxt(os.path.join(parent_dir, 'test_FRs', 'tv_inh.txt'))
    FR_exc = np.loadtxt(os.path.join(parent_dir, 'test_FRs', 'FR_exc.txt'))
    tv_FR_exc = np.loadtxt(os.path.join(parent_dir, 'test_FRs', 'tv_exc.txt'))



    """ Plot Figure 3 of the paper - TODO: Add DOI"""
    print('[+] Generating the figure...')

    # Make a figure
    fig = plt.figure(figsize=(fig_width,fig_height))

    # Use gridspecs
    G_outer = GridSpec(4, 2, left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.5, width_ratios=(0.5, 0.5), figure=fig)
    G_panel_A = G_outer[:,0].subgridspec(4, 1, hspace=0.1)
    G_outer.tight_layout(fig)


    # Organize axes
    #------------------------
    axs = []

    # Panel A - I/E rates | I_CAN | I_M - case w/ currents
    ax_A0 = fig.add_subplot(G_panel_A[0])
    ax_A1 = fig.add_subplot(G_panel_A[1])
    ax_A2 = fig.add_subplot(G_panel_A[2])
    ax_A3 = fig.add_subplot(G_panel_A[3])
    axs.append([ax_A0, ax_A1, ax_A2, ax_A3])


    # Panel B - I/E rates - case w/o currents
    ax_B0 = fig.add_subplot(G_panel_B[0])
    ax_B1 = fig.add_subplot(G_panel_B[1])
    axs.append([ax_B0, ax_B1])


    # Panel C - Quantification
    ax_C0 = fig.add_subplot(G_panel_B[0])
    axs.append([C0])






    plt.show()
