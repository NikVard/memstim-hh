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
from matplotlib import colors

# Other scripts and my stuff
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = Path(script_dir).parent
sys.path.insert(0, os.path.abspath(parent_dir))

from src.freq_analysis import *
from src.figure_plots_parameters import *
from src.annex_funcs import make_flat

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
    c_start = '#b3ff00'
    c_start = '#37961c'
    c_end = '#be36c3'

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

    # Parameters for rasters
    msize = 0.75

    # Parameters for FRs
    window_size = 1*ms
    bin_num = int(duration/window_size)
    fs_FR = 1/window_size

    # Parameters for spectrograms
    window_size_Pxx = 1.*second;
    window_width_Pxx = int(window_size_Pxx/second * fs_FR);
    overlap_Pxx = 0.9
    window_overlap_Pxx = int(window_width_Pxx*overlap_Pxx)
    vmin = 1e-12
    vmax = 1.
    norm_inh = colors.Normalize(vmin=vmin, vmax=vmax)
    norm_exc = colors.Normalize(vmin=vmin, vmax=vmax)
    specgram_kwargs = { 'return_onesided' : True,
                        'scaling' : 'density',
                        'mode' : 'magnitude' }

    # Load the spike data, case #1:
    conn_cases = ["CON_Amelie", "DC"]
    noise_E = 1000
    noise_I = 100
    ramp_E = True
    ramp_I = True

    # Filenames
    fname = f"input_ramp_E_{bool(ramp_E)}_I_{bool(ramp_I)}_noise_E_{noise_E:0.0f}uV_I_{noise_I:0.0f}uV"

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
    xlims_time = [0.5, duration-0.5]
    xlims_time_start = [1.1, 2.25]
    xlims_time_end = [3.6, 3.7]
    ylims_input = [0, 1]
    ylims_rasters = [-5, N_E+N_I+5]
    ylims_FRs = [-0.1, 2.1]

    # Make a figure
    fig = plt.figure(figsize=(fig_width,fig_height))

    G_outer = GridSpec(2,1, left=0.1, right=0.95, bottom=0.05, top=0.97,
               height_ratios=(0.5, 0.5),
               # hspace=0.5, wspace=0.5,
               hspace=0.2,
               figure=fig)
    G_inner = []
    G_inner.append(GridSpecFromSubplotSpec(3, 4,
                    height_ratios=(0.55, 0.15, 0.3),
                    width_ratios=(0.3, 0.4, 0.288, 0.012),
                    hspace=0.6,
                    wspace=0.1,
                    subplot_spec=G_outer[0]))
    G_inner.append(GridSpecFromSubplotSpec(3, 4,
                    height_ratios=(0.55, 0.15, 0.3),
                    width_ratios=(0.3, 0.4, 0.288, 0.012),
                    hspace=0.6,
                    wspace=0.1,
                    subplot_spec=G_outer[1]))
    ax_placeholders = []
    ax_placeholders.append(fig.add_subplot(G_inner[0][0,0]))
    ax_placeholders.append(fig.add_subplot(G_inner[1][0,0]))
    ax_rasters = []
    ax_rasters.append([fig.add_subplot(G_inner[0][0,1]), fig.add_subplot(G_inner[0][0,2])])
    ax_rasters.append([fig.add_subplot(G_inner[1][0,1]), fig.add_subplot(G_inner[1][0,2])])
    ax_inputs = []
    ax_inputs.append(fig.add_subplot(G_inner[0][1,:-1]))
    ax_inputs.append(fig.add_subplot(G_inner[1][1,:-1]))
    ax_specgrams = []
    ax_specgrams.append(fig.add_subplot(G_inner[0][2,:-1]))
    ax_specgrams.append(fig.add_subplot(G_inner[1][2,:-1]))
    ax_cbars = []
    ax_cbars.append(fig.add_subplot(G_inner[0][2,-1]))
    ax_cbars.append(fig.add_subplot(G_inner[1][2,-1]))

    # Set the titles
    ax_placeholders[0].set_title('Excitatory-Inhibitory Coupled', loc='center', fontsize=fsize_titles)
    ax_placeholders[1].set_title('Excitatory-Inhibitory Decoupled', loc='center', fontsize=fsize_titles)
    ax_rasters[0][0].set_title('Emergence of oscillations', loc='center', fontsize=fsize_titles)
    ax_rasters[0][1].set_title('Settling of gamma oscillations', loc='center', fontsize=fsize_titles)
    # ax_rasters[1][0].set_title('Raster 1', loc='center', fontsize=fsize_titles)
    # ax_rasters[1][1].set_title('Raster 2', loc='center', fontsize=fsize_titles)
    ax_inputs[0].set_title('Ramp Input (nA)', loc='center', fontsize=fsize_titles)
    ax_inputs[1].set_title('Ramp Input (nA)', loc='center', fontsize=fsize_titles)
    ax_specgrams[0].set_title('Spectrogram of excitatory firing rate', loc='center', fontsize=fsize_titles)
    ax_specgrams[1].set_title('Spectrogram of excitatory firing rate', loc='center', fontsize=fsize_titles)

    # Set the xlims/ylims
    for ax in make_flat(ax_inputs):
        ax.set_xlim([0,duration])

    for ax in make_flat(ax_rasters):
        ax.set_ylim(ylims_rasters)
    ax_rasters[0][0].set_xlim(xlims_time_start)
    ax_rasters[0][1].set_xlim(xlims_time_end)
    ax_rasters[1][0].set_xlim(xlims_time_start)
    ax_rasters[1][1].set_xlim(xlims_time_end)

    # Proper fontsizes
    for ax in make_flat(ax_inputs):
        ax.tick_params(axis='both', which='both', labelsize=fsize_ticks)
    for ax in make_flat(ax_rasters):
        ax.tick_params(axis='both', which='both', labelsize=fsize_ticks)
    for ax in make_flat(ax_specgrams):
        ax.tick_params(axis='both', which='both', labelsize=fsize_ticks)
        
    # Remove the box borders
    for ax in make_flat(ax_inputs):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)

    for ax in make_flat(ax_rasters):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    for ax in make_flat(ax_placeholders):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Remove ticks
    for ax in ax_inputs:
        ax.get_xaxis().set_ticks(np.arange(0,duration))
        ax.get_yaxis().set_ticks([0, 1])

    for ax in make_flat(ax_rasters):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    for ax in ax_placeholders:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    for ax in make_flat(ax_specgrams):
        ax.get_xaxis().set_ticks(np.arange(0,duration))
        ax.get_yaxis().set_ticks([40, 60, 100])

    # Panels text
    fig.text(0.01, 0.98, 'A.', weight='bold', fontsize=fsize_panels)
    fig.text(0.01, 0.48, 'B.', weight='bold', fontsize=fsize_panels)
    # fig.text(0.4, 0.98, 'B.', weight='bold', fontsize=fsize_panels)
    # fig.text(0.02, 0.48, 'C.', weight='bold', fontsize=fsize_panels)
    # fig.text(0.4, 0.48, 'D.', weight='bold', fontsize=fsize_panels)

    # Normalize the colors in the spectrograms
    cmesh_list = []
    vlow = []
    vhigh = []

    # Read the data
    for idx,conn in enumerate(conn_cases):

        # E
        try:
            E_i = np.loadtxt(os.path.join(spikes_dir, f"{conn}_{areas[0]}_{fname}_i.txt"))
        except OSError:
            print('File not found for parameter set:')
            print(os.path.join(spikes_dir, f"{conn}_{areas[0]}_{fname}_i.txt"))
            print(f'\tConnectivity: {conn}')
            print(f'\tramp_E: {ramp_E} | ramp_I {ramp_I}')
            print(f'\tnoise_E: {noise_E} | noise_I {noise_I}')
            print('Continuing...')
            print('-'*16)
            continue

        E_t = np.loadtxt(os.path.join(spikes_dir, f"{conn}_{areas[0]}_{fname}_t.txt"))
        E_t *= ms

        # I
        I_i = np.loadtxt(os.path.join(spikes_dir, f"{conn}_{areas[1]}_{fname}_i.txt"))
        I_t = np.loadtxt(os.path.join(spikes_dir, f"{conn}_{areas[1]}_{fname}_t.txt"))
        I_t *= ms

        # Compute the FRs
        tv_inh_FR, FR_inh, fs_FR2 = my_FR(spikes=I_t, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
        tv_exc_FR, FR_exc, _ = my_FR(spikes=E_t, duration=duration, window_size=winsize_FR, overlap=overlap_FR)

        # Normalize the FRs
        FR_inh_norm = (FR_inh/winsize_FR)/N_I
        FR_exc_norm = (FR_exc/winsize_FR)/N_E

        # Plot the rasters and FRs
        for ax in ax_rasters[idx]:
            ax.plot(E_t, E_i, linestyle='None', color=c_exc, marker='.', markersize=msize)
            ax.plot(I_t, I_i+N_E, linestyle='None', color=c_inh, marker='.', markersize=msize)

        # ax_FRs.plot(tv_E, (FR_E/N_E), color=c_exc, label='Excitatory')
        # ax_FRs.plot(tv_I, (FR_I/N_I)+1, color=c_inh, label='Inhibitory')

        # Get the specgrams
        fv_inh, tv_inh, pspec_inh = my_specgram(signal=FR_inh_norm, fs=fs_FR2, window_width=window_width_Pxx, window_overlap=window_overlap_Pxx, k=2,
                       **specgram_kwargs)
        fv_exc, tv_exc, pspec_exc = my_specgram(signal=FR_exc_norm, fs=fs_FR2, window_width=window_width_Pxx, window_overlap=window_overlap_Pxx, k=2,
                       **specgram_kwargs)

        vlow.append(pspec_exc.min())
        vhigh.append(pspec_exc.max())

        # im_inh = ax_specg_inh.pcolormesh(tv_inh, fv_inh, pspec_inh, cmap='inferno', shading='auto', rasterized=True)
        # im_exc = ax_specg_exc.pcolormesh(tv_exc, fv_exc, pspec_exc, cmap='inferno', shading='auto', rasterized=True)
        cmesh = ax_specgrams[idx].pcolormesh(tv_exc, fv_exc, pspec_exc, cmap='inferno', shading='auto', rasterized=True)
        cmesh_list.append(cmesh)

        ax_specgrams[idx].tick_params(axis='both',     # changes apply to both axes
                               which='both',      # both major and minor ticks are affected
                               bottom=True,       # ticks along the bottom edge are on
                               top=False,         # ticks along the top edge are on
                               right=False, 
                               left=True, 
                               labelbottom=True, # labels along the bottom edge are on
                               labelleft=True,
                               labelsize=fsize_ticks)
        ax_specgrams[idx].set_ylim([10,110])
        ax_specgrams[idx].set_xlabel('Time [s]', fontsize=fsize_xylabels)

    # Spectrograms common color scale
    # vlow = [1e-6]*2
    # vhigh = [4]*2
    for cmsh in make_flat(cmesh_list):
        cmsh.set_clim(min(vlow), max(vhigh))

    # Add colorbars
    for ax,cmsh in zip(make_flat(ax_cbars), make_flat(cmesh_list)):
        cbar = fig.colorbar(cmsh, cax=ax)
        cbar.outline.set_color('black')
        cbar.outline.set_linewidth(0.5)
        cbar.solids.set_rasterized(True)
        cbar.dividers.set_color('none')
        cbar.dividers.set_linewidth(5)
        cbar.ax.tick_params(labelsize=fsize_ticks)


    # Plot the common data
    for ax in make_flat(ax_inputs):
        ax.plot(tv_ramp, ramp, color='k')
        ax.set_ylim(ylims_input)

    # Add sizebars
    # for ax in make_flat(ax_inputs):
    #     # yaxis
    #     add_sizebar(ax, [duration, duration], [0, 0.5],
    #                 'black', ['0 nA', '0.5'], fsize=fsize_misc, rot=[0, 0], 
    #                 textx=[xlims_time[1]+0.15]*2, texty=[-0.1, 0.5],
    #                 ha='left', va='center')
    #     # xaxis
    #     xlims_sz = [xlims_time[1]-500*ms, xlims_time[1]]
    #     ylims_sz = [0]*2
    #     add_sizebar(ax, xlims_sz, ylims_sz,
    #                 'black', '500 ms', fsize=fsize_misc, rot=0,
    #                 textx=np.mean(xlims_sz), texty=ylims_sz[0]-0.15,
    #                 ha='center', va='top')

    for ax in ax_rasters:
        xlims_sz = [xlims_time_start[1]-510*ms, xlims_time_start[1]-10*ms]
        ylims_sz = [-50]*2
        add_sizebar(ax[0], xlims_sz, ylims_sz,
                    'black', '500 ms', fsize=fsize_misc, rot=0,
                    textx=np.mean(xlims_sz), texty=ylims_sz[0]-50,
                    ha='center', va='top')

        xlims_sz = [xlims_time_end[1]-30*ms, xlims_time_end[1]-10*ms]
        ylims_sz = [-50]*2
        add_sizebar(ax[1], xlims_sz, ylims_sz,
                    'black', '20 ms', fsize=fsize_misc, rot=0,
                    textx=np.mean(xlims_sz), texty=ylims_sz[0]-50,
                    ha='center', va='top')

    # Fill between
    for ax in ax_inputs:
        ax.fill_between(xlims_time_start, y1=0, y2=1, color=c_start, alpha=0.25)
        ax.fill_between(xlims_time_end, y1=0, y2=1, color=c_end, alpha=0.25)
        # ax.annotate('',
        #             xy=(1.25, 2), xycoords='data',
        #             xytext=(1, 0.75), textcoords='data', clip_on=False,
        #             arrowprops=dict(arrowstyle="->"))
        # ax.arrow(xlims_time_start[0],1.,dx=0.5,dy=1.1,fc='#ffcc00',ec='#ffcc00',width=0.,clip_on=False)
        # ax.arrow(xlims_time_start[1],1.,dx=1.5,dy=1.2,fc='#ffcc00',ec='#ffcc00',width=0.,clip_on=False)

    # Set the borders to a given color...
    for idx,ax in enumerate(make_flat(ax_rasters)):
        cval = c_end
        if not np.mod(idx,2):
            cval = c_start
        ax.tick_params(color=cval, labelcolor=cval)
        for spine in ax.spines.values():
            spine.set_edgecolor(cval)
            spine.set_visible(True)
            spine.set_linewidth(2)

    # add_sizebar(ax_FRs, [xlims_time[1]-0.25, xlims_time[1]-0.], [-0.1, -0.1], 'black', '250ms', fsize=fsize_misc, rot=0, 
    #             textx=np.mean([xlims_time[1]-0.25, xlims_time[1]-0.]), texty=-0.13, 
    #             ha='center', va='top')

    # Set the xlims
    for ax in make_flat(ax_inputs):
        ax.set_xlim(xlims_time)

    for ax in make_flat(ax_specgrams):
        ax.set_xlim(xlims_time)

    # Legends
    # lines_labels = [ax.get_legend_handles_labels() for ax in [ax_FI_E, ax_FI_I]]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # ax_FI_E.legend(lines, labels, loc='upper left')

    # Tight layout
    # G_outer.tight_layout(fig)

    # Save the figure
    fig.savefig(os.path.join('figures', 'rev_eLife', f"main_{ramp_E}_{areas[0]}_{ramp_I}_{areas[1]}_{fname}_spec_com_rev_eLife.png"))
    fig.savefig(os.path.join('figures', 'rev_eLife', f"main_{ramp_E}_{areas[0]}_{ramp_I}_{areas[1]}_{fname}_spec_com_rev_eLife.pdf"))

    # Show the figure
    plt.show()

    # Close the figure
    # plt.close()

    # Plotting done
    print('Done with generating figures.')

    # Exit properly
    sys.exit(0)