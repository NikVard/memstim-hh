# OS stuff
import os
import sys
import warnings
from pathlib import Path

# Computational stuff
import numpy as np
import matplotlib as mplb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib import ticker

# TensorPAC
from tensorpac import Pac, PreferredPhase
from tensorpac.utils import PSD

# Other scripts and my stuff
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = Path(script_dir).parent
sys.path.insert(0, os.path.abspath(parent_dir))

from src.freq_analysis import *
from src.figure_plots_parameters import *

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

# Error print!
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# Add sizebar to figure
def add_sizebar(ax, xlocs, ylocs, bcolor, text, textx, texty, fsize, rot, ha='center', va='center'):
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

    parser = argparse.ArgumentParser(description='Generate figure 5 of the paper')

    parser.add_argument('-op', '--order-parameter',
                        action='store_true',
                        default=False,
                        help='Set to plot the order parameter instead of the phase.')

    parser.add_argument('-fn', '--figure-name',
                        type=str,
                        default='fig_low_kN',
                        help='Name of the output figure [w/o file extension].')

    args = parser.parse_args()

    """ Parameters initialization """
    print('[+] Setting up parameters...')

    # Area names and sizes
    areas = [['EC_pyCAN', 'EC_inh'], ['DG_py', 'DG_inh'], ['CA3_pyCAN', 'CA3_inh'], ['CA1_pyCAN', 'CA1_inh']]
    area_labels = ['EC', 'DG', 'CA3', 'CA1']
    N_tot = [[10000, 1000], [10000, 100], [1000, 100], [10000, 1000]]

    # Timing
    second = 1.
    ms = 1e-3
    dt = 0.1*ms
    fs = int(1/dt)

    # duration = 8*second # change this if you change the dataset
    duration = 8*second
    tv = np.arange(0, duration, dt)

    # t_stim = 1800*ms # change this if you change the dataset
    t_lims = [1250*ms, 5000*ms] # ms : x-axs limits
    # t_lims_adj = [t_stim + 500*ms, t_stim+5500*ms] # ms : calculate mean FRs in a 5-sec window
    # t_lims_adj = [1000*ms, 4000*ms] # ms : calculate mean FRs in a 5-sec window
    # duration_adj = t_lims_adj[1] - t_lims_adj[0]

    # Frequency bands
    theta_band = [3,9]
    gamma_band = [40, 80]

    # Max rhythm value
    rhythm_gain_val = 0.18
    rhythm_max_val = 0.2 # 0.22 nA

    # FR calculation
    winsize_FR = 5*ms
    overlap_FR = 0.9
    winstep_FR = winsize_FR*round(1-overlap_FR,4)
    fs_FR = int(1/winstep_FR)

    # Noise for MI computation
    # noise - uniform function instead [min/max]
    curr_state = np.random.get_state() # get current state
    np.random.seed(42) # fix the noise - reproducibility
    noise = np.random.uniform(0, 10, int((duration-winsize_FR)*fs_FR)+1) # change max noise value
    np.random.set_state(curr_state) # resume state

    # Color selection
    c_inh = '#bf616a'
    c_exc = '#5e81ac'
    c_inh_RGB = np.array([191, 97, 106])/255
    c_exc_RGB = np.array([94, 129, 172])/255

    # Raster downsampling
    N_scaling = 100
    N_gap = 10

    # Firing rates plotting gap
    rates_gap = 10 # Hz

    # Set theta rhythm limits
    xlims_rhythm = [t for t in t_lims]
    ylims_rhythm = [-0.01, 0.21]

    # Set raster limits
    xlims_rasters = [t for t in t_lims]
    ylims_rasters = [0, 2*N_scaling+N_gap]

    # Set firing rate limits
    xlims_rates = [t for t in t_lims]
    ylims_rates = [-1, 500]

    # Text parameters
    sizebar_off = 175*ms # sizebar offset

    # Figure sizes
    fig_width = 7.5
    fig_height = 8.75
    dpi = 300

    # Data - point C
    results_in_phase_C_base_dir = os.path.join(parent_dir, 'results', 'fig_kN_single2', '8.0_nA', '0.00_1841.2_ms') # change this for new dataset
    results_in_phase_C_dirs = [os.path.join(results_in_phase_C_base_dir, 'w_phase_reset'),
                               os.path.join(results_in_phase_C_base_dir, 'wo_phase_reset')]
    data_in_phase_C_dirs = [os.path.join(results_in_phase_C_dirs[0], 'data'),
                            os.path.join(results_in_phase_C_dirs[1], 'data')]
    spikes_in_phase_C_dir = [os.path.join(data_in_phase_C_dirs[0], 'spikes'),
                             os.path.join(data_in_phase_C_dirs[1], 'spikes')]

    results_out_of_phase_C_base_dir = os.path.join(parent_dir, 'results', 'fig_kN_single2', '8.0_nA', '0.00_1924.7_ms') # change this for new dataset
    results_out_of_phase_C_dirs = [os.path.join(results_out_of_phase_C_base_dir, 'w_phase_reset'),
                               os.path.join(results_out_of_phase_C_base_dir, 'wo_phase_reset')]
    data_out_of_phase_C_dirs = [os.path.join(results_out_of_phase_C_dirs[0], 'data'),
                            os.path.join(results_out_of_phase_C_dirs[1], 'data')]
    spikes_out_of_phase_C_dir = [os.path.join(data_out_of_phase_C_dirs[0], 'spikes'),
                             os.path.join(data_out_of_phase_C_dirs[1], 'spikes')]

    # Data for pulse trains - point B
    results_trains_in_phase_B_base_dir = os.path.join(parent_dir, 'results', 'fig_kN_point_B_trains2', '8.0_nA', '0.00_1841.2_ms') # change this for new dataset
    results_trains_in_phase_B_dirs = [os.path.join(results_trains_in_phase_B_base_dir, 'w_phase_reset'),
                               os.path.join(results_trains_in_phase_B_base_dir, 'wo_phase_reset')]
    data_trains_in_phase_B_dirs = [os.path.join(results_trains_in_phase_B_dirs[0], 'data'),
                            os.path.join(results_trains_in_phase_B_dirs[1], 'data')]
    spikes_trains_in_phase_B_dir = [os.path.join(data_trains_in_phase_B_dirs[0], 'spikes'),
                             os.path.join(data_trains_in_phase_B_dirs[1], 'spikes')]

    results_trains_out_of_phase_B_base_dir = os.path.join(parent_dir, 'results', 'fig_kN_point_B_trains2', '8.0_nA', '0.00_1924.7_ms') # change this for new dataset
    results_trains_out_of_phase_B_dirs = [os.path.join(results_trains_out_of_phase_B_base_dir, 'w_phase_reset'),
                               os.path.join(results_trains_out_of_phase_B_base_dir, 'wo_phase_reset')]
    data_trains_out_of_phase_B_dirs = [os.path.join(results_trains_out_of_phase_B_dirs[0], 'data'),
                            os.path.join(results_trains_out_of_phase_B_dirs[1], 'data')]
    spikes_trains_out_of_phase_B_dir = [os.path.join(data_trains_out_of_phase_B_dirs[0], 'spikes'),
                             os.path.join(data_trains_out_of_phase_B_dirs[1], 'spikes')]


    """ Plot figures 5/6 w/ low kN of the paper - TODO: Add DOI"""
    print('[+] Generating the figure...')

    # Make a figure
    fig = plt.figure(figsize=(fig_width,fig_height))

    # Use gridspecs
    G_outer_figure = fig.add_gridspec(4, 3, left=0.01, right=0.95, bottom=0.1, top=0.995,
                                            height_ratios=(0.01, 0.405, 0.405, 0.18),
                                            width_ratios=(0.01, 0.495, 0.495),
                                            wspace=0.2, hspace=0.3)

    # Separate the bar graphs from the rest of the figure elements
    G_top_titles = G_outer_figure[0,1:].subgridspec(1,2)
    G_side_titles = G_outer_figure[1:3,0].subgridspec(2,1)
    G_outer_figure_nb = G_outer_figure[1:3,1:].subgridspec(2,2)
    G_barplots = G_outer_figure[3,1:].subgridspec(1,3, wspace=0.4)

    # Fix tight layout
    G_outer_figure.tight_layout(fig)

    # Dummy axes for titles
    # ax_left = fig.add_subplot(G_outer_figure[:,0])
    # ax_left.axis('off')
    # ax_left.set_title('Left title')
    #
    # axs_right = fig.add_subplot(G_outer_figure[:,1])
    # axs_right.axis('off')
    # axs_right.set_title('Right title')

    # Organize the directories
    analysis_dirs = []
    # analysis_dirs.extend(results_in_phase_C_dirs)
    # analysis_dirs.extend(results_out_of_phase_C_dirs)
    analysis_dirs.extend(results_trains_in_phase_B_dirs)
    analysis_dirs.extend(results_trains_out_of_phase_B_dirs)

    # Theta power
    theta_band_power = []

    # Gamma power
    gamma_band_power = []

    # PAC
    pac_obj = Pac(idpac=(2,0,0), f_pha=theta_band, f_amp=gamma_band)
    PAC_metric = []

    for G_curr, panel_label, results_dir_curr, t_stim in zip(G_outer_figure_nb, ['A.', 'B.', 'C.', 'D.'], analysis_dirs, [1850.3*ms, 1850.3*ms, 1934.4*ms, 1934.4*ms]):
        print('[*] Panel', panel_label)

        # Adjust limits
        t_lims_pre = [t_stim-2050*ms, t_stim-50*ms] # ms : pre-stim window [2s]
        t_lims_post = [np.round(t_stim+500*ms,4), np.round(t_stim+2500*ms,4)] # ms : calculate mean FRs in a 2-sec window
        t_lims2 = [np.round(t_stim,4), np.round(t_stim+2000*ms,4)]
        duration_post = t_lims_post[1] - t_lims_post[0]
        duration_2 = t_lims2[1] - t_lims2[0]
        t_lims_post = t_lims2

        # Use gridspecs
        G_outer = GridSpecFromSubplotSpec(2, 1, height_ratios=(0.25, 0.75), subplot_spec=G_curr, hspace=0.1)
        G_oscillators = GridSpecFromSubplotSpec(2, 1, subplot_spec=G_outer[0], hspace=1.2)
        G_all_areas = GridSpecFromSubplotSpec(4, 1, hspace=0.25, subplot_spec=G_outer[1])
        # G_EC = GridSpecFromSubplotSpec(2, 1, hspace=0.1, subplot_spec=G_all_areas[0])
        # G_DG = GridSpecFromSubplotSpec(2, 1, hspace=0.1, subplot_spec=G_all_areas[1])
        # G_CA3 = GridSpecFromSubplotSpec(2, 1, hspace=0.1, subplot_spec=G_all_areas[2])
        # G_CA1 = GridSpecFromSubplotSpec(2, 1, hspace=0.1, subplot_spec=G_all_areas[3])
        G_EC = G_all_areas[0]
        G_DG = G_all_areas[1]
        G_CA3 = G_all_areas[2]
        G_CA1 = G_all_areas[3]


        # Organize axes
        #------------------------
        axs = []

        # Rhythm
        # ------------------------
        print('[>] Input rhythm')
        ax_rhythm = fig.add_subplot(G_oscillators[0])
        axs.append(ax_rhythm)

        # Set the title
        ax_rhythm.set_title('Theta Rhythm', fontsize=fsize_titles)

        # Set the x-y limits
        ax_rhythm.set_xlim(xlims_rhythm)
        ax_rhythm.set_ylim(ylims_rhythm)

        # Hide x-y axes
        ax_rhythm.xaxis.set_visible(False)
        ax_rhythm.yaxis.set_visible(False)

        # Hide some spines
        ax_rhythm.spines['top'].set_visible(False)
        ax_rhythm.spines['bottom'].set_visible(False)
        ax_rhythm.spines['left'].set_visible(False)
        ax_rhythm.spines['right'].set_visible(False)


        # Order Parameter / Phase
        # ------------------------
        if args.order_parameter:
            print('[>] Order Parameter')
            ax_common = fig.add_subplot(G_oscillators[1], sharex=ax_rhythm, sharey=ax_rhythm)

            # Set the title
            ax_common.set_title('Order Parameter', fontsize=fsize_titles)

        else:
            print('[>] Phase')
            ax_common = fig.add_subplot(G_oscillators[1], sharex=ax_rhythm)
            ylims_common = [-0.1, 2*np.pi+0.1]

            # Set the x-y limits
            ax_common.set_ylim(ylims_common)

            # Set the title
            ax_common.set_title('Phase', fontsize=fsize_titles)

        axs.append(ax_common)

        # Hide x-y axes
        ax_common.xaxis.set_visible(False)
        ax_common.yaxis.set_visible(False)

        # Hide some spines
        ax_common.spines['top'].set_visible(False)
        ax_common.spines['bottom'].set_visible(False)
        ax_common.spines['left'].set_visible(False)
        ax_common.spines['right'].set_visible(False)


        # Rasters
        # ------------------------
        print('[>] Rasters')

        # Create the axes and append them to the list
        ax_rasters_0 = fig.add_subplot(G_EC, sharex=ax_rhythm)
        ax_rasters_1 = fig.add_subplot(G_DG, sharex=ax_rhythm, sharey=ax_rasters_0)
        ax_rasters_2 = fig.add_subplot(G_CA3, sharex=ax_rhythm, sharey=ax_rasters_0)
        ax_rasters_3 = fig.add_subplot(G_CA1, sharex=ax_rhythm, sharey=ax_rasters_0)
        axs.append([ax_rasters_0, ax_rasters_1, ax_rasters_2, ax_rasters_3])

        # Set the titles
        # ax_rasters_0.set_title('Network Activity Overview')

        # Set limits
        ax_rasters_0.set_xlim(xlims_rasters)
        ax_rasters_0.set_ylim(ylims_rasters)

        # Formatting the axes
        for ax, label in zip([ax for ax in axs[2]], area_labels):
            # Titles
            # ax.set_title('{0} Activity'.format(label), fontsize=fsize_titles)

            # Labels
            ax.set_ylabel(label, rotation=90, labelpad=10., fontsize=fsize_xylabels)

            # Axes
            ax.xaxis.set_visible(False)
            # ax.yaxis.set_visible(False)

            # Spines
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Fix the ytick locators
            ax.yaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_minor_locator(ticker.NullLocator())

            # Remove tick labels
            # ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])


        # # Firing Rates
        # # ------------------------
        # print('[>] Firing Rates')
        # ax_rates_0 = fig.add_subplot(G_EC[1], sharex=ax_rhythm)
        # ax_rates_1 = fig.add_subplot(G_DG[1], sharex=ax_rhythm, sharey=ax_rates_0)
        # ax_rates_2 = fig.add_subplot(G_CA3[1], sharex=ax_rhythm, sharey=ax_rates_0)
        # ax_rates_3 = fig.add_subplot(G_CA1[1], sharex=ax_rhythm, sharey=ax_rates_0)
        # axs.append([ax_rates_0, ax_rates_1, ax_rates_2, ax_rates_3])
        #
        # # Set the title
        # # ax_rates_0.set_title('Firing Rates')
        #
        # # Set the x-y limits
        # ax_rates_0.set_ylim(ylims_rates)
        #
        # # Formatting the axes
        # for ax, label in zip([ax for ax in axs[3]], area_labels):
        #     # Labels
        #     ax.set_ylabel(label, rotation=0, labelpad=15., fontsize=fsize_xylabels)
        #
        #     # Axes
        #     ax.xaxis.set_visible(False)
        #     ax.yaxis.set_visible(False)
        #
        #     # Spines
        #     ax.spines['top'].set_visible(False)
        #     ax.spines['bottom'].set_visible(False)
        #     ax.spines['left'].set_visible(False)
        #     ax.spines['right'].set_visible(False)
        #
        #     # Fix the ytick locators
        #     ax.yaxis.set_major_locator(ticker.NullLocator())
        #     ax.yaxis.set_minor_locator(ticker.NullLocator())
        #
        #     # Remove tick labels
        #     # ax.xaxis.set_ticklabels([])
        #     ax.yaxis.set_ticklabels([])

        # Iterate over areas
        print('[!] Actually plotting stuff...')


        # =====================
        # Plot the input rhythm
        # =====================

        print('[+] Plotting rhythm...')

        rhythm = np.loadtxt(os.path.join(results_dir_curr, 'data', 'order_param_mon_rhythm.txt'))
        rhythm *= rhythm_gain_val*1e-9
        ax_rhythm.plot(tv, rhythm, ls='-', c='k', linewidth=1., rasterized=False, zorder=1)

        # vertical lines at x-points
        pks_post, _ = sig.find_peaks(rhythm, distance=int(50*ms*fs))

        # peak indices filtering (in adjusted window)
        pks_post_idx = np.logical_and(pks_post>t_lims_post[0]*fs, pks_post<=t_lims_post[1]*fs)
        pks_post_new = pks_post[pks_post_idx]

        # calculate the frequency in the adjusted window
        fval0 = 1/(np.mean(pks_post_new[1:] - pks_post_new[0:-1])/fs) if len(pks_post_new)>1 else 1/(pks_post_new[0]/fs)

        # find the oscillation frequency in the post-stim window using TensorPAC toolbox
        window_post_idx = np.logical_and(np.round(tv,4)>t_lims_post[0], np.round(tv,4)<=t_lims_post[1])
        rhythm_window = rhythm[window_post_idx]
        rhythm_PSD = PSD(rhythm_window[np.newaxis,:], fs)

        # peak indicates most prominent] oscillation
        max_val_psd_idx = np.argmax(rhythm_PSD.psd[0])
        fval1 = rhythm_PSD.freqs[max_val_psd_idx]

        # select which fval to show
        # fval_fin = fval0 # custom peak-based mean calculation
        fval_fin = fval1 # tensorpac PSD argmax calculation

        # text frequency label
        ax_rhythm.text(x=xlims_rhythm[1]-25*ms, y=ylims_rhythm[1]+0.03, s=r"$f_\theta={0:.1f}$ Hz".format(fval_fin), fontsize=fsize_misc, ha='right', color='k', clip_on=False)

        # add a sizebar for the y-axis
        add_sizebar(ax_rhythm, [xlims_rhythm[1]+sizebar_off, xlims_rhythm[1]+sizebar_off], [0, 0.1], 'black', ['0', '0.1 nA'], fsize=fsize_legends, rot=[0, 0], textx=[xlims_rhythm[1]+sizebar_off+20*ms]*2, texty=[0, 0.12], ha='left', va='center')

        # ================================
        # Plot the phase / order parameter
        # ================================

        if args.order_parameter:
            print('[+] Plotting order parameter...')
            data = np.loadtxt(os.path.join(results_dir_curr, 'data', 'order_param_mon_coherence.txt'))

            # asymptote
            ax_common.hlines(y=1., xmin=0., xmax=duration, color='k', ls='--', linewidth=0.5, zorder=11)

            # add a sizebar for the y-axis
            add_sizebar(ax_common, [xlims_rhythm[1]+sizebar_off, xlims_rhythm[1]+sizebar_off], [0, 1.], 'black', ['0', '1'], fsize=fsize_legends, rot=[0, 0], textx=[xlims_rhythm[1]+sizebar_off+20*ms]*2, texty=[0, 1], ha='left', va='center')

        else:
            print('[+] Plotting phase...')
            data = np.loadtxt(os.path.join(results_dir_curr, 'data', 'order_param_mon_phase.txt'))

            # data = (data + np.pi) % (2 * np.pi)
            data += (1.*(data<0)*2*np.pi)

            # add a sizebar for the y-axis
            add_sizebar(ax_common, [xlims_rhythm[1]+sizebar_off, xlims_rhythm[1]+sizebar_off], [0, 2*np.pi], 'black', ['0', '$2\pi$'], fsize=fsize_legends, rot=[0, 0], textx=[xlims_rhythm[1]+sizebar_off+20*ms]*2, texty=[0, 2*np.pi], ha='left', va='center')

            # text with stimulation phase [deg/rad]
            # ax_common.scatter(x=t_stim, y=data[int(t_stim*fs)], s=12, marker='o', c='k')
            # ax_common.text(x=t_stim-75*ms, y=data[int(t_stim*fs)]+0.45, s=r"$\pi/2$", fontsize=fsize_misc, ha='left', color='k', clip_on=False)

        # Data plotting
        ax_common.plot(np.arange(0.,duration,dt), data, ls='-', c='k', linewidth=1., rasterized=False, zorder=1)

        # ==================
        # Plot Rasters + FRs
        # ==================
        for area_name, fname, N_area, curr_ax_rasters in zip(area_labels, areas, N_tot, axs[2]):

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, append=1)

                # load t-i arrays for this area
                print('[+] Loading the spikes for area:', area_name)
                i_exc = np.loadtxt(os.path.join(results_dir_curr, 'data', 'spikes', '{0}_spikemon_i.txt'.format(fname[0])))
                t_exc = np.loadtxt(os.path.join(results_dir_curr, 'data', 'spikes', '{0}_spikemon_t.txt'.format(fname[0])))
                i_inh = np.loadtxt(os.path.join(results_dir_curr, 'data', 'spikes', '{0}_spikemon_i.txt'.format(fname[1])))
                t_inh = np.loadtxt(os.path.join(results_dir_curr, 'data', 'spikes', '{0}_spikemon_t.txt'.format(fname[1])))

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
            N_exc = N_area[0]
            N_inh = N_area[1]

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
            curr_ax_rasters.scatter(t_inh_sub, i_inh_sub, s=1.25, linewidth=1., marker='.', c=c_inh, edgecolors='none', alpha=1., rasterized=True)

            # excitatory
            curr_ax_rasters.scatter(t_exc_sub, i_exc_sub, s=1.25, linewidth=1., marker='.', c=c_exc, edgecolors='none', alpha=1., rasterized=True)

            # mean FRs
            FR_inh_mean = (sum((t_inh>=t_lims_post[0]) & (t_inh<t_lims_post[1]))/duration_post)/N_inh
            FR_exc_mean = (sum((t_exc>=t_lims_post[0]) & (t_exc<t_lims_post[1]))/duration_post)/N_exc

            # add it as a text
            # curr_ax_rasters.text(x=xlims_rates[1]+50*ms, y=1.75*N_scaling+N_gap, s=r'$\mu_I$: {0:.1f} Hz'.format(FR_inh_mean), fontsize=fsize_xylabels, ha='left', color=c_inh, clip_on=False)
            # curr_ax_rasters.text(x=xlims_rates[1]+50*ms, y=N_scaling//2, s=r'$\mu_E$: {0:.1f} Hz'.format(FR_exc_mean), fontsize=fsize_xylabels, ha='left', color=c_exc, clip_on=False)

            # calculate firing rates
            print('[>] Computing firing rates...')
            tv_inh_FR, FR_inh = my_FR(spikes=t_inh, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
            tv_exc_FR, FR_exc = my_FR(spikes=t_exc, duration=duration, window_size=winsize_FR, overlap=overlap_FR)

            # # SAVE THE NON-NORMALIZED FR SIGNALS
            # np.savetxt(os.path.join(results_dir_curr, 'data', 'FR_inh.txt'), FR_inh, fmt='%.8f')
            # np.savetxt(os.path.join(results_dir_curr, 'data', 'tv_inh.txt'), tv_inh_FR, fmt='%.8f')
            # np.savetxt(os.path.join(results_dir_curr, 'data', 'FR_exc.txt'), FR_exc, fmt='%.8f')
            # np.savetxt(os.path.join(results_dir_curr, 'data', 'tv_exc.txt'), tv_exc_FR, fmt='%.8f')

            # Normalize the FRs
            FR_inh_norm = (FR_inh/winsize_FR)/N_inh
            FR_exc_norm = (FR_exc/winsize_FR)/N_exc

            # Add noise
            FR_inh_norm_noise = FR_inh_norm + noise
            FR_exc_norm_noise = FR_exc_norm + noise

            # # Plot the FRs
            # curr_ax_rates.plot(tv_inh_FR, FR_inh_norm+FR_exc_norm.max()+rates_gap, ls='-', linewidth=1.2, c=c_inh, label='inh', zorder=10, rasterized=False)
            # curr_ax_rates.plot(tv_exc_FR, FR_exc_norm, ls='-', linewidth=1.2, c=c_exc, label='exc', zorder=10, rasterized=False)

            # Analysis of FRs
            print('[>] Analysis of FRs...')

            # Calculate FR analysis windows
            window_FR_exc_idx = np.logical_and(np.round(tv_exc_FR,4)>t_lims_post[0], np.round(tv_exc_FR,4)<=t_lims_post[1])
            window_FR_inh_idx = np.logical_and(np.round(tv_inh_FR,4)>t_lims_post[0], np.round(tv_inh_FR,4)<=t_lims_post[1])

            # Calculate theta/gamma band power (post-stim)
            theta_band_pow_inh = bandpower(FR_inh_norm[np.newaxis, window_FR_inh_idx], fs_FR, theta_band, window_sec=1., overlap=overlap_FR, relative=False)
            theta_band_pow_exc = bandpower(FR_exc_norm[np.newaxis, window_FR_exc_idx], fs_FR, theta_band, window_sec=1., overlap=overlap_FR, relative=False)
            gamma_band_pow_inh = bandpower(FR_inh_norm[np.newaxis, window_FR_inh_idx], fs_FR, gamma_band, window_sec=1., overlap=overlap_FR, relative=False)
            gamma_band_pow_exc = bandpower(FR_exc_norm[np.newaxis, window_FR_exc_idx], fs_FR, gamma_band, window_sec=1., overlap=overlap_FR, relative=False)

            # Calculate modulation index
            MI_I = pac_obj.filterfit(fs_FR, FR_inh_norm_noise[np.newaxis, window_FR_inh_idx]).squeeze()
            MI_E = pac_obj.filterfit(fs_FR, FR_exc_norm_noise[np.newaxis, window_FR_exc_idx]).squeeze()

            # Append to the lists
            theta_band_power.append([theta_band_pow_exc, theta_band_pow_inh])
            gamma_band_power.append([gamma_band_pow_exc, gamma_band_pow_inh])
            PAC_metric.append([MI_E, MI_I])



        # axs[3][0].text(x=xlims_rates[1]+50*ms, y=ylims_rates[0]+150+FR_exc_norm.max()+rates_gap, s='Inhibitory', fontsize=fsize_legends, ha='left', color=c_inh, clip_on=False)
        # axs[3][0].text(x=xlims_rates[1]+50*ms, y=ylims_rates[0]-150, s='Excitatory', fontsize=fsize_legends, ha='left', color=c_exc, clip_on=False)
        #
        # # add a sizebar for the y-axis
        # add_sizebar(curr_ax_rates, [xlims_rates[1]+sizebar_off, xlims_rates[1]+sizebar_off], [-50, 50], 'black', '100 Hz', fsize=fsize_misc, rot=0, textx=xlims_rates[1]+sizebar_off+20*ms, texty=0, ha='left', va='center')
        #
        # # add a sizebar for the x-axis
        # xlims_sz = [xlims_rates[1]+sizebar_off, xlims_rates[1]+sizebar_off-200*ms]
        # add_sizebar(curr_ax_rates, xlims_sz, [-50, -50], 'black', '200 ms', fsize=fsize_misc, rot=0, textx=np.mean(xlims_sz), texty=-75, ha='center', va='top')

        # stimulation lines
        # ax_rhythm.vlines(x=t_stim, ymin=-0.1, ymax=1., color='gray', alpha=0.75, ls='--', linewidth=1.5, zorder=11, rasterized=False, clip_on=True)

        # stimulation lines
        ax_rhythm.axvline(x=t_stim, ymin=-1.5, ymax=1.1, color='gray', alpha=0.75, ls='-', linewidth=0.75, zorder=10, rasterized=False, clip_on=False)

        ax_common.axvline(x=t_stim, ymin=-0.5, ymax=1.5, color='gray', alpha=0.75, ls='-', linewidth=0.75, zorder=10, rasterized=False, clip_on=False)

        for ax in axs[2]:
            ln = ax.axvline(x=t_stim, ymin=-0.5, ymax=1.5, color='gray', alpha=0.75, ls='-', linewidth=0.75, zorder=10, rasterized=False, clip_on=False)

        # for ax in axs[3]:
        #     ln = ax.axvline(x=t_stim, ymin=-0.5, ymax=1.5, color='gray', alpha=0.75, ls='-', linewidth=0.75, zorder=10, rasterized=False, clip_on=False)

        ln.set_clip_on(True)

        # stimulation point(s)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, append=1)

            # load t-i arrays for this area
            print('[+] Loading the stimulation waveform:')
            xstim = np.loadtxt(os.path.join(results_dir_curr, 'data', 'stim_input.txt'))

        # stim onset
        ax_rhythm.scatter(x=t_stim, y=ylims_rhythm[1]+0.05, s=75, marker='v', edgecolors='white', facecolors='gray', rasterized=False, clip_on=False)

        # Find stimulation peaks
        idx_all = sig.find_peaks(xstim)
        idx_all = idx_all[0]
        if idx_all.size > 1:
            curr_idx = idx_all[0]
            for idx in idx_all[1:]:
                if idx - curr_idx > 1:
                    ax_rhythm.scatter(x=tv[idx], y=ylims_rhythm[1]+0.036, s=20, marker='o', edgecolors='none', facecolors='gray', rasterized=False, clip_on=False)
                curr_idx = idx

        # Shade the areas
        ax_rhythm.fill_betweenx(y=[-0.01,rhythm_max_val], x1=t_lims_post[0], x2=t_lims_post[1], color='k', alpha=0.25)
        axs[2][3].fill_betweenx(y=[0,N_scaling], x1=t_lims_post[0], x2=t_lims_post[1], color=c_exc, alpha=0.25)
        axs[2][3].fill_betweenx(y=[N_scaling+N_gap, ylims_rasters[1]], x1=t_lims_post[0], x2=t_lims_post[1], color=c_inh, alpha=0.25)

        # Add text to the top-left to indicate panel A.
        ax_rhythm.text(x=-0.07, y=1.6, s=panel_label, fontsize=fsize_panels, weight='bold', ha='left', va='top', transform=ax_rhythm.transAxes)


    # add a sizebar for the x-axis
    xlims_sz = [xlims_rasters[1]+sizebar_off, xlims_rasters[1]+sizebar_off-250*ms]
    add_sizebar(curr_ax_rasters, xlims_sz, [-50, -50], 'black', '250 ms', fsize=fsize_misc, rot=0, textx=np.mean(xlims_sz), texty=-75, ha='center', va='top')


    # Add the bar plots
    bar_axs = []

    for G_curr, data, label in zip(G_barplots, [theta_band_power, gamma_band_power, PAC_metric], ["CA1 Theta Power", "CA1 Gamma Power", "CA1 Phase-Amplitude Coupling"]):

        # X/Y axes
        X = np.arange(0, 4)
        data_arr = np.array(data)

        # Add an axis
        ax = fig.add_subplot(G_curr)
        bar_axs.append(ax)

        # Barplot
        bars_E = ax.bar(X*3, data_arr[3::4,0], color=c_exc)
        bars_I = ax.bar(X*3+1, data_arr[3::4,1], color=c_inh)

        # Set x-ticks and x-labels
        ax.tick_params(axis='x', size=0, labelsize=fsize_ticks, labelrotation=45)
        ax.set_xticks(X*3+0.5)
        ax.set_xticklabels(["A. Peak Stim.\nPhase Reset", "B. Peak Stim.\nNo Phase Reset", "C. Trough Stim.\nPhase Reset", "D. Trough Stim.\nNo Phase Reset"])
        ax.set_title(label, fontsize=fsize_titles)

        # Add text for units
        ax.text(x=-0.08, y=-0.2, s='a.u.', fontsize=fsize_xylabels, ha='center', va='center', transform=ax.transAxes)

        # remove y-ticks
        # ax.set_yticks([])

        # Hide some spines
        ax.spines['top'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

    bar_axs[0].text(x=-0.07, y=1.3, s='E.', fontsize=fsize_panels, weight='bold', ha='left', va='top', transform=bar_axs[0].transAxes)


    # Add top titles
    ax_top_titles_left = fig.add_subplot(G_top_titles[0])
    ax_top_titles_right = fig.add_subplot(G_top_titles[1])

    # Add side titles
    ax_side_titles_top = fig.add_subplot(G_side_titles[0])
    ax_side_titles_bottom = fig.add_subplot(G_side_titles[1])

    # Hide spines
    ax_top_titles_left.spines['top'].set_visible(False)
    ax_top_titles_left.spines['bottom'].set_visible(False)
    ax_top_titles_left.spines['left'].set_visible(False)
    ax_top_titles_left.spines['right'].set_visible(False)
    ax_top_titles_right.spines['top'].set_visible(False)
    ax_top_titles_right.spines['bottom'].set_visible(False)
    ax_top_titles_right.spines['left'].set_visible(False)
    ax_top_titles_right.spines['right'].set_visible(False)

    ax_side_titles_top.spines['top'].set_visible(False)
    ax_side_titles_top.spines['bottom'].set_visible(False)
    ax_side_titles_top.spines['left'].set_visible(False)
    ax_side_titles_top.spines['right'].set_visible(False)
    ax_side_titles_bottom.spines['top'].set_visible(False)
    ax_side_titles_bottom.spines['bottom'].set_visible(False)
    ax_side_titles_bottom.spines['left'].set_visible(False)
    ax_side_titles_bottom.spines['right'].set_visible(False)

    # Hide x-y axes
    ax_top_titles_left.xaxis.set_visible(False)
    ax_top_titles_left.yaxis.set_visible(False)
    ax_top_titles_right.xaxis.set_visible(False)
    ax_top_titles_right.yaxis.set_visible(False)

    ax_side_titles_top.xaxis.set_visible(False)
    ax_side_titles_top.yaxis.set_visible(False)
    ax_side_titles_bottom.xaxis.set_visible(False)
    ax_side_titles_bottom.yaxis.set_visible(False)

    # Add the text
    ax_top_titles_left.text(x=0.5, y=0.5, rotation=0, s='Phase Reset Enabled', fontsize=fsize_panels, va='top', ha='center', transform=ax_top_titles_left.transAxes)
    ax_top_titles_right.text(x=0.5, y=0.5, rotation=0, s='Phase Reset Disabled', fontsize=fsize_panels, va='top', ha='center', transform=ax_top_titles_right.transAxes)
    ax_side_titles_top.text(x=0.5, y=0.5, rotation=90, s='Peak Stimulation', fontsize=fsize_panels, va='center', ha='left', transform=ax_side_titles_top.transAxes)
    ax_side_titles_bottom.text(x=0.5, y=0.5, rotation=90, s='Trough Stimulation', fontsize=fsize_panels, va='center', ha='left', transform=ax_side_titles_bottom.transAxes)

    # Set tight_layout
    # G_outer_figure.tight_layout(fig)
    # fig.tight_layout()


    """ Saving the figure """
    print('[+] Saving the figure...')
    fig.savefig(os.path.join(parent_dir, 'figures', 'fig5_low_kN', args.figure_name + '.png'), transparent=True, dpi=300, format='png')
    fig.savefig(os.path.join(parent_dir, 'figures', 'fig5_low_kN', args.figure_name + '.pdf'), transparent=True, dpi=300, format='pdf')

    print('[!] Done')

    # fig.tight_layout()
    plt.show()

    # Exit - no errors
    sys.exit(0)
