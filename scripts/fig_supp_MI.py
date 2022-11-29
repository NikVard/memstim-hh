# -*- coding: utf-8 -*-

# OS stuff
import os
import sys
import warnings
from pathlib import Path
from tqdm import tqdm

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

# Timing
second = 1
ms = 1e-3
duration = 10*second
dt = 0.1*ms
fs = int(1/dt)
winsize_FR = 5*ms
overlap_FR = 0.9
winstep_FR = winsize_FR*round(1-overlap_FR,4)
fs_FR = int(1/winstep_FR)
binnum = int(duration/winsize_FR)
t_stim = 1800*ms

# Bands for PAC / PreferredPhase
f_pha_PAC = (1, 12, 1, .2)
f_amp_PAC = (30, 100, 10, 1)
f_pha_PP = [3, 9]
f_amp_PP = (20, 100, 10, 1)
f_pha_MI = [3, 9]
f_amp_MI = [40, 80]

# Area names and sizes
areas = [['EC_pyCAN', 'EC_inh'], ['DG_py', 'DG_inh'], ['CA3_pyCAN', 'CA3_inh'], ['CA1_pyCAN', 'CA1_inh']]
area_labels = ['EC', 'DG', 'CA3', 'CA1']
N_tot = [[10000, 1000], [10000, 100], [1000, 100], [10000, 1000]]

# Color selection
c_inh = '#bf616a'
c_exc = '#5e81ac'

# Firing rates plotting gap
rates_gap = 225 # Hz

# main program
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate Figure_2B from paper')

    parser.add_argument('-fn', '--figure-name',
                        type=str,
                        default='fig2_A',
                        help='Name of the output figure [without extension].')

    parser.add_argument('--noise',
                        action='store_true',
                        default=False,
                        help='Set to produce the noisy figures.')

    args = parser.parse_args()

    """ Figure preparation """
    print('[+] Preparing figure outline...')

    # Make figure outline
    fig_width = 7.5
    fig_height = 5.0
    fig = plt.figure(figsize=(fig_width, fig_height), tight_layout=False)
    fig.subplots_adjust(left=0.2, right = 0.95, top=0.9, bottom=0.1)

    # Add gridspecs
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[0.4, 0.6]) # if you add wspace=<%> remove the tight_layout!!
    gs_top = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_outer[0])
    gs_bottom = GridSpecFromSubplotSpec(1, 3, wspace=0.52, subplot_spec=gs_outer[1])

    # Organize the axes
    axs_top = []
    axs_bottom = []
    ax_FRs = fig.add_subplot(gs_top[0])
    ax_PSD = fig.add_subplot(gs_bottom[0])
    axin_PSD = ax_PSD.inset_axes([0.3, 0.3, 0.65, 0.4])
    ax_prefp = fig.add_subplot(gs_bottom[1], projection='polar')
    ax_PAC = fig.add_subplot(gs_bottom[2])
    axs_top.append(ax_FRs)
    axs_bottom.append([ax_PSD, axin_PSD])
    axs_bottom.append(ax_prefp)
    axs_bottom.append(ax_PAC)

    """ Setting data dir """
    print('[+] Setting the data directory...')
    # osc 0.22nA / stim 10nA / stim onset 1800ms / kN 15 / reset 4.0
    # dir_data = os.path.join(parent_dir, 'results_cluster', 'results_fig4', 'K_0.22', '10.0_nA', '0.00_1800.0_ms', '14-10-2022 11H39M47S', 'data')
    # dir_data = '/home/nikos/Documents/projects/Python/memstim-hh/test_12Hz/10.0_nA/0.00_1800.0_ms/24-11-2022 17H29M25S/data'
    # dir_data = '/home/nikos/Documents/projects/Python/memstim-hh/test_8Hz/10.0_nA/0.00_1800.0_ms/25-11-2022 09H58M40S/data'
    # dir_data = '/home/nikos/Documents/projects/Python/memstim-hh/test_8Hz/10.0_nA/0.00_1800.0_ms/25-11-2022 12H02M48S/data'
    dir_data = '/home/nikos/Documents/projects/Python/memstim-hh/test_6Hz/10.0_nA/0.00_1800.0_ms/25-11-2022 12H35M59S/data'

    """ Loading the spikes from disk """
    print('[+] Loading the spikes for area', areas[3][0].split('_')[0])

    # Load the spikes for CA1
    i_exc = np.loadtxt(os.path.join(dir_data, 'spikes/CA1_pyCAN_spikemon_i.txt'))
    t_exc = np.loadtxt(os.path.join(dir_data, 'spikes/CA1_pyCAN_spikemon_t.txt'))
    i_inh = np.loadtxt(os.path.join(dir_data, 'spikes/CA1_inh_spikemon_i.txt'))
    t_inh = np.loadtxt(os.path.join(dir_data, 'spikes/CA1_inh_spikemon_t.txt'))

    # Fix the timings -> from ms to sec
    i_exc = i_exc.astype(int)
    t_exc = t_exc*ms
    i_inh = i_inh.astype(int)
    t_inh = t_inh*ms

    """ Calculate the FRs """
    print('[+] Calculating FRs...')

    # get the firing rate (spikes-to-rates)
    tv_inh_FR, FR_inh = my_FR(spikes=t_inh, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
    tv_exc_FR, FR_exc = my_FR(spikes=t_exc, duration=duration, window_size=winsize_FR, overlap=overlap_FR)

    # Normalize the FRs
    FR_inh_norm = (FR_inh/winsize_FR)/N_tot[3][1]
    FR_exc_norm = (FR_exc/winsize_FR)/N_tot[3][0]

    # get the post-stim indices
    tlims_post = np.array([500, 5500])*ms + t_stim # avoid stim artifact
    tidx_post = np.logical_and(tv_exc_FR>=tlims_post[0], tv_exc_FR<=tlims_post[1])

    """ Add noise? """
    noise = np.zeros(int((duration-winsize_FR)*fs_FR)+2)

    if args.noise:
        print('[!] Adding noise to the FRs...')

        # make the noise
        noise += np.random.uniform(0, 50, int((duration-winsize_FR)*fs_FR)+2) # change max noise value

    # add it to the FRs
    FR_inh_proc = FR_inh_norm + noise
    FR_exc_proc = FR_exc_norm + noise

    """ PAC objects """
    print('[+] Making the PAC objects...')
    pac_obj = Pac(idpac=(2, 0, 0), f_pha=f_pha_PAC, f_amp=f_amp_PAC)
    prefp_obj = PreferredPhase(f_pha=f_pha_PP, f_amp=f_amp_PP)
    MI_obj = Pac(idpac=(2, 0, 0), f_pha=f_pha_MI, f_amp=f_amp_MI)

    """ PSD calculations """
    print('[+] Calculating the PSDs using TensorPAC...')

    # Use the TensorPAC module to calculate the PSDs
    psd_pac_inh = PSD(FR_inh_proc[np.newaxis, tidx_post], fs_FR)
    psd_pac_exc = PSD(FR_exc_proc[np.newaxis, tidx_post], fs_FR)

    """ PAC calculations """
    print('[+] Calculating the PAC...')

    # Calculate PAC (post-stim)
    pac_inh = pac_obj.filterfit(fs_FR, FR_inh_proc[np.newaxis, tidx_post])
    pac_exc = pac_obj.filterfit(fs_FR, FR_exc_proc[np.newaxis, tidx_post])

    """ PreferredPhase calculations """
    print('[+] Calculating the Preferred Phase...')

    # Calculate PreferredPhase (post_stim)
    ampbin_inh, PP_I, polarvec_inh = prefp_obj.filterfit(fs_FR, FR_inh_proc[np.newaxis, tidx_post])
    ampbin_exc, PP_E, polarvec_exc = prefp_obj.filterfit(fs_FR, FR_exc_proc[np.newaxis, tidx_post])

    """ MI calculation """
    print('[+] Calculating MI for theta-gamma...')

    # compute the MI in the [40, 80] Hz band
    xpac_inh = MI_obj.filterfit(fs_FR, FR_inh_proc[np.newaxis, tidx_post])
    xpac_exc = MI_obj.filterfit(fs_FR, FR_exc_proc[np.newaxis, tidx_post])

    """ Plotting the FRs """
    print('[+] Plotting FRs...')

    # Actual plotting
    ax_FRs.plot(tv_inh_FR, FR_inh_proc+rates_gap, ls='-', linewidth=1.2, c=c_inh, label='inh', zorder=10, rasterized=False)
    ax_FRs.plot(tv_exc_FR, FR_exc_proc, ls='-', linewidth=1.2, c=c_exc, label='exc', zorder=10, rasterized=False)

    # Hide some spines
    ax_FRs.spines['top'].set_visible(False)
    ax_FRs.spines['bottom'].set_visible(False)
    ax_FRs.spines['left'].set_visible(False)
    ax_FRs.spines['right'].set_visible(False)

    # Hide x-y axes
    ax_FRs.xaxis.set_visible(False)
    ax_FRs.yaxis.set_visible(False)

    # Set the x-y limits
    ax_FRs.set_xlim([t_stim-200*ms, t_stim+1600*ms])

    # Set the title
    ax_FRs.set_title('CA1 Firing Rates')

    # Add text to indicate E-I
    ax_FRs.text(x=t_stim-250*ms, y=325, s='Inhibitory', fontsize=fsize_legends, ha='center', color=c_inh, clip_on=False)
    ax_FRs.text(x=t_stim-250*ms, y=100, s='Excitatory', fontsize=fsize_legends, ha='center', color=c_exc, clip_on=False)


    """ Plotting the PSDs """
    print('[+] Plotting PSDs...')

    # Actually plotting
    ax_PSD.plot(psd_pac_inh.freqs, psd_pac_inh.psd[0], c=c_inh, label='Inhibitory', zorder=1, rasterized=False)
    ax_PSD.plot(psd_pac_exc.freqs, psd_pac_exc.psd[0], c=c_exc, label='Excitatory', zorder=2, rasterized=False)

    # inset
    axin_PSD.plot(psd_pac_inh.freqs, psd_pac_inh.psd[0], c=c_inh, label='Inhibitory', zorder=1, rasterized=False)
    axin_PSD.plot(psd_pac_exc.freqs, psd_pac_exc.psd[0], c=c_exc, label='Excitatory', zorder=2, rasterized=False)

    # Setting xlims
    ax_PSD.set_xlim([0, 120])
    axin_PSD.set_xlim(37, 81)
    axin_PSD.set_ylim(0, 60)

    # Hide some spines
    ax_PSD.spines['top'].set_visible(False)
    # ax_PSD.spines['bottom'].set_visible(False)
    # ax_PSD.spines['left'].set_visible(False)
    ax_PSD.spines['right'].set_visible(False)

    # gridlines
    axin_PSD.grid(which='major', alpha=0.5, linestyle='dotted', linewidth=1)

    # Color the spines of the inset
    for spine in axin_PSD.spines.values():
        spine.set_edgecolor('#cdcdcd')

    # Add the lines to indicate where the inset axis is coming from
    ax_PSD.indicate_inset_zoom(axin_PSD)

    # Legend
    ax_PSD.legend(fontsize=fsize_legends, frameon=False)

    # Title
    ax_PSD.set_title('CA1 Spectral Analysis', fontsize=fsize_titles)

    # Labels
    ax_PSD.set_ylabel('PSD', labelpad=0., fontsize=fsize_xylabels)
    ax_PSD.set_xlabel('Frequency [Hz]', fontsize=fsize_xylabels)

    # Tick fontsizes
    ax_PSD.tick_params(axis='both', which='both', labelsize=fsize_ticks)
    axin_PSD.tick_params(axis='both', which='both', labelsize=fsize_ticks)

    # Remove some ticks
    ax_PSD.yaxis.set_major_locator(ticker.FixedLocator([0, 500, 1000, 1500, 2000]))
    ax_PSD.yaxis.set_minor_locator(ticker.FixedLocator([250, 750, 1250, 1750]))
    # ax_PSD.yaxis.set_ticklabels(['0', 'Max'])
    axin_PSD.xaxis.set_minor_locator(ticker.FixedLocator([50, 70]))
    axin_PSD.yaxis.set_major_locator(ticker.FixedLocator([0, 50]))
    axin_PSD.yaxis.set_minor_locator(ticker.FixedLocator([25]))
    axin_PSD.yaxis.set_ticklabels(['', ''])
    axin_PSD.tick_params(axis='y', which='both', labelsize=fsize_ticks, length=0, width=0)

    """ Plotting the PreferredPhase (polar plot) """
    print('[+] Plotting Preferred Phase...')

    # ==========================================================================#
    # Important! To use the .polar() to plot in a selected ax, remove the line: #
    # 114:      plt.subplot(subplot, projection='polar')                        #
    # in the visu.py file of the TensorPAC module!                              #
    # ==========================================================================#

    kw_plt = dict(plotas='pcolor', cmap='viridis', interp=.1, cblabel='Amplitude bins',
                  vmin=0.012, vmax=0.18, y=1., fz_title=11, fz_labels=9)

    plt.sca(axs_bottom[1]) # sets current axis (sca) to the bottom-middle
    ax_prefp = prefp_obj.polar(ampbin_exc.squeeze().T, polarvec_exc, prefp_obj.yvec, title='Phase-Amplitude Coupling', colorbar=False, **kw_plt)
    axs_bottom[1].yaxis.grid(linewidth=0.5, alpha=0.9)

    # Add MI as text overlay
    mi_text = r'  MI: {0:.4f}'.format(xpac_exc.squeeze())
    ax_prefp.text(x=np.deg2rad(-90), y=70, s=mi_text, fontsize=fsize_misc, ha='center', va='bottom', color='white', clip_on=True)

    # Fix the label positions
    ax_prefp.set_rlabel_position(135)
    for label in ax_prefp.get_ymajorticklabels():
        label.set_color('white')

    # Set the ticks
    ax_prefp.xaxis.set_ticklabels(['', '', '', r'$\theta_{peak}$', '', '', '', r'$\theta_{trough}$'])
    ax_prefp.set_yticks([40, 60, 80, 100])
    ax_prefp.yaxis.set_ticklabels(['40 Hz', '60 Hz', '80 Hz', '120 Hz'])
    ax_prefp.tick_params(axis='both', which='both', labelsize=fsize_ticks)
    ax_prefp.tick_params(axis='x', which='both', rotation=90, pad=-0.5, labelsize=fsize_misc)
    ax_prefp.set_thetalim(-np.pi, np.pi)

    # Set the xy-lims
    # ax_prefp.set_xlim(37, 81)
    ax_prefp.set_ylim(30, 90)

    # Set origins
    ax_prefp.set_rorigin(10)
    # ax_prefp.set_theta_zero_location('W', offset=10)

    # plot the colorbar
    im = ax_prefp.collections[-1]
    cb = plt.colorbar(im, cax=None, ax=ax_prefp, pad=0.05, location='bottom', orientation='horizontal')
    # cb.ax.set_yticks([0.012, 0.03])
    # cb.ax.set_yticklabels(['0', '1'])
    clims = im.get_clim()
    cb.set_ticks([min(clims), max(clims)])
    cb.set_ticklabels(['min', 'max'])
    cb.ax.tick_params(labelsize=fsize_ticks)
    cb.set_label('')

    cb.outline.set_visible(False)

    # perfect circles
    ax_prefp.set_aspect(1)


    """ Plotting the PAC (comodulogram) """
    print('[+] Plotting comodulogram...')

    # plot
    vmax = np.max([pac_inh.max(), pac_exc.max()])
    # kw = dict(vmax=vmax, vmin=.04, cmap='viridis', interp=None, plotas='imshow')
    kw = dict(vmax=vmax+0.01, vmin=.04, cmap='viridis', interp=(0.5, 0.5), plotas='imshow', fz_title=fsize_titles, fz_labels=fsize_xylabels)

    # Set current axes
    plt.sca(ax_PAC)
    pac_obj.comodulogram(pac_exc.mean(-1), title='PAC Excitatory [-1, 0]s', colorbar=False, **kw)


    # Set tight_layout
    # gs_outer.tight_layout(fig)

    """ Saving the figure """
    print('[+] Saving the figure...')

    # figure name
    fname = 'fig_supp_MI'
    if args.noise:
        fname += '_noise'

    plt.savefig(os.path.join(parent_dir, 'figures', 'supplementary', fname+'.png'), transparent=True, dpi=300, format='png', bbox_inches='tight')
    plt.savefig(os.path.join(parent_dir, 'figures', 'supplementary', fname+'.pdf'), transparent=True, dpi=300, format='pdf', bbox_inches='tight')


    # Show the figure
    plt.show()

    # Exit normally
    sys.exit(0)
