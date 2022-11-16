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

# TensorPAC
from tensorpac import Pac
from tensorpac.utils import PSD

# Other scripts and my stuff
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = Path(script_dir).parent
sys.path.insert(0, os.path.abspath(parent_dir))

from src.freq_analysis import *

# get the root directory for the simulations
root_cluster = os.path.join(parent_dir, 'results_cluster', 'results_fig4')
# print(root_cluster)

# get a list of the directories with oscillator amplitudes
osc_amplitude_dirs = sorted(next(os.walk(root_cluster))[1])
# print(osc_amplitude_dirs)

def check_MI(signal, theta_freqs=[3, 5], gamma_freqs=[30, 60], return_distance=False):
    """
        PAC check using Modulation Index (MI)
          Wrapper function that filters and analyzes the input signal for phase-amplitude coupling (PAC) using the Modulation Index (MI) - ported function from Matlab.
    """
    # 1. Filtering at theta/gamma bands
    sig_filt_gamma = butter_bandpass_filter(signal, gamma_freqs[0], gamma_freqs[1], fs_FR, sos=True)
    sig_filt_theta = butter_bandpass_filter(signal, theta_freqs[0], theta_freqs[1], fs_FR, sos=True)

    # 2. Time series of phases / amplitudes using Hilbert Transform
    xfp = sig.hilbert(sig_filt_theta)
    xfA = sig.hilbert(sig_filt_gamma)
    sig_amp = np.abs(xfA)
    sig_phase = np.angle(xfp)

    # 3./4. Calculate Modulation Index (MI)
    MI, dist_KL = my_modulation_index(sig_phase, sig_amp)

    if return_distance:
        return MI, dist_KL
    else:
        return MI


# Timing parameters
second = 1
ms = 1e-3
duration = 4*second
dt = 0.1*ms
fs = int(1*second/dt)
tv = np.linspace(0, duration, fs*duration)/ms
winsize_FR = 5*ms
overlap_FR = 0.9
winstep_FR = winsize_FR*round(1-overlap_FR,4)
fs_FR = int(1/winstep_FR)

# Area names and sizes
areas = [['EC_pyCAN', 'EC_inh'], ['DG_py', 'DG_inh'], ['CA3_pyCAN', 'CA3_inh'], ['CA1_pyCAN', 'CA1_inh']]
area_labels = ['EC', 'DG', 'CA3', 'CA1']
N_tot = [[10000, 1000], [10000, 100], [1000, 100], [10000, 1000]]

# get the oscillator amplitude values
# osc_amps = np.arange(0.01, 0.23, 0.01)
osc_amps = []
for val in osc_amplitude_dirs:
    osc_amps.append(float(val.split('_')[1]))
osc_amps.sort()
osc_amps = np.array(osc_amps)

# stim_amps = np.arange(1, 11, 1)
stim_amps = []
for val in next(os.walk(os.path.join(root_cluster, osc_amplitude_dirs[0])))[1]:
    stim_amps.append(float(val.split('_')[0]))
stim_amps.sort()
stim_amps = np.array(stim_amps)

# make a matrix
X, Y = np.meshgrid(osc_amps, stim_amps)
# MI_heatmap_E = np.zeros(X.shape)
# MI_heatmap_I = np.zeros(X.shape)
MI_heatmap_E = np.zeros((len(osc_amps), len(stim_amps)))
MI_heatmap_I = np.zeros((len(osc_amps), len(stim_amps)))

# freq bins
peak_freqs_inh = []
peak_freqs_exc = []

# theta / gamma power matrices
theta_band = [3, 9]
gamma_band = [40, 80]
theta_heatmap_E = np.zeros((len(osc_amps), len(stim_amps)))
theta_heatmap_I = np.zeros((len(osc_amps), len(stim_amps)))
gamma_heatmap_E = np.zeros((len(osc_amps), len(stim_amps)))
gamma_heatmap_I = np.zeros((len(osc_amps), len(stim_amps)))

# noise - uniform function instead [min/max]
curr_state = np.random.get_state() # get current state
np.random.seed(42) # fix the noise - reproducibility
noise = np.random.uniform(0, 10, int((duration-0.005)*fs_FR)+1) # change max noise value
np.random.set_state(curr_state) # resume state

# Make a PAC object
f_pha_PAC = theta_band
f_amp_PAC = gamma_band
pac_obj = Pac(idpac=(2, 0, 0), f_pha=f_pha_PAC, f_amp=f_amp_PAC)

# iterate over oscillator amplitudes
for osc_amp_dir in tqdm(osc_amplitude_dirs, desc='Computing the modulation index...'):
    curr_osc_amp = float(osc_amp_dir.split('_')[1])
    idx_osc_amp = np.where(osc_amps == curr_osc_amp) # index for heatmap

    # one step in
    curr_osc_dir = os.path.join(root_cluster, osc_amp_dir)
    print('osc: ', osc_amp_dir)

    # iterate over stimulation amplitudes
    stim_amp_dirs = [dirname for dirname in sorted(os.listdir(curr_osc_dir)) if os.path.isdir(os.path.join(curr_osc_dir, dirname))]

    for stim_amp_dir in stim_amp_dirs:
        curr_stim_amp = float(stim_amp_dir.split('_')[0])
        idx_stim_amp = np.where(stim_amps == curr_stim_amp) # index for heatmap

        # one step in
        curr_stim_dir = os.path.join(curr_osc_dir, stim_amp_dir)
        # print('|')
        # print('-> stim: ', stim_amp_dir)

        # go through the stimulation directory
        stim_time_dir = next(os.walk(curr_stim_dir))[1][0]
        t_stim = float(stim_time_dir.split('_')[1]) # in ms
        # print('|')
        # print('---> stim_on: ', stim_time_dir)

        # traverse the next directory too (simulation)
        curr_stim_time_dir = os.path.join(curr_stim_dir, stim_time_dir)
        sim_dir = next(os.walk(curr_stim_time_dir))[1][0]
        curr_sim_dir = os.path.join(curr_stim_time_dir, sim_dir)
        # print('|')
        # print('-----> sim: ', sim_dir)

        # load the data (spikes) for the CA1 E-group
        curr_data_dir = os.path.join(curr_sim_dir, 'data')
        curr_spikes_dir = os.path.join(curr_data_dir, 'spikes')
        # print('[L5]-----data: ', curr_data_dir)
        # print('[L6]------spikes: ', curr_spikes_dir)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, append=1)

            if len([entry for entry in os.listdir(curr_spikes_dir) if os.path.isfile(os.path.join(curr_spikes_dir, entry))]) == 16:
                i_exc = np.loadtxt(os.path.join(curr_spikes_dir, 'CA1_pyCAN_spikemon_i.txt'))
                t_exc = np.loadtxt(os.path.join(curr_spikes_dir, 'CA1_pyCAN_spikemon_t.txt'))
                i_inh = np.loadtxt(os.path.join(curr_spikes_dir, 'CA1_inh_spikemon_i.txt'))
                t_inh = np.loadtxt(os.path.join(curr_spikes_dir, 'CA1_inh_spikemon_t.txt'))
            else:
                print("[!] Warning: files missing!", "osc_amp: ", curr_osc_amp, " stim_amp: ", curr_stim_amp)
                continue

        # Fix the timings -> from ms to sec
        i_exc = i_exc.astype(int)
        t_exc = t_exc*ms
        i_inh = i_inh.astype(int)
        t_inh = t_inh*ms

        # get the firing rate (spikes-to-rates)
        tv_inh_FR, FR_inh = my_FR(spikes=t_inh, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
        tv_exc_FR, FR_exc = my_FR(spikes=t_exc, duration=duration, window_size=winsize_FR, overlap=overlap_FR)

        # Normalize the FRs
        FR_inh_norm = (FR_inh/winsize_FR)/N_tot[3][1]
        FR_exc_norm = (FR_exc/winsize_FR)/N_tot[3][0]

        # avoid divisions by zero
        # FR_inh_norm[FR_inh_norm<=1.e-5] = 1.e-5
        # FR_exc_norm[FR_exc_norm<=1.e-5] = 1.e-5

        # Add noise
        # FR_inh_norm += noise
        # FR_exc_norm += noise

        # get the post-stim indices
        tlims_post = np.array([500, 1500]) + t_stim # avoid stim artifact
        tidx_post = np.logical_and(tv_exc_FR/ms>=tlims_post[0], tv_exc_FR/ms<=tlims_post[1])

        # Use the TensorPAC module to calculate the PSDs
        psd_pac_inh = PSD(FR_inh_norm[np.newaxis,tidx_post], fs_FR)
        psd_pac_exc = PSD(FR_exc_norm[np.newaxis,tidx_post], fs_FR)

        # Theta band indices
        f_theta_idxs = np.logical_and(psd_pac_inh.freqs>=theta_band[0], psd_pac_inh.freqs<=theta_band[1])
        f_gamma_idxs = np.logical_and(psd_pac_inh.freqs>=gamma_band[0], psd_pac_inh.freqs<=gamma_band[1])

        # Calculate peaks - can use argmax or np.argwhere(np.abs(x-x.max())<1e-6)
        peaks_inh, props_inh = sig.find_peaks(psd_pac_inh.psd[0], prominence=1, threshold=psd_pac_inh.psd[0].max()*0.2)
        peaks_exc, props_exc = sig.find_peaks(psd_pac_exc.psd[0], prominence=1, threshold=psd_pac_exc.psd[0].max()*0.2)

        # Check if we have ~any~ peaks
        if peaks_inh.size > 0:
            # All peaks to theta peaks
            idx_theta_peaks = np.argwhere(np.logical_and(psd_pac_inh.freqs[peaks_inh]>=theta_band[0], psd_pac_inh.freqs[peaks_inh]<=theta_band[1]))

            # Idx of max theta power
            idx_max_theta_peak = np.argmax(psd_pac_inh.psd[0][peaks_inh[idx_theta_peaks]])

            # Which frequency is the max?
            peak_theta_freq_inh = psd_pac_inh.freqs[peaks_inh[idx_theta_peaks[idx_max_theta_peak]]].tolist()

            # Find peak frequency
            # peak_freq_inh = psd_pac_inh.freqs[peaks_inh][np.argwhere(psd_pac_inh.psd[0][peaks_inh] == psd_pac_inh.psd[0][peaks_inh].max())].squeeze().tolist()

            # Add to the list for later (hist calc)
            peak_freqs_inh.extend(peak_theta_freq_inh)

        if peaks_exc.size > 0:
            # All peaks to theta peaks
            idx_theta_peaks = np.argwhere(np.logical_and(psd_pac_exc.freqs[peaks_exc]>=theta_band[0], psd_pac_exc.freqs[peaks_exc]<=theta_band[1]))

            # Idx of max theta power
            idx_max_theta_peak = np.argmax(psd_pac_exc.psd[0][peaks_exc[idx_theta_peaks]])

            # Which frequency is the max?
            peak_theta_freq_exc = psd_pac_exc.freqs[peaks_exc[idx_theta_peaks[idx_max_theta_peak]]].tolist()

            # Find peak frequency
            # peak_freq_exc = psd_pac_exc.freqs[peaks_exc][np.argwhere(psd_pac_exc.psd[0][peaks_exc] == psd_pac_exc.psd[0][peaks_exc].max())].squeeze().tolist()

            # Add to the list for later (hist calc)
            peak_freqs_exc.extend(peak_theta_freq_exc)

        # if peaks_inh.size > 0 and peaks_exc.size > 0:
        #     exit()
        # else:
        #     continue

        # Print peak frequencies
        print("="*24)
        if (peaks_inh.size > 0):
            print("Inh. Peak Freqs: [", psd_pac_inh.freqs[peaks_inh], "] Hz")

        if (peaks_inh.size > 0):
            print("Exc. Peak Freqs: [", psd_pac_exc.freqs[peaks_exc], "] Hz")
        print("="*24, "\n")

        # if there is theta activity
        theta_flag_inh = np.any(np.isin(psd_pac_inh.freqs[peaks_inh], psd_pac_inh.freqs[f_theta_idxs], assume_unique=True))
        theta_flag_exc = np.any(np.isin(psd_pac_exc.freqs[peaks_exc], psd_pac_exc.freqs[f_theta_idxs], assume_unique=True))

        # if there is gamma activity
        gamma_flag_inh = np.any(np.isin(psd_pac_inh.freqs[peaks_inh], psd_pac_inh.freqs[f_gamma_idxs], assume_unique=True))
        gamma_flag_exc = np.any(np.isin(psd_pac_exc.freqs[peaks_exc], psd_pac_exc.freqs[f_gamma_idxs], assume_unique=True))

        # # calculate the modulation indexes
        # if theta_flag_inh and gamma_flag_inh:
        #     # MI_I = check_MI(FR_inh_norm[np.newaxis, tidx_post])
        #     MI_I = pac_obj.filterfit(fs_FR, FR_inh_norm[np.newaxis, tidx_post])
        #     MI_heatmap_I[idx_osc_amp, idx_stim_amp] = MI_I
        #
        #     # Calculate theta/gamma power (post-stim)
        #     theta_band_pow_inh = bandpower(FR_inh_norm[np.newaxis, tidx_post], fs_FR, theta_band, window_sec=1., overlap=0.9, relative=False)
        #     gamma_band_pow_inh = bandpower(FR_inh_norm[np.newaxis, tidx_post], fs_FR, gamma_band, window_sec=1., overlap=0.9, relative=False)
        #
        #     # Add the values to the heatmaps
        #     theta_heatmap_I[idx_osc_amp, idx_stim_amp] = theta_band_pow_inh
        #     gamma_heatmap_I[idx_osc_amp, idx_stim_amp] = gamma_band_pow_inh
        #
        # if theta_flag_exc and gamma_flag_exc:
        #     # MI_E = check_MI(FR_exc_norm[np.newaxis, tidx_post])
        #     MI_E = pac_obj.filterfit(fs_FR, FR_exc_norm[np.newaxis, tidx_post])
        #     MI_heatmap_E[idx_osc_amp, idx_stim_amp] = MI_E
        #
        #     # Calculate theta/gamma power (post-stim)
        #     theta_band_pow_exc = bandpower(FR_exc_norm[np.newaxis, tidx_post], fs_FR, theta_band, window_sec=1., overlap=0.9, relative=False)
        #     gamma_band_pow_exc = bandpower(FR_exc_norm[np.newaxis, tidx_post], fs_FR, gamma_band, window_sec=1., overlap=0.9, relative=False)
        #
        #     # Add the values to the heatmaps
        #     theta_heatmap_E[idx_osc_amp, idx_stim_amp] = theta_band_pow_exc
        #     gamma_heatmap_E[idx_osc_amp, idx_stim_amp] = gamma_band_pow_exc

        # Calculate modulation index
        MI_I = pac_obj.filterfit(fs_FR, FR_inh_norm[np.newaxis, tidx_post])
        MI_E = pac_obj.filterfit(fs_FR, FR_exc_norm[np.newaxis, tidx_post])

        # Calculate theta/gamma power (post-stim)
        theta_band_pow_inh = bandpower(FR_inh_norm[np.newaxis, tidx_post], fs_FR, theta_band, window_sec=1., overlap=0.9, relative=False)
        theta_band_pow_exc = bandpower(FR_exc_norm[np.newaxis, tidx_post], fs_FR, theta_band, window_sec=1., overlap=0.9, relative=False)
        gamma_band_pow_inh = bandpower(FR_inh_norm[np.newaxis, tidx_post], fs_FR, gamma_band, window_sec=1., overlap=0.9, relative=False)
        gamma_band_pow_exc = bandpower(FR_exc_norm[np.newaxis, tidx_post], fs_FR, gamma_band, window_sec=1., overlap=0.9, relative=False)

        # Add the values to the heatmaps
        MI_heatmap_I[idx_osc_amp, idx_stim_amp] = MI_I
        MI_heatmap_E[idx_osc_amp, idx_stim_amp] = MI_E

        theta_heatmap_I[idx_osc_amp, idx_stim_amp] = theta_band_pow_inh
        gamma_heatmap_I[idx_osc_amp, idx_stim_amp] = gamma_band_pow_inh
        theta_heatmap_E[idx_osc_amp, idx_stim_amp] = theta_band_pow_exc
        gamma_heatmap_E[idx_osc_amp, idx_stim_amp] = gamma_band_pow_exc


# Done with the iterations
print("Done.")

# nans to nums
# np.nan_to_num(MI_heatmap_E, copy=False)
# np.nan_to_num(MI_heatmap_I, copy=False)
# np.nan_to_num(theta_heatmap_E, copy=False)
# np.nan_to_num(theta_heatmap_I, copy=False)
# np.nan_to_num(gamma_heatmap_E, copy=False)
# np.nan_to_num(gamma_heatmap_I, copy=False)

# set vmin/vmax for plotting
# vmin = 1e-12
# vmax = 1.
vmin_MI = min(MI_heatmap_I.min(), MI_heatmap_E.min())
vmax_MI = max(MI_heatmap_I.max(), MI_heatmap_E.max())
vmin_theta = min(theta_heatmap_I.min(), theta_heatmap_E.min())
vmax_theta = max(theta_heatmap_I.max(), theta_heatmap_E.max())
vmin_gamma = min(gamma_heatmap_I.min(), gamma_heatmap_E.min())
vmax_gamma = max(gamma_heatmap_I.max(), gamma_heatmap_E.max())

# normalize the colors
norm_MI = mplb.colors.Normalize(vmin=vmin_MI, vmax=vmax_MI)
norm_theta = mplb.colors.Normalize(vmin=vmin_theta, vmax=vmax_theta)
norm_gamma = mplb.colors.Normalize(vmin=vmin_gamma, vmax=vmax_gamma)

# Plot the heatmaps
fig_MI, axs_MI = plt.subplots(2,1)
fig_theta, axs_theta = plt.subplots(2,1)
fig_gamma, axs_gamma = plt.subplots(2,1)

# MI images
im1 = axs_MI[0].pcolormesh(osc_amps, stim_amps, MI_heatmap_I.T, norm=norm_MI, shading='auto', rasterized=True)
im2 = axs_MI[1].pcolormesh(osc_amps, stim_amps, MI_heatmap_E.T, norm=norm_MI, shading='auto', rasterized=True)

# Theta power images
im3 = axs_theta[0].pcolormesh(osc_amps, stim_amps, theta_heatmap_I.T, norm=norm_theta, shading='auto', rasterized=True)
im4 = axs_theta[1].pcolormesh(osc_amps, stim_amps, theta_heatmap_E.T, norm=norm_theta, shading='auto', rasterized=True)

# Gamma power images
im5 = axs_gamma[0].pcolormesh(osc_amps, stim_amps, gamma_heatmap_I.T, norm=norm_gamma, shading='auto', rasterized=True)
im6 = axs_gamma[1].pcolormesh(osc_amps, stim_amps, gamma_heatmap_E.T, norm=norm_gamma, shading='auto', rasterized=True)

# Set x-y labels
axs_MI[0].set_ylabel('Stim. amp.')
axs_MI[1].set_ylabel('Stim. amp.')
axs_MI[1].set_xlabel('Osc. amp.')

axs_theta[0].set_ylabel('Stim. amp.')
axs_theta[1].set_ylabel('Stim. amp.')
axs_theta[1].set_xlabel('Osc. amp.')

axs_gamma[0].set_ylabel('Stim. amp.')
axs_gamma[1].set_ylabel('Stim. amp.')
axs_gamma[1].set_xlabel('Osc. amp.')

# Set titles
axs_MI[0].set_title('Inhibitory')
axs_MI[1].set_title('Excitatory')

axs_gamma[0].set_title('Inhibitory')
axs_gamma[1].set_title('Excitatory')

axs_gamma[0].set_title('Inhibitory')
axs_gamma[1].set_title('Excitatory')

# Add the colorbars
plt.colorbar(im2, label="Modulation Index", ax=axs_MI.ravel().tolist())
plt.colorbar(im4, label="Theta Power", ax=axs_theta.ravel().tolist())
plt.colorbar(im6, label="Gamma Power", ax=axs_gamma.ravel().tolist())

# Save the figure
print('[+] Saving the figure...')
fig_MI.savefig(os.path.join(parent_dir, 'figures', 'fig4', 'fig4_B.png'), transparent=True, dpi=300, format='png', bbox_inches='tight')
fig_MI.savefig(os.path.join(parent_dir, 'figures', 'fig4', 'fig4_B.pdf'), transparent=True, dpi=300, format='pdf', bbox_inches='tight')

fig_theta.savefig(os.path.join(parent_dir, 'figures', 'fig4', 'fig4_B_theta.png'), transparent=True, dpi=300, format='png', bbox_inches='tight')
fig_theta.savefig(os.path.join(parent_dir, 'figures', 'fig4', 'fig4_B_theta.pdf'), transparent=True, dpi=300, format='pdf', bbox_inches='tight')

fig_gamma.savefig(os.path.join(parent_dir, 'figures', 'fig4', 'fig4_B_gamma.png'), transparent=True, dpi=300, format='png', bbox_inches='tight')
fig_gamma.savefig(os.path.join(parent_dir, 'figures', 'fig4', 'fig4_B_gamma.pdf'), transparent=True, dpi=300, format='pdf', bbox_inches='tight')

# Show the images
plt.show()

# Exit regularly
sys.exit(0)
