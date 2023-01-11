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

# TensorPAC
from tensorpac import Pac
from tensorpac.utils import PSD

# Other scripts and my stuff
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = Path(script_dir).parent
sys.path.insert(0, os.path.abspath(parent_dir))

import parameters
from src.freq_analysis import *

# get the root directory for the simulations
root_cluster = os.path.join(parent_dir, 'results_cluster', 'results_fig4_low_kN')

# get a list of the directories with oscillator amplitudes
osc_amplitude_dirs = sorted(next(os.walk(root_cluster))[1])

# Timing parameters
second = 1
ms = 1e-3
duration = 10*second
stim_onset = 1800*ms
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

# kN values = np.arange(1, 11, 1)
kN_vals = []
for kN_dir_curr in next(os.walk(os.path.join(root_cluster, osc_amplitude_dirs[1], '8.0_nA', '0.00_2000.0_ms')))[1]:

    print(kN_dir_curr)

    pdir = os.path.join(root_cluster, osc_amplitude_dirs[1], '8.0_nA', '0.00_2000.0_ms', kN_dir_curr)
    fname = 'parameters_bak.json'

    try:
        data = parameters.load(os.path.join(pdir, fname))
        kN_curr = data['Kuramoto']['kN']
        print('Current kN: "{0}"'.format(kN_curr))
        kN_vals.append(kN_curr)
    except Exception as e:
        print(bcolors.RED + '[!]' + "Error code " + str(e.errno) + ": " + e.strerror + ' | Parameters file not found!' + bcolors.ENDC)

kN_vals.sort()
kN_vals = np.array(kN_vals)

# make a meshgrid for plotting heatmaps
X, Y = np.meshgrid(osc_amps, kN_vals)
# MI_heatmap_E = np.zeros(X.shape)
# MI_heatmap_I = np.zeros(X.shape)
MI_heatmap_E = np.zeros((len(osc_amps), len(kN_vals)))
MI_heatmap_I = np.zeros((len(osc_amps), len(kN_vals)))

# peak frequencies
peak_freqs_inh = []
peak_freqs_exc = []

# theta / gamma power matrices
theta_band = [3, 9]
gamma_band = [40, 80]
theta_heatmap_E = np.zeros((len(osc_amps), len(kN_vals)))
theta_heatmap_I = np.zeros((len(osc_amps), len(kN_vals)))
gamma_heatmap_E = np.zeros((len(osc_amps), len(kN_vals)))
gamma_heatmap_I = np.zeros((len(osc_amps), len(kN_vals)))

# noise - uniform function instead [min/max]
curr_state = np.random.get_state() # get current state
np.random.seed(42) # fix the noise - reproducibility
noise = np.random.uniform(0, 10, int((duration-winsize_FR)*fs_FR)+2) # change max noise value
np.random.set_state(curr_state) # resume state

# Make a PAC object
f_pha_PAC = theta_band
f_amp_PAC = gamma_band
pac_obj = Pac(idpac=(2, 0, 0), f_pha=f_pha_PAC, f_amp=f_amp_PAC)

# Parameters (backup) filename
fname = 'parameters_bak.json'
# t_stim = data['stimulation']['onset']*1000
t_stim = 2000.0 # ms

# iterate over oscillator amplitudes
for osc_amp_dir in tqdm(osc_amplitude_dirs, desc='Computing the modulation index...'):
    curr_osc_amp = float(osc_amp_dir.split('_')[1])
    idx_osc_amp = np.where(osc_amps == curr_osc_amp)[0] # index for heatmap

    # one step in
    curr_osc_dir = os.path.join(root_cluster, osc_amp_dir)
    print('osc: ', osc_amp_dir)

    # two steps in - stim 8.0nA @ 2000ms
    for kN_dir in next(os.walk(os.path.join(curr_osc_dir, '8.0_nA', '0.00_2000.0_ms')))[1]:

        # one more step in - kN
        curr_kN_dir = os.path.join(curr_osc_dir, '8.0_nA', '0.00_2000.0_ms', kN_dir)

        # load parameters backup data
        try:
            data = parameters.load(os.path.join(curr_kN_dir, fname))
        except Exception as e:
            print('[!]' + "Error code " + str(e.errno) + ": " + e.strerror + ' | Parameters file not found!')
            continue;

        curr_kN_val = data['Kuramoto']['kN']
        idx_kN_val = np.where(kN_vals == curr_kN_val)[0] # index for heatmap

        # load the data (spikes) for the CA1 E-group
        curr_data_dir = os.path.join(curr_kN_dir, 'data')
        curr_spikes_dir = os.path.join(curr_data_dir, 'spikes')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, append=1)

            if len([entry for entry in os.listdir(curr_spikes_dir) if os.path.isfile(os.path.join(curr_spikes_dir, entry))]) == 16:
                i_exc = np.loadtxt(os.path.join(curr_spikes_dir, 'CA1_pyCAN_spikemon_i.txt'))
                t_exc = np.loadtxt(os.path.join(curr_spikes_dir, 'CA1_pyCAN_spikemon_t.txt'))
                i_inh = np.loadtxt(os.path.join(curr_spikes_dir, 'CA1_inh_spikemon_i.txt'))
                t_inh = np.loadtxt(os.path.join(curr_spikes_dir, 'CA1_inh_spikemon_t.txt'))
            else:
                print("[!] Warning: files missing!", "osc_amp: ", curr_osc_amp, " kN_val: ", curr_kN_val)
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

        # Add noise
        FR_inh_norm_noise = FR_inh_norm + noise
        FR_exc_norm_noise = FR_exc_norm + noise

        # get the post-stim indices
        tlims_post = np.array([500, 5500]) + t_stim # avoid stim artifact | 5s window
        tidx_post = np.logical_and(np.round(tv_exc_FR/ms,4)>tlims_post[0], np.round(tv_exc_FR/ms,4)<=tlims_post[1])

        # Use the TensorPAC module to calculate the PSDs
        psd_pac_inh = PSD(FR_inh_norm[np.newaxis,tidx_post], fs_FR)
        psd_pac_exc = PSD(FR_exc_norm[np.newaxis,tidx_post], fs_FR)

        # Theta band indices
        f_theta_idxs = np.logical_and(psd_pac_inh.freqs>=theta_band[0], psd_pac_inh.freqs<=theta_band[1])
        f_gamma_idxs = np.logical_and(psd_pac_inh.freqs>=gamma_band[0], psd_pac_inh.freqs<=gamma_band[1])

        # Calculate peaks - can use argmax or np.argwhere(np.abs(x-x.max())<1e-6)
        peaks_inh, props_inh = sig.find_peaks(psd_pac_inh.psd[0], prominence=1)
        peaks_exc, props_exc = sig.find_peaks(psd_pac_exc.psd[0], prominence=1)

        # Check if we have ~any~ peaks
        if peaks_inh.size > 0:
            # Filter out peaks lower than 10% of signal max
            threshold_inh = psd_pac_inh.psd[0].max()*0.1
            idx_peaks_inh_filt = psd_pac_inh.psd[0][peaks_inh] > threshold_inh
            peaks_inh = peaks_inh[idx_peaks_inh_filt]

            # All peaks to theta peaks
            idx_theta_peaks = np.argwhere(np.logical_and(psd_pac_inh.freqs[peaks_inh]>=theta_band[0], psd_pac_inh.freqs[peaks_inh]<=theta_band[1]))

            if idx_theta_peaks.size > 0:
                # Idx of max theta power
                idx_max_theta_peak = np.argmax(psd_pac_inh.psd[0][peaks_inh[idx_theta_peaks]])

                # Which frequency is the max?
                peak_theta_freq_inh = psd_pac_inh.freqs[peaks_inh[idx_theta_peaks[idx_max_theta_peak]]].tolist()

                # Find peak frequency
                # peak_freq_inh = psd_pac_inh.freqs[peaks_inh][np.argwhere(psd_pac_inh.psd[0][peaks_inh] == psd_pac_inh.psd[0][peaks_inh].max())].squeeze().tolist()

                # Add to the list for later (hist calc)
                peak_freqs_inh.extend(peak_theta_freq_inh)

        if peaks_exc.size > 0:
            # Filter out peaks lower than 10% of signal max
            threshold_exc = psd_pac_exc.psd[0].max()*0.1
            idx_peaks_exc_filt = psd_pac_exc.psd[0][peaks_exc] > threshold_exc
            peaks_exc = peaks_exc[idx_peaks_exc_filt]

            # All peaks to theta peaks
            idx_theta_peaks = np.argwhere(np.logical_and(psd_pac_exc.freqs[peaks_exc]>=theta_band[0], psd_pac_exc.freqs[peaks_exc]<=theta_band[1]))

            if idx_theta_peaks.size > 0:
                # Idx of max theta power
                idx_max_theta_peak = np.argmax(psd_pac_exc.psd[0][peaks_exc[idx_theta_peaks]])

                # Which frequency is the max?
                peak_theta_freq_exc = psd_pac_exc.freqs[peaks_exc[idx_theta_peaks[idx_max_theta_peak]]].tolist()

                # Find peak frequency
                # peak_freq_exc = psd_pac_exc.freqs[peaks_exc][np.argwhere(psd_pac_exc.psd[0][peaks_exc] == psd_pac_exc.psd[0][peaks_exc].max())].squeeze().tolist()

                # Add to the list for later (hist calc)
                peak_freqs_exc.extend(peak_theta_freq_exc)

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

        # Calculate modulation index
        MI_I = pac_obj.filterfit(fs_FR, FR_inh_norm_noise[np.newaxis, tidx_post])
        MI_E = pac_obj.filterfit(fs_FR, FR_exc_norm_noise[np.newaxis, tidx_post])

        # Calculate theta/gamma power (post-stim)
        theta_band_pow_inh = bandpower(FR_inh_norm[np.newaxis, tidx_post], fs_FR, theta_band, window_sec=1., overlap=0.9, relative=False)
        theta_band_pow_exc = bandpower(FR_exc_norm[np.newaxis, tidx_post], fs_FR, theta_band, window_sec=1., overlap=0.9, relative=False)
        gamma_band_pow_inh = bandpower(FR_inh_norm[np.newaxis, tidx_post], fs_FR, gamma_band, window_sec=1., overlap=0.9, relative=False)
        gamma_band_pow_exc = bandpower(FR_exc_norm[np.newaxis, tidx_post], fs_FR, gamma_band, window_sec=1., overlap=0.9, relative=False)

        # Add the values to the heatmaps
        MI_heatmap_I[idx_osc_amp, idx_kN_val] = MI_I
        MI_heatmap_E[idx_osc_amp, idx_kN_val] = MI_E

        theta_heatmap_I[idx_osc_amp, idx_kN_val] = theta_band_pow_inh
        gamma_heatmap_I[idx_osc_amp, idx_kN_val] = gamma_band_pow_inh
        theta_heatmap_E[idx_osc_amp, idx_kN_val] = theta_band_pow_exc
        gamma_heatmap_E[idx_osc_amp, idx_kN_val] = gamma_band_pow_exc


# Done with the iterations
print("Done.")

# Save the figure
print('[+] Saving the data (.npy objects)...')
print('[*] Modulation Index E/I...')
np.save(os.path.join(parent_dir, 'figures', 'fig4_low_kN', 'data', 'MI_E_heatmap.npy'), MI_heatmap_E, allow_pickle=False, fix_imports=False)
np.save(os.path.join(parent_dir, 'figures', 'fig4_low_kN', 'data', 'MI_I_heatmap.npy'), MI_heatmap_I, allow_pickle=False, fix_imports=False)
print('[*] Theta-Gamma heatmaps...')
np.save(os.path.join(parent_dir, 'figures', 'fig4_low_kN', 'data', 'theta_E_heatmap.npy'), theta_heatmap_E, allow_pickle=False, fix_imports=False)
np.save(os.path.join(parent_dir, 'figures', 'fig4_low_kN', 'data', 'theta_I_heatmap.npy'), theta_heatmap_I, allow_pickle=False, fix_imports=False)
np.save(os.path.join(parent_dir, 'figures', 'fig4_low_kN', 'data', 'gamma_E_heatmap.npy'), gamma_heatmap_E, allow_pickle=False, fix_imports=False)
np.save(os.path.join(parent_dir, 'figures', 'fig4_low_kN', 'data', 'gamma_I_heatmap.npy'), gamma_heatmap_I, allow_pickle=False, fix_imports=False)

# Done with the iterations
print("[V] Completed. Exiting...")

# Exit regularly
sys.exit(0)
