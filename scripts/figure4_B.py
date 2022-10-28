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
        print('-> stim: ', stim_amp_dir)

        # go through the stimulation directory
        stim_time_dir = next(os.walk(curr_stim_dir))[1][0]
        t_stim = float(stim_time_dir.split('_')[1]) # in ms
        # print('|')
        print('---> stim_on: ', stim_time_dir)

        # traverse the next directory too (simulation)
        curr_stim_time_dir = os.path.join(curr_stim_dir, stim_time_dir)
        sim_dir = next(os.walk(curr_stim_time_dir))[1][0]
        curr_sim_dir = os.path.join(curr_stim_time_dir, sim_dir)
        # print('|')
        print('-----> sim: ', sim_dir)

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

        # get the post-stim indices
        tlims_post = np.array([0, 1000]) + t_stim
        tidx_post = np.logical_and(tv_exc_FR/ms>=tlims_post[0], tv_exc_FR/ms<=tlims_post[1])

        # calculate the modulation index
        MI_I = check_MI(FR_inh_norm[np.newaxis, tidx_post])
        MI_E = check_MI(FR_exc_norm[np.newaxis, tidx_post])

        # Add the MI in the MI matrix
        MI_heatmap_I[idx_osc_amp, idx_stim_amp] = MI_I
        MI_heatmap_E[idx_osc_amp, idx_stim_amp] = MI_E

# Done with the iterations
print("Done.")

# nans to nums
np.nan_to_num(MI_heatmap_E, copy=False)
np.nan_to_num(MI_heatmap_I, copy=False)

# set vmin/vmax for plotting
# vmin = 1e-12
# vmax = 1.
vmin = min(MI_heatmap_I.min(), MI_heatmap_E.min())
vmax = max(MI_heatmap_I.max(), MI_heatmap_E.max())

# normalize the colors
norm_inh = mplb.colors.Normalize(vmin=vmin, vmax=vmax)
norm_exc = mplb.colors.Normalize(vmin=vmin, vmax=vmax)

# Plot the two heatmaps
fig, axs = plt.subplots(2,1)

im1 = axs[0].pcolormesh(osc_amps, stim_amps, MI_heatmap_I.T, norm=norm_inh, shading='auto', rasterized=True)
im2 = axs[1].pcolormesh(osc_amps, stim_amps, MI_heatmap_E.T, norm=norm_exc, shading='auto', rasterized=True)

# Set x-y labels
axs[0].set_ylabel('Stim. amp.')
axs[1].set_ylabel('Stim. amp.')
axs[1].set_xlabel('Osc. amp.')

# Set titles
axs[0].set_title('Inhibitory')
axs[1].set_title('Excitatory')

# Add the colorbar
plt.colorbar(im2, label="Modulation Index", ax=axs.ravel().tolist())

# Show the image
plt.show()

# Exit regularly
sys.exit(0)
