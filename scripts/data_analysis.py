#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import numpy as np
# import matplotlib.pyplot as plt

from brian2 import *

# Parameters
# ------------------------------------------------------------------------------
areas = ['EC', 'DG', 'CA3', 'CA1']
duration = 2*second
dt = 0.1*ms
bin_size = 5*ms

# Directories
# ------------------------------------------------------------------------------
dirs = {}

# Results path
dirs['results'] = 'results/analysis/'
dirs['data'] = dirs['results'] + 'data/'

# Spikes
dirs['spikes'] = dirs['data'] + 'spikes/'


# Load the rasters
# ------------------------------------------------------------------------------
print(">> Loading rasters...")

EC_exc_t = np.loadtxt(dirs['spikes'] + 'EC_pyCAN_spikemon_t', dtype=np.float32)
EC_exc_i = np.loadtxt(dirs['spikes'] + 'EC_pyCAN_spikemon_i', dtype=int)
EC_inh_t = np.loadtxt(dirs['spikes'] + 'EC_inh_spikemon_t', dtype=np.float32)
EC_inh_i = np.loadtxt(dirs['spikes'] + 'EC_inh_spikemon_i', dtype=int)

DG_exc_t = np.loadtxt(dirs['spikes'] + 'DG_py_spikemon_t', dtype=np.float32)
DG_exc_i = np.loadtxt(dirs['spikes'] + 'DG_py_spikemon_i', dtype=int)
DG_inh_t = np.loadtxt(dirs['spikes'] + 'DG_inh_spikemon_t', dtype=np.float32)
DG_inh_i = np.loadtxt(dirs['spikes'] + 'DG_inh_spikemon_i', dtype=int)

CA3_exc_t = np.loadtxt(dirs['spikes'] + 'CA3_pyCAN_spikemon_t', dtype=np.float32)
CA3_exc_i = np.loadtxt(dirs['spikes'] + 'CA3_pyCAN_spikemon_i', dtype=int)
CA3_inh_t = np.loadtxt(dirs['spikes'] + 'CA3_inh_spikemon_t', dtype=np.float32)
CA3_inh_i = np.loadtxt(dirs['spikes'] + 'CA3_inh_spikemon_i', dtype=int)

CA1_exc_t = np.loadtxt(dirs['spikes'] + 'CA1_pyCAN_spikemon_t', dtype=np.float32)
CA1_exc_i = np.loadtxt(dirs['spikes'] + 'CA1_pyCAN_spikemon_i', dtype=int)
CA1_inh_t = np.loadtxt(dirs['spikes'] + 'CA1_inh_spikemon_t', dtype=np.float32)
CA1_inh_i = np.loadtxt(dirs['spikes'] + 'CA1_inh_spikemon_i', dtype=int)

EC_exc_N = 10000
EC_inh_N = 1000
DG_exc_N = 10000
DG_inh_N = 100
CA3_exc_N = 1000
CA3_inh_N = 100
CA1_exc_N = 10000
CA1_inh_N = 1000
N_tot = EC_exc_N + EC_inh_N + DG_exc_N + DG_inh_N + CA3_exc_N + CA3_inh_N + CA1_exc_N + CA1_inh_N

# Generate firing rates
# COBA example, brian2 docs
# https://brian2.readthedocs.io/en/stable/examples/frompapers.Stimberg_et_al_2018.example_1_COBA.html
# ------------------------------------------------------------------------------
print(">> Generating firing rates from rasters...")

EC_exc_spk_cnt, EC_exc_bin_edges = np.histogram(EC_exc_t, int(duration/ms))
EC_exc_rate = double(EC_exc_spk_cnt)/(EC_exc_N)/bin_size/Hz
EC_inh_spk_cnt, EC_inh_bin_edges = np.histogram(EC_inh_t, int(duration/ms))
EC_inh_rate = double(EC_inh_spk_cnt)/(EC_inh_N)/bin_size/Hz

DG_exc_spk_cnt, DG_exc_bin_edges = np.histogram(DG_exc_t, int(duration/ms))
DG_exc_rate = double(DG_exc_spk_cnt)/(DG_exc_N)/bin_size/Hz
DG_inh_spk_cnt, DG_inh_bin_edges = np.histogram(DG_inh_t, int(duration/ms))
DG_inh_rate = double(DG_inh_spk_cnt)/(DG_inh_N)/bin_size/Hz

CA3_exc_spk_cnt, CA3_exc_bin_edges = np.histogram(CA3_exc_t, int(duration/ms))
CA3_exc_rate = double(CA3_exc_spk_cnt)/(CA3_exc_N)/bin_size/Hz
CA3_inh_spk_cnt, CA3_inh_bin_edges = np.histogram(CA3_inh_t, int(duration/ms))
CA3_inh_rate = double(CA3_inh_spk_cnt)/(CA3_inh_N)/bin_size/Hz

CA1_exc_spk_cnt, CA1_exc_bin_edges = np.histogram(CA1_exc_t, int(duration/ms))
CA1_exc_rate = double(CA1_exc_spk_cnt)/(CA1_exc_N)/bin_size/Hz
CA1_inh_spk_cnt, CA1_inh_bin_edges = np.histogram(CA1_inh_t, int(duration/ms))
CA1_inh_rate = double(CA1_inh_spk_cnt)/(CA1_inh_N)/bin_size/Hz

# Plot the firing rates
# ------------------------------------------------------------------------------
fig1, axs1 = plt.subplots(4, sharex=True, sharey=True, figsize=(16,8))
axs1[0].plot(EC_exc_bin_edges[:-1], EC_exc_rate, linestyle='-', color='b', label= 'exc')
axs1[0].plot(EC_inh_bin_edges[:-1], EC_inh_rate, linestyle='--', color='r', label='inh')
axs1[1].plot(DG_exc_bin_edges[:-1], DG_exc_rate, linestyle='-', color='b', label='exc')
axs1[1].plot(DG_inh_bin_edges[:-1], DG_inh_rate, linestyle='--', color='r', label='inh')
axs1[2].plot(CA3_exc_bin_edges[:-1], CA3_exc_rate, linestyle='-', color='b', label='exc')
axs1[2].plot(CA3_inh_bin_edges[:-1], CA3_inh_rate, linestyle='--', color='r', label='inh')
axs1[3].plot(CA1_exc_bin_edges[:-1], CA1_exc_rate, linestyle='-', color='b', label='exc')
axs1[3].plot(CA1_inh_bin_edges[:-1], CA1_inh_rate, linestyle='--', color='r', label='inh')

for (ax, lbl) in zip(axs1, areas):
    ax.set_ylabel(lbl)
    ax.set_ylim([0, 1000])

axs1[3].set_xlabel('Time [ms]')
axs1[3].legend()


plt.show()
