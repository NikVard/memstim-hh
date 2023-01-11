import os
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

# Parameters
T = 3
ms = 1e-3
dt = 0.1*ms
fs = int(1/dt)
tv = np.arange(0, T, dt)

# file path
dir = 'results_PRC_new/None/22-06-2022 12H35M46S/data'
fname = 'order_param_mon_phase.txt'
full_path = os.path.join(dir, fname)

# target phases
rdec = 6
phase_step = np.pi/8
phase_target = np.linspace(-np.pi, np.pi, 11)
phase_target = np.arange(-np.pi, np.pi+phase_step, phase_step)
phase_target = np.round(phase_target, rdec)

# load the phase vector
phase = np.loadtxt(full_path)

# isolate a phase cycle
t_low = 1500*ms
t_high = 2000*ms
t_low_s = t_low*fs
t_high_s = t_high*fs

# find peaks
phase_pks, _ = sig.find_peaks(phase, distance=120*ms*fs)
mask_pks = (phase_pks > t_low*fs) & (phase_pks < t_high*fs)
phase_idxs = phase_pks[mask_pks]

# stimulation times/samples
t_stims = []
s_stims = []
for pt in phase_target:
    tmp = np.argmin(np.abs(np.round(phase[phase_idxs[0]:phase_idxs[1]+1], rdec) - pt))
    s_stims.append(tmp)
    t_stims.append((phase_idxs[0] + tmp)/fs)

# plot the phase
print("[+] Plotting...")
plt.plot(tv, phase)
for val in t_stims:
    plt.plot(val, phase[int(val*fs)], color='r', marker='o')
plt.show()

# save the times
print("[+] Saving stimulus timings...")
np.savetxt('stim_times.txt', t_stims, fmt='%.6f')


# generate the simulation config files
