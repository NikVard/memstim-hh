# -*- coding: utf-8 -*-

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib as mplb
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = Path(script_dir).parent
sys.path.insert(0, os.path.abspath(parent_dir))

from src.freq_analysis import *

# PAC check using Modulation Index (MI)
# 1. Filtering at theta/gamma bands
theta_freqs = [4, 12]
gamma_freqs = [30, 80]
FR_exc_filt_gamma = butter_bandpass_filter(FR_exc_norm, gamma_freqs[0], gamma_freqs[1], fs_FR, sos=True)
FR_exc_filt_theta = butter_bandpass_filter(FR_exc_norm, theta_freqs[0], theta_freqs[1] fs_FR, sos=True)

# 2. Time series of phases / amplitudes using Hilbert Transform
xfp = sig.hilbert(FR_exc_filt_theta)
xfA = sig.hilbert(FR_exc_filt_gamma)
sig_amp = np.abs(xfA)
sig_phase = np.angle(xfp)

# 3./4. Calculate Modulation Index (MI)
MI, dist_KL = my_modulation_index(sig_phase, sig_amp)
