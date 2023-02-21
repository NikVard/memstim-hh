"""
--------------------------------------------------------------------------------
Date: 11/10/2021

@author: Nikolaos Vardalakis
--------------------------------------------------------------------------------

Implementation Notes
--------------------------------------------------------------------------------
    | 1: The simplest (dumbest) way to get the firing rate of a population of neurons is to low-pass filter the spikes it produces. Thus, we will be using a neuron group as a real-time exponential filter to get the firing rate; this implementation is condoned by @adam and @mstimberg here: https://brian.discourse.group/t/how-can-i-get-the-population-firing-rate-from-a-spiking-hh-network-during-simulation/496
    | 2: A demo of this approach is shown in the jupyter notebook 'feedback.ipynb'
"""

# Spikes-2-Rates
firing_rate_filter_eqs = '''
    dY/dt = -Y/tauFR : 1/second
    drive = Y/Hz : 1
'''
