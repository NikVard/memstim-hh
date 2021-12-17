"""
--------------------------------------------------------------------------------
Date: 09/12/2021

@author: Nikolaos Vardalakis
--------------------------------------------------------------------------------

Implementation Notes
--------------------------------------------------------------------------------
    | 1. Includes functions to generate stimulation waveforms. The functions are
    | not restricted to brian2, i.e. the functions return numpy arrays which can
    | later be transformed into TimedArrays in the main program.
"""

import numpy as np

def generate_stim(duration, dt=1e-4, stim_on=0., nr_of_trains=5, nr_of_pulses=4, stim_freq=5, stim_type="monophasic", pulse_width=.2e-3, pulse_freq=100, ipi=.1e-3):
    ''' Generate a pulse train to be used as the stimulation waveform.

    Parameters:
    ---------------------------------------------------------------------------
    *       duration    :   stimulation waveform duration - [sec]
    |
    *           dt      :   sampling rate - [sec]
    |
    *       stim_on     :   stimulation start time - [sec]
    |
    *   nr_of_trains    :   number of pulse trains i.e. groups of pulses - [1]
    |
    *   nr_of_pulses    :   number of pulses per group - [1]
    |
    *       stim_freq   :   stimulation frequency - [Hz]
    |
    *       stim_type   :   "monophasic" | "anodic-first" | "cathodic-first"
    |
    *       pulse_width :   width (in time) of pulse ON phase - [sec]
    |
    *       pulse_freq  :   pulse frequency; determines ON duration - [Hz]
    |
    *               ipi :   inter-pulse interval - [sec]
    ---------------------------------------------------------------------------
    '''

    # step 0: parse arguments and evaluate
    if pulse_width > 1/pulse_freq:
        raise ValueError('Pulse width is too large for given pulse frequency.')


    # calculate single pulse duration
    pd = 1./pulse_freq                  # pulse duration [sec]
    pd_samples = int(pd/dt)             # pulse duration [samples]
    pw_samples = int(pulse_width/dt)    # pulse width [samples]
    ipi_samples = int(ipi/dt)           # inter-pulse interval [samples]

    # step 1: create a single pulse
    if stim_type.lower() == "monophasic":
        # monophasic pulse
        pulse = np.zeros((1,pd_samples), dtype="int")
        pulse[0,:pw_samples] = 1        # ON state
    else:
        # biphasic pulse
        pulse = np.zeros((1,pd_samples), dtype="int")
        pulse[0,:pw_samples]            # ON state
        idx = 0
        pulse[0,0:idx+pw_samples] = 1
        idx += pw_samples
        pulse[0,idx:idx+ipi_samples] = 0
        idx += ipi_samples
        pulse[0,idx:idx+pw_samples] = -1

        if stim_type.lower()=="cathodic-first":
            pulse *= -1

    # step 2: repeat the pulse and add the delay between bursts
    pulse_train = np.tile(pulse, nr_of_pulses)
    delay = int(1/stim_freq/dt - len(pulse_train[0]))
    pulse_train = np.append(pulse_train, np.zeros((1,delay)), axis=1)

    # step 3: repeat the whole group
    waveform_ = np.tile(pulse_train, nr_of_trains) # temp array

    # step 4: create the stimulation waveform
    tv = np.linspace(0, duration, int(duration/dt)+1)
    stimulation = np.zeros(tv.shape, dtype="int")

    # find the nearest time value to the stimulation start time
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

    v, i = find_nearest(tv, stim_on)

    # step 5: clip the waveform, removing unused trailing zeros
    idxs = np.where(waveform_.flatten() != 0)
    idx = np.max(idxs)
    waveform = waveform_[0,:idx+10]

    # check that waveform is not too large
    if (i+len(waveform) > len(tv)):
        raise ValueError('Generated signal too large for given duration.')

    stimulation[i:i+len(waveform)] = waveform

    return stimulation
