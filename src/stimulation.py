import numpy as np

def generate_stim(duration, dt=1e-4, I_stim=[1.], stim_on=0., nr_of_trains=5, nr_of_pulses=4, stim_freq=5, pulse_width=[.2e-3], pulse_freq=100, ipi=.1e-3):
    ''' Generate a pulse train to be used as the stimulation waveform.

    Parameters:
    ---------------------------------------------------------------------------
    *       duration    :   stimulation waveform duration - [sec]
    |
    *           dt      :   sampling rate - [sec]
    |
    *        I_stim     :   stimulation amplitudes, given in list format [I_first, I_second] - [nA]
    |                       I < 0 -> cathodic pulse (e.g. I_stim = [-1.,1.] means biphasic, cathodic-first)
    |
    *       stim_on     :   stimulation start time - [sec]
    |
    *   nr_of_trains    :   number of pulse trains i.e. groups of pulses - [1]
    |
    *   nr_of_pulses    :   number of pulses per group - [1]
    |
    *       stim_freq   :   stimulation frequency - [Hz]
    |
    *       pulse_width :   width (in time) of pulse ON phase, given in list format [pw_first, pw_second] - [sec]
    |                       I[i] < 0 means pw[i] is the width of the cathodic phase
    |
    *       pulse_freq  :   pulse frequency; determines ON duration - [Hz]
    |
    *               ipi :   inter-pulse interval - [sec]
    ---------------------------------------------------------------------------
    '''
    # calculate single pulse duration
    pd = 1./pulse_freq                  # pulse duration [sec]
    pd_samples = int(pd/dt)             # pulse duration [samples]

    # step 1: create a single pulse
    if len(I_stim) == 1:
        # monophasic pulse
        I0 = I_stim[0] # get pulse amplitude
        pw = pulse_width[0] # get pulse width
        pw_samples = int(pw/dt) # pulse width [samples]

        # evaluation
        if (pw_samples > pd_samples):
            raise ValueError('Pulse width is too large for given pulse frequency.')

        # make the pulse
        pulse = np.zeros((1,pd_samples), dtype="int")
        pulse[0,:pw_samples] = I0           # ON state

    elif len(I_stim) == 2:
        # biphasic pulse
        I0 = np.array(I_stim)
        pw = np.array(pulse_width)

        pw_samples = pw/dt    # pulse width [samples]
        pw_samples = pw_samples.astype(int)
        ipi_samples = int(ipi/dt)           # inter-pulse interval [samples]

        # evaluation
        if ((np.sum(pw_samples) + ipi_samples) > pd_samples):
            raise ValueError('Pulse width is too large for given pulse frequency.')

        if np.sum(np.multiply(I0, pw)) != 0:
            raise ValueError('Current settings do not lead to charge-balanced pulses.')

        # make the pulse
        pulse = np.zeros((1,pd_samples), dtype="int")
        idx = 0
        pulse[0,0:idx+pw_samples[0]] = I0[0]
        idx += pw_samples[0]
        pulse[0,idx:idx+ipi_samples] = 0
        idx += ipi_samples
        pulse[0,idx:idx+pw_samples[1]] = I0[1]

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
