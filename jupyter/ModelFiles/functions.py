from brian2 import *

def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')
    

    
def generate_stim(duration, dt=1e-4, stim_on=.2, I_stim=[1.], nr_of_trains=5, nr_of_pulses=4, stim_freq=5, pulse_width=[.2e-3], pulse_freq=100, ipi=.1e-3):
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
        pulse = np.zeros((1,pd_samples), dtype="float")
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
        pulse = np.zeros((1,pd_samples), dtype="float")
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
    stimulation = np.zeros(tv.shape, dtype="float")

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

    return stimulation, tv


# Add sizebar to figure
def add_sizebar(ax, xlocs, ylocs, bcolor, text, textx, texty, fsize, rot, ha='center', va='center'):
    """ Add a sizebar to the provided axis """
    ax.plot(xlocs, ylocs, ls='-', c=bcolor, linewidth=1., rasterized=False, clip_on=False)

    # add the text
    if type(text) is list:
        # bottom text
        ax.text(x=textx[0], y=texty[0], s=text[0], rotation=rot[0], fontsize=fsize, va=va, ha=ha, clip_on=False)

        # top text
        ax.text(x=textx[1], y=texty[1], s=text[1], rotation=rot[1], fontsize=fsize, va=va, ha=ha, clip_on=False)
    else:
        ax.text(x=textx, y=texty, s=text, rotation=rot, fontsize=fsize, va=va, ha=ha, clip_on=False)

# Manual FR computation
def my_FR(spikes: np.ndarray,
            duration: float,
            window_size: float,
            overlap: float) -> (np.ndarray, np.ndarray):
    """
    Compute the firing rate using a windowed moving average.

    Parameters
    ----------
    spikes: numpy.ndarray
        The spike times (Brian2 format, in _seconds_)
    duration: int
        The duration of the recording (in seconds)
    window_size: float
        Width of the moving average window (in seconds)
    overlap: float
        Desired overlap between the windows (percentage, in [0., 1.))

    Returns
    -------
    t: numpy.ndarray
        Array of time values for the computed firing rate. These are the window centers.
    FR: numpy.ndarray
        Spikes per window (needs to be normalized)
    """

    # Calculate new sampling times
    win_step = window_size * round(1. - overlap, 4)
    fs_n = int(1/win_step)

    # First center is at the middle of the first window
    c0 = window_size/2
    cN = duration-c0

    # centers
    centers = np.arange(c0, cN+win_step, win_step)

    # Calculate windowed FR
    counts = []
    for center in centers:
        cl = center - c0
        ch = center + c0
        spike_cnt = np.count_nonzero(np.where((spikes >= cl) & (spikes < ch)))
        counts.append(spike_cnt)

    FR = (np.array(counts)/window_size)

    # return centers, spike counts, and adjusted sampling rates per window
    return centers, FR, fs_n