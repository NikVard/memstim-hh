from brian2 import *

from mpl3d import glm               # https://github.com/rougier/matplotlib-3d
from mpl3d.camera import Camera

# -----------
from scipy import signal as sig

from matplotlib import ticker
from matplotlib import font_manager as fm
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# -----------

from model import settings
from model import globals
from model.globals import *
from src import plot3d

def subplot(index):
    ax = plt.subplot(index, xlim=[-1,1], ylim=[-1,1], aspect=1)
    ax.set_axisbelow(True)
    ax.set_xticks(np.linspace(-1,1,11,endpoint=True)[1:-1])
    ax.set_yticks(np.linspace(-1,1,11,endpoint=True)[1:-1])
    ax.grid(True)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.label.set_visible(False)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.label.set_visible(False)
    return ax


def plot_watermark(fig, script_filename, config_filename, branch, hash):
    """ Add simulation infomation on the figure """

    plt.text(.995, .99, '{0}\n {1} ({2}, {3})\n using "{4}"'.format(
        time.ctime(), script_filename, branch, hash, config_filename),
             transform=fig.transFigure, ha="right", va="top", clip_on=False,
             color = "black", family="Roboto Mono", weight="400", size="xx-small")


def generate_fig_name(initials=''):
    """ Generate the figure's name for truncating code """
    # initialize the figure name string
    fig_name = initials

    # phase offset
    if settings.offset:
        fig_name += 'offset_{offset:.2f}_'.format(offset=settings.offset)

    # stimulation
    if settings.I_stim[0]:
        fig_name += 'stim_on_'
        if len(settings.I_stim)==1:
            fig_name += 'mono_{stim:.2f}_nA_'.format(stim=settings.I_stim[0])
        else:
            fig_name += 'biphasic_{stimA:.2f}_nA_{stimB:.2f}_nA_'.format(stimA=settings.I_stim[0], stimB=settings.I_stim[1])

        # stimulation frequency
        if settings.stim_freq:
            fig_name += 'fstim_{fstim:d}_Hz_'.format(fstim=int(settings.stim_freq))

        if settings.nr_of_trains:
            fig_name += 'trains_{train_nr:d}_'.format(train_nr=int(settings.nr_of_trains))

        if settings.nr_of_pulses:
            fig_name += 'pulses_{pulse_nr:d}_'.format(pulse_nr=int(settings.nr_of_pulses))

        if settings.stim_onset:
            fig_name += 'ton_{stim_on:.1f}_ms'.format(stim_on=settings.stim_onset*1000)
    else:
        fig_name += 'stim_off'

    return fig_name+'.png'


def plot_raster_all(spike_mon_E_all, spike_mon_I_all):
    """ Plots the activity of all the excitatory and inhibitory populations in the network in a raster plot.
    Returns a figure handler """

    # raster plot of all regions
    fig, axs = subplots(len(areas),2) # from globals.py
    fig.set_figheight(16)
    fig.set_figwidth(12)

    for ii in range(4):
        axs[ii][0].plot(spike_mon_E_all[ii][0].t/msecond, spike_mon_E_all[ii][0].i, '.g', markersize=.5, alpha=0.5)
        axs[ii][0].set_title('raster '+areas[ii]+' exc')
        axs[ii][0].set_xlim(0, settings.duration/msecond)
        axs[ii][0].set_ylim(0, settings.N_all[ii][0])
        axs[ii][0].set_xlabel('Time (ms)')
        axs[ii][0].set_ylabel('Neuron index')

        axs[ii][1].set_title('raster '+areas[ii]+' inh')
        axs[ii][1].plot(spike_mon_I_all[ii][0].t/msecond, spike_mon_I_all[ii][0].i, '.r', markersize=.5, alpha=0.5)
        axs[ii][1].set_xlim(0, settings.duration/msecond)
        axs[ii][1].set_ylim(0, settings.N_all[ii][1])
        axs[ii][1].set_xlabel('Time (ms)')
        axs[ii][1].set_ylabel('Neuron index')

    # Generate the figure's name
    fig_name = generate_fig_name('Raster_')

    return fig, axs, fig_name


def plot_raster_grid_all(spike_mon_E_all, spike_mon_I_all, platesize=100, winsize = 10*ms, interpolation=None):
    """ Plots the activity of all the excitatory and inhibitory populations in the network in a binned raster plot.
    Returns a figure handler """

    # Plot in a grid, raster of all regions
    fig, axs = plt.subplots(nrows=4, ncols=2)
    fig.set_figheight(12)
    fig.set_figwidth(10)

    # all the monitors in one array, easy access
    popmons = [spike_mon_E_all, spike_mon_I_all]

    # Iterate over areas
    for area_idx in range(4):

        # Iterate over populations
        for pop_idx in range(2):

            t = popmons[pop_idx][area_idx][0].t/msecond
            i = popmons[pop_idx][area_idx][0].i

            # Parameters
            N = settings.N_all[area_idx][pop_idx]
            duration = settings.duration
            dt = defaultclock.dt

            xedges = np.arange(0, duration/dt, winsize/ms)
            yedges = np.arange(0, N, platesize)
            H, xedges, yedges = np.histogram2d(t*ms/dt, i, bins=(xedges,yedges))
            I = axs[area_idx,pop_idx].imshow(H.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='viridis', interpolation=interpolation, vmin=0, vmax=100, origin='lower', aspect='auto')

            axs[area_idx,pop_idx].set_xticklabels([])
            axs[area_idx,pop_idx].set_yticklabels([])

    # Fix labels to appear in ms


    # Colorbar
    cbar_ax = fig.add_axes([0.09, 0.05, 0.84, 0.01])
    fig.colorbar(I, cax=cbar_ax, orientation='horizontal')

    tight_layout()

    # Generate the figure's name
    fig_name = generate_fig_name('Raster_')

    return fig, axs, fig_name


def plot_kuramoto(order_param_mon):
    """ Plots the average phase and the phase coherence of the ensemble of Kuramoto oscillators. Also plots the generated theta rhythm from the MS. """

    # Kuramoto oscillators plots
    fig, axs = subplots(3,1) # [0] output rhythm, [1] avg. phase, [2] phase coherence
    fig.set_figheight(12)
    fig.set_figwidth(16)
    fig.suptitle("Kuramoto Oscillators")

    #axs[0].plot(kuramoto_mon.t/second, mean(sin(kuramoto_mon.Theta), axis=0), label='Mean theta')
    #axs[0].plot(kuramoto_mon.t/second, imag(r), '--', label='sin(Im(r))')
    axs[0].plot(order_param_mon.t/second, order_param_mon.rhythm_rect[0], '-', label='Generated theta rhythm')
    axs[1].plot(order_param_mon.t/second, order_param_mon.phase[0], '-', label='Avg. phase')
    axs[2].plot(order_param_mon.t/second, order_param_mon.coherence[0], '-', label='Coherence (|r|)')

    axs[0].set_ylabel("Ensemble Theta Rhythm")
    axs[1].set_ylabel("Average Phase")
    axs[2].set_ylabel("Phase Coherence")
    axs[2].set_xlabel("Time [s]")
    #axs[0].set_ylim([-1,1])
    axs[1].set_ylim([-pi,pi])
    axs[2].set_ylim([0,1])

    if settings.I_stim[0]:
        axs[0].axvline(x=settings.stim_onset, ymin=-1, ymax=1, c="red", linewidth=2, zorder=0, clip_on=False)
        axs[1].axvline(x=settings.stim_onset, ymin=-1, ymax=1, c="red", linewidth=2, zorder=0, clip_on=False)
        axs[2].axvline(x=settings.stim_onset, ymin=-1, ymax=1, c="red", linewidth=2, zorder=0, clip_on=True)

    # make things pretty
    axs[0].legend()
    axs[0].grid()
    axs[1].legend()
    axs[1].grid()
    axs[2].legend()
    axs[2].grid()

    # Generate the figure's name
    fig_name = generate_fig_name('Kuramoto_rhythms_')

    return fig, axs, fig_name


def plot_network_output(spike_mon_E, spike_mon_I, rate_mon, order_param_mon, tv, stim_inp):
    """ Plots the network output (population CA1 E/I) activity along with the Kuramoto oscillators and the stimulation input. """

    fig, axs = subplots(nrows=5, ncols=1)
    fig.set_figheight(12)
    fig.set_figwidth(16)

    axs[0].plot(spike_mon_E.t/second, spike_mon_E.i, '.g', markersize=.5,alpha=0.5)
    axs[0].set_title('CA1 Excitatory Neurons')
    axs[0].set_xlim(0, settings.duration/second)
    axs[0].set_ylim(0, settings.N_CA1[0])
    axs[0].set_ylabel('Neuron index')

    axs[1].plot(spike_mon_I.t/second, spike_mon_I.i, '.r', markersize=.5,alpha=0.5)
    axs[1].set_title('CA1 Inhibitory Neurons')
    axs[1].set_xlim(0, settings.duration/second)
    axs[1].set_ylim(0, settings.N_CA1[1])
    axs[1].set_ylabel('Neuron index')

    # axs[1].plot(tv*1000, stim_inp)
    # axs[1].set_title('Stimulation Input')
    # axs[1].set_xlim(0, settings.duration/second)
    # axs[1].set_ylabel('Current [nA]')
    # axs[1].grid()

    axs[2].plot(rate_mon.t/second, rate_mon.drive[0], label='LPF Output')
    axs[2].set_title('CA1 Population Firing Rates')
    axs[2].set_ylabel('Rate [Hz]')
    axs[2].set_xlim(0, settings.duration/second)
    axs[2].legend()
    axs[2].grid()

    axs[3].plot(order_param_mon.t/second, order_param_mon.coherence[0], '-', label='Order Param')
    axs[3].set_title('Kuramoto Oscillators Order Parameter')
    axs[3].set_xlim(0, settings.duration/second)
    axs[3].set_ylabel('r')
    axs[3].legend()
    axs[3].grid()

    axs[4].plot(order_param_mon.t/second, order_param_mon.rhythm_rect[0]/nA, '-', label='Theta Rhythm (Ensemble)')
    axs[4].set_title('Generated Theta Rhythm')
    axs[4].set_xlim(0, settings.duration/second)
    axs[4].set_ylabel('Theta rhythm (corr)')
    axs[4].set_xlabel('Time (ms)')
    axs[4].legend()
    axs[4].grid()

    if settings.I_stim[0]:
        axs[0].axvline(x=settings.stim_onset, ymin=-1, ymax=1, c="red", linewidth=2, zorder=0, clip_on=False)
        axs[1].axvline(x=settings.stim_onset, ymin=-1, ymax=1, c="red", linewidth=2, zorder=0, clip_on=False)
        axs[2].axvline(x=settings.stim_onset, ymin=-1, ymax=1, c="red", linewidth=2, zorder=0, clip_on=False)
        axs[3].axvline(x=settings.stim_onset, ymin=-1, ymax=1, c="red", linewidth=2, zorder=0, clip_on=False)
        axs[4].axvline(x=settings.stim_onset, ymin=-1, ymax=1, c="red", linewidth=2, zorder=0, clip_on=True)

    # Generate the figure's name
    fig_name = generate_fig_name('Kuramoto_extra_')

    return fig, axs, fig_name


def plot_structure_3D(neuron_groups):
    """ Plot neuron positions in 3D. """

    # make a figure
    fig = figure(figsize=(8,8))

    # add an axis (main plot)
    #ax = subplot(221)
    ax = fig.add_axes([0,0,1,1], xlim=[-1,+1], ylim=[-1,+1], aspect=1)
    ax.axis("off")

    # initialize the numpy arrays
    V = empty((1,3))
    C = empty((1,3))
    S = empty((1,1))

    for ng in neuron_groups:
        gname = ng.name.lower()
        if (gname.find("pycan") >= 0):
            color = (0,0,1)
        elif (gname.find("py") >= 0):
            color = (0,1,0)
        else:
            color = (1,0,0)

        nsize = len(ng.x_soma)
        psize = int(.1*nsize)
        #psize = 100
        idxs = permutation(nsize)[:psize]
        V_tmp = zeros((psize,3))
        V_tmp[:,0] = ng.x_soma[idxs]
        V_tmp[:,1] = ng.y_soma[idxs]
        V_tmp[:,2] = ng.z_soma[idxs]

        C_tmp = ones((len(V_tmp),3))
        C_tmp[:] = color

        S_tmp = ones((len(V_tmp),1))


        V = concatenate((V,V_tmp), axis=0)
        C = concatenate((C,C_tmp), axis=0)
        S = concatenate((S,S_tmp), axis=0)

    #S = 100+S*300

    FC = ones((len(V),4)) # face colors
    EC = ones((len(V),4)) # edge colors
    FC[:,:3] = C
    EC[:] = 0,0,0,.75

    # initialize the camera
    camera = Camera("ortho", 10, 0, scale=100)

    # scatter and connect to camera
    scatter = plot3d.Scatter(ax, camera.transform, V,
                            facecolors=FC, edgecolors=EC, sizes=squeeze(S))
    camera.connect(ax, scatter.update)

    show()


def plot_network_output2(spike_mon_E, spike_mon_I, rate_mon, order_param_mon, tv, stim_inp):
    """ Plot CA1 rasters, rhythm, phase reset using Nord theme colors. """

    # color selection
    c_inh = '#bf616a'
    c_exc = '#5e81ac'
    c_inh_RGB = np.array([191, 97, 106])/255
    c_exc_RGB = np.array([94, 129, 172])/255

    # create the figure
    fig, axs = subplots(nrows=6, ncols=1)
    fig.set_figheight(18)
    fig.set_figwidth(16)

    # axes aliases
    ax_rhythm = axs[0]
    ax_CA1_E_raster = axs[1]
    ax_CA1_I_raster = axs[2]
    ax_CA1_E_FR = axs[3]
    ax_order_param = axs[4]
    ax_phase = axs[5]

    # axes parameterizations
    # spines
    # rhythm
    ax_rhythm.spines['top'].set_visible(False)
    # ax_rhythm.spines['bottom'].set_visible(False)
    # ax_rhythm.spines['left'].set_visible(False)
    ax_rhythm.spines['right'].set_visible(False)
    ax_rhythm.xaxis.set_ticklabels([])

    # CA1
    # E / I
    ax_CA1_E_raster.spines['top'].set_visible(False)
    # ax_CA1_E_raster.spines['bottom'].set_visible(False)
    # ax_CA1_E_raster.spines['left'].set_visible(False)
    ax_CA1_E_raster.spines['right'].set_visible(False)
    ax_CA1_E_raster.xaxis.set_ticklabels([])

    ax_CA1_I_raster.spines['top'].set_visible(False)
    # ax_CA1_I_raster.spines['bottom'].set_visible(False)
    # ax_CA1_I_raster.spines['left'].set_visible(False)
    ax_CA1_I_raster.spines['right'].set_visible(False)
    ax_CA1_I_raster.xaxis.set_ticklabels([])

    # FR
    ax_CA1_E_FR.spines['top'].set_visible(False)
    # ax_CA1_E_FR.spines['bottom'].set_visible(False)
    # ax_CA1_E_FR.spines['left'].set_visible(False)
    ax_CA1_E_FR.spines['right'].set_visible(False)
    ax_CA1_E_FR.xaxis.set_ticklabels([])

    # Phase and order parameter
    ax_phase.spines['top'].set_visible(False)
    # ax_phase.spines['bottom'].set_visible(False)
    # ax_phase.spines['left'].set_visible(False)
    ax_phase.spines['right'].set_visible(False)
    ax_CA1_E_FR.xaxis.set_ticklabels([])

    ax_order_param.spines['top'].set_visible(False)
    # ax_order_param.spines['bottom'].set_visible(False)
    ax_order_param.spines['left'].set_visible(False)
    ax_order_param.spines['right'].set_visible(False)

    ax_phase.set_xlabel('Time [s]')


    # Data plotting
    ax_rhythm.plot(order_param_mon.t/second, order_param_mon.rhythm_rect[0]/nA, '-', label='Theta Rhythm (Ensemble)')
    ax_rhythm.set_title('Generated Theta Rhythm')
    ax_rhythm.set_xlim(0, settings.duration/second)
    # ax_rhythm.set_ylabel('Theta rhythm (corr)')
    # ax_rhythm.legend()
    ax_rhythm.grid()

    # ax_CA1_E_raster.plot(spike_mon_E.t/second, spike_mon_E.i, '.g', markersize=.5,alpha=0.5)
    ax_CA1_E_raster.scatter(spike_mon_E.t/second, spike_mon_E.i, s=1, marker='o', c=c_exc, edgecolors=None, alpha=.5, zorder=1, rasterized=False)
    ax_CA1_E_raster.set_title('CA1 Excitatory Neurons')
    ax_CA1_E_raster.set_xlim(0, settings.duration/second)
    ax_CA1_E_raster.set_ylim(0, settings.N_CA1[0])
    ax_CA1_E_raster.set_ylabel('Neuron index')

    # ax_CA1_I_raster.plot(spike_mon_I.t/second, spike_mon_I.i, '.r', markersize=.5,alpha=0.5)
    ax_CA1_I_raster.scatter(spike_mon_I.t/second, spike_mon_I.i, s=1, marker='o', c=c_inh, edgecolors=None, alpha=.5, zorder=1, rasterized=False)
    ax_CA1_I_raster.set_title('CA1 Inhibitory Neurons')
    ax_CA1_I_raster.set_xlim(0, settings.duration/second)
    ax_CA1_I_raster.set_ylim(0, settings.N_CA1[1])
    ax_CA1_I_raster.set_ylabel('Neuron index')

    ax_CA1_E_FR.plot(rate_mon.t/second, rate_mon.drive[0], label='LPF Output')
    ax_CA1_E_FR.set_title('CA1 Population Firing Rates')
    ax_CA1_E_FR.set_ylabel('Rate [Hz]')
    ax_CA1_E_FR.set_xlim(0, settings.duration/second)
    ax_CA1_E_FR.grid()

    ax_order_param.plot(order_param_mon.t/second, order_param_mon.coherence[0], '-', label='Order Param')
    ax_order_param.set_title('Synchronization level')
    ax_order_param.set_xlim(0, settings.duration/second)
    # ax_rhythm.set_ylabel('Theta rhythm (corr)')
    # ax_rhythm.legend()
    ax_order_param.grid()


    ax_phase.plot(order_param_mon.t/second, order_param_mon.phase[0], '-', label='Ensemble phase')
    ax_phase.set_title('Oscillators\' phase')
    ax_phase.set_xlim(0, settings.duration/second)
    ax_phase.grid()

    # stim pulse
    if settings.I_stim[0]:
        axs[0].axvline(x=settings.stim_onset, ymin=-1, ymax=1, c="red", linewidth=2, zorder=0, clip_on=False)
        axs[1].axvline(x=settings.stim_onset, ymin=-1, ymax=1, c="red", linewidth=2, zorder=0, clip_on=False)
        axs[2].axvline(x=settings.stim_onset, ymin=-1, ymax=1, c="red", linewidth=2, zorder=0, clip_on=False)
        axs[3].axvline(x=settings.stim_onset, ymin=-1, ymax=1, c="red", linewidth=2, zorder=0, clip_on=False)
        axs[4].axvline(x=settings.stim_onset, ymin=-1, ymax=1, c="red", linewidth=2, zorder=0, clip_on=True)


    # Generate the figure's name
    fig_name = generate_fig_name('Kuramoto_extra_new_')

    return fig, axs, fig_name


def plot_fig2(spike_mon_E_all, spike_mon_I_all, rate_mon, order_param_mon, tv, stim_inp):
    """ Plot all rasters, osc. rhythm, phase reset using Nord theme colors.
    This goes in the Fig2 section of the paper! """

    # Helper functions
    def my_FR(spikes: np.ndarray,
                duration: int,
                window_size: float,
                overlap: float) -> (np.ndarray, np.ndarray):
        """
        Compute the firing rate using a windowed moving average.

        Parameters
        ----------
        spikes: numpy.ndarray
            The spike times (Brian2 format, in ms)
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
        # fs_n = int(1/win_step)

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

        # return centers and spike counts per window
        return centers, np.array(counts)


    def my_specgram(signal: np.ndarray,
                       fs: int,
                       window_width: int,
                       window_overlap: int) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Computes the power spectrum of the specified signal.

        A periodic Hann window with the specified width and overlap is used.

        Parameters
        ----------
        signal: numpy.ndarray
            The input signal
        fs: int
            Sampling frequency of the input signal
        window_width: int
            Width of the Hann windows in samples
        window_overlap: int
            Overlap between Hann windows in samples

        Returns
        -------
        f: numpy.ndarray
            Array of frequency values for the first axis of the returned spectrogram
        t: numpy.ndarray
            Array of time values for the second axis of the returned spectrogram
        sxx: numpy.ndarray
            Power spectrogram of the input signal with axes [frequency, time]
        """
        f, t, Sxx = sig.spectrogram(x=signal,
                                    # nfft=2048,
                                    detrend=False,
                                    fs=fs,
                                    window=sig.windows.hann(M=window_width, sym=False),
                                    nperseg=window_width,
                                    noverlap=window_overlap,
                                    return_onesided=True,
                                    # scaling='spectrum',
                                    mode='magnitude')

        return f, t, (1.0 / window_width) * (Sxx ** 2)


    def add_sizebar(ax, xlocs, ylocs, bcolor, text):
        """ Add a sizebar to the provided axis """
        ax.plot(xlocs, ylocs, ls='-', c=bcolor, linewidth=1., rasterized=True, clip_on=False)
        ax.text(x=xlocs[0]+10*ms, y=ylocs[0], s=text, va='center', ha='left', clip_on=False)


    # Fonts
    fontprops = fm.FontProperties(size=12, family='monospace')

    # Color selection
    c_inh = '#bf616a'
    c_exc = '#5e81ac'
    c_inh_RGB = np.array([191, 97, 106])/255
    c_exc_RGB = np.array([94, 129, 172])/255

    N = 256
    cvals_inh = np.ones((N,4))
    cvals_exc = np.ones((N,4))

    # inhibitory cmap
    reds = np.linspace(1., c_inh_RGB[0], N)
    greens = np.linspace(1., c_inh_RGB[1], N)
    blues = np.linspace(1., c_inh_RGB[2], N)
    cvals_inh[:,0] = reds[...]
    cvals_inh[:,1] = greens[...]
    cvals_inh[:,2] = blues[...]
    newcmp_inh = ListedColormap(cvals_inh)

    # excitatory cmap
    reds = np.linspace(1., c_exc_RGB[0], N)
    greens = np.linspace(1., c_exc_RGB[1], N)
    blues = np.linspace(1., c_exc_RGB[2], N)
    cvals_exc[:,0] = reds[...]
    cvals_exc[:,1] = greens[...]
    cvals_exc[:,2] = blues[...]
    newcmp_exc = ListedColormap(cvals_exc)

    # Timing
    duration = settings.duration
    dt = defaultclock.dt
    fs = int(1/dt)

    t_stim = settings.stim_onset
    t_lims = [1250*ms, 2250*ms] # ms

    # FR parameters
    winsize_FR = globals.filter_params['tauFR']
    overlap_FR = 0.9
    winstep_FR = winsize_FR*round(1-overlap_FR,4)
    fs_FR = int(1/winstep_FR)
    binnum = int(duration/winsize_FR)

    interp = 'nearest'

    # Raster downsampling
    N_scaling = 100
    N_gap = 1

    # Firing rates plotting gap
    rates_gap = 125*Hz

    # Power spectra parameters
    fs2 = int(1/winstep_FR)
    window_size = 100*ms
    window_width = int(fs2*window_size) # samples
    overlap = 0.99 # percentage
    noverlap = int(overlap*window_width)

    # Set theta rhythm limits
    # xlims_rhythm = [t for t in t_lims]
    xlims_rhythm = [0*ms, duration+50*ms]
    ylims_rhythm = [-0.1, 1.1]

    # Set raster limits
    # xlims_raster = [t for t in t_lims]
    xlims_raster = [0*ms, duration+50*ms]
    ylims_raster = [0, 2*N_scaling]

    # Set firing rate limits
    # xlims_rates = [t for t in t_lims]
    xlims_rates = [0*ms, duration+50*ms]
    ylims_rates = [-1, 200]

    # Set spectrogram limits
    # xlims_freq = [t for t in t_lims]
    xlims_freq = [0*ms, duration+50*ms]
    ylims_freq = [0, 120] # Hz
    zlims_freq = [0.1, 10] # cmap limits [vmin vmax]

    # Make the figure
    fig = plt.figure(figsize=(10,14))

    # Use gridspecs
    G_outer = GridSpec(4, 2, left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.2, height_ratios=(0.15, 0.4, 0.2, 0.25), width_ratios=(0.99,0.01))
    G_rhythm = GridSpecFromSubplotSpec(1, 1, hspace=0., subplot_spec=G_outer[0,0])
    G_rasters = GridSpecFromSubplotSpec(4, 1, hspace=0.4, subplot_spec=G_outer[1,0])
    G_rates = GridSpecFromSubplotSpec(1, 1, hspace=0.6, subplot_spec=G_outer[2,0])
    G_specg = GridSpecFromSubplotSpec(2, 1, hspace=0.1, subplot_spec=G_outer[3,0])
    G_specg_cbars = GridSpecFromSubplotSpec(2, 1, hspace=0.1, subplot_spec=G_outer[3,1])

    G_outer.tight_layout(fig)


    # Organize axes
    #------------------------
    axs = []


    # Rasters
    # ------------------------
    ax0 = fig.add_subplot(G_rasters[0]) # EC E/I rasters
    ax1 = fig.add_subplot(G_rasters[1], sharex=ax0, sharey=ax0) # DG E/I rasters
    ax2 = fig.add_subplot(G_rasters[2], sharex=ax0, sharey=ax0) # CA3 E/I rasters
    ax3 = fig.add_subplot(G_rasters[3], sharex=ax0, sharey=ax0) # CA1 E/I rasters
    axs.append([ax0, ax1, ax2, ax3])

    # Set label as area name
    ax0.set_ylabel("EC", rotation=0, labelpad=25.)
    ax1.set_ylabel("DG", rotation=0, labelpad=25.)
    ax2.set_ylabel("CA3", rotation=0, labelpad=25.)
    ax3.set_ylabel("CA1", rotation=0, labelpad=25.)

    # Set limits
    ax0.set_xlim(xlims_raster)
    ax0.set_ylim(ylims_raster)

    # Set the ticks
    ax_raster_majors = np.arange(0., (duration+50*ms)/second, .5) #[0.5, 1.0, 1.5...]
    ax3.xaxis.set_major_locator(ticker.FixedLocator(ax_raster_majors))
    ax3.xaxis.set_minor_locator(ticker.NullLocator())

    # Hide x-y axes
    ax0.xaxis.set_visible(False)
    # ax0.yaxis.set_visible(False)
    ax1.xaxis.set_visible(False)
    # ax1.yaxis.set_visible(False)
    ax2.xaxis.set_visible(False)
    # ax2.yaxis.set_visible(False)
    # ax3.xaxis.set_visible(False)
    # ax3.yaxis.set_visible(False)

    # Hide some spines
    ax0.spines['top'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    # ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Fix the ytick locators
    ax0.yaxis.set_major_locator(ticker.NullLocator())
    ax0.yaxis.set_minor_locator(ticker.NullLocator())
    ax1.yaxis.set_major_locator(ticker.NullLocator())
    ax1.yaxis.set_minor_locator(ticker.NullLocator())
    ax2.yaxis.set_major_locator(ticker.NullLocator())
    ax2.yaxis.set_minor_locator(ticker.NullLocator())
    ax3.yaxis.set_major_locator(ticker.NullLocator())
    ax3.yaxis.set_minor_locator(ticker.NullLocator())

    # Remove tick labels
    # ax0.xaxis.set_ticklabels([])
    ax0.yaxis.set_ticklabels([])
    # ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    # ax2.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])
    # ax3.xaxis.set_ticklabels([])
    ax3.yaxis.set_ticklabels([])


    # Firing Rates
    # ------------------------
    # ax_rate_inh = fig.add_subplot(G_rates[0], sharex=ax0)
    # ax_rate_exc = fig.add_subplot(G_rates[1], sharex=ax0, sharey=ax_rate_inh)
    # axs.append([ax_rate_exc, ax_rate_inh])
    ax_rates = fig.add_subplot(G_rates[0], sharex=ax0)
    axs.append([ax_rates])

    # Set the title
    # ax_rate_exc.set_title('CA1 Firing Rate')

    # Set the x-y limits
    # ax_rate_exc.set_ylim(ylims_rates)
    ax_rates.set_ylim([0, 300])

    # Set the ticks
    ax_rate_majors = np.arange(0., (duration+50*ms)/second, .5) #[0.5, 1.0, 1.5...]
    # ax_rate_exc.xaxis.set_major_locator(ticker.FixedLocator(ax_rate_majors))
    # ax_rate_exc.xaxis.set_minor_locator(ticker.NullLocator())
    ax_rates.xaxis.set_major_locator(ticker.FixedLocator(ax_rate_majors))
    ax_rates.xaxis.set_minor_locator(ticker.NullLocator())

    # Hide x-y axes
    # # ax_rate_exc.xaxis.set_visible(False)
    # ax_rate_exc.yaxis.set_visible(False)
    # ax_rate_inh.xaxis.set_visible(False)
    # ax_rate_inh.yaxis.set_visible(False)
    ax_rates.yaxis.set_visible(False)

    # Hide some spines
    # ax_rate_exc.spines['top'].set_visible(False)
    # # ax_rate_exc.spines['bottom'].set_visible(False)
    # ax_rate_exc.spines['left'].set_visible(False)
    # ax_rate_exc.spines['right'].set_visible(False)
    # ax_rate_inh.spines['top'].set_visible(False)
    # ax_rate_inh.spines['bottom'].set_visible(False)
    # ax_rate_inh.spines['left'].set_visible(False)
    # ax_rate_inh.spines['right'].set_visible(False)
    ax_rates.spines['top'].set_visible(False)
    # ax_rates.spines['bottom'].set_visible(False)
    ax_rates.spines['left'].set_visible(False)
    ax_rates.spines['right'].set_visible(False)


    # Spectrograms
    # ------------------------
    ax_specg_inh = fig.add_subplot(G_specg[0])
    ax_specg_exc = fig.add_subplot(G_specg[1])
    axs.append([ax_specg_exc, ax_specg_inh])

    # Set the title
    # ax_specg_exc.set_title('Spectrogram')

    # Set the x-y limits
    ax_specg_inh.set_xlim(xlims_freq)
    ax_specg_inh.set_ylim(ylims_freq)
    ax_specg_exc.set_xlim(xlims_freq)
    ax_specg_exc.set_ylim(ylims_freq)

    # Set the ticks
    specg_freq_majors = [10, 40, 100]
    ax_specg_inh.yaxis.set_major_locator(ticker.FixedLocator(specg_freq_majors))
    ax_specg_exc.yaxis.set_major_locator(ticker.FixedLocator(specg_freq_majors))

    specg_freq_majors = np.arange(0., (duration+50*ms)/second, .5) #[0.5, 0.6, 0.7, 1., 1.25, 1.5]
    ax_specg_exc.xaxis.set_major_locator(ticker.FixedLocator(specg_freq_majors))
    ax_specg_exc.xaxis.set_minor_locator(ticker.NullLocator())

    # Hide x axis for inh
    ax_specg_inh.xaxis.set_visible(False)

    # Set xlabel
    ax_specg_exc.set_xlabel('Time [s]')

    # Hide some spines
    ax_specg_exc.spines['top'].set_visible(False)
    # ax_specg_exc.spines['bottom'].set_visible(False)
    # ax_specg_exc.spines['left'].set_visible(False)
    ax_specg_exc.spines['right'].set_visible(False)
    ax_specg_inh.spines['top'].set_visible(False)
    # ax_specg_inh.spines['bottom'].set_visible(False)
    # ax_specg_inh.spines['left'].set_visible(False)
    ax_specg_inh.spines['right'].set_visible(False)


    # Rhythm
    # ------------------------
    ax_rhythm = fig.add_subplot(G_rhythm[0])
    axs.append(ax_rhythm)

    # Set the title
    # ax_rhythm.set_title('Theta Input')

    # Set the x-y limits
    ax_rhythm.set_xlim(xlims_rhythm)
    ax_rhythm.set_ylim(ylims_rhythm)

    # Set x-y ticks
    ax_rhythm_majors = np.arange(0., (duration+50*ms)/second, .5)
    ax_rhythm.xaxis.set_major_locator(ticker.FixedLocator(ax_rhythm_majors))
    ax_rhythm.xaxis.set_minor_locator(ticker.NullLocator())

    # Hide x-y axes
    # ax_rhythm.xaxis.set_visible(False)
    ax_rhythm.yaxis.set_visible(False)

    # Hide some spines
    ax_rhythm.spines['top'].set_visible(False)
    # ax_rhythm.spines['bottom'].set_visible(False)
    ax_rhythm.spines['left'].set_visible(False)
    ax_rhythm.spines['right'].set_visible(False)


    # Other stuff
    # ------------------------
    # legend patches
    # print('[>] Legend')
    # leg_lines = [Line2D([0], [0], color=c_inh, lw=4),
                # Line2D([0], [0], color=c_exc, lw=4)]
    # labels = ['Inhibitory', 'Excitatory']
    # fig.legend(leg_lines, labels, bbox_transform=fig.transFigure, bbox_to_anchor=(0.15,0.5), loc='upper right', ncol=1)
    # fig.legend(leg_lines, labels, loc='upper right', ncol=1)


    # Actually plotting stuff
    # ------------------------
    # all the monitors in one array, easy access
    popmons = [spike_mon_E_all, spike_mon_I_all]


    # ==================
    # Plot the rasters
    # ==================
    # Iterate over areas
    for area_idx in range(len(globals.areas)):

        t_exc = spike_mon_E_all[area_idx][0].t
        i_exc = spike_mon_E_all[area_idx][0].i
        t_inh = spike_mon_I_all[area_idx][0].t
        i_inh = spike_mon_I_all[area_idx][0].i

        # sort based on index number (lower to higher)
        idx_sort_exc = np.argsort(i_exc)
        idx_sort_inh = np.argsort(i_inh)
        i_exc = i_exc[idx_sort_exc]
        t_exc = t_exc[idx_sort_exc]
        i_inh = i_inh[idx_sort_inh]
        t_inh = t_inh[idx_sort_inh]

        # set number of neurons
        N_exc = settings.N_all[area_idx][0]
        N_inh = settings.N_all[area_idx][1]

        # select some neurons randomly, subscaling
        exc_mixed = np.arange(0, N_exc, int(N_exc/N_scaling))
        inh_mixed = np.arange(0, N_inh, int(N_inh/N_scaling))

        idx_exc = np.in1d(i_exc, exc_mixed)
        idx_inh = np.in1d(i_inh, inh_mixed)

        i_exc_sub = i_exc[idx_exc]
        t_exc_sub = t_exc[idx_exc]
        i_inh_sub = i_inh[idx_inh]
        t_inh_sub = t_inh[idx_inh]

        # assign new neuron count numbers
        cnt = 0
        for ii in exc_mixed:
            idx_tmp = np.where(i_exc_sub == ii)
            i_exc_sub[idx_tmp] = cnt
            cnt += 1

        # cnt = 0
        for jj in inh_mixed:
            idx_tmp = np.where(i_inh_sub == jj)
            i_inh_sub[idx_tmp] = cnt
            cnt += 1

        # select axis
        ax_curr = axs[0][area_idx]

        # inhibitory
        ax_curr.scatter(t_inh_sub, i_inh_sub, s=1, marker='o', c=c_inh, edgecolors=None, alpha=.25, zorder=1, rasterized=False)

        # excitatory
        ax_curr.scatter(t_exc_sub, i_exc_sub, s=1, marker='o', c=c_exc, edgecolors=None, alpha=.25, zorder=1, rasterized=False)

        # Calculate mean firing rates
        t_lims_adj = [2000*ms, 3000*ms]
        duration_adj = t_lims_adj[1] - t_lims_adj[0]

        # FR_inh_mean = (len(t_inh)/duration)/N_inh
        # FR_exc_mean = (len(t_exc)/duration)/N_exc
        FR_inh_mean = (sum((t_inh>=t_lims_adj[0]) & (t_inh<t_lims_adj[1]))/duration_adj)/N_inh
        FR_exc_mean = (sum((t_exc>=t_lims_adj[0]) & (t_exc<t_lims_adj[1]))/duration_adj)/N_exc

        # add it as a text
        ax_curr.text(x=duration+200*ms, y=150, s=r'$\mu_I$: {0:.1f}Hz'.format(FR_inh_mean), ha='center', color=c_inh, clip_on=False)
        ax_curr.text(x=duration+200*ms, y=50, s=r'$\mu_E$: {0:.1f}Hz'.format(FR_exc_mean), ha='center', color=c_exc, clip_on=False)


    # ==================
    # Plot the input rhythm
    # ==================
    rhythm = order_param_mon.rhythm_rect[0]/nA

    # Data plotting
    ax_rhythm.plot(np.arange(0.,duration,dt), rhythm/(np.max(rhythm)), ls='-', c='k', linewidth=1.2, rasterized=True, zorder=1)
    # ax_rhythm.set_ylabel('Theta rhythm (corr)')
    # ax_rhythm.legend()
    # ax_rhythm.grid()

    # vertical lines at x-points
    pks, _ = sig.find_peaks(rhythm, distance=int(80*ms*fs))
    fval = 1/(mean(pks[1:] - pks[0:-1])/fs) if len(pks)>1 else 1/(pks[0]/fs)

    # for peak in pks:
        # if (peak*dt >= t_lims[0]) & (peak*dt <= t_lims[1]):
            # "X" marks the spot
            # ax_rhythm.plot(peak*dt, rhythm[peak], 'C1x', markersize=12, zorder=10, clip_on=False)

            # trace the treasure
            # ax_rhythm.vlines(x=peak*dt, ymin=-15.85, ymax=rhythm[peak], color='black', ls='--', linewidth=0.5, zorder=11, clip_on=False)

    # stimulation line
    ax_rhythm.vlines(x=t_stim, ymin=-15.85, ymax=1., color='red', ls='-', linewidth=0.5, zorder=11, clip_on=False)

    # text frequency label
    ax_rhythm.text(x=duration+100*ms, y=1.1, s=r"$f_\theta={0:.2f}$Hz".format(fval), ha='left', color='k', clip_on=False)

    # add a sizebar for the y-axis
    add_sizebar(ax_rhythm, [duration+100*ms, duration+100*ms], [0, 0.5], 'black', '0.5nA')


    # ==================
    # Plot CA1 FRs
    # ==================
    tv_inh_FR, FR_inh = my_FR(spikes=t_inh, duration=duration, window_size=winsize_FR, overlap=overlap_FR)
    tv_exc_FR, FR_exc = my_FR(spikes=t_exc, duration=duration, window_size=winsize_FR, overlap=overlap_FR)

    FR_inh_norm = (FR_inh/winsize_FR)/N_inh
    FR_exc_norm = (FR_exc/winsize_FR)/N_exc

    # ax_rate_inh.plot(tv_inh_FR, FR_inh_norm, ls='-', linewidth=1., c=c_inh, label='inh', zorder=10, rasterized=True)
    # ax_rate_exc.plot(tv_exc_FR, FR_exc_norm, ls='-', linewidth=1., c=c_exc, label='exc', zorder=10, rasterized=True)
    ax_rates.plot(tv_inh_FR, FR_inh_norm+rates_gap, ls='-', linewidth=1.2, c=c_inh, label='inh', zorder=10, rasterized=True)
    ax_rates.plot(tv_exc_FR, FR_exc_norm, ls='-', linewidth=1.2, c=c_exc, label='exc', zorder=10, rasterized=True)

    # add labels
    # ax_rate_inh.set_title('Inhibitory', color=c_inh, loc='left')
    # ax_rate_exc.set_title('Excitatory', color=c_exc, loc='left')

    # ax_rate_inh.text(x=-10*ms, y=ylims_rates[1]//2, s='Inhibitory', ha='center', color=c_inh, clip_on=False)
    # ax_rate_exc.text(x=-10*ms, y=ylims_rates[1]//2, s='Excitatory', ha='center', color=c_exc, clip_on=False)
    ax_rates.text(x=-100*ms, y=ylims_rates[1]-50, s='Inhibitory', ha='center', color=c_inh, clip_on=False)
    ax_rates.text(x=-100*ms, y=ylims_rates[0]+50, s='Excitatory', ha='center', color=c_exc, clip_on=False)

    # add a sizebar for the y-axis
    # add_sizebar(ax_rate_exc, [duration+100*ms, duration+100*ms], [0, 50], 'black', '50Hz')
    add_sizebar(ax_rates, [duration+100*ms, duration+100*ms], [0, 50], 'black', '50Hz')

    # ==================
    # Plot the spectrograms
    # ==================
    fv_inh, tv_inh, pspec_inh = my_specgram(signal=FR_inh_norm,
                                fs=fs2,
                                window_width=window_width,
                                window_overlap=noverlap)

    fv_exc, tv_exc, pspec_exc = my_specgram(signal=FR_exc_norm,
                                fs=fs2,
                                window_width=window_width,
                                window_overlap=noverlap)

    # avoid division by zero in log transform
    pspec_inh[np.where(pspec_inh<1e-10)] = 1e-10
    pspec_exc[np.where(pspec_exc<1e-10)] = 1e-10

    # get log power
    # pspec_inh_dB = 10*np.log10(pspec_inh)
    # pspec_exc_dB = 10*np.log10(pspec_exc)

    # find min/max for plotting
    # maxval_dB = max(pspec_inh_dB.max(), pspec_exc_dB.max())
    # minval_dB = min(pspec_inh_dB.min(), pspec_exc_dB.min())

    # pcm_inh = ax_specg_inh.pcolormesh(tv_inh, fv_inh, pspec_inh_dB, vmin=minval_dB, vmax=maxval_dB, cmap=newcmp_inh, shading='gouraud')
    # pcm_exc = ax_specg_exc.pcolormesh(tv_exc, fv_exc, pspec_exc_dB, vmin=minval_dB, vmax=maxval_dB, cmap=newcmp_exc, shading='gouraud')

    im_inh = ax_specg_inh.pcolormesh(tv_inh, fv_inh, pspec_inh, cmap=newcmp_inh, shading='auto')
    im_exc = ax_specg_exc.pcolormesh(tv_exc, fv_exc, pspec_exc, cmap=newcmp_exc, shading='auto')
    # im_inh = ax_specg_inh.imshow(pspec_inh_dB, cmap=newcmp_inh, interpolation='nearest', vmin=minval_dB, vmax=maxval_dB, origin='lower', aspect='auto')
    # im_exc = ax_specg_exc.imshow(pspec_exc_dB, cmap=newcmp_exc, interpolation='nearest', vmin=minval_dB, vmax=maxval_dB, origin='lower', aspect='auto')

    # pcm_inh= ax_specg_inh.specgram(FR_exc_norm, NFFT=window_width, detrend='none', Fs=fs2, window=sig.windows.hann(M=window_width, sym=False), noverlap=noverlap, scale_by_freq=False, mode='magnitude', scale='dB', sides='onesided')

    # Colorbars
    # fig.colorbar(pcm_exc, ax=ax_specg_exc)
    # fig.colorbar(pcm_inh, ax=ax_specg_inh)
    # cbar_inh_ax = fig.add_subplot(G_specg_cbars[0])
    # cbar_exc_ax = fig.add_subplot(G_specg_cbars[1])
    # fig.colorbar(im_inh, cax=cbar_inh_ax, aspect=1)
    # fig.colorbar(im_exc, cax=cbar_exc_ax, aspect=1)

    # sizebars
    # ax_specg_exc.plot([550*ms, 600*ms, None, 550*ms, 550*ms], [-60, -60, None, 0, 0], ls='-', c='r', linewidth=1., rasterized=True, clip_on=False)
    # ax_specg_exc.text(x=575*ms, y=-100, s='50ms', ha='center', fontproperties=fontprops, clip_on=False)


    # Generate the figure's name
    fig_name = generate_fig_name('Fig2_')

    # fig.savefig('figures/fig2_A.svg', dpi=200, format='svg')
    # fig.savefig('figures/fig2_A.pdf', dpi=200, format='pdf')
    # fig.savefig('figures/fig2_A.png', dpi=200, format='png')

    return fig, axs, fig_name
