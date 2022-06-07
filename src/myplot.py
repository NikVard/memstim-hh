from brian2 import *

from mpl3d import glm               # https://github.com/rougier/matplotlib-3d
from mpl3d.camera import Camera

from model import settings
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
    """ Print CA1 rasters, rhythm, phase reset using Nord theme colors. """

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
