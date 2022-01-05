from brian2 import *

from model import settings
from model.globals import *

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

<<<<<<< Updated upstream
    if settings.stim_onset:
        fig_name += 'ton_{stim_on:d}_ms'.format(stim_on=int(settings.stim_onset*1000))
=======
        if settings.stim_onset:
            fig_name += 'ton_{stim_on:.1f}_ms'.format(stim_on=settings.stim_onset*1000)
    else:
        fig_name += 'stim_off'
>>>>>>> Stashed changes

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
