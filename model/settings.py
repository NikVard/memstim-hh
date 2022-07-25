from brian2.units import *
from brian2 import seed

""" JSON PARAMETERS HERE (+DEFAULTS) """
# Simulation
duration = 1*second # simulation duration
debugging = False # run in debugging mode (numpy vs cython, no optimization, reporting)

# population sizes per area | [E, I]
N_EC = [] # def: [10e3, 1e3]
N_DG = [] # def: [10e3, 0.1e3]
N_CA3 = [] # def: [1e3, 0.1e3]
N_CA1 = [] # def: [10e3, 1e3]
N_all = None

# population noise levels per area | [E, I]
sigma_EC = [] # def: [100.e-6, 1.e-6]
sigma_DG = [] # def: [100.e-6, 1.e-6]
sigma_CA3 = [] # def: [100.e-6, 1.e-6]
sigma_CA1 = [] # def: [100.e-6, 1.e-6]
sigma_all = None

# intra-area conn. probabilities per area | [[E-E, E-I], [I-E, I-I]]
p_EC_all = [[],[]] # def:[[0., 0.37], [0.54, 0.]]
p_DG_all = [[],[]] # def: [[0., 0.06], [0.14, 0.]]
p_CA3_all = [[],[]] # def: [[0.56, 0.75], [0.75, 0.]]
p_CA1_all = [[],[]] # def: [[0., 0.28], [0.3, 0.7]]
p_intra_all = None

# inter-area conn. probabilities per area
p_inter_all = None

# inter-area conn. probabilities
p_mono = None # def: 0.2 # monosynaptic pathway connectivity
p_tri = None # def: 0.45 # trisynaptic pathway connectivity

# Kuramoto settings
N_Kur = None
f0 = 4 # Hz
sigma = 0.5 # std of Gaussian for phase/ang.vel. initialization
kN_frac = 0. # synchronization parameter (k/N factor)
k_gain = 0. # phase reset gain
r_gain = 0.*nA # output sin rhythm gain (scaling, in nA)

# Stimulation settings - stimulation module is not brian2-dependent!
stim_target = "" # [EC | DG | CA1 | CA3]
stim_coordinates = (0., 0., 0.) # (x,y,z) [mm]
stim_sigma = 0. # [S/m]
I_stim = [1.] # [nA]
pulse_width = [.2e-3]
stim_freq = 5. # [Hz]
stim_duration = 1. # [sec]
stim_dt = .1e-3 # [sec]
stim_onset = 200e-3 # [sec]
nr_of_trains = 5
nr_of_pulses = 4
pulse_freq = 100. # [Hz]
stim_ipi = .1e-3 # [sec]

# Reproducibility settings
timestamp = None
git_branch = None
git_hash = None
git_short_hash = None

def init(data):
    """ This is used to set the global variables according to the JSON file parameters """

    # Neuronal population sizes > [E, I]
    global N_EC, N_DG, N_CA3, N_CA1, N_all
    N_EC = [data['areas']['EC']['E']['N'], data['areas']['EC']['I']['N']]
    N_DG = [data['areas']['DG']['E']['N'], data['areas']['DG']['I']['N']]
    N_CA3 = [data['areas']['CA3']['E']['N'], data['areas']['CA3']['I']['N']]
    N_CA1 = [data['areas']['CA1']['E']['N'], data['areas']['CA1']['I']['N']]
    N_all = [N_EC, N_DG, N_CA3, N_CA1]

    # Population noise
    global sigma_EC, sigma_DG, sigma_CA3, sigma_CA1, sigma_all
    sigma_EC = [data['areas']['EC']['E']['noise'], data['areas']['EC']['I']['noise']]
    sigma_DG = [data['areas']['DG']['E']['noise'], data['areas']['DG']['I']['noise']]
    sigma_CA3 = [data['areas']['CA3']['E']['noise'], data['areas']['CA3']['I']['noise']]
    sigma_CA1 = [data['areas']['CA1']['E']['noise'], data['areas']['CA1']['I']['noise']]
    sigma_all = [sigma_EC, sigma_DG, sigma_CA3, sigma_CA1]

    # Intra-conn. probabilities | [[E-E, E-I], [I-E, I-I]]
    global p_EC_all, p_DG_all, p_CA3_all, p_CA1_all, p_intra_all
    p_EC_all = data['connectivity']['intra']['EC']
    p_DG_all = data['connectivity']['intra']['DG']
    p_CA3_all = data['connectivity']['intra']['CA3']
    p_CA1_all = data['connectivity']['intra']['CA1']
    p_intra_all = [p_EC_all, p_DG_all, p_CA3_all, p_CA1_all]

    # Inter-conn. probabilities | p_mono / p_tri
    global p_inter_all
    custom_conn = False

    if 'inter_custom' in data['connectivity'].keys():
        if data['connectivity']['inter_custom']:
            custom_conn = True

    if custom_conn:
        # Custom connectivity
        print(bcolors.YELLOW + '[!]' + bcolors.ENDC + 'Custom inter-connectivity being used')
        p_custom = data['connectivity']['inter_custom']

        p_inter_all = []
        for k in p_custom.keys():
            v1 = p_custom[k]

            l0 = []
            for idx in range(4):
                v2 = [v1['E'][idx], v1['I'][idx]]
                l0.append(v2)

            p_inter_all.append(l0)
    else:
        # Default mono/tri connectivity
        print('[*] Default inter-connectivity for mono-/tri-synaptic pathways')

        p_mono = data['connectivity']['inter']['p_mono'] # monosynaptic pathway connectivity
        p_tri = data['connectivity']['inter']['p_tri'] # trisynaptic pathway connectivity

        p_inter_all = [[[[],[]],[[],[]]],[[[],[]],[[],[]]],[[[],[]],[[],[]]],[[[],[]],[[],[]]]]
        p_inter_all = [[[[0,0] for ii in range(2)] for jj in range(4)] for kk in range(4)]
        p_inter_all[0][1][0] = [p_tri for ii in range(2)] # EC_E to DG_E | DG_I
        p_inter_all[0][2][0] = [p_mono for ii in range(2)] # EC_E to CA3_E | CA3_I
        p_inter_all[0][3][0] = [p_mono for ii in range(2)] # EC_E to CA1_E | CA1_I
        p_inter_all[1][2][0] = [p_tri for ii in range(2)] # DG_E to CA3_E | CA3_I
        p_inter_all[2][3][0] = [p_tri for ii in range(2)] # CA3_E to CA1_E | CA1_I
        p_inter_all[3][0][0] = [p_tri for ii in range(2)] # CA1_E to EC_E | EC_I

    global duration, dt, debugging
    duration = data['simulation']['duration']*second
    dt = data['simulation']['dt']*second
    debugging = data['simulation']['debugging']

    global N_Kur, f0, sigma, kN_frac, k_gain, r_gain, offset
    N_Kur = data['Kuramoto']['N']
    f0 = data['Kuramoto']['f0']
    sigma = data['Kuramoto']['sigma']
    kN_frac = data['Kuramoto']['kN']
    k_gain = data['Kuramoto']['gain_reset']
    r_gain = data['Kuramoto']['gain_rhythm']*nA
    offset = data['Kuramoto']['offset']

    # Stimulation
    global stim_target, stim_coordinates, stim_rho, stim_duration, stim_dt, stim_onset, I_stim, pulse_width, stim_freq, pulse_freq, nr_of_trains, nr_of_pulses, stim_ipi
    stim_target = data['stimulation']['target']
    stim_coordinates = tuple(data['stimulation']['coordinates']) # immutable
    stim_sigma = data['stimulation']['sigma']
    stim_duration = data['stimulation']['duration']
    stim_dt = data['stimulation']['dt']
    stim_onset = data['stimulation']['onset']
    I_stim = data['stimulation']['I']
    pulse_width = data['stimulation']['pulse_width']
    pulse_freq = data['stimulation']['pulse_freq']
    stim_freq = data['stimulation']['stim_freq']
    nr_of_trains = data['stimulation']['nr_of_trains']
    nr_of_pulses = data['stimulation']['nr_of_pulses']
    stim_ipi = data['stimulation']['ipi']

    global timestamp, git_branch, git_hash, git_short_hash
    timestamp = data['timestamp']
    git_branch = data['git_branch']
    git_hash = data['git_hash']
    git_short_hash = data['git_hash']

    seed(data['seed_val'])
    seed(0)
