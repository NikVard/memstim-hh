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

# intra-area conn. probabilities per area | [[E-E, E-I], [I-E, I-I]]
p_EC_all = [[],[]] # def:[[0., 0.37], [0.54, 0.]]
p_DG_all = [[],[]] # def: [[0., 0.06], [0.14, 0.]]
p_CA3_all = [[],[]] # def: [[0.56, 0.75], [0.75, 0.]]
p_CA1_all = [[],[]] # def: [[0., 0.28], [0.3, 0.7]]
p_intra_all = None

# inter-area conn. probabilities
p_mono = None # def: 0.2 # monosynaptic pathway connectivity
p_tri = None # def: 0.45 # trisynaptic pathway connectivity

# Kuramoto settings
N_Kur = None
f0 = 4 # Hz
sigma = 0.5 # std of Gaussian for phase/ang.vel. initialization
kN_frac = 0 # synchronization parameter (k/N factor)

# Stimulation settings
I_stim = 0*nA
t_stim = 0*msecond

def init(data):
    """ This is used to set the global variables according to the JSON file parameters """

    # Neuronal population sizes > [E, I]
    global N_EC, N_DG, N_CA3, N_CA1, N_all
    N_EC = [data['areas']['EC']['E']['N'], data['areas']['EC']['I']['N']]
    N_DG = [data['areas']['DG']['E']['N'], data['areas']['DG']['I']['N']]
    N_CA3 = [data['areas']['CA3']['E']['N'], data['areas']['CA3']['I']['N']]
    N_CA1 = [data['areas']['CA1']['E']['N'], data['areas']['CA1']['I']['N']]
    N_all = [N_EC, N_DG, N_CA3, N_CA1]

    # Intra-conn. probabilities | [[E-E, E-I], [I-E, I-I]]
    global p_EC_all, p_DG_all, p_CA3_all, p_CA1_all, p_intra_all
    p_EC_all = data['connectivity']['intra']['EC']
    p_DG_all = data['connectivity']['intra']['DG']
    p_CA3_all = data['connectivity']['intra']['CA3']
    p_CA1_all = data['connectivity']['intra']['CA1']
    p_intra_all = [p_EC_all, p_DG_all, p_CA3_all, p_CA1_all]

    # Inter-conn. probabilities | p_mono / p_tri
    global p_mono, p_tri
    p_mono = data['connectivity']['inter']['p_mono'] # monosynaptic pathway connectivity
    p_tri = data['connectivity']['inter']['p_tri'] # trisynaptic pathway connectivity

    global duration, debugging
    duration = data['simulation']['duration']*ms
    debugging = data['simulation']['debugging']

    global N_Kur, f0, sigma, kN_frac, k_gain, offset
    N_Kur = data['Kuramoto']['N']
    f0 = data['Kuramoto']['f0']
    sigma = data['Kuramoto']['sigma']
    kN_frac = data['Kuramoto']['kN']
    k_gain = data['Kuramoto']['gain']
    offset = data['Kuramoto']['offset']

    global I_stim, t_stim, dt_stim
    I_stim = data['stimulation']['I_stim']*nA
    t_stim = data['stimulation']['t_stim']*ms
    dt_stim = data['stimulation']['dt_stim']*ms

    seed(data['seed_val'])
