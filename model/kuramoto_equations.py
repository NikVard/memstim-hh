"""
--------------------------------------------------------------------------------
Date: 11/10/2021

@author: Nikolaos Vardalakis
--------------------------------------------------------------------------------

Implementation Notes
--------------------------------------------------------------------------------
    | 1: For more information, refer to: https://michaelsmclayton.github.io/travellingWaves.html
    | 2: For the details regarding the (summed) keyword and the synapses, refer to the question I posed on the Brian2 forum, here: https://brian.discourse.group/t/how-can-i-get-the-population-firing-rate-from-a-spiking-hh-network-during-simulation/496
    | 3: A demo of the Kuramoto Brian2 model is implemented in the jupyter notebook titled 'Stimberg_Oscillators.ipynb'
"""

# Kuramoto oscillators
kuramoto_eqs_stim = '''
    dTheta/dt = ((omega + (kN * PIF) - G_in*X*sin(Theta + offset)) * second**-1) : 1
    PIF = .5 * (sin(ThetaPreInput - Theta)) : 1

    ThetaPreInput : 1
    omega : 1 (constant)
    kN : 1 (shared)         # k/N ratio, affects sync.
    G_in : 1 (shared)       # input gain, affects the phase reset aggressiveness
    offset : 1 (shared)     # range [0, 2*pi], controls phase reset curve
    X : 1 (linked)          # this is linked to the firing rates
'''

# synapses
syn_kuramoto_eqs = '''
    ThetaPreInput_post = Theta_pre
'''

# Order parameter group calculation equations
pop_avg_eqs = '''
    coherence = sqrt(x**2 + y**2) : 1
    phase = arctan(y/x) + int(x<0 and y>0)*pi - int(x<0 and y<0)*pi: 1

    # Rhythms
    rhythm_default = coherence * cos(phase) : 1
    rhythm_positive = coherence * (cos(phase)+1)/2 : 1
    rhythm_abs = abs(rhythm_default) : amp
    rhythm_rect = rhythm_positive : amp
    rhythm_zero = 0*rhythm : amp   # for debugging

    # Output selection
    rhythm = G_out*rhythm_rect : amp

    x : 1
    y : 1
    G_out : amp                 # output rhythm gain
'''

syn_avg_eqs = '''
    x_post = cos(Theta_pre)/N_incoming : 1 (summed)
    y_post = sin(Theta_pre)/N_incoming : 1 (summed)
'''



''' Keep for later
# Define Gaussian function
def gaussian(distance, sig):
    return np.exp(-np.power(distance, 2.) / (2 * np.power(sig, 2.)))

# Get Euclidean distance
def euclideanDistance(postSynapticLocation, preSynapticLocation):
    return np.linalg.norm(postSynapticLocation - preSynapticLocation)

# Define function to get weight between two neurons
locations = [[x,y] for x in range(groupLength) for y in range(groupLength)]
@implementation('numpy', discard_units=True)
@check_units(i=1, j=1, sig=1, result=1)
def getDistance(i, j, sig=3):
    preSynapticLocation = np.array(locations[int(i)])
    postSynapticLocation = np.array(locations[int(j)])
    distance = euclideanDistance(postSynapticLocation, preSynapticLocation)
    return gaussian(distance, sig)
'''
