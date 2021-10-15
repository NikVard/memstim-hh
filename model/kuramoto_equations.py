"""
--------------------------------------------------------------------------------
Date: 11/10/2021

@author: Nikolaos Vardalakis
--------------------------------------------------------------------------------

Implementation Notes
--------------------------------------------------------------------------------
    | 1: For more information, refer to: https://michaelsmclayton.github.io/travellingWaves.html
    | 2:
"""

kuramoto_eqs = '''
    dTheta/dt = ((omega + (kN * PIF)) * second**-1) : 1
    PIF = .5 * (sin(ThetaPreInput - Theta)) : 1
    Vm = sin(Theta)*mV : volt
    ThetaPreInput : 1
    omega : 1
    kN : 1
'''


kuramoto_eqs_stim = '''
    dTheta/dt = ((omega + (kN * PIF) - I_stim*X*sin(Theta)) * second**-1) : 1
    PIF = .5 * (sin(ThetaPreInput - Theta)) : 1
    Vm = sin(Theta)*mV : volt
    ThetaPreInput : 1
    omega : 1
    kN : 1
    I_stim : amp
    X = pulse_train(t) : amp**-1
'''

kuramoto_eqs_stim_test = '''
    dTheta/dt = ((omega + (kN * PIF) - I_stim*X*sin(Theta)) * second**-1) : 1
    PIF = .5 * (sin(ThetaPreInput - Theta)) : 1
    Vm = sin(Theta)*mV : volt
    ThetaPreInput : 1
    omega : 1
    kN : 1
    I_stim : 1
    X : 1 (linked)
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
