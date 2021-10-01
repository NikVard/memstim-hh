from brian2 import *
from model.globals import *
from model.HH_equations import *

def create_group(parameters, name):
    """ Universal function that creates a group of neurons according to input parameters """
    pop_size = parameters['N']

    if parameters['type'] == 'PyCAN':
        eqs = py_CAN_eqs

        G = NeuronGroup(N=pop_size,
            model=py_CAN_eqs,
            threshold='v>V_th',
            reset=reset_eqs,
            refractory=refractory_time,
            namespace={'size':cell_size_py, 'glu':1},
            method=integ_method,
            name=name)

    elif parameters['type'] == 'Py':
        eqs = py_eqs

        G = NeuronGroup(N=pop_size,
            model=py_eqs,
            threshold='v>V_th',
            reset=reset_eqs,
            refractory=refractory_time,
            namespace={'size' : cell_size_py},
            method=integ_method,
            name=name)

    else:
        eqs = inh_eqs

        G = NeuronGroup(N=pop_size,
            model=inh_eqs,
            threshold='v>V_th',
            refractory=refractory_time,
            namespace={'size' : cell_size_inh},
            method=integ_method,
            name=name)

    # random initialization of neuron membrane potential
    G.v = '-60*mvolt-rand()*10*mvolt' # str -> individual init. val. per neuron

    # no stimulation
    G.r = 0

    return G
