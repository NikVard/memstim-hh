import sys
from brian2 import *
from model.globals import *
from model.HH_equations import *

def create_group(parameters, name):
    """ Universal function that creates a group of neurons according to input parameters
    TODO: Need to add distance-based connectivity on the synapses i.e. in the _connect_ statement, {p=... + <distance>} """
    pop_size = parameters['N']
    try:
        if parameters['type'].lower() == 'pycan':
            eqs = py_CAN_eqs

            G = NeuronGroup(N=pop_size,
                model=py_CAN_eqs,
                threshold='v>V_th',
                reset=reset_eqs,
                refractory=refractory_time,
                namespace={'size':cell_size_py, 'glu':1},
                method=integ_method,
                name=name)

        elif parameters['type'].lower() == 'py':
            eqs = py_eqs

            G = NeuronGroup(N=pop_size,
                model=py_eqs,
                threshold='v>V_th',
                reset=reset_eqs,
                refractory=refractory_time,
                namespace={'size' : cell_size_py},
                method=integ_method,
                name=name)

        elif parameters['type'].lower() == 'inh':
            eqs = inh_eqs

            G = NeuronGroup(N=pop_size,
                model=inh_eqs,
                threshold='v>V_th',
                refractory=refractory_time,
                namespace={'size' : cell_size_inh},
                method=integ_method,
                name=name)
        else:
            raise RuntimeError('Cannot create group ' + name + ': Incorrect neuron type [' + parameters['type'] + ']')
    except Exception as e:
        sys.stderr.write('Exception: ' + str(e))
        sys.exit(1)

    # random initialization of neuron membrane potential
    G.v = '-60*mvolt-rand()*10*mvolt' # str -> individual init. val. per neuron

    # TODO: fix later (spatial connectivity)
    # no stimulation for now
    G.r = 0

    return G


def connect_intra(G_py, G_inh, p_conn, gains):
    """ Function that takes care of intra-connectivity between populations in an area
        Added (Gaussian) distance-based connectivity on the synapses in the `connect()` statement, {p=... + <distance>} """

    N_py = len(G_py)
    N_inh = len(G_inh)
    syn_all = [[0]*(N_py + N_inh) for i in range(N_py + N_inh)] # for N populations we have N**2 synapses

    for pypop in range(N_py): # iterate over pyramidal populations in the current area
        # self connection | is there a connection probability? does the group exist?
        if p_conn[pypop][pypop]!=0. and G_py[pypop]:
            syn_EE = Synapses(G_py[pypop], G_py[pypop], on_pre="he_post+="+str(gains[pypop])+"*"+str(g_max_e/siemens)+"*siemens*glu_pre", name=G_py[pypop].name+"to"+G_py[pypop].name) # create synapse
            syn_EE.connect(condition='i!=j', p=str(p_conn[pypop][pypop])+'*exp(-((x_soma_pre-x_soma_post)**2+(y_soma_pre-y_soma_post)**2+(z_soma_pre-z_soma_post)**2)/(2*(2500*umetre)**2))') # connect E2E
            syn_all[pypop][pypop] = syn_EE

        # other connections
        for inhpop in range(N_inh):
            if p_conn[pypop][N_py+inhpop]!=0. and G_py[pypop] and G_inh[inhpop]: # connection is valid (p>0); exc pop exists; inh pop exists. # the addition at the start makes sure we go to the inhibitory section of the connections
                syn_EI = Synapses(G_py[pypop], G_inh[inhpop], on_pre="he_post+="+str(gains[pypop])+"*"+str(g_max_e/siemens)+"*siemens*glu_pre", name=G_py[pypop].name+"to"+G_inh[inhpop].name) # create the excitatory synapse on the inhibitory population
                syn_EI.connect(p=str(p_conn[pypop][N_py+inhpop])+'*exp(-((x_soma_pre-x_soma_post)**2+(y_soma_pre-y_soma_post)**2+(z_soma_pre-z_soma_post)**2)/(2*(2500*umetre)**2))') # connect E2I
                syn_all[pypop][N_py+inhpop] = syn_EI

            if p_conn[N_py+inhpop][pypop]!=0 and G_inh[inhpop] and G_py[pypop]: # same as before, but the other way around, I2E connection
                syn_IE = Synapses(G_inh[inhpop], G_py[pypop], on_pre="hi_post+="+str(gains[N_py+inhpop])+"*"+str(g_max_i/siemens)+"*siemens", name=G_inh[inhpop].name+"to"+G_py[pypop].name) # create the inhibitory synapse on the excitatory population
                syn_IE.connect(p=str(p_conn[N_py+inhpop][pypop])+'*exp(-((x_soma_pre-x_soma_post)**2+(y_soma_pre-y_soma_post)**2+(z_soma_pre-z_soma_post)**2)/(2*(350*umetre)**2))') # connect I2E
                syn_all[N_py+inhpop][pypop] = syn_IE

    for inhpop in range(N_inh): # iterate over the inhibitory populations in the current area; handles I2I self connections
        if p_conn[N_py+inhpop][N_py+inhpop]!=0 and G_inh[inhpop]:
            syn_II = Synapses(G_inh[inhpop], G_inh[inhpop], on_pre="hi_post+="+str(gains[N_py+inhpop])+"*"+str(g_max_i/siemens)+"*siemens", name=G_inh[inhpop].name+"to"+G_inh[inhpop].name)
            syn_II.connect(condition='i!=j', p=str(p_conn[N_py+inhpop][N_py+inhpop])+'*exp(-((x_soma_pre-x_soma_post)**2+(y_soma_pre-y_soma_post)**2+(z_soma_pre-z_soma_post)**2)/(2*(350*umetre)**2))') # connect I2I
            syn_all[N_py+inhpop][N_py+inhpop] = syn_II

    # done, return the synapses list
    return syn_all


def connect_inter(all_G_py_from, all_G_py_to, all_G_inh_to, all_p, gains):
    """ Function that takes care of inter-connectivity between areas.
        Added (Gaussian) distance-based connectivity between the areas """
    N_py_from = len(all_G_py_from)
    N_py_to = len(all_G_py_to)
    N_inh_to = len(all_G_inh_to)
    syn_all = [[0]*(N_py_to+N_inh_to) for ii in range(N_py_from)] # each pyramidal pop from the origin projects to a set of pyramidal/inhibitory populations in the destination

    for origin in range(N_py_from):
        G_py_from = all_G_py_from[origin]
        if G_py_from:

            for destination in range(N_py_to):  # from Py (origin) to Py (destination)
                G_py_to = all_G_py_to[destination]
                if G_py_to:
                    syn_E = Synapses(G_py_from, G_py_to, on_pre="he_ext_post+="+str(gains[origin])+"*"+str(g_max_e/siemens)+"*siemens*glu_pre", name=G_py_from.name+"to"+G_py_to.name)
                    syn_E.connect(p=str(all_p[origin][destination])+'*exp(-((z_soma_pre-z_soma_post)**2)/(2*(1000*umetre)**2))')
                    syn_all[origin][destination] = syn_E

            for destination in range(N_inh_to): # from Py (origin) to Inh (destination)
                G_inh_to = all_G_inh_to[destination]
                if G_inh_to:
                    syn_I = Synapses(G_py_from, G_inh_to, on_pre="he_ext_post+="+str(gains[origin])+"*"+str(g_max_e/siemens)+"*siemens*glu_pre", name=G_py_from.name+"to"+G_inh_to.name)
                    syn_I.connect(p=str(all_p[origin][N_py_to+destination])+'*exp(-((z_soma_pre-z_soma_post)**2)/(2*(1000*umetre)**2))')

                    syn_all[origin][N_py_to+destination] = syn_I

    return syn_all
