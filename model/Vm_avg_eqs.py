"""
--------------------------------------------------------------------------------
Date: 16/06/2022

@author: Nikolaos Vardalakis
--------------------------------------------------------------------------------

Implementation Notes
--------------------------------------------------------------------------------
    | 1: For the details regarding the (summed) keyword and the synapses, refer to the question I posed on the Brian2 forum, here: https://brian.discourse.group/t/how-can-i-get-the-population-firing-rate-from-a-spiking-hh-network-during-simulation/496
"""

eq_record_neurons = '''
    sum_v : volt
'''

eq_record_synapses = '''
    sum_v_post = v_pre/N_incoming : volt (summed)
'''
