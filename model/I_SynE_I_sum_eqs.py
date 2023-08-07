"""
--------------------------------------------------------------------------------
Date: 30/06/2023

@author: Nikolaos Vardalakis
--------------------------------------------------------------------------------

Implementation Notes
--------------------------------------------------------------------------------
    | 1: For the details regarding the (summed) keyword and the synapses, refer to the question I posed on the Brian2 forum, here: https://brian.discourse.group/t/how-can-i-get-the-population-firing-rate-from-a-spiking-hh-network-during-simulation/496
    | 2: Each unit (of the post-synaptic group) records both I_SynE and I_SynI (from the pre-synaptic group). Then sums the results. It is used as a proxy to the LFP.
"""

eq_record_LFP_neurons = '''
    sum_I_SynE : amp
    sum_I_SynI : amp
'''

eq_record_LFP_synapses = '''
    sum_I_SynE_post = I_SynE_pre : amp (summed)
    sum_I_SynI_post = I_SynI_pre : amp (summed)
'''