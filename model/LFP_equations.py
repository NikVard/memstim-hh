"""
--------------------------------------------------------------------------------
Date: 19/04/2022

@author: Nikolaos Vardalakis
--------------------------------------------------------------------------------

Implementation Notes
--------------------------------------------------------------------------------
    | 1: Calculating LFPs is done by averaging the membrane potential per group in each area
    | 2: We separate E/I groups to get individual signals
"""

pop_LFP_eqs = '''
    LFP_E : volt
    LFP_I : volt
'''

syn_E_LFP_eqs = '''
    LFP_E_post = v_pre / N_incoming : volt (summed)
'''

syn_I_LFP_eqs = '''
    LFP_I_post = v_pre / N_incoming : volt (summed)
'''
