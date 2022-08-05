"""
--------------------------------------------------------------------------------
Date: 05/08/2022

@author: Nikolaos Vardalakis
--------------------------------------------------------------------------------

Implementation Notes
--------------------------------------------------------------------------------
    | 1: A fixed input can be implemented in a number of ways. I chose to create a neuron group with a single neuron that reads a TimedArray with the fixed input and links it to a new variable. It simplifies the code.
    | 2: Alternatively, we can make a new group that generates the input by itself (i.e. rhythm = sin(....)*nA), but that has the inherent problem of not easily allowing for a fixed delay until the input starts!
"""

# Fixed input eqs using note #1 (TimedArray)
fixed_input_TA_eqs = '''
    rhythm = inp_theta(t): amp
'''
