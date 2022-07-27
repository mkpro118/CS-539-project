# will finish cleaning up 

import numpy as np

def tanh(x):
    """Compute tanh slope coefficients for each set of values in x"""
    return (np.exp(2*x) - 1)/(np.exp(2*x) + 1)