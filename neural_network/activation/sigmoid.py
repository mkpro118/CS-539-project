# will finish cleaning up 

import numpy as np

def sigmoid(x):
    """Compute sigmoid slope coefficients for each set of values in x"""
    return 1/(1 + np.exp(-x))