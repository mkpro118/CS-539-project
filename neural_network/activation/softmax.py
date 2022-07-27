# will finish cleaning up 

import numpy as np

def softmax(x):
    """Compute softmax slope coefficients for each sets of scores in x"""
    return np.exp(x) / np.sum(np.exp(x), axis=0)