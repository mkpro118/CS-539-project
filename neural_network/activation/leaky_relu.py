# will finish cleaning up 

def leaky(x):
    """Compute leaky relu slope coefficients for each set of values in x"""
    if x > 0:
        return x
    else:
        return 0.01*x  