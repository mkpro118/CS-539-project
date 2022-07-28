import numpy as np
from sklearn import metrics


def accuracy_score(cmat, K):
    '''
    Calculates overall accuracy of the model

    Parameters:
        cmat: confusion matrix (n_labels, n_labels)
        K: total amount of samples

    Returns:
        float: 0 <= accuracy of the model <= 1
    '''
    accuracy = float(np.trace(cmat) / (K))

    return accuracy
