import numpy as np
from sklearn import metrics


def precision_score(cmat):
    '''
    Calculates overall precision score of the model

    Parameters:
        cmat: confusion matrix (2d: n_labels, n_labels)

    Returns:
        float: precision score
    '''
    rows, cols = cmat.shape

    tp = np.trace(cmat)
    sumCols = 0

    # get tp+fp values for each label
    for i in range(cols):
        for j in range(rows):
            if (j >= i):
                sumCols += cmat[j, i]

    precision = tp / sumCols

    return precision
