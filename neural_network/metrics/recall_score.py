import numpy as np


def recall_score(cmat):
    '''
    Calculates overall recall score of the model

    Parameters:
        cmat: confusion matrix (2d: n_labels, n_labels)

    Returns:
        float: recall score
    '''
    rows, cols = cmat.shape

    tp = np.trace(cmat)
    sumRows = 0

    # get tp+fn values for each label
    for i in range(rows):
        for j in range(cols):
            if (j >= i):
                sumRows += cmat[i, j]

    recall = tp / sumRows

    return recall
