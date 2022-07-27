import matplotlib.pyplot as plt
import numpy as np 
from sklearn import metrics

from ..utils.exports import export

@export
def cmat(y_true, y_pred):
    """Returns confusion matrix from classification"""
    
    # could use sklearn: 
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)

    cm_display.plot()
    plt.show()

    return confusion_matrix

    # otherwise: 
    # might need to refactor y_pred using idx and Z before hand 
    # return y_true.T @ y_pred

