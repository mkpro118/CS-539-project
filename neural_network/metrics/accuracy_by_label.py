# will finish cleaning up
import numpy as np

class labelAccuracy:

    def labelAccuracy(cmat): 
        '''
        Calculates accuracy of the model by label

        Parameters: 
            cmat: confusion matrix (2d array: n_labels, n_labels)

        Returns: 
            float array: each element of the array has an accuracy between
            0 and 1, corresponding with it's label position in the confusion
            matrix 
        '''
        rows, cols = cmat.shape

        accuracy = np.zeros(rows, dtype = float)
        truePositive = np.zeros(rows, dtype = float)
        rowSum = np.zeros(rows, dtype = float)
        colSum = np.zeros(rows, dtype = float)

        diagSum = np.trace(cmat)

        # get row sum
        for i in range(rows):
            truePositive[i] = cmat[i,i]
            for j in range(cols):
                rowSum[i] += cmat[i, j]
        
        # get col sum 
        for i in range(cols):
            for j in range(rows):
                colSum[i] += cmat[j, i] 

        # get accuracy
        for i in range(rows):
            accuracy[i] = float(diagSum / ((rowSum[i] - truePositive[i]) + (colSum[i] - truePositive[i]) + diagSum))

        return accuracy

