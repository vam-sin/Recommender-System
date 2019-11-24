import numpy as np

def rmse(pred_M, actual_M):

    error = 0.0
    for i in range(actual_M.shape[0]):
        for j in range(actual_M.shape[1]):
            error += ((actual_M[i][j] - pred_M[i][j]) ** 2) / (actual_M.shape[0] * actual_M.shape[1])

    error = np.sqrt(error)

    return error

def mae(pred_M, actual_M):

    error = 0.0
    for i in range(actual_M.shape[0]):
        for j in range(actual_M.shape[1]):
            error += abs(actual_M[i][j] - pred_M[i][j])

    return error / (actual_M.shape[0] * actual_M.shape[1])
