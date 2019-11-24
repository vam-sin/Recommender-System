# libraries
import numpy as np
from numpy.linalg import svd as actualsvd
from error_measures import rmse, mae
import pandas as pd
import time

def eigen_decomp(Amult):
    w, v = np.linalg.eig(Amult)

    # w is the eigen values, v is the eigen vectors
    # Real parts
    w = w.real
    v = v.real

    # Rounding
    for i in range(len(w)):
        w[i] = round(w[i], 2)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            v[i][j] = round(v[i][j], 2)

    eigen = dict() # Maps eigen values to eigen vectors
    for i in range(len(w)):
        if w[i] != 0:
            eigen[w[i]] = v[:, i]

    # Sort in descending order according to the eigen values
    sorted_eigen_values = sorted(list(eigen.keys()), reverse = True)
    sorted_eigen_vectors = np.zeros_like(v)

    for i in range(len(sorted_eigen_values)):
        sorted_eigen_vectors[:, i] = eigen[sorted_eigen_values[i]]
    sorted_eigen_vectors = sorted_eigen_vectors[:, :len(sorted_eigen_values)]

    return sorted_eigen_values, sorted_eigen_vectors

def svd(A, dim_red):
    # Both have same eigen values, use non zero ones to make s.
    # U
    eigen_u, U = eigen_decomp(np.dot(A, A.T))
    print("SVD: Calculated U")
    # V
    eigen_v, V = eigen_decomp(np.dot(A.T, A))
    VT = V.T
    print("SVD: Calculated VT")

    # Sigma
    s = np.diag([np.sqrt(i) for i in eigen_u])
    print("SVD: Calculated s")

    if dim_red == 1.0:
        return U, s, VT
    else:
        # Energy based.
        # Ex: if 90% energy, choose Singular values such that their sum of squares is atleast 90% of the original sum of squares
        total_s = np.sum(s ** 0.5)
        for i in range(s.shape[0]):
            s_sum = np.sum(s[:i+1, :i+1])
            if s_sum > dim_red * total_s:
                s = s[:i, :i]
                U = U[:, :i]
                VT = VT[:i, :]

                return U, s, VT

if __name__ == '__main__':
    # Utility Matrix
    A = pd.read_csv('normalized_utility_matrix.csv')
    A = np.asarray(A)
    print("Imported Matrix")

    # Decomposition
    # A matrix X is column orthonormal if X.T . X is I
    # U is mXr (Left Singular Vectors) - Column Orthonormal (Rows to concepts) - (Unit rotation matrices)
    # Sigma is rXr diagonal matrix - positive values in descending order (strength of concepts) - (Stretch matrix)
    # V is nXr (Right Singular Vectors) - Column Orthonormal (connects concepts to columns) - (Rotation matrix)

    # Full SVD:
    # U is a square matrix

    # Singular Value Decomposition
    print("Calculating SVD")
    start = time.time()
    U, s, VT = svd(A, 1.0)
    print("Time taken for SVD: ", time.time() - start)
    print("Calculated Shapes ", U.shape, s.shape, VT.shape)
    B = np.dot(U, (s.dot(VT)))
    print("Calc: ", B)
    print(A)

    # Calculating Error
    rmse = rmse(B, A)
    print("RMSE Value: ", rmse)

    mae = mae(B, A)
    print("MAE Value: ", mae)

    # SVD results
    # Time: 308.8444137573242, RMSE Value: 0.28308750006854566, MAE Value: 0.14049277643419675
    # Reduced SVD results
    # Time: 321.66628861427307, RMSE Value: 0.23311788798688335, MAE Value: 0.085797425129047
