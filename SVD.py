# libraries
import numpy as np

def gso(E):
    # gram-schmidt orthonormalization
    Y = []
    for i in E:
        a = i
        for j in Y:
            i = np.squeeze(np.asarray(i))
            j = np.squeeze(np.asarray(j))
            a = a - (np.dot(j, i))*j
        a = a/np.linalg.norm(a)

        Y.append(a)
    Y = np.asarray(Y)

    return Y

def svd(A):
    # U
    Amult = np.matmul(A, A.T)
    w, v = np.linalg.eig(Amult)
    U = gso(v.T)

    # s
    s = np.zeros((len(w), len(w)))
    for i in range(len(w)):
        s[i][i] = np.sqrt(w[i])

    # V
    Amult = np.matmul(A.T, A)
    w, v = np.linalg.eig(Amult)
    V = gso(v.T)

    return U, s, V.T


if __name__ == '__main__':
    # Utility Matrix
    m = 4 # Number of users
    n = 5 # Number of movies
    # r is the rank of matrix A
    A = np.random.randint(5, size=(m, n))   # Input data matrix

    # Decomposition
    # A matrix X is column orthonormal if X.T . X is I
    # U is mXr (Left Singular Vectors) - Column Orthonormal
    # Sigma is rXr diagonal matrix - positive values in descending order
    # V is nXr (Right Singular Vectors) - Column Orthonormal

    # Calculate U, s, VT
    U, s, VT = svd(A)
    print(U, s, VT)
