# libraries
import numpy as np
from numpy.linalg import svd as actualsvd

# tasks
# s: Done
# U
# V

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

    # V
    eigen_v, V = eigen_decomp(np.dot(A.T, A))
    VT = V.T

    # Sigma
    s = np.diag([np.sqrt(i) for i in eigen_u])

    if dim_red == 1.0:
        return U, s, VT
    else:
        total_s = np.sum(s ** 0.5)
        for i in range(s.shape[0]):
            s_sum = np.sum(s[:i+1, :i+1])
            if s_sum > dim_red * total_s:
                s = s[:i, :i]
                U = U[:, :i]
                VT = VT[:i, :]

                return U, s, VT



    # # print(w, v)
    # inds = w.argsort()
    # w = sorted(w)
    # v = v[inds]
    # # print(w, v)
    # U = gso(v.T)
    # # U = -1.0 * U
    #
    # # s
    # sing_values = []
    # for i in w:
    #     if i>0.0:
    #         sing_values.append(np.sqrt(i))
    #
    # sing_values = sorted(sing_values, reverse = True)
    # s = np.zeros((A.shape[0], A.shape[1]))
    # for i in range(len(sing_values)):
    #     s[i][i] = sing_values[i]
    #
    # # V
    # Amult = np.matmul(A.T, A)
    # w, v = np.linalg.eig(Amult)
    # # print(w, v)
    # # print(v, w)
    # # print(w, v)
    # w_mod = sorted(w, reverse = True)
    # # print(v, w_mod)
    # vmod = []
    # w = list(w)
    # for i in w_mod:
    #     vmod.append(v[w.index(i)])
    # # print(v, vmod)
    # vmod = np.asarray(vmod)
    # # print(w_mod, vmod)
    # VT = gso(vmod.T)

    # return U, s, VT



if __name__ == '__main__':
    # Utility Matrix
    m = 4 # Number of users
    n = 5 # Number of movies
    # r is the rank of matrix A
    A = np.random.randint(5, size=(m, n))   # Input data matrix
    # A = np.array([[3, 2, 2], [2, 3, -2]])

    # Decomposition
    # A matrix X is column orthonormal if X.T . X is I
    # U is mXr (Left Singular Vectors) - Column Orthonormal
    # Sigma is rXr diagonal matrix - positive values in descending order
    # V is nXr (Right Singular Vectors) - Column Orthonormal

    U, s, VT = svd(A, 1.0)
    print("Calculated Shapes ", U.shape, s.shape, VT.shape)
    B = np.dot(U, (s.dot(VT)))
    print("Calc: ", -B)
    print(A)
