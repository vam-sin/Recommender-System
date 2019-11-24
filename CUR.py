import pandas as pd
import numpy as np
import random
import time
import csv
from SVD import svd

NUM_MOVIES = 3952
NUM_USERS = 6040
PRINT_FREQUENCY = 500

#number or rows / columns to be selected from M
r = 2

def create_utility_matrix():
    # read from file and make utility matrix
    utility_matrix = np.zeros((NUM_USERS, NUM_MOVIES), np.float32)
    with open('./ratings.dat', 'r') as datafile:
            dataReader = csv.DictReader(datafile, delimiter=':')
            i = 0
            for row in dataReader:
                if i%PRINT_FREQUENCY == 0:
                    print("[Utility Matrix - Input] Rating# ", i)
                for key in row:
                    row[key] = int(row[key])
                utility_matrix[row['UserID']-1][row['MovieID']-1] = row['Rating']
                i+=1
    print("Inputted total of ", i, " ratings from dataset")
    return utility_matrix


M = create_utility_matrix()

#creating the list of random rows/columns to select
random_rows = random.sample(range(0,M.shape[1]),r)
random_cols = random.sample(range(0,M.shape[1]),r)



def frobenius_norm(P):
    """
    Input: A numpy array P
    Output: Frobenius norm of P
    """

    X = np.square(P)
    sum = np.sum(X)
    return sum

def make_C(M,r):
    
    """
    Input: Utility Matrix M and dimension r
    Output: The 'C' matrix in CUR decomposition
    """

    #Initialising an empty numpy array
    C = np.empty([M.shape[0],1])
    
    #appending r randomly selected columns from M to C
    for i in range(len(random_cols)):
        col_num = random_cols[i]
        C = np.concatenate((C,M[:,col_num:col_num+1]),axis=1)

    #removing the first random column
    C = np.delete(C,0,axis=1)

    overall_frobenius_norm = frobenius_norm(M)

    for i in range(C.shape[1]):
        
        #normalizing the columns
        prob = frobenius_norm(C[:,i:i+1])/overall_frobenius_norm
        
        temp = np.sqrt(r*prob)
        
        if temp != 0:
            C[:,i:i+1] /= temp
    
    return C

def make_R(M,r):
    """
    Input: Utility Matrix M and dimension r
    Output: The R matrix in CUR decomposition
    """

    #Initializing a random empty numpy array
    R = np.empty([1,M.shape[1]])


    #Appending r randomly selected rows
    for i in range(len(random_rows)):
        row_num = random_rows[i]
        R = np.concatenate((R,M[row_num:row_num+1,:]),axis=0)

    #Deleting the first random row
    R = np.delete(R,0,axis=0)

    overall_frobenius_norm = frobenius_norm(M)

    #normalizing the rows
    for i in range(R.shape[1]):
        prob = frobenius_norm(R[i:i+1,:])/overall_frobenius_norm
        temp = np.sqrt(r*prob)
        if temp != 0:
            R[i:i+1,:] /= temp
    return R

def make_W(C,R,r):
    """
    Input: The C and R matrices of CUR decomposition and dimension r
    Output: The W matrix in CUR decomposition which is used to make U
    """

    #Initializing a random numpy array
    W = np.random.randint(low=0,high=5,size=(r,r))

    #Algorithm to construct the W matrix
    for i in range(r):
        for j in range(r):
            W[i,j] = M[random_rows[i],random_cols[j]]

    return W

def calculate_rmse(X,Y):
    """
    Input: 2 numpy array
    Output: -1 if any of their dimensions don't match. Else, Root Mean Square Error between the 2 arrays
    """

    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        print("ERROR: Shapes don't match")
        return -1

    n = X.size

    error = X - Y
    square_error = np.square(error)
    mean_square_error = (1/n)*(np.sum(square_error))
    root_mean_square_error = np.sqrt(mean_square_error)
    return root_mean_square_error

def calculate_mae(X,Y):
    """
    Input: 2 numpy array
    Output: -1 if any of their dimensions don't match. Else, Mean Average Error between the 2 arrays
    """
    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        print("ERROR: Shapes don't match")
        return -1

    n = X.size

    error = np.sum(np.absolute(X - Y))
    mean_av_error = (1/n)*error

    return mean_av_error


start =  time.time()

C = make_C(M,r)
R = make_R(M,r)
W = make_W(C,R,r)

X,sigma,YT = svd(W,1.0)

t = np.matmul(X,sigma)
w = np.matmul(t,YT)


sigma_plus = np.linalg.pinv(sigma)

sigma_plus_square = np.matmul(sigma,sigma)

temp = np.matmul(YT.T,sigma_plus_square)

U = np.matmul(temp,X.T)

predicted = np.matmul(C,U)
predicted = np.matmul(predicted,R)

end = time.time()
print("\n-------------------STATISTICS--------------------\n")
print("RMSE for CUR:",calculate_rmse(M,predicted))
print("MAE for CUR:",calculate_mae(M,predicted))
print("Time taken to run:",(end-start)*1000,"milliseconds\n")
