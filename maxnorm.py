import numpy as np
import cvxpy as cp
import time

def gen_ratings(n,m,p):
    """
    Inputs:
        n: the number of users
        m: the number of movies
        p: the number of observed ratings
    Returns:
        Y: a list of p ratings in {-1, 1}
        S: a list of p tuples (i,j), 0 <= i < n, 0 <= j < m

        For 0 <= k < p, y[k] is the rating given by user S[k][0]
        for movie S[k][1]
    """
    if p > m*n:
        print('You asked for more ratings than m*n!')
        return [],[]
    np.random.seed(0) # makes this deterministic
    Y = 2.*np.random.randint(0,2,(n,m)) - 1. # generates +1/-1 ratings
    mask = np.random.rand(n,m) # samples an nxm array of random numbers in [0,1]
    cutoff = np.sort(mask.reshape(-1))[-p] # finds the pth largest number in mask
    # mask = 1*(mask <= cutoff) # mask[i,j] = 1 for p of the i,j, 0 everywhere else
    Y = Y[mask >= cutoff] # ratings to be returned
    S = np.argwhere(mask >= cutoff) # indices of ratings
    return Y, S

def solving_SDP(n,m,p,Y,S):
    start_time = time.time()


    A = cp.Variable((n,n), PSD = True)  
    B = cp.Variable((m,m), PSD = True)  
    X = cp.Variable((n,m))  
    V = cp.bmat([[A, X], [X.T, B]])
    objective =  cp.maximum(cp.max(cp.diag(A)),cp.max(cp.diag(B)))

    constraints = [V >> 0]
    constraints += [
    X[S[i][0],S[i][1]] * Y[i] >= 1 for i in range(p)
]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()
    print("Optimal value of the rate matrix is \n", X.value)



    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f'elapsed time is \n {elapsed_time}')


    





n=3
m=3
p=3
Y,S = gen_ratings(n,m,p)
solving_SDP(n,m,p,Y,S)