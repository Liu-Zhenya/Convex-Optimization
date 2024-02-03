import cvxpy as cvx
import numpy as np

means = np.asmatrix(np.genfromtxt('crypto_means.csv', delimiter=',').reshape(100,1))
covs = np.asmatrix(np.genfromtxt('crypto_covs.csv', delimiter=',').reshape(100,100))
names = []
with open('crypto_names.csv', 'r') as f:
    for line in f.readlines():
        names.append(line.strip())

### YOUR CODE HERE ###
