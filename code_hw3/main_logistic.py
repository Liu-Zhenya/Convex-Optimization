######
######  This script compares different optimization methods on logistic
######  regression problem
######

import numpy as np
import algorithms as alg
import hw3_functions as hw3
import matplotlib.pyplot as plt

# logistic_small contains 1000 10-dimensional examples
# logistic_large contains 1000 50-dimensional examples
#
#data = np.genfromtxt('logistic_small.csv', delimiter=',')
data = np.genfromtxt('logistic_small.csv', delimiter=',')

# extract the labels y and features X from dataset
y = np.asmatrix(data[ :, 0]).T
X = data[:,1:data.shape[1]]
d = X.shape[1]

# Choose the logistic objective. This notation defines a "closure", returning
# an oracle function which takes w (and order) as its only parameter, and calls
# obj.logistic with parameters y and X defined above, and the parameters of 
# the closure (w and order)
func = lambda w, order: hw3.logistic( y, X, w, order )

initial_w = np.asmatrix( np.zeros( shape=(d,1) ) )

initial_inverse_hessian = np.eye( d )
# Find the (1e-4)-suboptimal solution
eps = 1e-4
maximum_iterations = 65536

# Setting the backtracking line search parameters
alpha = 0.4
beta = 0.9

# Find the optimal solution (it is really suboptimal but with a higher accuracy!)
w, values, runtimes, newton_ws = alg.newton( func, initial_w, eps, maximum_iterations, alg.backtracking, alpha, beta )
minimum_f = min( values)


# Run algorithms and draw plots
w, values, runtimes, gd_ws = alg.gradient_descent( func, initial_w, eps, maximum_iterations, alg.backtracking, alpha, beta )


iterations = np.arange(len(values))
plt.semilogy( iterations, values-minimum_f, linewidth=2, color='r', label='GD' )


w, values, runtimes, newton_ws = alg.newton( func, initial_w, eps, maximum_iterations, alg.backtracking, alpha, beta )


iterations = np.arange(len(values))
plt.semilogy( iterations, values-minimum_f, linewidth=2, color='m', label='Newton')

plt.xlabel('time')
plt.ylabel('suboptimality')
plt.legend()
plt.savefig('logistics_small_iterations.jpg')