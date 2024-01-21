import numpy as np
import algorithms as alg
import functions as functions

d=1
y=np.matrix('1.0;1.0;1.0;1.0;-1.0;-1.0;-1.0;-1.0')
X=np.matrix('8.0;7.0;6.0;5.0;4.0;3.0;2.0;1.0')



###########Form the oracle function of the objective
def func(w, order):
    return functions.logistic(y, X, w, order)


#############Intialized at 0
initial_w = np.asmatrix( np.zeros( shape=(d,1) ) )
err = 1e-4
maximum_iterations = 65536

M = 1000 # set M to be large

# Run the algorithms with backtracking line search
x_2, values_2, runtimes_2, xs_2 = alg.gradient_descent(func, initial_w, err, maximum_iterations, alg.backtracking)

# Run the algorithms with fixed step size 1/M 
x_t, values_t, runtimes_t, xs_t = alg.gradient_descent(func, initial_w, err, maximum_iterations, alg.poly, 1/M, 0)