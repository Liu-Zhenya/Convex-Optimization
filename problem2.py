import numpy as np
import algorithms as alg
import functions as functions





# the hessian matrix of the quadratic function
H = np.matrix('1 0; 0 30')

# the vector of linear coefficient of the quadratic function
b = np.matrix('0; 0')

# define the oracle function
def func(x, order):
    return functions.quadratic(H, b, x, order)


# For example start at (4,0.3)
initial_x = np.matrix('4; 0.3')
err = 1e-5
maximum_iterations = 65536




# set smoothness constant M and strong convexity constant m
m=1
M= 30

# Run the algorithms with fixed step size 1/M 1/(100M) and 4/m
x, values, runtimes, xs = alg.gradient_descent(func, initial_x, err, maximum_iterations, alg.poly, 1/M, 0)


# TODO:Experiment with other step size choices as directed in the homework
# For this you should also first set the error and maximum iteration for inner loop and also parameters for backtracking


