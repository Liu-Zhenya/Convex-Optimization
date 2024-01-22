######
######  This script compares different optimization methods on optimizating
######  a sum-exp Objective of the form sum_i exp( < a_i , x > + b_i )
######


import numpy as np
import hw3_functions as hw3
import algorithms as alg


# a is the vector of coefficients
a = np.matrix('1 3; 1 -3; -1 0')
# b is the vector of bias terms
b = np.matrix('-0.1; -0.1; -0.1')

# Choose the sum-exp objective. This notation defines a "closure", returning
# an oracle function which takes x (and order) as its only parameter, and calls
# obj.sum_exp with parameters a and b defined above, and the parameters of 
# the closure (x and order)
func = lambda x, order: hw3.sum_exp( a, b, x, order )

# Use backtracking line search. This defines a closure which takes three
# parameters. Note that the alpha and beta parameters to the backtracking line
# search are 0.4 and 0.9, respectively
linesearch = lambda f, x, direction: alg.backtracking( f, x, direction, 0.4, 0.9 )

# Start at (-1,1.5), with (for BFGS) an identity inverse Hessian
initial_x = np.matrix('-1; 1.5')
initial_inverse_hessian = np.eye( 2 )
# Find the (1e-4)-suboptimal solution
eps = 1e-4
maximum_iterations = 65536

# Setting the backtracking line search parameters
alpha = 0.4
beta = 0.9

# Run the algorithms
x, values, runtimes, gd_xs = alg.gradient_descent( func, initial_x, eps, maximum_iterations, alg.backtracking, alpha, beta )

x, values, runtimes, newton_xs = alg.newton( func, initial_x, eps, maximum_iterations, alg.backtracking, alpha, beta )

# Draw contour plots
hw3.draw_contour_new( func, gd_xs, newton_xs, levels=np.arange(5, 1000, 10), x=np.arange(-3, 1.04, 0.04), y=np.arange(-2, 2.04, 0.04),title='sum-exp')
