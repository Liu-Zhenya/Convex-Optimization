import numpy as np
import algorithms as alg
import functions as functions
import matplotlib.pyplot as plt





# the hessian matrix of the quadratic function
H = np.matrix('1 0; 0 30')

# the vector of linear coefficient of the quadratic function
b = np.matrix('0; 0')

# define the oracle function
def func(x, order):
    return functions.quadratic(H, b, x, order)

def func_exact_linesearch(x, order):
    return functions.quadratic_exact_linesearch(H, b, x, order)
# For example start at (4,0.3)
initial_x = np.matrix('4; 0.3')
err = 1e-5
maximum_iterations = 65536




# set smoothness constant M and strong convexity constant m
m=1
M= 30

# Run the algorithms with fixed step size 1/M 1/(100M) and 4/m
x, values, runtimes, xs = alg.gradient_descent(func, initial_x, err, maximum_iterations, alg.poly, 1/M, 0)
x_2, values_2, runtimes_2, xs_2 = alg.gradient_descent(func, initial_x, err, maximum_iterations, alg.poly, 1/(100*M), 0)
x_3, values_3, runtimes_3, xs_3 = alg.gradient_descent(func, initial_x, err, maximum_iterations, alg.poly, 5/m, 0)


fig, axs = plt.subplots(nrows=5,figsize=(10, 20))

axs[0].plot(values,label='1/M')
axs[0].legend()
axs[0].set_xlabel('iterations')
axs[0].set_ylabel('values')


axs[1].plot(values_2,label='1/100M')
axs[1].legend()
axs[1].set_xlabel('iterations')
axs[1].set_ylabel('values')


axs[2].plot(values_3,label='5/m')
axs[2].legend()
axs[2].set_xlabel('iterations')
axs[2].set_ylabel('values')




# TODO:Experiment with other step size choices as directed in the homework
# For this you should also first set the error and maximum iteration for inner loop and also parameters for backtracking

x_4, values_4, runtimes_4, xs_4 = alg.gradient_descent(func, initial_x, err, maximum_iterations, alg.backtracking, 0.4,0.9)
axs[3].plot(values_4,label='backtracking,alpha=0.4,beta=0.9')
axs[3].legend()
axs[3].set_xlabel('iterations')
axs[3].set_ylabel('values')



x_5, values_5, runtimes_5, xs_5 = alg.gradient_descent_exact_linesearch(func_exact_linesearch, initial_x, err, maximum_iterations)
axs[4].plot(values_4,label='Exact Line Search')
axs[4].legend()
axs[4].set_xlabel('iterations')
axs[4].set_ylabel('values')



plt.savefig('plot_A')