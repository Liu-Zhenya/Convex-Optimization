import numpy as np
import algorithms as alg
import functions as functions
import matplotlib.pyplot as plt

###############################################################################################toy example
###############################################################################################

############## 1 dimensional data toy example
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


# TODO：Run gradient descent with backtracking line search and with fixed step sizes as directed in the homework
M = 30

x, values, runtimes, xs = alg.gradient_descent(func, initial_w, err, maximum_iterations, alg.backtracking)
fig, axs = plt.subplots(nrows=4,figsize=(10, 20))

axs[0].plot(values,label='backtracking')
axs[0].legend()
axs[0].set_xlabel('iterations')
axs[0].set_ylabel('values')



x_2, values_2, runtimes_2, xs_2 = alg.gradient_descent(func, initial_w, err, maximum_iterations, alg.poly, 1/M, 0)

axs[1].plot(values_2,label='1/M')
axs[1].legend()
axs[1].set_xlabel('iterations')
axs[1].set_ylabel('values')






#####################################################################################################################
######################################################################################################################




################################################################################A larger Example
###################################################################################################


# read the data from csv file
data = np.genfromtxt('logisticdat.csv', delimiter=',')

# extract the labels y and features X from dataset (y is in the first column)
y = np.asmatrix(data[:, 0]).T
X = data[:,1:data.shape[1]]
d = X.shape[1]

#Form the oracle function of the objective
def func(w, order):
    return functions.logistic(y, X, w, order)

# initialization from 0
initial_w = np.asmatrix( np.zeros( shape=(d,1) ) )
epsilon =0.001
err = 1e-5
maximum_iterations = 65536

# TODO: Try backtracking linesearch

M = 30

x_3, values_3, runtimes_3, xs_3 = alg.gradient_descent(func, initial_w, err, maximum_iterations, alg.backtracking)

axs[2].plot(values_3,label='backtracking non-toy')
axs[2].legend()
axs[2].set_xlabel('iterations')
axs[2].set_ylabel('values')



x_4, values_4, runtimes_4, xs_4 = alg.gradient_descent(func, initial_w, err, maximum_iterations, alg.poly, 1/M, 0)

axs[3].plot(values_4,label='1/M non-toy')
axs[3].legend()
axs[3].set_xlabel('iterations')
axs[3].set_ylabel('values')







plt.savefig('plot_B')
#############################################################################################
#############################################################################################

