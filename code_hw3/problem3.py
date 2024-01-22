import numpy as np
import algorithms as alg
import functions as functions

############################################################################################
############################################################################################ Toy example

#####################Construct a two dimensional toy example of y and X
X=np.matrix('1. 0.;0. 1.')
y=np.matrix('0. ; 0.')


####################Form the oracle function of the objective
def func(w, order):
    return functions.l1(y, X, w, order)

# initialization
initial_w = np.matrix('2; 1.5')
epsilon=0.001
err = 1e-5
maximum_iterations = 65536

# TODO: Experiment as directed in the homework

###########################################################################################################
###########################################################################################################





################################################################################A larger Example
###################################################################################################


# read the data from csv file
data = np.genfromtxt('l1dat.csv', delimiter=',')

# extract the labels y and features X from dataset (y is in the first column)
y = np.asmatrix(data[:, 0]).T
X = data[:,1:data.shape[1]]
d = X.shape[1]

#Form the oracle function of the objective
def func(w, order):
    return functions.l1(y, X, w, order)

# initialization from 0
initial_w = np.asmatrix( np.zeros( shape=(d,1) ) )
epsilon =0.001
err = 1e-5
maximum_iterations = 65536

#set the error and maximum iteration for inner loop and parameter for backtracking
err_inner=1e-7
maximum_iteration_inner=65536
alpha=0.4
beta=0.9


# TODO: Experiment as directed in the homework

###########################################################################################
###########################################################################################