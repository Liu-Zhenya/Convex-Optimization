import numpy as np
import algorithms as alg
import functions as functions

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




#############################################################################################
#############################################################################################

