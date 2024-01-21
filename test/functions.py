import numpy as np


############################################quadratic objective xHx/2 + <b,x>
###############################################################################
def quadratic(H, b, x, order=0):
    """
    Quadratic Objective
    H:          the Hessian matrix
    b:          the vector of linear coefficients
    x:          the current iterate
    order:      the order of the oracle. For example, order=1 returns the value of the function and its gradient
    """
    H = np.asmatrix(H)
    b = np.asmatrix(b)
    x = np.asmatrix(x)

    value = 0.5 * x.T * H * x + b.T * x

    if order == 0:
        return value

    elif order == 1:

        #gradient = (TODO: Calculate the Gradient)
        gradient = H*x + b

        return (value, gradient)
    

# Define for calculating exact line search
def quadratic_exact_linesearch(H, b, x, order=0):
    """
    Quadratic Objective
    H:          the Hessian matrix
    b:          the vector of linear coefficients
    x:          the current iterate
    order:      the order of the oracle. For example, order=1 returns the value of the function and its gradient
    """
    H = np.asmatrix(H)
    b = np.asmatrix(b)
    x = np.asmatrix(x)

    value = 0.5 * x.T * H * x + b.T * x

    if order == 0:
        return value

    elif order == 1:

        #gradient = (TODO: Calculate the Gradient)
        gradient = H*x + b
        step_length = (gradient.T * gradient)/(gradient.T * H *  gradient)
        return (value, gradient,step_length)


################################logistic objective  f(w)=Sum_i [-y_i<w,x_i>+ln(1+exp(y_i<w,x_i>))]
###############################################################################
def logistic(y, X, w, order=0):
    """
    Logistic Regression Objective
    y:          an n-dimenstional vector of labels
    X:          an n*d dimenstional matrix of features
    w:          the current iterate
    order:      the order of the oracle. For example, order=1 returns the value of the function and its gradient
    """

    y = np.asmatrix(y)
    X = np.asmatrix(X)
    w = np.asmatrix(w)

    z = np.multiply(X * w, y)
    s = np.log(1 + np.exp(z)) - z
    value = s.sum()

    if order == 0:
        return value

    elif order == 1:
        s = - np.divide(y, 1 + np.exp(z))
        gradient = X.T * s

        return (value, gradient)