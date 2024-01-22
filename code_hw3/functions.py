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

        return (value, gradient)


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


#############################################l_1 regression objective f(w)=|y-Xw|_1
###########################################################################################
def l1(y,X,w,order=0):
    """
        l_1 Regression Objective
        y:          an n-dimenstional vector of labels
        X:          an n*d dimenstional matrix of features
        w:          the current iterate of dimension d*1
        order:      the order of the oracle. For example, order=1 returns the value of the function and its subgradient
    """
    y = np.asmatrix(y)
    X = np.asmatrix(X)
    w = np.asmatrix(w)

    z = X*w-y
    z_abs=np.absolute(X*w-y)
    value = z_abs.sum()

    if order == 0:
        return value

    elif order == 1:
        s = np.sign(z)
        #gradient = (TODO: Calculate a Subgradient)

        return (value, gradient)