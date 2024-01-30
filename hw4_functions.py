######
######  This file includes different functions used in HW4
######

import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import time


###############################################################################
def quadratic( H, b, x, order=0 ):
    """
    Quadratic Objective
    H:          the Hessian matrix
    b:          the vector of linear coefficients
    x:          the current iterate
    order:      the order of the oracle. For example, order=1 returns the value of the function and its gradient while order=2 will also return the hessian
    """
    H = np.asmatrix(H)
    b = np.asmatrix(b)
    x = np.asmatrix(x)

    value = 0.5 * x.T * H * x + b.T * x

    if order == 0:
        return value

    elif order == 1:

        gradient = H * x + b

        return (value, gradient)

    elif order == 2:

        gradient = H * x + b

        hessian = H

        return (value, gradient, hessian)

    else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")




###############################################################################
def sum_exp( a, b, x, order=0 ):
    """
    Sum-exp Objective :     sum_i exp( < a_i , x > + b_i )
    a:          the vector of coefficient
    b:          the vector of bias terms
    x:          the current iterate
    order:      the order of the oracle. For example, order=1 returns the value of the function and its gradient while order=2 will also return the hessian
    """

    exp_x = np.exp( a * x + b )

    value = exp_x.sum()

    if order == 0:
        return value

    elif order == 1:

        gradient = a.T * exp_x

        return (value, gradient)

    elif order == 2:

        gradient = a.T * exp_x

        hessian = a.T * np.asmatrix( np.asarray(a) * np.asarray(exp_x) )

        return (value, gradient, hessian)

    else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")




###############################################################################
def logistic( y, X, w, order=0 ):
    """
    Logistic Regression Objective
    y:          an n-dimenstional vector of labels
    X:          an n*d dimenstional matrix of features
    w:          the current iterate
    order:      the order of the oracle. For example, order=1 returns the value of the function and its gradient while order=2 will also return the hessian
    """

    y = np.asmatrix(y)
    X = np.asmatrix(X)
    w = np.asmatrix(w)

    # value = TODO: FUNCTION VALUE

    if order == 0:
        return value

    elif order == 1:

        # gradient = TODO: GRADIENT

        return (value, gradient)

    elif order == 2:

        # gradient = TODO: GRADIENT

        #hessian = TODO: HESSIAN

        return (value, gradient, hessian)

    else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")




###############################################################################
def draw_contour( func, gd_xs, cg_xs, bfgs_xs, newton_xs, levels=np.arange(5, 1000, 10), x=np.arange(-5, 5.1, 0.05), y=np.arange(-5, 5.1, 0.05) ):
    """
    Draws a contour plot of given iterations for a function
    func:       the contour levels will be drawn based on the values of func
    gd_xs:      gradient descent iterates
    cg_xs:      conjugate gradient iterates
    bfgs_xs:    BFGS iterates
    newton_xs:  Newton's method iterates
    levels:     levels of the contour plot
    x:          x coordinates to evaluate func and draw the plot
    y:          y coordinates to evaluate func and draw the plot
    """
    Z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j] = func( np.matrix([x[i],y[j]]).T , 0 )

    plt.contour( x, y, Z.T, levels, colors='0.75')
    plt.ion()
    plt.show()

    line_gd, = plt.plot( gd_xs[0][0,0], gd_xs[0][1,0], linewidth=2, color='r', marker='o', label='GD' )
    line_cg, = plt.plot( cg_xs[0][0,0], cg_xs[0][1,0], linewidth=2, color='g', marker='o', label='CG' )
    line_bfgs, = plt.plot( bfgs_xs[0][0,0], bfgs_xs[0][1,0], linewidth=2, color='b', marker='o', label='BFGS' )
    line_newton, = plt.plot( newton_xs[0][0,0], newton_xs[0][1,0], linewidth=2, color='m', marker='o',label='Newton' )

    L = plt.legend(handles=[line_gd, line_cg, line_bfgs, line_newton])
    plt.draw()
    time.sleep(1)

    for i in range( 1, max(len(gd_xs), len(newton_xs), len(cg_xs), len(bfgs_xs)) ):

        line_gd.set_xdata( np.append( line_gd.get_xdata(), gd_xs[ min(i,len(gd_xs)-1) ][0,0] ) )
        line_gd.set_ydata( np.append( line_gd.get_ydata(), gd_xs[ min(i,len(gd_xs)-1) ][1,0] ) )

        line_cg.set_xdata( np.append( line_cg.get_xdata(), cg_xs[ min(i,len(cg_xs)-1) ][0,0] ) )
        line_cg.set_ydata( np.append( line_cg.get_ydata(), cg_xs[ min(i,len(cg_xs)-1) ][1,0] ) )

        line_bfgs.set_xdata( np.append( line_bfgs.get_xdata(), bfgs_xs[ min(i,len(bfgs_xs)-1) ][0,0] ) )
        line_bfgs.set_ydata( np.append( line_bfgs.get_ydata(), bfgs_xs[ min(i,len(bfgs_xs)-1) ][1,0] ) )

        line_newton.set_xdata( np.append( line_newton.get_xdata(), newton_xs[ min(i,len(newton_xs)-1) ][0,0] ) )
        line_newton.set_ydata( np.append( line_newton.get_ydata(), newton_xs[ min(i,len(newton_xs)-1) ][1,0] ) )


        L.get_texts()[0].set_text( " GD, %d iterations" % min(i,len(gd_xs)-1) )
        L.get_texts()[1].set_text( " CG, %d iterations" % min(i,len(cg_xs)-1) )
        L.get_texts()[2].set_text( " BFGS, %d iterations" % min(i,len(bfgs_xs)-1) )
        L.get_texts()[3].set_text( " Newton, %d iterations" % min(i,len(newton_xs)-1) )

        plt.draw()
        if i < 10:
            input("Press Enter to continue...")
        else:
            time.sleep(0.05)
