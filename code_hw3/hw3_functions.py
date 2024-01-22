######
######  This file includes different functions used in HW3
######

import numpy as np
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

    y = np.asarray(y)
    X = np.asarray(X)
    w = np.asarray(w)
    
    # value = TODO: FUNCTION VALUE
    value = np.sum((-X*y)@w+np.log(1+np.exp((X*y)@w)),axis=0)



    if order == 0:
        return value
    
    elif order == 1:
        
        # gradient = TODO: GRADIENT
        gradient= np.sum((-X*y)+X*y*np.exp((X*y)@w)/(1+np.exp((X*y)@w)),axis=0)

        
        return (value, np.asmatrix(gradient).T)
        
    elif order == 2:
        
        # gradient = TODO: GRADIENT
        gradient=np.sum((-X*y)+X*y*np.exp((X*y)@w)/(1+np.exp((X*y)@w)),axis=0)
    
        
        #hessian = TODO: HESSIAN
        weight = (np.exp((X*y)@w)/np.multiply(1+np.exp((X*y)@w),1+np.exp((X*y)@w)))*np.multiply(y,y)
        result_matrix = np.einsum('ij,ik->ijk', X, X)
        hessian = np.sum(weight[:, :,np.newaxis] * result_matrix, axis=0)
      

        return (np.asmatrix(value), np.asmatrix(gradient).T,np.asmatrix(hessian))
        
    else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")




###############################################################################
def draw_contour( func, gd_xs, newton_xs, levels=np.arange(5, 1000, 10), x=np.arange(-5, 5.1, 0.05), y=np.arange(-5, 5.1, 0.05) ):
    """ 
    Draws a contour plot of given iterations for a function
    func:       the contour levels will be drawn based on the values of func
    gd_xs:      gradient descent iterates
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
    
    line_gd, = plt.plot( gd_xs[0][0,0], gd_xs[0][1,0], linewidth=2, color='r', marker='o', label='GD' )
    line_newton, = plt.plot( newton_xs[0][0,0], newton_xs[0][1,0], linewidth=2, color='m', marker='o',label='Newton' )
    
    # L = plt.legend([line_gd, line_newton])
    L=plt.legend()
    plt.draw()
    time.sleep(1)
    
    for i in range( 1, max(len(gd_xs), len(newton_xs)) ):
        
        line_gd.set_xdata( np.append( line_gd.get_xdata(), gd_xs[ min(i,len(gd_xs)-1) ][0,0] ) )
        line_gd.set_ydata( np.append( line_gd.get_ydata(), gd_xs[ min(i,len(gd_xs)-1) ][1,0] ) )
         
        
        line_newton.set_xdata( np.append( line_newton.get_xdata(), newton_xs[ min(i,len(newton_xs)-1) ][0,0] ) )
        line_newton.set_ydata( np.append( line_newton.get_ydata(), newton_xs[ min(i,len(newton_xs)-1) ][1,0] ) )
         
        
        L.get_texts()[0].set_text( " GD, %d iterations" % min(i,len(gd_xs)-1) )
        L.get_texts()[1].set_text( " Newton, %d iterations" % min(i,len(newton_xs)-1) )    
        
        plt.draw()
        if i > 100:
            input("Press Enter to continue...")
        else:
            time.sleep(0.05)
    plt.show()
    plt.savefig('3.2(a).jpg')



def draw_contour_new( func, gd_xs, newton_xs, levels=np.arange(5, 1000, 10), x=np.arange(-5, 5.1, 0.05), y=np.arange(-5, 5.1, 0.05),title='default'):
    plt.figure()
    """ 
    Draws a contour plot of given iterations for a function
    func:       the contour levels will be drawn based on the values of func
    gd_xs:      gradient descent iterates
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

    line_gd, = plt.plot( gd_xs[0][0,0], gd_xs[0][1,0], linewidth=2, color='r', marker='o', label='GD' )
    line_newton, = plt.plot( newton_xs[0][0,0], newton_xs[0][1,0], linewidth=2, color='m', marker='o',label='Newton' )
    
    # L = plt.legend([line_gd, line_newton])
    L=plt.legend()
    plt.draw()
    time.sleep(1)
    
    for i in range( 1, max(len(gd_xs), len(newton_xs)) ):
        
        line_gd.set_xdata( np.append( line_gd.get_xdata(), gd_xs[ min(i,len(gd_xs)-1) ][0,0] ) )
        line_gd.set_ydata( np.append( line_gd.get_ydata(), gd_xs[ min(i,len(gd_xs)-1) ][1,0] ) )
         
        
        line_newton.set_xdata( np.append( line_newton.get_xdata(), newton_xs[ min(i,len(newton_xs)-1) ][0,0] ) )
        line_newton.set_ydata( np.append( line_newton.get_ydata(), newton_xs[ min(i,len(newton_xs)-1) ][1,0] ) )
         
        
        L.get_texts()[0].set_text( " GD, %d iterations" % min(i,len(gd_xs)-1) )
        L.get_texts()[1].set_text( " Newton, %d iterations" % min(i,len(newton_xs)-1) )    
        
        plt.draw()
        if i > 100:
            input("Press Enter to continue...")
        else:
            time.sleep(0.05)
    plt.show()
    plt.savefig(f'{title}.jpg')    