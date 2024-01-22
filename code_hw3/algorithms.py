######
######  This file includes different optimization methods
######

import time
import numpy as np

###############################################################################
def bisection( func, x, direction, eps=1e-9, maximum_iterations=65536 ):
    """ 
    'Exact' linesearch (using bisection method)
    func:               the function to optimize It is called as "value, gradient = func( x, 1 )
    x:                  the current iterate
    direction:          the direction along which to perform the linesearch
    eps:                the maximum allowed error in the resulting stepsize t
    maximum_iterations: the maximum allowed number of iterations
    """
    
    x = np.asarray( x )
    direction = np.asarray( direction )
    
    if eps <= 0:
        raise ValueError("Epsilon must be positive")
        
    _, gradient = func( x , 1 )
    gradient = np.asarray( gradient )
    
    # checking that the given direction is indeed a descent direciton
    if np.vdot( direction, gradient )  >= 0:
        return 0
    
    else:
        
        # setting an upper bound on the optimum.
        MIN_t = 0
        MAX_t = 1
        iterations = 0
        
        value, gradient = func( x + MAX_t * direction, 1 )
        value = np.double( value )
        gradient = np.asarray( gradient )
        
        while np.vdot( direction, gradient ) < 0:
            
            MAX_t *= 2
            
            value, gradient = func( x + MAX_t * direction, 1 )
            
            iterations += 1
            
            if iterations >= maximum_iterations:
                raise ValueError("Too many iterations")
        
        # bisection search in the interval (MIN_t, MAX_t)
        iterations = 0
        
        while True:        
        
            t = ( MAX_t + MIN_t ) / 2
            
            value, gradient = func( x + t * direction, 1 )
            value = np.double( value )
            gradient = np.asarray( gradient )
        
            suboptimality = abs( np.vdot( direction, gradient ) ) * (MAX_t - t)
            
            if suboptimality <= eps:
                break
            
            if np.vdot( direction, gradient ) < 0:
                MIN_t = t
            else:
                MAX_t = t
            
            iterations += 1
            if iterations >= maximum_iterations:
                raise ValueError("Too many iterations")
                
        return t
    



###############################################################################
def backtracking( func, x, direction, alpha=0.4, beta=0.9, maximum_iterations=65536 ):
    """ 
    Backtracking linesearch
    func:               the function to optimize It is called as "value, gradient = func( x, 1 )
    x:                  the current iterate
    direction:          the direction along which to perform the linesearch
    alpha:              the alpha parameter to backtracking linesearch
    beta:               the beta parameter to backtracking linesearch
    eps:                the maximum allowed error in the resulting stepsize t
    maximum_iterations: the maximum allowed number of iterations
    """

    if alpha <= 0:
        raise ValueError("Alpha must be positive")
    if alpha >= 0.5:
        raise ValueError("Alpha must be less than 0.5")
    if beta <= 0:
        raise ValueError("Beta must be positive")
    if beta >= 1:
        raise ValueError("Beta must be less than 1")
        
        
    x = np.asarray( x )
    direction = np.asarray( direction )
    
    
    value, gradient = func( x , 1 )
    value = np.double( value )
    gradient = np.asarray( gradient )
    
    derivative = np.vdot( direction, gradient )
    
    # checking that the given direction is indeed a descent direciton
    if derivative >= 0:
        return 0
        
    else:
        t = 1
        iterations = 0
        while True:        
        
            if func( x + t * direction, 0 ) <= ( value + alpha * t * derivative):   
                break                                                               
   
            t *= beta                                                               
            
            iterations += 1
            if iterations >= maximum_iterations:
                raise ValueError("Too many iterations")
      
        return t




###############################################################################
def gradient_descent( func, initial_x, eps=1e-5, maximum_iterations=65536, linesearch=bisection, *linesearch_args ):
    """ 
    Gradient Descent
    func:               the function to optimize It is called as "value, gradient = func( x, 1 )
    initial_x:          the starting point
    eps:                the maximum allowed error in the resulting stepsize t
    maximum_iterations: the maximum allowed number of iterations
    linesearch:         the linesearch routine
    *linesearch_args:   the extra arguments of linesearch routine
    """
    
    if eps <= 0:
        raise ValueError("Epsilon must be positive")
    x = np.asarray( initial_x.copy() )
    
    # initialization
    values = []
    runtimes = []
    xs = []
    start_time = time.time()
    iterations = 0
    
    # gradient updates
    while True:
        
        value, gradient = func( x , 1 )

       

        value = np.double( value )
        gradient = np.asarray( gradient )
    
        # updating the logs
        values.append( value )
        runtimes.append( time.time() - start_time )
        xs.append( x.copy() )
        
        
        if np.vdot( gradient, gradient ) <= eps:        
            break                               
            
        
        
        direction = - gradient                  
        
        t = linesearch( func, x, direction, *linesearch_args )
        
        x += t * direction
        
        iterations += 1
        if iterations >= maximum_iterations:
            raise ValueError("Too many iterations")
                
    return (x, values, runtimes, xs)
    
    


###############################################################################
def newton( func, initial_x, eps=1e-5, maximum_iterations=65536, linesearch=bisection, *linesearch_args  ):
    """ 
    Newton's Method
    func:               the function to optimize It is called as "value, gradient, hessian = func( x, 2 )
    initial_x:          the starting point
    eps:                the maximum allowed error in the resulting stepsize t
    maximum_iterations: the maximum allowed number of iterations
    linesearch:         the linesearch routine
    *linesearch_args:   the extra arguments of linesearch routine
    """
    
    if eps <= 0:
        raise ValueError("Epsilon must be positive")
    x = np.asarray( initial_x.copy() )
    
    # initialization
    values = []
    runtimes = []
    xs = []
    start_time = time.time()
    iterations = 0
    
    # newton's method updates
    while True:
        
        value, gradient, hessian = func( x , 2 )
        value = np.double( value )
        gradient = np.matrix( gradient )
        hessian = np.matrix( hessian ) 
        
        # updating the logs
        values.append( value )
        runtimes.append( time.time() - start_time )
        xs.append( x.copy() )

        ### TODO: Compute the Newton update direction

        direction = -np.linalg.inv(hessian)@gradient
        ### TODO: Compute the Newton decrement
        decrement = np.sqrt(gradient.T @ np.linalg.inv(hessian) @ gradient)
        if (0.5*(decrement**2)<= eps):
            break
        
        t = linesearch( func, x, direction )

        x += t * np.asarray( direction )

        iterations += 1
        if iterations >= maximum_iterations:
            raise ValueError("Too many iterations")
    
    return (x, values, runtimes, xs)
    