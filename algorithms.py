######
######  This file includes gradient descent method
######

import time
import numpy as np


###############################################################################
def bisection(func, x, direction, k, eps=1e-9, maximum_iterations=65536):
    """
    'Exact' linesearch (using bisection method)
    func:               the function to optimize It is called as "value, gradient = func( x, 1 )
    x:                  the current iterate
    direction:          the direction along which to perform the linesearch
    eps:                the maximum allowed error in the resulting stepsize t
    maximum_iterations: the maximum allowed number of iterations
    k:                  k is the iteration counter that is unused
    """

    x = np.asarray(x)
    direction = np.asarray(direction)

    if eps <= 0:
        raise ValueError("Epsilon must be positive")

    _, gradient = func(x, 1)
    gradient = np.asarray(gradient)

    # checking that the given direction is indeed a descent direction
    if np.vdot(direction, gradient) >= 0:
        return 0

    else:

        # setting an upper bound on the optimum.
        MIN_t = 0
        MAX_t = 1
        iterations = 0

        value, gradient = func(x + MAX_t * direction, 1)
        value = np.double(value)
        gradient = np.asarray(gradient)

        #  increase Max_t to pin down the search region
        while np.vdot(direction, gradient) < 0:

            MAX_t *= 2

            value, gradient = func(x + MAX_t * direction, 1)
            gradient = np.asarray(gradient)

            iterations += 1

            if iterations >= maximum_iterations:
                raise ValueError("Too many iterations")

        # bisection search in the interval (MIN_t, MAX_t)
        iterations = 0

        while True:

            t = (MAX_t + MIN_t) / 2

            value, gradient = func(x + t * direction, 1)
            value = np.double(value)
            gradient = np.asarray(gradient)

            suboptimality = abs(np.vdot(direction, gradient)) * (MAX_t - t)

            if suboptimality <= eps:
                break

            if np.vdot(direction, gradient) < 0:
                MIN_t = t
            else:
                MAX_t = t

            iterations += 1
            if iterations >= maximum_iterations:
                raise ValueError("Too many iterations")

        return t


###############################################################################
def backtracking(func, x, direction, k, alpha=0.4, beta=0.9, maximum_iterations=65536):
    """
    Backtracking linesearch
    func:               the function to optimize It is called as "value, gradient = func( x, 1 )
    x:                  the current iterate
    direction:          the direction along which to perform the linesearch
    k:                  k is the iteration counter that is unused
    alpha:              the alpha parameter to backtracking linesearch
    beta:               the beta parameter to backtracking linesearch
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

    x = np.asarray(x)
    direction = np.asarray(direction)

    value, gradient = func(x, 1)
    value = np.double(value)
    gradient = np.asarray(gradient)

    derivative = np.vdot(direction, gradient)

    # checking that the given direction is indeed a descent direction
    if derivative >= 0:
        return 0

    else:
        t = 1
        iterations = 0
        while True:

            # if (TODO: TERMINATION CRITERION): break
            if func( x + t * direction, 0 ) <= ( value + alpha * t * derivative):
                break

            # t = TODO: BACKTRACKING LINE SEARCH
            t *= beta 

            iterations += 1
            if iterations >= maximum_iterations:
                raise ValueError("Too many iterations")

        return t


###############################################################################
def poly(func, x, direction, k, coefficient=1, exponent=0):
    """
        Polynomially decay stepsize
        func ,x,direction: These are unused argument since we are using deterministic stepsize
        k:                 Current iteration number
        coefficient:       stepsize is coefficient*T^{exponent}
        exponent:          stepsize is coefficient*T^{exponent}, exponent=0 corresponds to fixed stepsize
        """

    return coefficient * k ** exponent


# gradient_descent1 is for line search method: either bisection or backtracking

##############################################################################
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

        # if ( TODO: TERMINATION CRITERION ): break
        if np.vdot( gradient, gradient ) <= eps:        
            break  
        # direction = TODO: GRADIENT DESCENT UPDATE Direction
        direction = - gradient

        t = linesearch(func, x, direction, iterations, *linesearch_args)

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
        newton_decrement = np.sqrt(gradient.T @ np.linalg.inv(hessian) @ gradient)
        if newton_decrement <= np.sqrt(eps):
            break

        t = linesearch(func, x, direction, iterations, *linesearch_args)

        x += t * np.asarray( direction )

        iterations += 1
        if iterations >= maximum_iterations:
            raise ValueError("Too many iterations")

    return (x, values, runtimes, xs)


###############################################################################

def cg(func, initial_x, eps=1e-5, maximum_iterations=65536, linesearch=bisection, *linesearch_args  ):
    """
    Conjugate Gradient
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
    m = len( initial_x )
    iterations = 0
    direction = np.asmatrix( np.zeros( initial_x.shape ) )
    old_gradient=1

    # conjugate gradient updates
    while True:

        value, gradient = func( x , 1 )
        value = np.double( value )
        gradient = np.asarray( gradient )

        # updating the logs
        values.append( value )
        runtimes.append( time.time() - start_time )
        xs.append( x.copy() )

        # if ( TODO: TERMINATION CRITERION ): break
        if(np.linalg.norm(gradient)<=eps):
            break
        # beta = TODO: UPDATE BETA

        beta = np.vdot(gradient,gradient)/np.vdot(old_gradient,old_gradient)
        # reset after #(dimensions) iterations
        if iterations % m == 0:
            beta = 0

        # direction = TODO: FLETCHER-REEVES CONJUGATE GRADIENT UPDATE

        direction = -gradient + beta*direction
        t = linesearch(func, x, direction, iterations, *linesearch_args)

        x += t * direction

        # update old gradient
        old_gradient = np.asarray( gradient.copy() )

        iterations += 1
        if iterations >= maximum_iterations:
            raise ValueError("Too many iterations")

    return (x, values, runtimes, xs)

###############################################################################
def bfgs( func, initial_x, initial_inverse_hessian=1, eps=1e-5, maximum_iterations=65536, linesearch=bisection, *linesearch_args  ):
    """
    BFGS
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

    if( np.isscalar( initial_inverse_hessian ) ):
        inverse_hessian = initial_inverse_hessian
    else:
        inverse_hessian = np.asmatrix( initial_inverse_hessian.copy() )

    # initialization
    values = []
    runtimes = []
    xs = []
    start_time = time.time()
    m = len( initial_x )
    iterations = 0
    old_x = np.zeros( initial_x.shape )
    old_gradient = np.zeros( initial_x.shape )
    direction = np.zeros( initial_x.shape )

    # BFGS gradient updates
    while True:

        value, gradient = func( x , 1 )
        value = np.double( value )
        gradient = np.asarray( gradient )

        # updating the logs
        values.append( value )
        runtimes.append( time.time() - start_time )
        xs.append( x.copy() )

        # termination criterion
        if np.vdot( gradient, gradient ) <= eps:
            break

        # BFGS: estimating the hessian
        if iterations > 0:
            s = x - old_x
            y = gradient - old_gradient
            tau = 1.0 / np.vdot( y, s )
            inverse_hessian = np.asmatrix( np.eye(m) - tau * np.outer( s, y ) ) * inverse_hessian * np.asmatrix( np.eye(m) - tau * np.outer( y, s ) ) + np.asmatrix( tau * np.outer( s, s) )
        old_x = x.copy()
        old_gradient = gradient.copy()


        # direction of update
        gradient = np.matrix( gradient )
        direction = - np.asarray( inverse_hessian * gradient )

        t = linesearch(func, x, direction, iterations, *linesearch_args)

        x += t * direction

        iterations += 1
        if iterations >= maximum_iterations:
            raise ValueError("Too many iterations")

    return (x, values, runtimes, xs)
