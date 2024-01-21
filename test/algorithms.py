######
######  This file includes gradient descent methods
######

import time
import numpy as np


###############################################################################
def bisection(func, x, direction,T, eps=1e-9, maximum_iterations=65536):
    """ 
    'Exact' linesearch (using bisection method)
    func:               the function to optimize It is called as "value, gradient = func( x, 1 )
    x:                  the current iterate
    direction:          the direction along which to perform the linesearch
    eps:                the maximum allowed error in the resulting stepsize t
    maximum_iterations: the maximum allowed number of iterations
    T:                  T is the iteration counter that is unused
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
def backtracking(func, x, direction, T, alpha=0.4, beta=0.9, maximum_iterations=65536):
    """ 
    Backtracking linesearch
    func:               the function to optimize It is called as "value, gradient = func( x, 1 )
    x:                  the current iterate
    direction:          the direction along which to perform the linesearch
    alpha:              the alpha parameter to backtracking linesearch
    beta:               the beta parameter to backtracking linesearch
    eps:                the maximum allowed error in the resulting stepsize t
    maximum_iterations: the maximum allowed number of iterations
    T:                  T is the iteration counter that is unused
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
            # TO DO ################################check Armijo's

            if func(x+t*direction,1)[0] < value+alpha*t*derivative:
                break
            t *= beta

            iterations += 1
            if iterations >= maximum_iterations:
                raise ValueError("Too many iterations")

        return t


###############################################################################
def poly(func, x, direction, T, coefficient=1, exponent=0):
    """
        Polynomially decay stepsize
        func ,x,direction: These are unused argument since we are using deterministic stepsize
        T:                 Current iteration number
        coefficient:       stepsize is coefficient*T^{exponent}
        exponent:          stepsize is coefficient*T^{exponent}, exponent=0 corresponds to fixed stepsize
        """

    return coefficient * T ** exponent


# gradient_descent1 is for line search method: either bisection or backtracking

##############################################################################
def gradient_descent(func, initial_x, err=1e-5, maximum_iterations=65536,linesearch=bisection, *linesearch_args):
    """
    Gradient Descent Using different stepsize choices: backtracking or bisection or polynomially decay or fixed stepsize
    func:               the function to optimize It is called as "value, gradient = func( x, 1 )
    initial_x:          the starting point
    err:                the stopping criteria for the squared norm of the gradient
    maximum_iterations: the maximum allowed number of iterations
    linesearch:         the linesearch routine, possible choice are: bisection, backtracking, poly.
    *linesearch_args:   the extra arguments of linesearch routine, for bisection, these are: eps and
                        maximum_iterations, for backtracking, these are  alpha, beta,
                        maximum_iterations, for poly, these are coefficient, exponent. For fixed stepsize, just
                        set exponent=0, and coefficient=stepsize.
    """

    if err <= 0:
        raise ValueError("Err must be positive")
    x = np.asarray(initial_x.copy())

    # initialization the list that respectively contains objective vales, runtimes, x's at each iteration
    values = []
    runtimes = []
    xs = []
    start_time = time.time()
    iterations = 1

    # gradient updates
    while True:

        value, gradient = func(x, 1)
        value = np.double(value)
        gradient = np.asarray(gradient)

        # updating the logs
        values.append(value)
        runtimes.append(time.time() - start_time)
        xs.append(x.copy())

        #if (TODO: Check Termination Criterion): ###################
        # if np.dot(np.transpose(gradient),gradient)<err:
        if np.linalg.norm(gradient)**2<= err:
            break

        #direction =  TODO: Search Direction #######################
        direction = -gradient

        t = linesearch(func, x, direction, iterations, *linesearch_args)

        x += t * direction

        iterations += 1
        if iterations >= maximum_iterations:
            print("Too many iterations")
            break

    return (x, values, runtimes, xs)



