from math import exp
import numpy as np

# bisection gets handle `fun' to the oracle of a function f, performs
# bisection search on interval [MIN,MAX] and returns an 
# eps-suboptimal solution x; i.e. f(x)-f(x^*) <= eps .
# You only need to add three lines to the code:
#   1) upper bound the suboptimality of MID
#   2,3) update the search interval 

def bisection( fun, MIN, MAX, eps=1e-5 ):

  # counting the number of iterations
  counter = 0
  while True:
    counter +=1

    MID = ( MAX + MIN ) / 2

    # Oracle access to the function value and gradient
    gradient, value = fun( MID )

    # provide an upper bound for the suboptimality of MID in terms of
    # the magnitude of the gradient and distance from the optimum
    ###############################
    # TODO: suboptimality = ???
    ###############################

    if suboptimality <= eps:
      break

    if gradient > 0:
      ###############################
      # TODO: Updating the interval #
      ###############################
    else:
      ###############################
      # TODO: Updating the interval #
      ###############################

  print( 'Number of Iterations: %d' %counter )
  print( 'Suboptimal point: %1.15f' %MID )
  print( 'Suboptimal value: %1.15f' %value )
  return MID

###############################################################
# weird_func gets scalar x as input and returns f(x) and f'(x)

def weird_func( x ):

  # f(x) = x^4 + 6x^2 + 12(x-4)e^(x-1)
  value = pow(x, 4) + 6 * pow(x, 2) + 12 * (x - 4) * exp(x - 1)

  # f'(x) = 4x^3 + 12x + 12(x-3)e^(x-1)
  gradient = 4 * pow(x, 3) + 12 * x + 12 * (x - 3) * exp(x - 1)

  return (gradient,value)

###############################################################

# You may need to define other functions below






