import cvxpy as cp
import numpy as np

means = np.asmatrix(np.genfromtxt('crypto_means.csv', delimiter=',').reshape(100,1))
covs = np.asmatrix(np.genfromtxt('crypto_covs.csv', delimiter=',').reshape(100,100))
names = []
with open('crypto_names.csv', 'r') as f:
    for line in f.readlines():
        names.append(line.strip())

### YOUR CODE HERE ###
# Generate a random non-trivial quadratic program.

n = 100



# Define and solve the CVXPY problem.
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(-means.T @ x), [x >= 0,x <=1, cp.quad_form(x, covs) <= 1])
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution p is")
print(x.value)