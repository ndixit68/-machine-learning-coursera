import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import fmin_bfgs  # imports the BFGS algorithm to minimize
import numpy as np


# Clearing all variables from python interpreter
def clear_all():
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue
        del globals()[var]


clear_all();

# loading Data from the input file
data = np.mat(np.genfromtxt('ex2data1.txt', delimiter=','))

X = data[:, 0:data.shape[1] - 1]
Y = data[:, data.shape[1] - 1]

# Plotting Data

positives = np.nonzero(Y == 1)[0]  # indices of positive integers
negatives = np.nonzero(Y == 0)[0]  # indices of negative integers

fig, ax = plt.subplots(sharex=True, sharey=True)
ax.plot(X[positives, 0], X[positives, 1], marker='+', linestyle='', label='Admitted')
ax.plot(X[negatives, 0], X[negatives, 1], marker='o', linestyle='', label='Rejected')
ax.set_xlabel('Exam Score 1')
ax.set_ylabel('Exam Score 2')
plt.legend()
plt.show()

raw_input("Hit enter to continue for Logistic Regression")


# ============ Part 2: Compute Cost and Gradient ============
# define a function to calculate gradient and cost


def CostFunction(theta, X, Y):
    hypothesis = (theta.transpose() * X.transpose()).transpose()
    sigmoidal = 1 / (1 + np.exp(-hypothesis))
    J = -sum((np.multiply(Y, np.log(sigmoidal))) + (np.multiply((1 - Y), np.log(1 - sigmoidal)))) / m
    # grad = np.sum(np.multiply((sigmoidal - Y), X), axis=0) / m
    return J


def GradientFunction(theta, X, Y):
    hypothesis = (theta.transpose() * X.transpose()).transpose()
    sigmoidal = 1 / (1 + np.exp(-hypothesis))
    grad = np.sum(np.multiply((sigmoidal - Y), X), axis=0) / m
    return grad.transpose()


# fetching number of training sets and number of features
(m, n) = X.shape

# Adding intercept term to X
X = np.append(np.ones(shape=(X.shape[0], 1)), X, axis=1)

# define initial theta

initial_theta = np.zeros(shape=(X.shape[1], 1))

cost = CostFunction(initial_theta, X, Y)
print "Cost at initial theta (zeros):", str(cost)
print "Expected Cost (approx) : 0.693 "

grad = GradientFunction(initial_theta, X, Y)
print "Gradient at initial theta (zeros):"
print grad

test_theta = np.mat([[-24], [0.2], [0.2]])

cost = CostFunction(test_theta, X, Y)
print "Cost at initial theta (zeros):", str(cost)
print "Expected Cost (approx) : 0.218 "

grad = GradientFunction(test_theta, X, Y)
print "Gradient at initial theta (zeros): \n 0.043\n 2.566\n 2.647\n"
print str(grad)

raw_input("Hit enter to continue for optimization of cost in Logistic Regression")
# ============= Part 3: Optimizing using  BFGS =============

theta_opt = fmin_bfgs(CostFunction, initial_theta, fprime=GradientFunction, args=(X, Y) )
print theta_opt
