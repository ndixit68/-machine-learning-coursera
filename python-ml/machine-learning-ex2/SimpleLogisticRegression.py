import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.optimize as opt
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

def plot_data(X, Y):
    positives = np.nonzero(Y == 1)[0]  # indices of positive integers
    negatives = np.nonzero(Y == 0)[0]  # indices of negative integers

    fig, ax = plt.subplots(sharex=True, sharey=True)
    ax.plot(X[positives, 0], X[positives, 1], marker='+', linestyle='', label='Admitted')
    ax.plot(X[negatives, 0], X[negatives, 1], marker='o', linestyle='', label='Rejected')
    ax.set_xlabel('Exam Score 1')
    ax.set_ylabel('Exam Score 2')
    plt.legend()
    #plt.show()
    return fig, ax


#fig, ax = plot_data(X, Y)

#raw_input("Hit enter to continue for Logistic Regression")


# ============ Part 2: Compute Cost and Gradient ============
# define a function to calculate gradient and cost

def sigmoid(X):
    return 1 / (1 + np.exp(- X))


def CostFunction(theta, X, Y):
    hypothesis = sigmoid(np.dot(X, theta))
    J = -np.multiply(Y.transpose(), np.log(hypothesis)) - np.multiply((1 - Y).transpose(), np.log(1 - hypothesis))
    return J.mean()


def GradientFunction(theta, X, Y):
    hypothesis = sigmoid(np.dot(X, theta))
    error = hypothesis - Y.transpose()
    grad = np.dot(error, X) / Y.size
    return grad


def PlotDecisonBoundary(theta, X, Y):
    fig, ax = plot_data(X[:, 1:3], Y)

    if X.shape[1] == 3:
        plot_x = np.array([min(X[:, 1])[0,0], max(X[:, 1])[0,0]])
        plot_y = (-1/theta[2])*(theta[1]*plot_x + theta[0])
        ax.plot(plot_x, plot_y)

    else:
        

    plt.show()

    return


# fetching number of training sets and number of features
(m, n) = X.shape

# Adding intercept term to X
X = np.append(np.ones(shape=(X.shape[0], 1)), X, axis=1)

# define initial theta

initial_theta = np.zeros(shape=(X.shape[1], 1))
initial_theta = np.zeros(X.shape[1])

cost = CostFunction(initial_theta, X, Y)
print "Cost at initial theta (zeros):", str(cost)
print "Expected Cost (approx) : 0.693 "

grad = GradientFunction(initial_theta, X, Y)
print "Gradient at initial theta (zeros):"
print grad

test_theta = np.array([-24, 0.2, 0.2])

cost = CostFunction(test_theta, X, Y)
print "Cost at initial theta (zeros):", str(cost)
print "Expected Cost (approx) : 0.218 "

grad = GradientFunction(test_theta, X, Y)
print "Gradient at initial theta (zeros): \n 0.043\n 2.566\n 2.647\n"
print str(grad)

#raw_input("Hit enter to continue for optimization of cost in Logistic Regression")

# ============= Part 3: Optimizing using fmin_tnc =============

theta_opt = opt.fmin_tnc(func=CostFunction, x0=test_theta, fprime=GradientFunction, args=(X, Y), messages=0)
print "Optimized theta", theta_opt[0]
print "Cost with Optimized theta", CostFunction(theta_opt[0], X, Y)

# ============== Part 4: Predict and Accuracies ==============

PlotDecisonBoundary(theta_opt[0], X, Y)
