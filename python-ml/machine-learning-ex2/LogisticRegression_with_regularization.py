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
data = np.mat(np.genfromtxt('ex2data2.txt', delimiter=','))

X = data[:, 0:data.shape[1] - 1]
Y = data[:, data.shape[1] - 1]

# degree of polynomial term - dopt
dopt = 6

# Set regularization parameter lambda
lmbd = 0.7

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
    # plt.show()
    return fig, ax


fig, ax = plot_data(X, Y)


# raw_input("Hit enter to continue for Logistic Regression")


# ============ Part 2: Compute Cost and Gradient ============
# define a function to calculate gradient and cost

def sigmoid(X):
    return 1 / (1 + np.exp(- X))


def CostFunction(theta, X, Y, lmbd):
    hypothesis = sigmoid(np.dot(X, theta))
    J = np.mean(
        -np.multiply(Y.transpose(), np.log(hypothesis)) - np.multiply((1 - Y).transpose(), np.log(1 - hypothesis)))
    reg_term = lmbd * np.sum(np.square(theta)[1:theta.shape[0] + 1]) / (
                2.0 * X.shape[0])  # mind that we don't have to include theta0
    cost = J + reg_term
    return cost


def GradientFunction(theta, X, Y, lmbd):
    hypothesis = sigmoid(np.dot(X, theta))
    error = hypothesis - Y.transpose()
    grad = np.dot(error, X) / Y.size
    reg_term = lmbd * theta[1:theta.shape[0] + 1] / np.float(X.shape[0])  # mind that we don't have to include theta0
    # print reg_term.shape # this should be one less than size of theta
    gradient = grad + np.append(np.array([0]), reg_term)
    # print np.append(np.array([0]),reg_term)
    return gradient


# mapping features = adding polynomial terms to the data set

def MapFeature(X1, X2, degree):
    Mapped_X = np.ones(shape=(X1.shape[0], 1))
    for i in range(1, degree):
        for j in range(0, i + 1):
            Mapped_X = np.append(Mapped_X, np.multiply(np.power(X1, i - j), np.power(X2, j)), axis=1)
    return Mapped_X


def PlotDecisonBoundary(theta, X, Y, degree=1):
    fig, ax = plot_data(X[:, 1:3], Y)

    if X.shape[1] == 3:
        plot_x1 = np.array([min(X[:, 1])[0, 0], max(X[:, 1])[0, 0]])
        plot_x2 = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
        ax.plot(plot_x1, plot_x2)
        plt.show()

    else:
        u = np.linspace(-1, 1.5, 50);
        v = np.linspace(-1, 1.5, 50);

        z = np.zeros(shape=(u.size, v.size))

        for i in range(0, u.size):
            for j in range(0, v.size):
                z[i, j] = np.dot(MapFeature(np.mat(u[i]), np.mat(v[j]), degree), theta)
        ax.contour(u, v, z, np.logspace(-2, -2, 1.99999))
        plt.show()

    return


def predict(theta, X):
    m = X.shape[0]
    p = np.zeros(shape=(m, 1))
    for iter in range(0, m):
        probability = float(sigmoid(np.dot(X[iter, :], theta)))
        if probability > 0.5:
            p[iter] = 1
        else:
            p[iter] = 0
    return p


# adding polynomial terms in data set
X = MapFeature(X[:, 0], X[:, 1], dopt)

# define initial theta
initial_theta = np.zeros(X.shape[1])

# ============= Part 1: Compute and display initial cost and gradient for regularized logistic regression =============

cost = CostFunction(initial_theta, X, Y, lmbd)
print "Cost at initial theta (zeros):", str(cost)
print "Expected Cost (approx) : 0.693 "

grad = GradientFunction(initial_theta, X, Y, lmbd)
print "Gradient at initial theta (zeros), first five thetas:"
print grad[0, 0:5]

test_theta = np.ones(X.shape[1])

cost = CostFunction(test_theta, X, Y, 10)
print "Cost at initial theta (ones):", str(cost)
print "Expected Cost (approx) : 3.16 "

grad = GradientFunction(test_theta, X, Y, 10)
print "Gradient at initial theta (ones), first five thetas:"
print grad[0, 0:5]

# ============= Part 2: Optimizing using fmin_tnc =============

theta_opt = opt.fmin_tnc(func=CostFunction, x0=test_theta, fprime=GradientFunction, args=(X, Y, lmbd), messages=0)
print "Optimized theta", theta_opt[0][1:5]
print "Cost with Optimized theta", CostFunction(theta_opt[0], X, Y, lmbd)

PlotDecisonBoundary(theta_opt[0], X, Y, dopt);

# Compute accuracy on our training set
p = predict(theta_opt[0], X);

print 'Train Accuracy:', np.mean(np.double(p == Y)) * 100
print ('Expected accuracy (with lambda = 1): 83.1 (approx)\n');
