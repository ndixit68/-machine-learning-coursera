import cs as cs
import matplotlib
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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
data = np.mat(np.genfromtxt('ex1data2.txt', delimiter=','))  # loading data into an array
print 'Shape of array, or we can say dimension of matrix stored = ', data.shape

# loading the variables from the data matrix

X = data[:, 0:data.shape[1] - 1]  # loading X into an m x n matrix
Y = data[:, data.shape[1] - 1]  # loading Y into an m x n matrix

# counting the number of sample
m = X.shape[0];


# Defining Normalizing function
def NormalizeFeatures(X):
    sigma = X.std(0);
    mu = X.mean(0);
    X_norm = (X - mu) / sigma;
    return X_norm, mu, sigma


print("Normalizing Features...")

X, mu, sigma = NormalizeFeatures(X)
# append a row of ones in X
X = np.append(np.ones(shape=(X.shape[0], 1)), X, axis=1)

print('\nTesting the cost function ...\n')


# compute and display initial cost
def ComputeCost(X, Y, theta, m):
    J = sum((np.square((theta.transpose() * X.transpose()).transpose() - Y))) / (2 * m);
    return J


def GradientDescent(X, Y, theta, alpha, iterations, m):
    J_history = np.zeros(shape=(iterations, 1));
    temp = np.zeros(shape=(theta.size, iterations));
    temp = np.mat(temp);

    for iter_item in range(0, iterations):
        for theta_value in range(0, theta.shape[0]):
            temp[theta_value, iter_item] = theta[theta_value, :] - alpha * (
                ((X[:, theta_value]).transpose()) * ((theta.transpose() * X.transpose()).transpose() - Y)) / m;
        theta = temp[:, iter_item]
        J_history[iter_item] = ComputeCost(X, Y, theta, m);
    return theta, J_history


# Some gradient descent settings
iterations = 400;
alpha = 0.01;

# defining theta
theta = np.zeros(shape=(X.shape[1], 1));  # defining a column vector of length same as number of features

print 'Running Gradient Descent to find best Theta'

theta, J_history = GradientDescent(X, Y, theta, alpha, iterations, m);

print '\n' + 'Theta found by gradient Descent: '
print theta;

fig, ax = plt.subplots()
ax.plot(range(0, J_history.shape[0]), J_history[:, 0], label="Gradient Descent")
plt.legend();
leg = ax.legend();
ax.set_xlabel('Iterations')
ax.set_ylabel('J(theta)')
plt.show()

# Do the prediction

prediction = np.mat([1650, 3]);
prediction = (prediction - mu) / sigma;
prediction = np.append(np.ones(shape=(1, 1)), prediction)

predicted_price = prediction * theta

print "Predicted price for a 1650 sq-ft, 3 br house is", str(predicted_price[0, 0])

