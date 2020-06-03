import importlib
import sys
import matplotlib.pyplot as plt
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!
import matplotlib



# Clearing all variables from python interpreter
def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue
        del globals()[var]


clear_all();

# loading Data from the input file
data = np.genfromtxt('ex1data1.txt', delimiter=',')  # loading data into an array
print 'Shape of array, or we can say dimension of matrix stored = ', data.shape

# loading the variables from the data matrix

X = data[:, 0]  # loading X into an array
X = np.reshape(X, (X.size, 1))  # reshaping the array to a n x 1
Y = data[:, 1]  # loading X
Y = np.reshape(Y, (Y.size, 1))  # reshaping the array to a n x 1


# =======================Plotting Data =========================

# defining Plot data Function
def plotdata(X, Y, ):
    plt.plot(X, Y, marker='x', linestyle='')
    plt.legend
    plt.show()
    return


print('Plotting Data...\n')
plotdata(X, Y)

# =======================Implement Cost Function=================
# Some gradient descent settings

iterations = 1500;
alpha = 0.01;

# counting the number of sample
m = X.size;

# append a row of ones in X
X = np.append(np.ones(shape=(X.size, 1)), X, axis=1)

# defining theta
theta = np.array([[0], [0]])  # defining a column vector theta

print('\nTesting the cost function ...\n')


# compute and display initial cost
def ComputeCost(X, Y, theta, m):
    theta = np.mat(theta);
    X = np.mat(X);
    Y = np.mat(Y);
    J = sum((np.square((theta.transpose() * X.transpose()).transpose() - Y))) / (2 * m);
    return J


J = ComputeCost(X, Y, theta, m);

print 'With theta = [0 ; 0]..' + "\n" 'Cost computed = ' + str(J[0, 0]);
print 'Expected cost value (approx) 32.07';

# further testing of the cost function
J = ComputeCost(X, Y, [[-1], [2]], m)
print 'With theta = [-1 ; 2]..' + "\n" + 'Cost computed = ' + str(J[0, 0]);

print('Expected cost value (approx) 54.24');

raw_input("Hit enter to continue for gradient Descent")  # this will make user to input key before the program continues


# Compute the best theta by running gradient Descent

def GradientDescent(X, Y, theta, alpha, iterations, m):
    J_history = np.zeros(shape=(iterations, 1));  # to be able to plot convergence of cost
    temp = np.zeros(shape=(theta.size, iterations));
    theta = np.mat(theta);
    X = np.mat(X);
    Y = np.mat(Y);
    temp = np.mat(temp);

    print theta.size
    for iter_item in range(0, iterations):
        for theta_value in range(0, theta.shape[0]):
            temp[theta_value, iter_item] = theta[theta_value, :] - alpha * (
                    ((X[:, theta_value]).transpose()) * ((theta.transpose() * X.transpose()).transpose() - Y)) / m;  # no sum function here because the multiplication will automatically sum the  values

            # print theta_value
        theta = temp[:, iter_item]
        J_history[iter_item] = ComputeCost(X, Y, theta, m);
    return theta


print 'Running Gradient Descent to find best Theta'

theta = GradientDescent(X, Y, theta, alpha, iterations, m);
print '\n' + 'Theta found by gradient Descent: '
print theta;
print theta

# Plot the linear fit on the data.

fig, ax = plt.subplots()
ax.plot(data[:, 0], data[:, 1], 'rx', label="Training Data")
ax.plot(X[:, 1], np.array(X * theta), label="Linear Fit")
plt.legend();
leg = ax.legend();
plt.show()
#plt.close()

# Predict values for population sizes of 35,000 and 70,000

Predict1 = [1, 3.5] * theta
print 'For population = 35,000, we predict a profit of ', Predict1[0, 0]

Predict2 = [1, 7] * theta;
print 'For population = 70,000, we predict a profit of ', Predict2[0, 0]

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...' + "\n")

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100);
theta1_vals = np.linspace(-1, 4, 100);

# initialize J_vals to a matrix of 0's
J_vals = np.zeros(shape=(theta0_vals.size, theta1_vals.size));

# Fill out J_vals
for i in range(0, theta0_vals.size):
    for j in range(0, theta1_vals.size):
        t = [[theta0_vals[i]], [theta1_vals[j]]]
        J_vals[i, j] = ComputeCost(X, Y, t, m);

fig = plt.figure();
#ax = Axes3D(fig)
ax = fig.add_subplot(111, projection='3d')
X_axis, Y_axis = np.meshgrid(theta0_vals, theta1_vals)
Z_axis = J_vals.reshape(X_axis.shape)

ax.plot_surface(X_axis, Y_axis, Z_axis)
ax.set_xlabel('Theta_1')
ax.set_ylabel('Theta_2')
ax.set_zlabel('J(theta)')
plt.show()

raw_input(
    "Hit enter to continue to plot J_vals as 15 contours spaced logarithmically between 0.01 and 100");  # this will make user to input key before the program continues

# plot the contour

fig1 = plt.figure();
ax1 = fig1.add_subplot(111)
X_axis, Y_axis = np.meshgrid(theta0_vals, theta1_vals)
Z_axis = J_vals.reshape(X_axis.shape)

ax1.contour(X_axis, Y_axis, Z_axis, np.logspace(-2, 3, 20))
ax1.set_xlabel('Theta_1')
ax1.set_ylabel('Theta_2')
plt.show()
