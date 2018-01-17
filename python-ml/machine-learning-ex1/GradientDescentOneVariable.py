import numpy as np
import matplotlib.pyplot as plt

# loading Data from the input file
data = np.genfromtxt('ex1data1.txt', delimiter=',')  # loading data into an array
print 'Shape of array, or we can say dimension of matrix stored = ', data.shape

# loading the variables from the data matrix

X = data[:, 0]  # loading X into an array
X = np.reshape(X, (X.size, 1))  # reshaping the array to a n x 1
Y = data[:, 1]  # loading X
Y = np.reshape(Y, (Y.size, 1))  # reshaping the array to a n x 1

# m denotes the number of examples here, not the number of features
m = X.size


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

    J = sum((np.square((theta.transpose() * X.transpose()).transpose() - Y))) / (2*m);

    return J


J = ComputeCost(X, Y, theta, m);

print 'With theta = [0 ; 0]..' + "\n" 'Cost computed = ' + str(J[0,0]);
print 'Expected cost value (approx) 32.07' ;

# further testing of the cost function
J = ComputeCost(X, Y, [[-1], [2]], m)
print 'With theta = [-1 ; 2]..' + "\n" + 'Cost computed = ' + str(J[0,0]) ;

print('Expected cost value (approx) 54.24');

raw_input("Hit enter to continue for gradient Descent") # this will make user to input key before the program continues

# Compute the best theta by running gradient Descent

def GradientDescent(X, Y, theta, alpha, iterations, m):

    J_history =  np.zeros(shape=(iterations,1));
    temp = np.zeros(shape=(theta.size,iterations));
    for iter_item in range(1,iterations):
        for theta_value in theta:
            temp [theta_value,iterations] =  (theta[theta_value, :] - alpha * (sum(((theta.transpose()*X.transpose()).transpose - Y)*(X[:,theta_value]).transpose())/m));
        theta = temp[:, iterations]
        J_history[iter_item] = ComputeCost(X, Y, theta, m);


print 'Running Gradient Descent to find best Theta'

theta = GradientDescent(X, Y, theta, alpha, iterations, m);
