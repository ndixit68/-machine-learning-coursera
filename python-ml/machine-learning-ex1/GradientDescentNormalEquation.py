import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy.linalg import inv


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
# append a row of ones in X
X = np.append(np.ones(shape=(X.shape[0], 1)), X, axis=1)
Y = data[:, data.shape[1] - 1]  # loading Y into an m x n matrix

# counting the number of sample
m = X.shape[0];


# ================ Implementing Normal Equations ================

def NormalEquation(X, Y):
    theta = np.zeros(shape=(X.shape[1], 1))
    theta = inv((X.transpose()) * X) * X.transpose() * Y;
    return theta


theta = NormalEquation(X, Y);
print "Theta calcualatedd by Normal Equation" "\n", theta;

# Do the prediction

prediction = np.mat([1650, 3]);
prediction = np.append(np.ones(shape=(1,1)),prediction)

predicted_price = prediction * theta

print "\n" "Predicted price for a 1650 sq-ft, 3 br house is", str(predicted_price[0,0])

