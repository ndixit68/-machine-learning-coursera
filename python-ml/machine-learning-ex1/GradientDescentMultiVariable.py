import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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
data = np.mat(np.genfromtxt('ex1data2.txt', delimiter=','))  # loading data into an array
print 'Shape of array, or we can say dimension of matrix stored = ', data.shape

# loading the variables from the data matrix

X = data[:, 0:data.shape[1]-1]  # loading X into an m x n matrix

Y = data[:, 1]  # loading X
Y = np.reshape(Y, (Y.size, 1))  # reshaping the array to a n x 1

