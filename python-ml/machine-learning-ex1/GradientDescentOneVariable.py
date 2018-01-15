import numpy as np
import matplotlib.pyplot as plt

#defining Plot data Function
def plotdata(X,Y):
    plt.plot(X, Y, marker='x', linestyle='')
    plt.legend
    plt.show()

def computecost(X, Y, theta):
    m = X.size;
    J = sum((np.square((theta * X)- Y)))/(2*m)
    return J

# loading Data from the input file
data = np.genfromtxt('ex1data1.txt',delimiter=',') # loading data into an array
print 'Shape of array, or we can say dimension of matrix stored = ', data.shape

# loading the variables from the data matrix

X = data[:,0] #loading X
Y = data[:,1] #loading X

# m denotes the number of examples here, not the number of features
m = X.size

#=======================Plotting Data =========================
print('Plotting Data...\n')
plotdata(X,Y)
theta = np.array([0,0])


