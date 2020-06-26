import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io
import operator
import scipy.optimize as opt
from scipy.optimize import fmin_bfgs, minimize  # imports the BFGS algorithm to minimize
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Clearing all variables from python interpreter
def clear_all():
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue
        del globals()[var]

clear_all();


def check_n_load_dotmat_file(filename):
    data = scipy.io.loadmat(filename)
    returndict = {}
    count = 0
    for i in sorted(data.keys()):
        if os.path.exists(i.upper() + ".csv"):
            returndict[i.upper()] = np.mat(np.genfromtxt((i.upper() + ".csv"), delimiter=','))
            continue
        if '__' not in i and 'readme' not in i:
            np.savetxt((i.upper() + ".csv"), data[i], delimiter=',')
            print "Creating variable " + i.upper()
            returndict[i.upper()] = np.mat(np.genfromtxt((i.upper() + ".csv"), delimiter=','))
            count = count + 1
    variables = [item.strip().upper() for item in sorted(data.keys()) if "__" not in item]
    print len(variables).__str__() + " elements found in the matlab data file, " + str(count) + " loaded"
    return returndict

# Plotting Data

def plot_data(X, Y, xlabel, ylabel):
    fig, ax = plt.subplots(sharex=True, sharey=True)
    ax.plot(np.array(X[:,0].flatten()), np.array(Y[:,0].flatten()), marker='x')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend()
    plt.show()
    return fig, ax


def linearRegCostFunction(X, Y, theta, lmbd=0):
    h_theta = (theta.transpose() * X.transpose()).transpose()

    reg_term = lmbd * np.sum(np.square(theta)[1:theta.shape[0] + 1]) / (
                2.0 * X.shape[0])  # mind that we don't have to include theta0
    J = sum(np.square((h_theta - Y))) / (2 * m) + reg_term

    return J


if __name__ == "__main__":

    # check if the variables in the matlab file are not converted to csv , if not load data in matlab file
    data = check_n_load_dotmat_file("ex5data1.mat")

    # check some images from the dataset loaded
    X = np.transpose(data['X'])   # m X number_of_features matrix
    Y = np.transpose(data['Y'])   # 1 x m matrix
    Xval = np.transpose(data['XVAL']) # m se thodi badi matrix, basicaly X of validation test set
    yval = np.transpose(['YVAL']) # y of validation test set y
    Xtest = np.transpose(['XTEST']) # m se thodi badi matrix, basicaly X of validation test set
    ytest = np.transpose(['YTEST']) # y of validation test set y


    m = X.shape[0]

    raw_input("Hit enter to continue and display data")


    # get function to display a few images
    xlabel = 'Change in water level (x)'
    ylabel = 'Water flowing out of the dam (y)'
    plot_data(X, Y, xlabel, ylabel)

    raw_input("Hit enter to continue and test the Cost Function")


    # add a column to the left
    X = np.append(np.ones(shape=(m, 1)), X, 1)

    #initialize Theta
    theta = np.ones(X.shape[1])

    J = linearRegCostFunction(X, Y, theta, 1);
    print J
