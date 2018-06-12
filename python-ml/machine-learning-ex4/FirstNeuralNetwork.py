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


# function to display given number of example set images in given format(rows & Columns)

def display_sample_images(X, sample_size, sample_shape, image_resolution, pad):
    # validate imputs
    if sample_shape[0] * sample_shape[1] != sample_size:
        print "Sorry, sample size does not match with the sample shape"
        return
    elif image_resolution[0] * image_resolution[1] != X.shape[1]:
        print "Sorry, image resolution does not match with the image size"
        return
    random_indices = np.random.randint(low=0, high=X.shape[0], size=sample_size)
    sel = X[random_indices[:], :]
    image_height = image_resolution[0]
    image_width = image_resolution[1]
    sample_rows = sample_shape[0]
    sample_cols = sample_shape[1]
    # set up blank display frame
    display_frame = - np.ones(shape=((pad + image_height) * sample_rows + pad, (pad + image_width) * sample_cols + pad))
    # putting each patch in the display_frame
    current_image = 0
    for row in range(0, sample_rows):
        for col in range(0, sample_cols):
            max_val = np.amax(abs(sel[current_image, :]))
            display_frame[(image_height + pad) * row + 1: (image_height + pad) * (row + 1),
            (image_width + pad) * col + 1: (image_width + pad) * (1 + col)] \
                = np.reshape(sel[current_image, :], (image_height, image_width))
            current_image = current_image + 1
    plt.imshow(np.transpose(display_frame), origin='upper', cmap='gray')
    plt.show()
    return


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

# Theta is a an array of dimension (1,), X is a mXn matrix, Y = mX1 matrix lmbd is a float value

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lmbd):

    # unroll thetas

    Theta1 = np.reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)], newshape=(hidden_layer_size, (input_layer_size + 1)), order='F')
    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):], newshape=(num_labels, (hidden_layer_size + 1)), order='F')

    m = X.shape[0]
    Theta1_grad = np.zeros(shape=(Theta1.shape))
    Theta2_grad = np.zeros(shape=(Theta2.shape))

    # for input Layer l=1
    A1 = np.append(np.ones(shape=(m,1)),X, 1)


    # for hidden layer l=2
    Z2 = np.dot(A1, Theta1.transpose())
    A2 = sigmoid(Z2)
    A2 = np.append(np.ones(shape=(m,1)),A2, 1)

    # for output layer l=3
    Z3 = np.dot(A2, Theta2.transpose())
    h_theta = sigmoid(Z3)

    # convert y into m x K array
    new_y = np.zeros(shape=(m, num_labels))
    for i in range(0, Y.shape[0]):
        new_y[i, int(Y[i]%num_labels)] = 1.0

    #cost =

    J = np.sum(np.mean(-np.multiply(new_y.transpose(), np.log(h_theta)) - np.multiply((1 - new_y).transpose(), np.log(1 - h_theta))))

    #reg_term = lmbd * np.sum(np.square(theta)[1:theta.shape[0] + 1]) / (2.0 * X.shape[0])
    # mind that we don't have to include theta0
    #cost = J + reg_term
    return J


def sigmoid(X):
    return 1 / (1 + np.exp(- X))


# Theta is a an array of dimension (1,), X is a mXn matrix, Y = mX1 matrix lmbd is a float value

def GradientFunction(theta, X, Y, lmbd):
    hypothesis = sigmoid(np.dot(X, theta))
    error = hypothesis - Y.transpose()
    grad = np.dot(error, X) / Y.size
    reg_term = lmbd * theta[1:theta.shape[0] + 1] / np.float(X.shape[0])  # mind that we don't have to include theta0
    # print reg_term.shape # this should be one less than size of theta
    gradient = grad + np.append(np.array([0]), reg_term)
    return gradient.transpose()


if __name__ == "__main__":

    #Defining size of neural network
    input_layer_size = 400   # 20x20 Input Images of Digits
    hidden_layer_size = 25   # 25 hidden units
    num_labels = 10          # 10 labels, from 1 to 10   (note that we have mapped "0" to label 10)

    # check if the variables in the matlab file are not converted to csv , if not load data in matlab file
    data = check_n_load_dotmat_file("ex4data1.mat")

    # check some images from the dataset loaded
    X = data['X']
    Y = data['Y']

    raw_input("Hit enter to continue and display 100 randomly picked images")
    # get function to display a few images

    display_sample_images(X, 100, [10, 10], [20, 20], 1)

    raw_input("Hit enter to continue and load parameters")

    print('\nLoading parameters.....')

    params = check_n_load_dotmat_file("ex4weights.mat")

    Theta1 = np.array(params['THETA1'])
    Theta2 = np.array(params['THETA2'])

    print "Theta1 shape: " + str(Theta1.shape)
    print "Theta2 shape: " + str(Theta2.shape)
    print "Theta1 shape: " + str(Theta1.flatten('F').shape)
    print "Theta2 shape: " + str(Theta2.flatten('F').shape)

    nn_params = np.append(Theta1.flatten('F'), Theta2.flatten('F'), axis=0)

    raw_input("Hit enter to feed forward using Neural Network")

    lmbd = 0

    J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lmbd)
    print J



