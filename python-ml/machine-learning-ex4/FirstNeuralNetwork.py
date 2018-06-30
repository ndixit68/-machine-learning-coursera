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
    #print "Y.shape =", Y.shape
    #print "Y.type = ", type(Y)
    #print "Y.property = ", property(Y)
    new_y = np.zeros(shape=(m, num_labels))
    #print "New_Y.shape =", new_y.shape
    #print "New_Y.type = ", type(new_y)
    #print "New_Y.property = ", property(new_y)
    for i in range(0, int(Y.shape[0])):
        new_y[i, int((Y[i,:]-1))] = 1.0
    cost = np.sum((np.multiply(new_y, np.log(h_theta)) + np.multiply((1 - new_y), np.log(1 - h_theta))),1)
    theta1ExcludingBias = Theta1[:, 2:]
    theta2ExcludingBias = Theta2[:, 2:]
    reg1 = np.sum(np.sum(np.square(theta1ExcludingBias)))
    reg2 = np.sum(np.sum(np.square(theta2ExcludingBias)))
    regularization_term = lmbd * (reg1+reg2) / (2 * m)
    J = -np.sum(cost,0) / m + regularization_term
    grad = backpropagation(Theta1,Theta2,m,X,Y,new_y,h_theta)
    return J, grad

def sigmoidgradient(z):
    g = np.zeros(shape=(z.shape))
    A = sigmoid(z)
    g = np.multiply(A,(1-A))
    return g


def backpropagation(Theta1, Theta2, m, X, Y, new_y, h_theta):
    capitaldelta1 = np.zeros(shape=(Theta1.shape))
    capitaldelta2 = np.zeros(shape=(Theta2.shape))
    for i in range(0, m):    # for each sample
        # for input Layer l=1
        X1_ith = np.mat(X[i,:])
        A1_ith = np.append([[1]],X1_ith,1)
        # for hidden layer l=2
        Z2_ith = np.dot(A1_ith, Theta1.transpose())
        A2_ith = sigmoid(Z2_ith)
        A2_ith = np.append([[1]], A2_ith, 1)
        # for output layer l=3
        Z3_ith = np.dot(A2_ith, Theta2.transpose())
        A3_ith = sigmoid(Z3_ith)
        # for small delta values
        delta3_ith = A3_ith - new_y[i,:]
        delta2_ith = np.multiply((np.dot(Theta2.transpose(),delta3_ith.transpose())).transpose(),sigmoidgradient(np.append([[1]],Z2_ith,1).transpose())).transpose()
        delta2_ith = delta2_ith[1:]  #taking off the biased row
        capitaldelta2 = np.dot(capitaldelta2 + delta3_ith.transpose(), A2_ith.transpose())
        capitaldelta1 = np.dot(capitaldelta1 + delta2_ith.transpose(), A1_ith.transpose())
    theta1ExcludingBias = Theta1[:, 1:]
    theta2ExcludingBias = Theta2[:, 1:]
    Theta1ZeroedBias = np.append(np.zeros(shape=(Theta1.shape[0], 1)), theta1ExcludingBias,1)
    Theta2ZeroedBias = np.append(np.zeros(shape=(Theta2.shape[0], 1)), theta2ExcludingBias, 1)
    Theta1_grad = (1 / m) * capitaldelta1 + (lmbd / m) * Theta1ZeroedBias
    Theta2_grad = (1 / m) * capitaldelta2 + (lmbd / m) * Theta2ZeroedBias
    # Unroll gradients
    grad = np.append(Theta1_grad.flatten('F'), Theta2_grad.flatten('F'), axis=0)
    return grad


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


def randInitializeWeights(l_in, l_out):
    W = np.zeros(shape=(l_out, l_in+1))
    INIT_EPSILON = pow(10,-4)
    random_matrix = np.random.randint(low=0, high=1, size=(l_out, l_in+1))
    W = np.dot(random_matrix, (2 * INIT_EPSILON) - INIT_EPSILON)
    return W


def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros(shape=(fan_out, fan_in+1))
    random_matrix = np.reshape(np.sin(range(0,W.size,1)), newshape=(W.shape))/10
    return random_matrix


def checkNNGradients():
    lmbd = 0
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    # We generate some 'random'test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    X = debugInitializeWeights(m, input_layer_size - 1)
    Y = np.mat(1+np.mod(range(1,m+1),num_labels)).transpose()
    nn_params = np.append(Theta1.flatten('F'), Theta2.flatten('F'), axis=0)
    (cost, grad) = nnCostFunction(nn_params,input_layer_size,hidden_layer_size, num_labels, X, Y, lmbd)
    print grad
    return


if __name__ == "__main__":

    #Defining size of neural network
    input_layer_size = 400   # 20x20 Input Images of Digits
    hidden_layer_size = 25   # 25 hidden units
    num_labels = 10          # 10 labels, from 1 to 10   (note that we have mapped "0" to label 10)

    # check if the variables in the matlab file are not converted to csv , if not load data in matlab file
    data = check_n_load_dotmat_file("ex4data1.mat")

    # check some images from the dataset loaded
    X = data['X']   # m X number_of_features matrix
    Y = data['Y']   # 1 x m matrix
    Y = Y.transpose() # converting it into m x 1 matrix

    #print "Y.shape =", Y.shape
    #print "X.shape =", X.shape

    raw_input("Hit enter to continue and display 100 randomly picked images")
    # get function to display a few images

    display_sample_images(X, 100, [10, 10], [20, 20], 1)

    raw_input("Hit enter to continue and load parameters")

    print('\nLoading parameters.....')

    params = check_n_load_dotmat_file("ex4weights.mat")

    Theta1 = np.array(params['THETA1'])
    Theta2 = np.array(params['THETA2'])

    #print "Theta1 shape: " + str(Theta1.shape)
    #print "Theta2 shape: " + str(Theta2.shape)
    #print "Theta1 shape: " + str(Theta1.flatten('F').shape)
    #print "Theta2 shape: " + str(Theta2.flatten('F').shape)

    nn_params = np.append(Theta1.flatten('F'), Theta2.flatten('F'), axis=0)

    raw_input("Hit enter to calculate cost without regularization (lambda=0)")

    lmbd = 0

    J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lmbd)

    print 'Cost at parameters (loaded from ex4weights): ', J ,'\n(this value should be about 0.287629)\n'

    raw_input('\nProgram paused. Press enter to continue to calculate cost with regularization.\n')

    lmbd = 1

    J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lmbd)

    print 'Cost at parameters (loaded from ex4weights): ', J, '\n(this value should be about 0.383770)\n'

    raw_input('\nProgram paused. Press enter to continue to test sigmoidalGradient function\n')

    print '\nEvaluating sigmoid gradient...\n'

    g = sigmoidgradient(np.array([-1,-0.5, 0, 0.5, 1]))
    print 'Sigmoid gradient evaluated at [-1, -0.5, 0, 0.5, 1]:\n', g, '\n\n'

    print 'Congratulations!! you\'re now ready to train your Neural Network from Scratch'
    raw_input('Program paused. Press enter to continue to initialize theta.\n')

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    #print initial_Theta1.shape
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    #print initial_Theta2.shape

    initial_nn_params = np.append(initial_Theta1.flatten('F'), initial_Theta2.flatten('F'), axis=0)

    print '\nChecking Backpropagation... \n'

    checkNNGradients()




