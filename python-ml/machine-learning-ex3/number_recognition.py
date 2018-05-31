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

def CostFunction(theta, X, Y, lmbd):
    hypothesis = sigmoid(np.dot(X, theta))
    J = np.mean(
        -np.multiply(Y.transpose(), np.log(hypothesis)) - np.multiply((1 - Y).transpose(), np.log(1 - hypothesis)))
    reg_term = lmbd * np.sum(np.square(theta)[1:theta.shape[0] + 1]) / (2.0 * X.shape[0])
    # mind that we don't have to include theta0
    cost = J + reg_term
    return cost


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


def train_one_vs_all_classifier(X, Y, num_labels, lmbd):
    m = X.shape[0]
    n = X.shape[1]
    all_theta = np.zeros(shape=(num_labels, n + 1))
    X = np.append(np.ones(shape=(m, 1)), X, 1)
    initial_theta = np.zeros(n + 1)
    """
    For Function opt.fmin_tnc used in below for loop, parameters required are :
    ---> Function CostFunction should return cost of having the hypothesis function at a given theta, usually its a float type number
    ---> x0 is intial theta, it is expected in (n, ) shape, which does not include theta0
    ---> GradientFunction should return a (1,n) matrix
    ---> X is (m, n) shape martix
    ---> Y is (m, 1) shape matrix
    ---> lmbd is lambda, its generally a float type value
    """
    for classifier in range(1,
                            num_labels + 1):  # we're taking range one to ten because in zero is represented as 10 in dataset
        Y_temp = (Y == classifier).astype(int).transpose()
        temp = opt.fmin_tnc(func=CostFunction, x0=initial_theta, fprime=GradientFunction, args=(X, Y_temp, lmbd),
                            messages=0)
        if classifier == num_labels:
            all_theta[0, :] = temp[0]
        else:
            all_theta[classifier, :] = temp[0]
    return all_theta


def predict_one_vs_all(X, all_theta):
    m = X.shape[0]
    n = X.shape[1]
    X = np.append(np.ones(shape=(m, 1)), X, 1)
    p = np.zeros(shape=(m, 1))
    for ex_set in range(0, m):
        predicted_values = sigmoid(np.dot(X[ex_set, :], all_theta.transpose()))
        index_of_max = np.argmax(predicted_values)
        max_value = predicted_values[0, index_of_max]
        # print index_of_max, max_value
        if max_value >= 0.5:
            if index_of_max == 0:
                p[ex_set, :] = 10  # because in zero is represented as 10
            else:
                p[ex_set, :] = index_of_max
    return p


def predict_using_nn(theta1, theta2, X):
    m = X.shape[0]
    X = np.append(np.ones(shape=(m, 1)), X, 1)
    pred = np.zeros(shape=(m, 1))
    for ex_set in range (0, m):
        A1 = sigmoid(np.dot(X[ex_set,:], theta1.transpose()))
        A1 = np.append(np.ones(shape=(A1.shape[0],1)),A1,1)
        A2 = sigmoid(np.dot(A1,theta2.transpose()))
        index_of_max = np.argmax(A2)
        max_value = A2[0, index_of_max]
        if max_value >= 0.5:
            pred[ex_set, :] = index_of_max + 1
    return pred


if __name__ == "__main__":
    # check if the variables in the matlab file are not converted to csv , if not load data in matlab file
    data = check_n_load_dotmat_file("ex3data1.mat")

    # check some images from the dataset loaded

    X = data['X']
    Y = data['Y']

    raw_input("Hit enter to continue and display 100 randomly picked images")
    # get function to display a few images
    #display_sample_images(X, 100, [10, 10], [20, 20], 1)

    raw_input("Hit enter to continue and test the logistic regression")

    print('\nTesting CostFunction with regularization')

    theta_t = np.array([-2, -1, 1, 2])
    X_t = np.mat(np.append(np.ones(shape=(5, 1)), np.reshape(range(1, 16), (5, 3), 1) / 10.0, 1))
    y_t = np.mat([[1], [0], [1], [0], [1]])
    lambda_t = 3

    J = CostFunction(theta_t, X_t, y_t, lambda_t)
    grad = GradientFunction(theta_t, X_t, y_t, lambda_t)

    print 'Cost: ', J
    print 'Expected cost: 2.534819\n'
    print 'Gradients: ', grad,
    print 'Expected gradients:\n'
    print ' 0.146561\n -0.548558\n 0.724722\n 1.398003\n'

    raw_input('Program paused. Press enter to continue to train a classifier')

    # train one vs all classifier using logistic regression
    num_labels = 10
    lmbd = 0.00001
    print "Training Classifiers using Truncated-Newton Algorithm in C wrapper....\n"
    all_theta = train_one_vs_all_classifier(X, Y, num_labels, lmbd)

    print "Classifiers Trained\n"
    raw_input('Press enter to continue to check accuracy\n')

    # predict the values using predictedvalue trained thetas to calculate accuracy
    p = predict_one_vs_all(X, all_theta)
    print "Training set accuracy :", np.mean(np.double(p == np.transpose(Y))) * 100

    # create a neural network using given set of thetas

    input_layer = 400
    hidden_layer1 = 25
    num_label = 10                           # represents the output layer as well
    theta1 = np.zeros(shape=(hidden_layer1, input_layer+1))
    theta2 = np.zeros(shape=(num_label, hidden_layer1+1))
    weights = check_n_load_dotmat_file("ex3weights.mat")
    theta1 = weights['THETA1']
    theta2 = weights['THETA2']

    # predict the values for each example set using neural network
    pred = predict_using_nn(theta1, theta2, X)
    print "Training set accuracy using neural network :", np.mean(np.double(pred == np.transpose(Y)))*100

    raw_input('Press enter to continue to check and view random images and their values\n')
    random_indices = np.random.randint(low=0, high=X.shape[0], size=X.shape[0])

    for i in range (0,X.shape[0]):
        print "Displaying image at "+ str(random_indices[i]) + " row"
        display_sample_images(X[random_indices[i],:],1,[1,1],[20,20],1)
        predicted_number = predict_using_nn(theta1, theta2, X[random_indices[i],:])
        print "number predicted by neural network = " + str(predicted_number%10)

        action = raw_input("Press Enter to continue predicting, Press q to quit program")
        if action == 'q':
            break




