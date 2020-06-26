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

def nnCostFunction(nn_params, args, lmbd=0):

    '''Function expects the parameters & arguments in below format
    input_layer_size, hidden_layer_size, num_labels, X, Y (mx1), lmbd
    '''

    input_layer_size, hidden_layer_size, num_labels, X, new_y = args

    # print "\n input layer = ",  input_layer_size
    # print "\n hidden layer = ",  hidden_layer_size
    # print "\n number label = ",  num_labels
    # print "\n Shape of X = ",  X.shape
    # print "\n Shape of Y = ",  new_y shape (m X num_label)
    # print "\n lambda = ",  lmbd


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

    #-new_y = np.zeros(shape=(m, num_labels))

    #-for i in range(0, int(Y.shape[0])):
    #-    new_y[i, int((Y[i,:]-1))] = 1.0

    # print "New_Y.shape =", new_y.shape
    # print "New_Y.type = ", type(new_y)
    # print "New_Y.property = ", property(new_y)

    cost = np.sum((np.multiply(new_y, np.log(h_theta)) + np.multiply((1 - new_y), np.log(1 - h_theta))),1)

    theta1ExcludingBias = Theta1[:, 1:]
    theta2ExcludingBias = Theta2[:, 1:]

    reg1 = np.sum(np.sum(np.square(theta1ExcludingBias)))
    reg2 = np.sum(np.sum(np.square(theta2ExcludingBias)))

    regularization_term = lmbd * (reg1+reg2) / (2 * m)

    J = -np.sum(cost,0) / m + regularization_term

    # print "\n This is the Cost:", J

    return J

def sigmoidgradient(z):
    g = np.zeros(shape=(z.shape))
    A = sigmoid(z)
    g = np.multiply(A,(1-A))
    return g


def backpropagation(Theta1, Theta2, m, X, new_y, lmbd=0):

    #delta2 = np.zeros(shape=(m, hidden_layer_size))
    #delta3 = np.zeros(shape=h_theta.shape)
    #delta3 = h_theta - new_y

    capitaldelta1 = np.zeros(shape=(Theta1.shape))
    capitaldelta2 = np.zeros(shape=(Theta2.shape))

    for i in range(0, m):    # for each sample
        # for input Layer l=1
        X1_ith = np.mat(X[i, :])
        A1_ith = np.append([[1]], X1_ith, 1)

        # for hidden layer l=2
        Z2_ith = np.dot(A1_ith, Theta1.transpose())
        A2_ith = sigmoid(Z2_ith)
        A2_ith = np.append([[1]], A2_ith, 1)

        # for output layer l=3
        Z3_ith = np.dot(A2_ith, Theta2.transpose())
        A3_ith = sigmoid(Z3_ith)

        # to verify the value of A3_ith for each item must be same as that calculated for hypothesis Function
        # print "\n \n This is value of A3_ith from backpropagation for ", i, "example set: \n",  A3_ith
        # print "\n \n This is the h_theta for ", i, "example set: \n", h_theta[i]

        # for small delta values
        delta3_ith = A3_ith - new_y[i,:]

        # print "\n \n Shape of Z2_ith \n", Z2_ith.shape
        # print "\n \n Value of Z2_ith after append \n", np.append([[1]], Z2_ith, 1)
        # print "\n \n Value of sigmoidal Gradient of Z2_ith after append \n", sigmoidgradient(np.append([[1]],Z2_ith,1))
        # print "\n \n Shape of delta3_ith \n", delta3_ith.shape
        # print "\n \n Shape of Theta2 \n", Theta2.shape

        delta2_ith = np.multiply((np.dot(delta3_ith, Theta2)), sigmoidgradient(np.append([[1]],Z2_ith,1)))
        delta2_ith = delta2_ith[:,1:]  # taking off the biased row
        capitaldelta2 = capitaldelta2 + np.dot(delta3_ith.transpose(), A2_ith)
        capitaldelta1 = capitaldelta1 + np.dot(delta2_ith.transpose(), A1_ith)

    #print "Size of CapitalDelta 1 :", capitaldelta1.shape
    #print "Size of CapitalDelta 2 :", capitaldelta2.shape

    #print "\n \n CapitalDelta 1 \n", capitaldelta1 # To ensure Capital Delta is getting some value calculated
    #print "\n \n CapitalDelta 2 \n ", capitaldelta2  # To ensure Capital Delta is getting some value calculated

    theta1ExcludingBias = Theta1[:, 1:]
    theta2ExcludingBias = Theta2[:, 1:]

    #print "theta1ExcludingBias \n \n", theta1ExcludingBias
    #print "\n \n theta2ExcludingBias  \n \n ", theta2ExcludingBias
    #print "Size of theta1ExcludingBias  :", theta1ExcludingBias.shape
    #print "Size of theta2ExcludingBias  :", theta2ExcludingBias.shape


    Theta1ZeroedBias = np.append(np.zeros(shape=(Theta1.shape[0], 1)), theta1ExcludingBias,1)
    Theta2ZeroedBias = np.append(np.zeros(shape=(Theta2.shape[0], 1)), theta2ExcludingBias, 1)

    #print "\n lambda = ",lmbd

    #print "\n \n Theta1ZeroedBias \n \n", Theta1ZeroedBias
    #print "\n \n Theta2ZeroedBias  \n \n ", Theta2ZeroedBias
    #print "Size of Theta1ZeroedBias  :", Theta1ZeroedBias.shape
    #print "Size of Theta2ZeroedBias  :", Theta2ZeroedBias.shape


    Theta1_grad = (1.0/m)*capitaldelta1 + (lmbd/m) * Theta1ZeroedBias
    Theta2_grad = (1.0/m)*capitaldelta2 + (lmbd/m) * Theta2ZeroedBias

    #print "Value of Theta1_grad \n \n", Theta1_grad
    #print "\n \n Value of Theta2_grad \n \n", Theta2_grad
    #print "\n \n Size of Theta1_grad  :", Theta1_grad.shape, type(Theta1_grad), np.array(Theta1_grad).flatten('F')
    #print "\n \n Size of Theta2_grad  :", Theta2_grad.shape, type(Theta2_grad), np.array(Theta2_grad).flatten('F')

    # Unroll gradients
    grad = np.append(np.array(Theta1_grad).flatten('F'), np.array(Theta2_grad).flatten('F'), axis=0)
    return grad


def sigmoid(X):
    return 1 / (1 + np.exp(- X))


# Theta is a an array of dimension (1,), X is a mXn matrix, Y = mX1 matrix lmbd is a float value

def GradientFunction(theta_unrolled, args, lmbd=0):

    '''Function expects the parameters & arguments in below format
    input_layer_size, hidden_layer_size, num_labels, X, Y, lmbd
    '''

    input_layer_size, hidden_layer_size, num_labels, X, new_y = args

    # print "\n input layer = ",  input_layer_size
    # print "\n hidden layer = ",  hidden_layer_size
    # print "\n number label = ",  num_labels
    # print "\n Shape of X = ",  X.shape
    # print "\n Shape of Y = ",  Y.shape
    # print "\n lambda = ",  lmbd

    m = X.shape[0]
    #-new_y = np.zeros(shape=(m, num_labels))
    #-for i in range(0, int(Y.shape[0])):
    #-    new_y[i, int((Y[i,:]-1))] = 1.0

    # roll thetas
    Theta1 = np.reshape(theta_unrolled[0:hidden_layer_size * (input_layer_size + 1)], newshape=(hidden_layer_size, (input_layer_size + 1)), order='F')
    Theta2 = np.reshape(theta_unrolled[(hidden_layer_size * (input_layer_size + 1)):], newshape=(num_labels, (hidden_layer_size + 1)), order='F')

    grad = backpropagation(Theta1, Theta2, m, X, new_y, lmbd)
    return grad


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


def computeNumericalGradient(J, theta, args):
    numgrad = np.zeros(shape=theta.shape)

    # initialize perturbation vector
    perturb = np.zeros(shape=theta.shape)

    e = 1e-4
    for p in range(0, theta.size):
        perturb[p] = e;
        loss1 = J(theta - perturb, args);
        loss2 = J(theta + perturb, args);
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2 * e);
        perturb[p] = 0;
    return numgrad


def checkNNGradients(lmbd=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    # We generate some 'random'test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to generate X
    X = debugInitializeWeights(m, input_layer_size - 1) #doing -1 because its python, indices starts from 0
    Y = np.mat(1+np.mod(range(1, m+1),num_labels))

    m = X.shape[0]
    new_y = np.zeros(shape=(m, num_labels))
    for i in range(0, int(Y.shape[1])):
        new_y[i, int(Y[:,i]-1)] = 1.0

    # UnRoll parameters
    #print "\n \n Shape of  Theta1 in checkNNGradients: ", Theta1.shape, type(Theta1), Theta1.flatten('F')
    #print "\n \n Shape of  Theta2 in checkNNGradients: ", Theta2.shape, type(Theta2), Theta2.flatten('F')

    nn_params = np.append(Theta1.flatten('F'), Theta2.flatten('F'), axis=0)

    #print "\n \n Shape of nn_params in checkNNGradients: ", nn_params.shape

    args = (input_layer_size,hidden_layer_size, num_labels, X, new_y)

    # cost = nnCostFunction(nn_params, args)
    grad = GradientFunction(nn_params, args)

    numgrad = computeNumericalGradient(nnCostFunction, nn_params, args)

    print "\n \n grad from Nural Network \n", grad.shape, "\n", grad
    print "\n \n printing numgrads:", numgrad.shape, "\n", numgrad

    return


def trainNN(initial_theta, args1, lmbd=0):

    '''Function expects the parameters & arguments in below format
    input_layer_size, hidden_layer_size, num_labels, X, Y, lmbd
    '''

    input_layer_size, hidden_layer_size, num_labels, X, Y = args
    m = X.shape[0]  # number of Examples
    n = X.shape[1]  # number of features = input_layer_size
    all_theta = np.zeros(shape=initial_theta.shape)
    X = np.append(np.ones(shape=(m, 1)), X, 1)

    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_tnc.html
    
    For Function opt.fmin_tnc used in below for loop, parameters required are :
    ---> Function CostFunction should return cost of having the hypothesis function at a given theta, usually its a float type number
    ---> x0 is intial theta, it is expected in (n, ) shape, which does not include theta0
    ---> GradientFunction should return a (1,n) matrix
    ---> X is (m, n) shape martix
    ---> Y is (m, 1) shape matrix
    ---> lmbd is lambda, its generally a float type value
    """
    all_theta = opt.fmin_tnc(func=nnCostFunction, x0=initial_theta, fprime=GradientFunction, args=((args1), lmbd), messages='MGS_ALL')
    return all_theta

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

    #Defining size of neural network
    input_layer_size = 400   # 20x20 Input Images of Digits
    hidden_layer_size = 25   # 25 hidden units
    num_labels = 10          # 10 labels, from 1 to 10   (note that we have mapped "0" to label 10)

    # check if the variables in the matlab file are not converted to csv , if not load data in matlab file
    data = check_n_load_dotmat_file("ex4data1.mat")

    # check some images from the dataset loaded
    X = data['X']   # m X number_of_features matrix
    Y = data['Y']   # 1 x m matrix

    #print "Y.shape =", Y.shape
    #print "X.shape =", X.shape

    m = X.shape[0]
    new_y = np.zeros(shape=(m, num_labels))
    for i in range(0, int(Y.shape[1])):
        new_y[i, int(Y[:,i]-1)] = 1.0


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

    args = (input_layer_size, hidden_layer_size, num_labels, X, new_y)

    J = nnCostFunction(nn_params, args, lmbd)

    print 'Cost at parameters (loaded from ex4weights): ', J ,'\n(this value should be about 0.287629)\n'

    raw_input('\nProgram paused. Press enter to continue to calculate cost with regularization.\n')

    lmbd = 1

    J = nnCostFunction(nn_params, args, lmbd)

    print 'Cost at parameters (loaded from ex4weights): ', J, '\n(this value should be about 0.383770)\n'

    raw_input('\nProgram paused. Press enter to continue to test sigmoidalGradient function\n')

    print '\nEvaluating sigmoid gradient...\n'

    g = sigmoidgradient(np.array([-1,-0.5, 0, 0.5, 1]))
    print 'Sigmoid gradient evaluated at [-1, -0.5, 0, 0.5, 1]:\n', g, '\n\n'

    print 'Congratulations!! you\'re now ready to train your Neural Network from Scratch'
    raw_input('Program paused. Press enter to continue to initialize theta.\n')


    print 'Initializing Neural Network Parameters ...'

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    print initial_Theta1.shape

    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
    print initial_Theta2.shape

    initial_nn_params = np.append(initial_Theta1.flatten('F'), initial_Theta2.flatten('F'), axis=0)

    print '\nChecking Backpropagation... \n'

    checkNNGradients(lmbd=0)

    print '\n Checking Backpropagation (w/ Regularization) ...\n'
    raw_input('Program paused. Press enter to continue to initialize theta.\n')

    checkNNGradients(lmbd=3)

    # Also output the costFunction debugging values
    debug_J = nnCostFunction(nn_params, args, 3)

    print '\n\nCost at (fixed) debugging parameters (w/ lambda = 3): %f \n ' \
          '(for lambda = 3, this value should be about 0.576051)\n\n' % (debug_J)


    raw_input('Program paused. Press enter to continue to start training Neural Network.\n')
    print '\n Training Neural Network...\n'

    lmbd = 0.25
    trained_params, num_of_evaluation, return_code = trainNN(initial_nn_params, args, lmbd)

    # unroll thetas
    Opt_Theta1 = np.reshape(trained_params[0:hidden_layer_size * (input_layer_size + 1)], newshape=(hidden_layer_size, (input_layer_size + 1)), order='F')
    Opt_Theta2 = np.reshape(trained_params[(hidden_layer_size * (input_layer_size + 1)):], newshape=(num_labels, (hidden_layer_size + 1)), order='F')

    print '\n \nVisualizing Neural Network... \n \n'

    # predict the values for each example set using neural network
    pred = predict_using_nn(Opt_Theta1, Opt_Theta2, X)
    print "Training set accuracy using neural network :", np.mean(np.double(pred == np.transpose(Y))) * 100

    raw_input('Press enter to continue to check and view random images and their values\n')
    random_indices = np.random.randint(low=0, high=X.shape[0], size=X.shape[0])

    for i in range(0, X.shape[0]):
        print "Displaying image at " + str(random_indices[i]) + " row"
        display_sample_images(X[random_indices[i], :], 1, [1, 1], [20, 20], 1)
        predicted_number = predict_using_nn(Opt_Theta1, Opt_Theta2, X[random_indices[i], :])
        print "number predicted by neural network = " + str(predicted_number % 10)

        action = raw_input("Press Enter to continue predicting, Press q to quit program")
        if action == 'q':
            break

    print nnCostFunction(nn_params, args, 1), nnCostFunction(trained_params, args, 1)









