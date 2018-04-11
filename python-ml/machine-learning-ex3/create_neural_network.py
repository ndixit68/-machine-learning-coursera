import number_recognition as nr
import matplotlib.pyplot as plt
import numpy as np


def load_weights():
    return predicted_values


if __name__=='__main__':
    # decide the layers
    input_layer = 400
    hidden_layer1 = 25
    num_label = 10                           # represents the output layer as well
    theta1 = np.zeros(shape=(hidden_layer1, input_layer+1))
    theta2 = np.zeros(shape=(num_label, hidden_layer1+1))
    lmbd = 0.00001

    XnY = nr.check_n_load_dotmat_file("ex3data1.mat")
    X = XnY['X']
    Y = XnY['Y']
    m = X.shape[1]

    weights = nr.check_n_load_dotmat_file("ex3weights.mat")
    theta1 = weights['THETA1']
    theta2 = weights['THETA2']








