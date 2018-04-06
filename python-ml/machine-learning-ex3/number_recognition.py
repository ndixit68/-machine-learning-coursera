import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io
import scipy.optimize as opt
from scipy.optimize import fmin_bfgs  # imports the BFGS algorithm to minimize
import numpy as np


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
    #validate imputs
    if sample_shape[0]*sample_shape[1] != sample_size:
        print "Sorry, sample size does not match with the sample shape"
        return
    elif image_resolution[0]*image_resolution[1] != X.shape[1]:
        print "Sorry, image resolution does not match with the image size"
        return
    random_indices = np.random.randint(low=0, high=X.shape[0], size=sample_size)
    sel = X[random_indices[:],:]
    image_height = image_resolution[0]
    image_width = image_resolution[1]
    sample_rows = sample_shape[0]
    sample_cols = sample_shape[1]

    # set up blank display frame
    display_frame = - np.ones(shape=((pad + image_height)* sample_rows +pad, (pad + image_width)* sample_cols +pad))

    # putting each patch in the display_frame
    current_image = 0
    for row in range(0, sample_rows):
        for col in range(0, sample_cols):
            max_val = np.amax(abs(sel[current_image,:]))
            display_frame[(image_height + pad)*row +1 : (image_height+pad)*(row+1) , (image_width + pad)*col + 1  : (image_width+pad)*(1+col)]\
            = np.reshape(sel[current_image, :], (image_height,image_width))
            current_image = current_image + 1

    plt.imshow(np.transpose(display_frame), origin='upper',cmap='gray')
    plt.show()
    return


def check_n_load_dotmat_file():
    data = scipy.io.loadmat("ex3data1.mat")
    returndict = {}
    count = 0
    for i in sorted(data.keys()):
        if os.path.exists(i + ".csv"):
            returndict[i.upper()] = np.mat(np.genfromtxt((i + ".csv"), delimiter=','))
            continue
        if '__' not in i and 'readme' not in i:
            np.savetxt((i.upper() + ".csv"), data[i], delimiter=',')
            print "Creating variable " + i.upper()
            returndict[i.upper()] = np.mat(np.genfromtxt((i + ".csv"), delimiter=','))
            count = count+1

    variables = [item.strip().upper() for item in sorted(data.keys()) if "__" not in item]
    print len(variables).__str__() + " elements found in the matlab data file, " + str(count) + " loaded"

    return returndict

if __name__ == "__main__":
    # check if the variables in the matlab file are not converted to csv , if not load data in matlab file
    data = check_n_load_dotmat_file()

    # check some images from the dataset loaded
    X = data['X']
    Y = data['Y']

    # get function to display a few images
    display_sample_images(X, 100, [10,10], [20,20], 1)






