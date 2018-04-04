import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

def display_sample_images(X, sample_size, )

