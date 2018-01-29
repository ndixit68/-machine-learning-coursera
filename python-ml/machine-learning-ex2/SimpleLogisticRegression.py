import matplotlib.pyplot as plt
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

# loading Data from the input file
data=np.mat(np.genfromtxt('ex2data1.txt', delimiter=','))

X = data[:,0:data.shape[1]-1]
Y = data[:,data.shape[1]-1]

# Plotting Data

fig, ax = plt.subplots()
ax.plot()