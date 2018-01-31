import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

positives = np.nonzero(Y==1)[0] # indices of positive integers
negatives = np.nonzero(Y==0)[0] # indices of negative integers

fig, ax = plt.subplots()
ax.plot(X[positives, 0],X[positives, 1], marker='+', linestyle='')
ax.plot(X[negatives, 0],X[negatives, 1], marker='o', linestyle='')
plt.legend()
leg = ax.legend(handles=[mpatches.Patch(color='green', label='Admitted'), mpatches.Patch(label='Rejected')])

ax.set_xlabel('Exam Score 1')
ax.set_ylabel('Exam Score 2')
plt.show()
