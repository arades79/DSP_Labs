
#%% imports
import matplotlib.pyplot as plt
import numpy as np


#%% plot continuous (step 6)
plt.plot(np.exp(-np.arange(0, 11, 1e-3)))

#%% plot a lot of stuff (step 7)
continuous = np.exp(-np.arange(0, 11, 1e-3))

def difference_func():
    pass
f, axes = plt.subplot(2, 3)
[ax.plot() for ax in axes]

