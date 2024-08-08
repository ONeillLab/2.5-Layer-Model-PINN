import numpy as np
import math
import time
import example_data as ed
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

control = np.load("zeta2.npy")
test = np.load("zeta2ML.npy")

fig = plt.figure()
ax = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
im = ax.imshow(control, cmap="bwr")
im2 = ax2.imshow(test, cmap = "bwr")
im3 = ax3.imshow(control-test, cmap="bwr")
plt.show()