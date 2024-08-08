import numpy as np
import math
import time
import example_data as ed
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow as tf
from keras import Sequential
from keras import layers as Layers
from keras import ops
from keras import Model
import keras
from name_list import *


gridsize = 36
resolution = gridsize // 36
examples = np.load("examples.npy")
model = keras.models.load_model('model2.keras')
print(np.max(examples[50]))
print(np.min(examples[50]))



zeta2mat = []
u2mat = []

for i in range(199):
    print(i)
    slice = i
    timeslice = examples[slice]
    tinp = []
    xinp = []
    yinp = []
    Sinp = []
    for i in range(gridsize):
        for k in range(gridsize):
            tinp.append(slice)
            xinp.append(i/resolution)
            yinp.append(k/resolution)
            Sinp.append(timeslice[i//resolution,k//resolution])
    tinp = np.array(tinp)
    xinp = np.array(xinp)
    yinp = np.array(yinp)
    Sinp = np.array(Sinp)

    tinp = tinp.astype('float64') 
    xinp = xinp.astype('float64') 
    yinp = yinp.astype('float64') 
    Sinp = Sinp.astype('float64') 

    tinp, xinp, yinp, Sinp = (
            tf.convert_to_tensor(tinp, dtype=tf.float64),
            tf.convert_to_tensor(xinp, dtype=tf.float64),
            tf.convert_to_tensor(yinp, dtype=tf.float64),
            tf.convert_to_tensor(Sinp, dtype=tf.float64),
        )
    grid = np.zeros((gridsize,gridsize))
    with tf.GradientTape(persistent=True) as g:
        g.watch(tinp), g.watch(xinp), g.watch(yinp)
        uvh = model([tinp, xinp, yinp, Sinp], training=True)
        u2, v2, h2 = uvh[:, 3:4], uvh[:, 4:5], uvh[:, 5:]
    v2g, u2g, h2g = g.gradient(v2, xinp), g.gradient(u2, yinp), g.gradient(h2, tinp)
    u2g = np.reshape(u2g,(gridsize,gridsize))
    v2g = np.reshape(v2g, (gridsize,gridsize))
    h2g = np.reshape(h2g, (gridsize,gridsize))
    u2 = np.reshape(u2,(gridsize,gridsize))
    v2 = np.reshape(v2, (gridsize,gridsize))
    h2 = np.reshape(h2, (gridsize,gridsize))

    for i in range(gridsize):
        for k in range(gridsize):
            grid[i,k] = 1 - (Bt * np.sqrt((i ** 2) + (k ** 2))) + v2g[i,k] - u2g[i,k]

    zeta2mat.append(grid)
    u2mat.append(h2g)

fig = plt.figure()

frames4 = np.asarray(u2mat)
ax4 = fig.add_subplot(111)
cv4 = frames4[0]
vminlist = []
vmaxlist = []
for j in frames4:
    vminlist.append(np.min(j))
    vmaxlist.append(np.max(j))
vmin = np.min(vminlist)
vmax = np.max(vmaxlist)
im4 = ax4.imshow(cv4, cmap="bwr")#, vmin=vmin, vmax=vmax)
cb = fig.colorbar(im4)
tx4 = ax4.set_title("main")

def animate(i):
    arr4 = frames4[i] 
    #vmax = np.max(arr4)
    #vmin = np.min(arr4)
    tx4.set_text(f"time: {i}")
    im4.set_data(arr4)

print("animating")
ani = animation.FuncAnimation(fig, animate, interval=25, frames=len(frames4))
plt.show()



""" fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(grid, cmap="bwr")
np.save("zeta2ML.npy", grid)
plt.show() """



