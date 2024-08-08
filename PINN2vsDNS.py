import numpy as np
import tensorflow as tf
from keras import Sequential
from keras import layers as Layers
from keras import ops
from keras import Model
from name_list import *
import keras
from netCDF4 import Dataset
import time
from random import sample
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable


gridsize=N
# epoch 59/90, may want to lower LR
# lowered to 0.0001 after 100 epochs; was effective, might wanna lower again after 140; no it's still decreasing at 170

u1_inp = np.reshape(np.load("u1mat2.npy")[:1], (1, gridsize, gridsize))
v1_inp = np.reshape(np.load("v1mat2.npy")[:1], (1, gridsize, gridsize))
u2_inp = np.reshape(np.load("u2mat2.npy")[:1], (1, gridsize, gridsize))
v2_inp = np.reshape(np.load("v2mat2.npy")[:1], (1, gridsize, gridsize))
h1_inp = np.reshape(np.load("h1mat2.npy")[:1], (1, gridsize, gridsize))
h2_inp = np.reshape(np.load("h2mat2.npy")[:1], (1, gridsize, gridsize))
Wmatmat = np.load("Wmatmat2.npy")
Wmat_inp = np.reshape(Wmatmat[:1], (1, gridsize, gridsize))
model = keras.models.load_model("PINN.keras")
zeta2mat = []
u2mat = []

for i in range(1000):
    Wmat_inp = np.reshape(Wmatmat[i], (1, gridsize, gridsize))
    uvh = model(
            [u1_inp, v1_inp, u2_inp, v2_inp, h1_inp, h2_inp, Wmat_inp]
        )
    u1_out, v1_out, u2_out, v2_out, h1_out, h2_out = (
            np.reshape(uvh[:, :1], (1, gridsize, gridsize)),
            np.reshape(uvh[:, 1:2], (1, gridsize, gridsize)),
            np.reshape(uvh[:, 2:3], (1, gridsize, gridsize)),
            np.reshape(uvh[:, 3:4], (1, gridsize, gridsize)),
            np.reshape(uvh[:, 4:5], (1, gridsize, gridsize)),
            np.reshape(uvh[:, 5:], (1, gridsize, gridsize)),
        )
    #v2 = v2_inp.copy()
    u2 = u2_out.copy()
    #v2 = np.reshape(v2, (gridsize, gridsize))
    u2 = np.reshape(u2, (gridsize, gridsize))
    #zeta2 = 1 - Bt * rdist**2 + (1 / dx) * (v2 - v2[:,l] + u2[l,:] - u2)
    #zeta2mat.append(zeta2)
    u2mat.append(u2)
    u1_inp, v1_inp, u2_inp, v2_inp, h1_inp, h2_inp = u1_out, v1_out, u2_out, v2_out, h1_out, h2_out 

""" uvh = model(
            [u1_inp, v1_inp, u2_inp, v2_inp, h1_inp, h2_inp, Wmat_inp]
        )
u1_inp, v1_inp, u2_inp, v2_inp, h1_inp, h2_inp = (
        np.reshape(uvh[:, :1], (1, gridsize, gridsize)),
        np.reshape(uvh[:, 1:2], (1, gridsize, gridsize)),
        np.reshape(uvh[:, 2:3], (1, gridsize, gridsize)),
        np.reshape(uvh[:, 3:4], (1, gridsize, gridsize)),
        np.reshape(uvh[:, 4:5], (1, gridsize, gridsize)),
        np.reshape(uvh[:, 5:], (1, gridsize, gridsize)),
    )

v2 = v2_inp.copy()
u2 = u2_inp.copy()
v2 = np.reshape(v2, (gridsize, gridsize))
u2 = np.reshape(u2, (gridsize, gridsize))
zeta2 = 1 - Bt * rdist**2 + (1 / dx) * (v2 - v2[:,l] + u2[l,:] - u2) """


frames = u2mat
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(frames[0], cmap="bwr")

vminlist = []
vmaxlist = []
for j in frames:
    vminlist.append(np.min(j))
    vmaxlist.append(np.max(j))
vmin = np.min(vminlist)
vmax = np.max(vmaxlist)
print(vmin)
print(vmax)
im4 = ax.imshow(frames[0], cmap="bwr", vmin=vmin, vmax=vmax)
cb = fig.colorbar(im4)
tx4 = ax.set_title("main")

def animate(i):
    arr4 = frames[i] 
    vmax = np.max(arr4)
    vmin = np.min(arr4)
    tx4.set_text(f"time: {i}")
    im4.set_data(arr4)

print("animating")
ani = animation.FuncAnimation(fig, animate, interval=ani_interval, frames=len(frames))
plt.show()


""" fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(u2, cmap="bwr")
plt.show() """