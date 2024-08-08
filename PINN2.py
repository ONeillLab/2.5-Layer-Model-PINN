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
import helper_functions as hf

restart = True

tf.config.run_functions_eagerly(True)

gridsize = N

def build_model():
    u1_inp = Layers.Input(
        shape=(
            gridsize,
            gridsize,
        )
    )
    v1_inp = Layers.Input(
        shape=(
            gridsize,
            gridsize,
        )
    )
    u2_inp = Layers.Input(
        shape=(
            gridsize,
            gridsize,
        )
    )
    v2_inp = Layers.Input(
        shape=(
            gridsize,
            gridsize,
        )
    )
    h1_inp = Layers.Input(
        shape=(
            gridsize,
            gridsize,
        )
    )
    h2_inp = Layers.Input(
        shape=(
            gridsize,
            gridsize,
        )
    )
    Wmat_inp = Layers.Input(
        shape=(
            gridsize,
            gridsize,
        )
    )


    u1_inp_flat = Layers.Flatten()(u1_inp)
    v1_inp_flat = Layers.Flatten()(v1_inp)
    u2_inp_flat = Layers.Flatten()(u2_inp)
    v2_inp_flat = Layers.Flatten()(v2_inp)
    h1_inp_flat = Layers.Flatten()(h1_inp)
    h2_inp_flat = Layers.Flatten()(h2_inp)
    Wmat_inp_flat = Layers.Flatten()(Wmat_inp)

    hidden = Layers.Concatenate()(
        [
            u1_inp_flat,
            v1_inp_flat,
            u2_inp_flat,
            v2_inp_flat,
            h1_inp_flat,
            h2_inp_flat,
            Wmat_inp_flat,
        ]
    )
    for i in range(8):
        hidden = Layers.Dense(1024, activation="relu")(hidden)
    output = Layers.Dense(6 * gridsize * gridsize, activation="linear")(hidden)
    output_reshaped = Layers.Reshape((6, gridsize, gridsize))(output)

    model = Model(
        [u1_inp, v1_inp, u2_inp, v2_inp, h1_inp, h2_inp, Wmat_inp], output_reshaped
    )

    model.summary()

    return model


# tanh, tanh seems to work well 360->320 in 1000 epochs
# relu, relu works even better 360->280 in 1000 epochs
# relu, linear also works even even better 360->117 in 1000 epochs

# relu, linear for 10000 epochs loss: 0.1 at the lowest; stalling at 1 (adjust LR; overshooting)

# training size and batch size both 500

#

def viscND(vel):
    field = (
        20 * vel
        + 2 * tf.roll(tf.roll(vel, shift=1, axis=0), shift=1, axis=2)
        + 2 * tf.roll(tf.roll(vel, shift=1, axis=0), shift=-1, axis=2)
        + 2 * tf.roll(tf.roll(vel, shift=-1, axis=0), shift=1, axis=2)
        + 2 * tf.roll(tf.roll(vel, shift=-1, axis=0), shift=-1, axis=2)
        - 8 * tf.roll(vel, shift=1, axis=1)
        - 8 * tf.roll(vel, shift=-1, axis=1)
        - 8 * tf.roll(vel, shift=1, axis=2)
        - 8 * tf.roll(vel, shift=-1, axis=2)
        + tf.roll(vel, shift=2, axis=1)
        + tf.roll(vel, shift=-2, axis=1)
        + tf.roll(vel, shift=2, axis=2)
        + tf.roll(vel, shift=-2, axis=2)
    )
    field = -1 / Re * (1 / dx**4) * field

    return field


def BernN2(u1, v1, u2, v2, gm, c22h, c12h, h1, h2):
    B1p = (
            c12h * h1
            + c22h * h2
            + 0.25
            * (
                u1**2
                + tf.roll(u1, shift=-1, axis=2) ** 2
                + v1**2
                + tf.roll(v1, shift=-1, axis=1) ** 2
            )
        )
    
    B2p = (
            gm * c12h * h1
            + c22h * h2
            + 0.25
            * (
                u2**2
                + tf.roll(u2, shift=-1, axis=2) ** 2
                + v2**2
                + tf.roll(v2, shift=-1, axis=1) ** 2
            )
        )
    return B1p, B2p


def xflux(h1,u1):
    Fx1 = 0.5 * u1 * (
            tf.roll(h1, shift=1, axis=2) + h1
        )
    return Fx1


def yflux(h1,v1):
    Fy1 = 0.5 * v1 * (
            tf.roll(h1, shift=1, axis=2) + h1
        )
    return Fy1



def update_model(
    model, u1, v1, u2, v2, h1, h2, Wmat, channels
):
    with tf.GradientTape(persistent=True) as wghts:
        uvh = model(
            [u1, v1, u2, v2, h1, h2, Wmat], training=True
        )

        u1_out, v1_out, u2_out, v2_out, h1_out, h2_out = (
            tf.reshape(uvh[:, :1], (channels, gridsize, gridsize)),
            tf.reshape(uvh[:, 1:2], (channels, gridsize, gridsize)),
            tf.reshape(uvh[:, 2:3], (channels, gridsize, gridsize)),
            tf.reshape(uvh[:, 3:4], (channels, gridsize, gridsize)),
            tf.reshape(uvh[:, 4:5], (channels, gridsize, gridsize)),
            tf.reshape(uvh[:, 5:], (channels, gridsize, gridsize)),
        )

        u1_out, v1_out, u2_out, v2_out, h1_out, h2_out = (
            tf.cast(u1_out, dtype=tf.float64),
            tf.cast(v1_out, dtype=tf.float64),
            tf.cast(u2_out, dtype=tf.float64),
            tf.cast(v2_out, dtype=tf.float64),
            tf.cast(h1_out, dtype=tf.float64),
            tf.cast(h2_out, dtype=tf.float64),
        )
        
        u1_p = tf.identity(u1)
        v1_p = tf.identity(v1)
        h1_p = tf.identity(h1)
        u2_p = tf.identity(u2)
        v2_p = tf.identity(v2)
        h2_p = tf.identity(h2)


        for i in range(256):
            tmp = tf.identity(u1)
            u1 = 1.5 * u1 - 0.5 * u1_p
            u1_p = tmp  #
            tmp = tf.identity(u2)
            u2 = 1.5 * u2 - 0.5 * u2_p
            u2_p = tmp  #
            tmp = tf.identity(v1)
            v1 = 1.5 * v1 - 0.5 * v1_p
            v1_p = tmp
            tmp = tf.identity(v2)
            v2 = 1.5 * v2 - 0.5 * v2_p
            v2_p = tmp
            tmp = tf.identity(h1)
            h1 = 1.5 * h1 - 0.5 * h1_p
            h1_p = tmp
            tmp = tf.identity(h2)
            h2 = 1.5 * h2 - 0.5 * h2_p
            h2_p = tmp

            du1dt = viscND(u1)
            du2dt = viscND(u2)
            dv1dt = viscND(v1)
            dv2dt = viscND(v2)

            zeta1 = (
            1
            - Bt * rdist**2
            + (1 / dx)
            * (
                v1
                - tf.roll(v1, shift=1, axis=2)
                + tf.roll(u1, shift=1, axis=1)
                - u1
                )
            )

            zeta2 = (
            1
            - Bt * rdist**2
            + (1 / dx)
            * (
                v2
                - tf.roll(v2, shift=1, axis=2)
                + tf.roll(u2, shift=1, axis=1)
                - u2
                )
            )

            zv1 = zeta1 * (v1 + tf.roll(v1, shift=1, axis=2))
            zv2 = zeta2 * (v2 + tf.roll(v2, shift=1, axis=2))

            zu1 = zeta1 * (u1 + tf.roll(u1, shift=1, axis=1))
            zu2 = zeta2 * (u2 + tf.roll(u2, shift=1, axis=1))

            B1p, B2p = BernN2(u1, v1, u2, v2, gm, c22h, c12h, h1, h2)

            du1dtsq = (
            0.25 * (zv1 + tf.roll(zv1, shift=-1, axis=1))
            - (1 / dx) * (B1p - tf.roll(B1p, shift=1, axis=2))
            ) + du1dt

            du2dtsq = (
            0.25 * (zv2 + tf.roll(zv2, shift=-1, axis=1))
            - (1 / dx) * (B2p - tf.roll(B2p, shift=1, axis=2))
            ) + du2dt

            dv1dtsq = (
            dv1dt - 0.25 * (zu1 + tf.roll(zu1, shift=-1, axis=2))
            - (1 / dx) * (B1p - tf.roll(B1p, shift=1, axis=1))
            ) 

            dv2dtsq = (
            dv2dt - 0.25 * (zu2 + tf.roll(zu2, shift=-1, axis=2))
            - (1 / dx) * (B2p - tf.roll(B2p, shift=1, axis=1))
            ) 

            u1sq = u1_p + dt * du1dtsq
            u2sq = u2_p + dt * du2dtsq

            v1sq = v1_p + dt * dv1dtsq
            v2sq = v2_p + dt * dv2dtsq

            Fx1 = xflux(h1, u1) - kappa / dx * (h1 - tf.roll(h1, shift=1, axis=2))
            Fy1 = yflux(h1, v1) - kappa / dx * (h1 - tf.roll(h1, shift=1, axis=1))

            Fx2 = xflux(h2, u2) - kappa / dx * (h2 - tf.roll(h2, shift=1, axis=2))
            Fy2 = yflux(h2, v2) - kappa / dx * (h2 - tf.roll(h2, shift=1, axis=1))

            dh1dt = -(1 / dx) * (tf.roll(Fx1, shift=-1, axis=2) - Fx1 + tf.roll(Fy1, shift=-1, axis=1) - Fy1) - 1 / tradf * (h1 - 1) + Wmat
            dh2dt = -(1 / dx) * (tf.roll(Fx2, shift=-1, axis=2) - Fx2 + tf.roll(Fy2, shift=-1, axis=1) - Fy2) - 1 / tradf * (h1 - 1) - H1H2 * Wmat

            h1 = h1_p + dt * dh1dt
            h2 = h2_p + dt * dh2dt
            u1 = u1sq
            u2 = u2sq
            v1 = v1sq
            v2 = v2sq

        loss_u1 = tf.reduce_sum(tf.square(u1 - u1_out)) #_control
        loss_v1 = tf.reduce_sum(tf.square(v1 - v1_out))
        loss_u2 = tf.reduce_sum(tf.square(u2 - u2_out))
        loss_v2 = tf.reduce_sum(tf.square(v2 - v2_out))
        loss_h1 = tf.reduce_sum(tf.square(h1 - h1_out))
        loss_h2 = tf.reduce_sum(tf.square(h2 - h2_out))

        loss = loss_u1 + loss_v1 + loss_u2 + loss_v2 + loss_h1 + loss_h2

    gradloss = wghts.gradient(loss, model.trainable_variables)
    grads = [g for g in gradloss]

    return loss, grads


def train_model(model, u1_inp, v1_inp, u2_inp, v2_inp, h1_inp, h2_inp, Wmat_inp, epochs, batch_size, rate):
    opt = tf.keras.optimizers.Adam(rate) #learning_rate=0.001
    channels = np.shape(u1_inp)[0]
    batches = channels // batch_size

    for i in range(epochs):
        timer = time.perf_counter()
        epoch_loss = 0
        for k in range(batches):
            loss, grads = update_model(
                model,
                u1_inp[k * batch_size : k * batch_size + batch_size],
                v1_inp[k * batch_size : k * batch_size + batch_size],
                u2_inp[k * batch_size : k * batch_size + batch_size],
                v2_inp[k * batch_size : k * batch_size + batch_size],
                h1_inp[k * batch_size : k * batch_size + batch_size],
                h2_inp[k * batch_size : k * batch_size + batch_size],
                Wmat_inp[k * batch_size : k * batch_size + batch_size],
                batch_size,
            )
            epoch_loss += loss
            opt.apply_gradients(zip(grads, model.trainable_variables))
            # print(
            #    f"Batches completed: {k+1}, loss: {loss.numpy()}"
            # )
        print(
            f"Epochs completed: {i+1}, loss: {epoch_loss}, Time elapsed, {round(time.perf_counter()-timer, 3)}s"
        )
        timer = time.perf_counter()
    model.save("NPINNmain.keras", overwrite=True)
    return model


def run_sim(u1, u2, v1, v2, h1, h2, locs, lasttime):
    if restart_name == None:
        locs = hf.genlocs(num, N, 0) ### use genlocs instead of paircount
        
    mode = 1

    wlayer = hf.pairshapeN2(locs, lasttime)
    Wmat = hf.pairfieldN2(L, h1, wlayer)

    # TIME STEPPING
    if AB == 2:
        u1_p = u1.copy()
        v1_p = v1.copy()
        h1_p = h1.copy()
        u2_p = u2.copy()
        v2_p = v2.copy()
        h2_p = h2.copy()

    ts = []
    zeta2mat = []
    psi2 = np.zeros_like(x)
    zeta1 = psi2.copy()
    zeta2 = psi2.copy()
    B2 = psi2.copy()
    B1p = B2.copy()

    ii = 0

    KEmat = []
    APEmat = []
    u1mat = [] 
    v1mat = []
    u2mat = []
    v2mat = []
    h1mat = []
    h2mat = []
    Wmatmat = []

    timer = time.perf_counter()

        
    t = lasttime
    tc = round(t/dt)

    #####

    counter = 0
    while t <= tmax + lasttime + dt / 2:

        if AB == 2:
            tmp = u1.copy()
            u1 = 1.5 * u1 - 0.5 * u1_p
            u1_p = tmp  #
            tmp = u2.copy()
            u2 = 1.5 * u2 - 0.5 * u2_p
            u2_p = tmp  #
            tmp = v1.copy()
            v1 = 1.5 * v1 - 0.5 * v1_p
            v1_p = tmp
            tmp = v2.copy()
            v2 = 1.5 * v2 - 0.5 * v2_p
            v2_p = tmp
            tmp = h1.copy()
            h1 = 1.5 * h1 - 0.5 * h1_p
            h1_p = tmp
            if layers == 2.5:
                tmp = h2.copy()
                h2 = 1.5 * h2 - 0.5 * h2_p
                h2_p = tmp

        # add friction
        du1dt = hf.viscND(u1, Re, n)
        du2dt = hf.viscND(u2, Re, n)
        dv1dt = hf.viscND(v1, Re, n)
        dv2dt = hf.viscND(v2, Re, n)

        """ if spongedrag1 > 0:
            du1dt = du1dt - spdrag1 * (u1)
            du2dt = du2dt - spdrag2 * (u2)
            dv1dt = dv1dt - spdrag1 * (v1)
            dv2dt = dv2dt - spdrag2 * (v2) """

        # absolute vorticity
        zeta1 = 1 - Bt * rdist**2 + (1 / dx) * (v1 - v1[:,l] + u1[l,:] - u1)
        
        zeta2 = 1 - Bt * rdist**2 + (1 / dx) * (v2 - v2[:,l] + u2[l,:] - u2)


        # add vorticity flux, zeta*u
        zv1 = zeta1 * (v1 + v1[:,l])
        zv2 = zeta2 * (v2 + v2[:,l])

        du1dt = du1dt + 0.25 * (zv1 + zv1[r,:])

        du2dt = du2dt + 0.25 * (zv2 + zv2[r,:])

        zu1 = zeta1 * (u1 + u1[l,:])
        zu2 = zeta2 * (u2 + u2[l,:])

        dv1dt = dv1dt - 0.25 * (zu1 + zu1[:,r])
        dv2dt = dv2dt - 0.25 * (zu2 + zu2[:,r])

        ### Cumulus Drag (D) ###
        """ du1dt = du1dt - (1 / dx) * u1 / dragf
        du2dt = du2dt - (1 / dx) * u2 / dragf
        dv1dt = dv1dt - (1 / dx) * v1 / dragf
        dv2dt = dv2dt - (1 / dx) * v2 / dragf """

        B1p, B2p = hf.BernN2(u1, v1, u2, v2, gm, c22h, c12h, h1, h2, ord)

        du1dtsq = du1dt - (1 / dx) * (B1p - B1p[:,l])
        du2dtsq = du2dt - (1 / dx) * (B2p - B2p[:,l])

        dv1dtsq = dv1dt - (1 / dx) * (B1p - B1p[l,:])
        dv2dtsq = dv2dt - (1 / dx) * (B2p - B2p[l,:])

        if AB == 2:
            u1sq = u1_p + dt * du1dtsq
            u2sq = u2_p + dt * du2dtsq

            v1sq = v1_p + dt * dv1dtsq
            v2sq = v2_p + dt * dv2dtsq


        ##### new storm forcing -P #####

        remove_layers = [] # store weather layers that need to be removed here

        if mode == 1:
            for i in range(len(locs)):
                if (t-locs[i][-1]) >= locs[i][3] and t != 0:
                    remove_layers.append(i) # tag layer for removal if a storm's 

            add = len(remove_layers) # number of storms that were removed

            if add != 0:
                newlocs = hf.genlocs(add, N, t)

                for i in range(len(remove_layers)):
                    locs[remove_layers[i]] = newlocs[i]

                wlayer = hf.pairshapeN2(locs, t) ### use pairshapeBEGIN instead of pairshape
                Wmat = hf.pairfieldN2(L, h1, wlayer)

        ##### new storm forcing -P #####

        Fx1 = hf.xflux(h1, u1) - kappa / dx * (h1 - h1[:,l])
        Fy1 = hf.yflux(h1, v1) - kappa / dx * (h1 - h1[l,:])
        dh1dt = -(1 / dx) * (Fx1[:,r] - Fx1 + Fy1[r,:] - Fy1)

        if layers == 2.5:
            Fx2 = hf.xflux(h2, u2) - kappa / dx * (h2 - h2[:,l])
            Fy2 = hf.yflux(h2, v2) - kappa / dx * (h2 - h2[l,:])

            dh2dt = -(1 / dx) * (Fx2[:,r] - Fx2 + Fy2[r,:] - Fy2)

        if tradf > 0:
            dh1dt = dh1dt - 1 / tradf * (h1 - 1)
            dh2dt = dh2dt - 1 / tradf * (h2 - 1)

        if mode == 1:
            dh1dt = dh1dt + Wmat.astype(np.float64)
            if layers == 2.5:
                dh2dt = dh2dt - H1H2 * Wmat.astype(np.float64)

        if AB == 2:
            h1 = h1_p + dt * dh1dt
            if layers == 2.5:
                h2 = h2_p + dt * dh2dt

        u1 = u1sq
        u2 = u2sq
        v1 = v1sq
        v2 = v2sq

        if tc % tpl == 0:
            print(f"t={t}, mean h1 is {round(np.mean(np.mean(h1)), 4)}, num storms {locs.shape[0]}. Time elapsed, {round(time.perf_counter()-timer, 3)}s.")
            timer = time.perf_counter()

            ii += 1

        
            #ts.append(t)
            u1mat.append(u1)
            v1mat.append(v1)
            u2mat.append(u2)
            v2mat.append(v2)
            h1mat.append(h1)
            h2mat.append(h2)
            Wmatmat.append(Wmat)
            zeta2mat.append(zeta2)

            """ with objmode():
                ad.save_data(u1,u2,v1,v2,h1,h2,locs,t,lasttime,new_name) """

            #KEmat.append(hf.calculate_KE(u1,u2,v1,v2,h1,h2))
            #APEmat.append(hf.calculate_APE(h1, h2))
                
        if math.isnan(h1[0, 0]):
            break
    

        tc += 1
        t = tc * dt
    
    return u1mat, v1mat, u2mat, v2mat, h1mat, h2mat, Wmatmat

model = build_model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=keras.losses.MeanSquaredError())
for i in range(100):
    lasttime = 0
    locs = np.zeros((num,5))
    u1_inp, v1_inp, u2_inp, v2_inp, h1_inp, h2_inp, Wmat_inp = run_sim(u1,u2,v1,v2,h1,h2,locs,lasttime)
    u1_inp, v1_inp, u2_inp, v2_inp, h1_inp, h2_inp, Wmat_inp = np.array(u1_inp), np.array(v1_inp), np.array(u2_inp), np.array(v2_inp), np.array(h1_inp), np.array(h2_inp), np.array(Wmat_inp)
    model.fit([u1_inp, v1_inp, u2_inp, v2_inp, h1_inp, h2_inp, Wmat_inp], [u1_inp, v1_inp, u2_inp, v2_inp, h1_inp, h2_inp], batch_size=100, epochs=1)




# [:,l] move columns 1 right np.roll(v1, 1, axis=1)  tf.roll(v1, shift=1, axis=2)
# [:,r] move columns 1 left  np.roll(v1, -1, axis=1) tf.roll(v1, shift=-1, axis=2)
# [l,:] move rows 1 down     np.roll(u1, 1, axis=0)  tf.roll(u1, shift=1, axis=1)
# [r,:] moue rows 1 up       np.roll(u1, -1, axis=0) tf.roll(u1, shift=-1, axis=1)


"""
zeta1 = (
            1
            - Bt * rdist**2
            + (1 / dx)
            * (
                v1
                - tf.roll(v1, shift=1, axis=2)
                + tf.roll(u1, shift=1, axis=1)
                - u1
            )
        )
        zv1 = zeta1 * (v1 + tf.roll(v1, shift=1, axis=2))
        B1p = (
            c12h * h1
            + c22h * h2
            + 0.25
            * (
                u1**2
                + tf.roll(u1, shift=-1, axis=2) ** 2
                + v1**2
                + tf.roll(v1, shift=-1, axis=1) ** 2
            )
        )
        vel = tf.zeros_like(u1)
        field = (
            20 * vel
            + 2 * tf.roll(tf.roll(vel, shift=1, axis=0), shift=1, axis=2)
            + 2 * tf.roll(tf.roll(vel, shift=1, axis=0), shift=-1, axis=2)
            + 2 * tf.roll(tf.roll(vel, shift=-1, axis=0), shift=1, axis=2)
            + 2 * tf.roll(tf.roll(vel, shift=-1, axis=0), shift=-1, axis=2)
            - 8 * tf.roll(vel, shift=1, axis=1)
            - 8 * tf.roll(vel, shift=-1, axis=1)
            - 8 * tf.roll(vel, shift=1, axis=2)
            - 8 * tf.roll(vel, shift=-1, axis=2)
            + tf.roll(vel, shift=2, axis=1)
            + tf.roll(vel, shift=-2, axis=1)
            + tf.roll(vel, shift=2, axis=2)
            + tf.roll(vel, shift=-2, axis=2)
        )
        field = -1 / Re * (1 / dx**4) * field
        eqnu1 = (
            0.25 * (zv1 + tf.roll(zv1, shift=-1, axis=1))
            - (1 / dx) * (B1p - tf.roll(B1p, shift=1, axis=2))
            + field
        ) - du1dt

        zu1 = zeta1 * (u1 + tf.roll(u1, shift=1, axis=1))

        vel = tf.zeros_like(v1)
        field = (
            20 * vel
            + 2 * tf.roll(tf.roll(vel, shift=1, axis=0), shift=1, axis=2)
            + 2 * tf.roll(tf.roll(vel, shift=1, axis=0), shift=-1, axis=2)
            + 2 * tf.roll(tf.roll(vel, shift=-1, axis=0), shift=1, axis=2)
            + 2 * tf.roll(tf.roll(vel, shift=-1, axis=0), shift=-1, axis=2)
            - 8 * tf.roll(vel, shift=1, axis=1)
            - 8 * tf.roll(vel, shift=-1, axis=1)
            - 8 * tf.roll(vel, shift=1, axis=2)
            - 8 * tf.roll(vel, shift=-1, axis=2)
            + tf.roll(vel, shift=2, axis=1)
            + tf.roll(vel, shift=-2, axis=1)
            + tf.roll(vel, shift=2, axis=2)
            + tf.roll(vel, shift=-2, axis=2)
        )
        field = -1 / Re * (1 / dx**4) * field
        eqnv1 = (
            0.25 * (zu1 + tf.roll(zu1, shift=-1, axis=2))
            - (1 / dx) * (B1p - tf.roll(B1p, shift=1, axis=1))
            + field
        ) - dv1dt

        zeta2 = (
            1
            - Bt * rdist**2
            + (1 / dx)
            * (
                v2
                - tf.roll(v2, shift=1, axis=2)
                + tf.roll(u2, shift=1, axis=1)
                - u2
            )
        )
        zv2 = zeta2 * (v2 + tf.roll(v2, shift=1, axis=2))
        B2p = (
            gm * c12h * h1
            + c22h * h2
            + 0.25
            * (
                u2**2
                + tf.roll(u2, shift=-1, axis=2) ** 2
                + v2**2
                + tf.roll(v2, shift=-1, axis=1) ** 2
            )
        )
        vel = tf.zeros_like(u2)
        field = (
            20 * vel
            + 2 * tf.roll(tf.roll(vel, shift=1, axis=0), shift=1, axis=2)
            + 2 * tf.roll(tf.roll(vel, shift=1, axis=0), shift=-1, axis=2)
            + 2 * tf.roll(tf.roll(vel, shift=-1, axis=0), shift=1, axis=2)
            + 2 * tf.roll(tf.roll(vel, shift=-1, axis=0), shift=-1, axis=2)
            - 8 * tf.roll(vel, shift=1, axis=1)
            - 8 * tf.roll(vel, shift=-1, axis=1)
            - 8 * tf.roll(vel, shift=1, axis=2)
            - 8 * tf.roll(vel, shift=-1, axis=2)
            + tf.roll(vel, shift=2, axis=1)
            + tf.roll(vel, shift=-2, axis=1)
            + tf.roll(vel, shift=2, axis=2)
            + tf.roll(vel, shift=-2, axis=2)
        )
        field = -1 / Re * (1 / dx**4) * field
        eqnu2 = (
            0.25 * (zv2 + tf.roll(zv2, shift=-1, axis=1))
            - (1 / dx) * (B2p - tf.roll(B2p, shift=1, axis=2))
            + field
        ) - du2dt

        zu2 = zeta2 * (u2 + tf.roll(u2, shift=1, axis=1))

        vel = tf.zeros_like(v2)
        field = (
            20 * vel
            + 2 * tf.roll(tf.roll(vel, shift=1, axis=0), shift=1, axis=2)
            + 2 * tf.roll(tf.roll(vel, shift=1, axis=0), shift=-1, axis=2)
            + 2 * tf.roll(tf.roll(vel, shift=-1, axis=0), shift=1, axis=2)
            + 2 * tf.roll(tf.roll(vel, shift=-1, axis=0), shift=-1, axis=2)
            - 8 * tf.roll(vel, shift=1, axis=1)
            - 8 * tf.roll(vel, shift=-1, axis=1)
            - 8 * tf.roll(vel, shift=1, axis=2)
            - 8 * tf.roll(vel, shift=-1, axis=2)
            + tf.roll(vel, shift=2, axis=1)
            + tf.roll(vel, shift=-2, axis=1)
            + tf.roll(vel, shift=2, axis=2)
            + tf.roll(vel, shift=-2, axis=2)
        )
        field = -1 / Re * (1 / dx**4) * field
        eqnv2 = (
            0.25 * (zu2 + tf.roll(zu2, shift=-1, axis=2))
            - (1 / dx) * (B2p - tf.roll(B2p, shift=1, axis=1))
            + field
        ) - dv2dt

        Fx1 = 0.5 * u1 * (
            tf.roll(h1, shift=1, axis=2) + h1
        ) - kappa / dx * (h1 - tf.roll(h1, shift=1, axis=2))
        Fy1 = 0.5 * v1 * (
            tf.roll(h1, shift=1, axis=2) + h1
        ) - kappa / dx * (h1 - tf.roll(h1, shift=1, axis=2))
        eqnh1 = (
            -(1 / dx)
            * (
                tf.roll(Fx1, shift=-1, axis=2)
                - Fx1
                + tf.roll(Fy1, shift=-1, axis=1)
                - Fy1
            )
            - 1 / tradf * (h1 - 1)
            + Wmat
        ) - dh1dt

        Fx2 = 0.5 * u2 * (
            tf.roll(h2, shift=1, axis=2) + h2
        ) - kappa / dx * (h2 - tf.roll(h2, shift=1, axis=2))
        Fy2 = 0.5 * v2 * (
            tf.roll(h2, shift=1, axis=2) + h2
        ) - kappa / dx * (h2 - tf.roll(h2, shift=1, axis=2))
        eqnh2 = (
            -(1 / dx)
            * (
                tf.roll(Fx2, shift=-1, axis=2)
                - Fx2
                + tf.roll(Fy2, shift=-1, axis=1)
                - Fy2
            )
            - 1 / tradf * (h2 - 1)
            + H1H2 * Wmat
        ) - dh2dt

        loss_u1 = tf.reduce_mean(tf.square(eqnu1))
        loss_v1 = tf.reduce_mean(tf.square(eqnv1))
        loss_u2 = tf.reduce_mean(tf.square(eqnu2))
        loss_v2 = tf.reduce_mean(tf.square(eqnv2))
        loss_h1 = tf.reduce_mean(tf.square(eqnh1))
        loss_h2 = tf.reduce_mean(tf.square(eqnh2))

        loss = loss_u1 + loss_v1 + loss_u2 + loss_v2 + loss_h1 + loss_h2

"""