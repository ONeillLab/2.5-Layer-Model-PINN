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

gridsize = N
model_name = "NN.keras"

def build_model2():
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
    for i in range(2):
        hidden = Layers.Dense(1024, activation="relu")(hidden)
    output = Layers.Dense(6 * gridsize * gridsize, activation="linear")(hidden)
    output_reshaped = Layers.Reshape((6, gridsize, gridsize))(output)

    model = Model(
        [u1_inp, v1_inp, u2_inp, v2_inp, h1_inp, h2_inp, Wmat_inp], output_reshaped
    )

    model.summary()

    return model


def update_model(
    model, u1_inp, v1_inp, u2_inp, v2_inp, h1_inp, h2_inp, Wmat_inp, u1_control, v1_control, u2_control, v2_control, h1_control, h2_control, channels
):
    with tf.GradientTape(persistent=True) as wghts:
        uvh = model(
            [u1_inp, v1_inp, u2_inp, v2_inp, h1_inp, h2_inp, Wmat_inp], training=True
        )

        u1, v1, u2, v2, h1, h2 = (
            tf.reshape(uvh[:, :1], (channels, gridsize, gridsize)),
            tf.reshape(uvh[:, 1:2], (channels, gridsize, gridsize)),
            tf.reshape(uvh[:, 2:3], (channels, gridsize, gridsize)),
            tf.reshape(uvh[:, 3:4], (channels, gridsize, gridsize)),
            tf.reshape(uvh[:, 4:5], (channels, gridsize, gridsize)),
            tf.reshape(uvh[:, 5:], (channels, gridsize, gridsize)),
        )

        u1, v1, u2, v2, h1, h2 = (
            tf.cast(u1, dtype=tf.float64),
            tf.cast(v1, dtype=tf.float64),
            tf.cast(u2, dtype=tf.float64),
            tf.cast(v2, dtype=tf.float64),
            tf.cast(h1, dtype=tf.float64),
            tf.cast(h2, dtype=tf.float64),
        )

        loss_u1 = tf.reduce_mean(tf.square(u1 - u1_control))
        loss_v1 = tf.reduce_mean(tf.square(v1 - v1_control))
        loss_u2 = tf.reduce_mean(tf.square(u2 - u2_control))
        loss_v2 = tf.reduce_mean(tf.square(v2 - v2_control))
        loss_h1 = tf.reduce_mean(tf.square(h1 - h1_control))
        loss_h2 = tf.reduce_mean(tf.square(h2 - h2_control))
        loss = loss_u1 + loss_v1 + loss_u2 + loss_v2 + loss_h1 + loss_h2

    gradloss = wghts.gradient(loss, model.trainable_variables)
    grads = [g for g in gradloss]

    return loss, grads


def train_model(
    model, u1_inp, v1_inp, u2_inp, v2_inp, h1_inp, h2_inp, Wmat_inp, u1_control, v1_control, u2_control, v2_control, h1_control, h2_control, epochs, batch_size
):
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
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
                u1_control[k * batch_size : k * batch_size + batch_size],
                v1_control[k * batch_size : k * batch_size + batch_size],
                u2_control[k * batch_size : k * batch_size + batch_size],
                v2_control[k * batch_size : k * batch_size + batch_size],
                h1_control[k * batch_size : k * batch_size + batch_size],
                h2_control[k * batch_size : k * batch_size + batch_size],
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
    model.save(model_name, overwrite=True)
    return model