import numpy as np
import tensorflow as tf
from keras import Sequential
from keras import layers as Layers
from keras import ops
from keras import Model
from model_helpers import build_model, build_model2, PINNloss
from name_list import *
import keras
from netCDF4 import Dataset
import time
from random import sample

restart = True

tf.config.run_functions_eagerly(True)


def build_model():
    t_inp = Layers.Input(shape=(1,))
    x_inp = Layers.Input(shape=(1,))
    y_inp = Layers.Input(shape=(1,))
    S_inp = Layers.Input(shape=(1,))
    hidden = Layers.Concatenate()([t_inp, x_inp, y_inp, S_inp])
    for i in range(8):
        hidden = Layers.Dense(100, activation="tanh")(hidden)
    output = Layers.Dense(6, activation="tanh")(hidden)

    model = Model([t_inp, x_inp, y_inp, S_inp], output)

    model.summary()

    return model


def update_model(model, input_data, initial_data, boundary_data):
    channels = np.shape(boundary_data)[0]
    t_inp, x_inp, y_inp, S_inp = (
        input_data[:, :1],
        input_data[:, 1:2],
        input_data[:, 2:3],
        input_data[:, 3:4],
    )
    t_inp, x_inp, y_inp, S_inp = (
        tf.convert_to_tensor(t_inp, dtype=tf.float64),
        tf.convert_to_tensor(x_inp, dtype=tf.float64),
        tf.convert_to_tensor(y_inp, dtype=tf.float64),
        tf.convert_to_tensor(S_inp, dtype=tf.float64),
    )
    t_int, x_int, y_int, S_int = (
        initial_data[:, :1] * 0,
        initial_data[:, 1:2],
        initial_data[:, 2:3],
        initial_data[:, 3:4],
    )
    t_int, x_int, y_int, S_int = (
        tf.convert_to_tensor(t_int, dtype=tf.float64),
        tf.convert_to_tensor(x_int, dtype=tf.float64),
        tf.convert_to_tensor(y_int, dtype=tf.float64),
        tf.convert_to_tensor(S_int, dtype=tf.float64),
    )
    t_bound, x_bound, y_bound, S_bound = (
        boundary_data[:, :1],
        boundary_data[:, 1:2],
        boundary_data[:, 2:3],
        boundary_data[:, 3:4],
    )
    t_bound, x_bound, y_bound, S_bound = (
        tf.convert_to_tensor(t_bound, dtype=tf.float64),
        tf.convert_to_tensor(x_bound, dtype=tf.float64),
        tf.convert_to_tensor(y_bound, dtype=tf.float64),
        tf.convert_to_tensor(S_bound, dtype=tf.float64),
    )
    constant0 = tf.convert_to_tensor(np.zeros((channels, 1)), dtype=tf.float64)
    constantXY = tf.convert_to_tensor(np.zeros((channels, 1)) + N, dtype=tf.float64)

    with tf.GradientTape(persistent=True) as wghts:
        with tf.GradientTape(persistent=True) as g:
            g.watch(t_inp), g.watch(x_inp), g.watch(y_inp)
            with tf.GradientTape(persistent=True) as gg:
                gg.watch(t_inp), gg.watch(x_inp), gg.watch(y_inp)
                with tf.GradientTape(persistent=True) as ggg:
                    ggg.watch(t_inp), ggg.watch(x_inp), ggg.watch(y_inp)
                    with tf.GradientTape(persistent=True) as gggg:
                        gggg.watch(t_inp), gggg.watch(x_inp), gggg.watch(y_inp)
                        uvh = model([t_inp, x_inp, y_inp, S_inp], training=True)
                        u1, v1, h1 = uvh[:, :1], uvh[:, 1:2], uvh[:, 2:3]
                        u2, v2, h2 = uvh[:, 3:4], uvh[:, 4:5], uvh[:, 5:6]
                    [du1dt, du1dx, du1dy] = gggg.gradient(u1, [t_inp, x_inp, y_inp])
                    [dv1dt, dv1dx, dv1dy] = gggg.gradient(v1, [t_inp, x_inp, y_inp])
                    [dh1dt, dh1dx, dh1dy] = gggg.gradient(h1, [t_inp, x_inp, y_inp])
                    [du2dt, du2dx, du2dy] = gggg.gradient(u2, [t_inp, x_inp, y_inp])
                    [dv2dt, dv2dx, dv2dy] = gggg.gradient(v2, [t_inp, x_inp, y_inp])
                    [dh2dt, dh2dx, dh2dy] = gggg.gradient(h2, [t_inp, x_inp, y_inp])
                [du1dx2, du1dy2] = ggg.gradient(du1dx, x_inp), ggg.gradient(
                    du1dy, y_inp
                )
                [dv1dx2, dv1dy2] = ggg.gradient(dv1dx, x_inp), ggg.gradient(
                    dv1dy, y_inp
                )
                [dh1dx2, dh1dy2] = ggg.gradient(dh1dx, x_inp), ggg.gradient(
                    dh1dy, y_inp
                )
                [du2dx2, du2dy2] = ggg.gradient(du2dx, x_inp), ggg.gradient(
                    du2dy, y_inp
                )
                [dv2dx2, dv2dy2] = ggg.gradient(dv2dx, x_inp), ggg.gradient(
                    dv2dy, y_inp
                )
                [dh2dx2, dh2dy2] = ggg.gradient(dh2dx, x_inp), ggg.gradient(
                    dh2dy, y_inp
                )
            [du1dx3, du1dy3, du1dx2dy] = (
                gg.gradient(du1dx2, x_inp),
                gg.gradient(du1dy2, y_inp),
                gg.gradient(du1dx2, y_inp),
            )
            [dv1dx3, dv1dy3, dv1dx2dy] = (
                gg.gradient(dv1dx2, x_inp),
                gg.gradient(dv1dy2, y_inp),
                gg.gradient(dv1dx2, y_inp),
            )
            [du2dx3, du2dy3, du2dx2dy] = (
                gg.gradient(du2dx2, x_inp),
                gg.gradient(du2dy2, y_inp),
                gg.gradient(du2dx2, y_inp),
            )
            [dv2dx3, dv2dy3, dv2dx2dy] = (
                gg.gradient(dv2dx2, x_inp),
                gg.gradient(dv2dy2, y_inp),
                gg.gradient(dv2dx2, y_inp),
            )
        [du1dx4, du1dy4, du1dx2dy2] = (
            g.gradient(du1dx3, x_inp),
            g.gradient(du1dy3, y_inp),
            g.gradient(du1dx2dy, y_inp),
        )
        [dv1dx4, dv1dy4, dv1dx2dy2] = (
            g.gradient(dv1dx3, x_inp),
            g.gradient(dv1dy3, y_inp),
            g.gradient(dv1dx2dy, y_inp),
        )
        [du2dx4, du2dy4, du2dx2dy2] = (
            g.gradient(du2dx3, x_inp),
            g.gradient(du2dy3, y_inp),
            g.gradient(du2dx2dy, y_inp),
        )
        [dv2dx4, dv2dy4, dv2dx2dy2] = (
            g.gradient(dv2dx3, x_inp),
            g.gradient(dv2dy3, y_inp),
            g.gradient(dv2dx2dy, y_inp),
        )

        u1, v1, h1, u2, v2, h2 = (
            tf.cast(u1, dtype=tf.float64),
            tf.cast(v1, dtype=tf.float64),
            tf.cast(h1, dtype=tf.float64),
            tf.cast(u2, dtype=tf.float64),
            tf.cast(v2, dtype=tf.float64),
            tf.cast(h2, dtype=tf.float64),
        )

        eqnu1 = (
            du1dt
            - (1 - Bt * tf.math.sqrt(x_inp * x_inp + y_inp * y_inp) + dv1dx - du1dy) * v1
            + c12h * dh1dx
            + c22h * dh2dx
            + u1 * du1dx
            + v1 * dv1dx
            + 1 / Re * (du1dx4 + 2 * du1dx2dy2 + du1dy4)
        )
        eqnv1 = (
            dv1dt
            + (1 - Bt * tf.math.sqrt(x_inp * x_inp + y_inp * y_inp) + dv1dx - du1dy) * u1
            + c12h * dh1dy
            + c22h * dh2dy
            + u1 * du1dy
            + v1 * dv1dy
            + 1 / Re * (dv1dx4 + 2 * dv1dx2dy2 + dv1dy4)
        )
        eqnh1 = (
            dh1dt
            + du1dx * h1
            + u1 * dh1dx
            + dv1dy * h1
            + v1 * dh1dy
            - S_inp
            + (h1 - 1) * (1 / tradf)
            - kappa * (dh1dx2 + dh1dy2)
        )
        eqnu2 = (
            du2dt
            - (1 - Bt * tf.math.sqrt(x_inp * x_inp + y_inp * y_inp) + dv2dx - du2dy) * v2
            + gm * c12h * dh1dx
            + c22h * dh2dx
            + u2 * du2dx
            + v2 * dv2dx
            + 1 / Re * (du2dx4 + 2 * du2dx2dy2 + du2dy4)
        )
        eqnv2 = (
            dv2dt
            + (1 - Bt * tf.math.sqrt(x_inp * x_inp + y_inp * y_inp) + dv2dx - du2dy) * u2
            + gm * c12h * dh1dy
            + c22h * dh2dy
            + u2 * du2dy
            + v2 * dv2dy
            + 1 / Re * (dv2dx4 + 2 * dv2dx2dy2 + dv2dy4)
        )
        eqnh2 = (
            dh2dt
            + du2dx * h2
            + u2 * dh2dx
            + dv2dy * h2
            + v2 * dh2dy
            + H1H2 * S_inp
            + (h2 - 1) * (1 / tradf)
            - kappa * (dh2dx2 + dh2dy2)
        )

        PDEloss_u1 = tf.reduce_mean(tf.square(eqnu1))
        PDEloss_v1 = tf.reduce_mean(tf.square(eqnv1))
        PDEloss_h1 = tf.reduce_mean(tf.square(eqnh1))
        PDEloss_u2 = tf.reduce_mean(tf.square(eqnu2))
        PDEloss_v2 = tf.reduce_mean(tf.square(eqnv2))
        PDEloss_h2 = tf.reduce_mean(tf.square(eqnh2))

        PDEloss = (
            PDEloss_u1 + PDEloss_v1 + PDEloss_h1 + PDEloss_u2 + PDEloss_v2 + PDEloss_h2
        )

        uvh_int = model([t_int, x_int, y_int, S_int], training=True)
        u1, v1, h1 = uvh_int[:, :1], uvh_int[:, 1:2], uvh_int[:, 2:3]
        u2, v2, h2 = uvh_int[:, 3:4], uvh_int[:, 4:5], uvh_int[:, 5:6]

        ICloss_u1 = tf.reduce_mean(tf.square(u1))
        ICloss_v1 = tf.reduce_mean(tf.square(v1))
        ICloss_h1 = tf.reduce_mean(tf.square(h1 - 1))
        ICloss_u2 = tf.reduce_mean(tf.square(u2))
        ICloss_v2 = tf.reduce_mean(tf.square(v2))
        ICloss_h2 = tf.reduce_mean(tf.square(h2 - 1))

        ICloss = ICloss_u1 + ICloss_v1 + ICloss_h1 + ICloss_u2 + ICloss_v2 + ICloss_h2

        uvh_boundx0 = model([t_bound, constant0, y_bound, S_bound], training=True)
        uvh_boundxX = model([t_bound, constantXY, y_bound, S_bound], training=True)
        u1x, v1x, h1x = (
            (uvh_boundx0[:, :1] - uvh_boundxX[:, :1]),
            (uvh_boundx0[:, 1:2] - uvh_boundxX[:, 1:2]),
            (uvh_boundx0[:, 2:3] - uvh_boundxX[:, 2:3]),
        )
        u2x, v2x, h2x = (
            (uvh_boundx0[:, 3:4] - uvh_boundxX[:, 3:4]),
            (uvh_boundx0[:, 4:5] - uvh_boundxX[:, 4:5]),
            (uvh_boundx0[:, 5:] - uvh_boundxX[:, 5:]),
        )

        uvh_boundy0 = model([t_bound, x_bound, constant0, S_bound], training=True)
        uvh_boundyY = model([t_bound, y_bound, constantXY, S_bound], training=True)
        u1y, v1y, h1y = (
            (uvh_boundy0[:, :1] - uvh_boundyY[:, :1]),
            (uvh_boundy0[:, 1:2] - uvh_boundyY[:, 1:2]),
            (uvh_boundy0[:, 2:3] - uvh_boundyY[:, 2:3]),
        )
        u2y, v2y, h2y = (
            (uvh_boundy0[:, 3:4] - uvh_boundyY[:, 3:4]),
            (uvh_boundy0[:, 4:5] - uvh_boundyY[:, 4:5]),
            (uvh_boundy0[:, 5:] - uvh_boundyY[:, 5:]),
        )

        BCloss_u1 = tf.reduce_mean(tf.square(u1x) + tf.square(u1y))
        BCloss_v1 = tf.reduce_mean(tf.square(v1x) + tf.square(v1y))
        BCloss_h1 = tf.reduce_mean(tf.square(h1x) + tf.square(h1y))
        BCloss_u2 = tf.reduce_mean(tf.square(u2x) + tf.square(u2y))
        BCloss_v2 = tf.reduce_mean(tf.square(v2x) + tf.square(v2y))
        BCloss_h2 = tf.reduce_mean(tf.square(h2x) + tf.square(h2y))

        BCloss = BCloss_u1 + BCloss_v1 + BCloss_h1 + BCloss_u2 + BCloss_v2 + BCloss_h2

    gradPDE = wghts.gradient(PDEloss, model.trainable_variables)
    gradIVP = wghts.gradient(ICloss, model.trainable_variables)
    gradBVP = wghts.gradient(BCloss, model.trainable_variables)

    grads = [gPDE + gIVP + gBVP for gPDE, gIVP, gBVP in zip(gradPDE, gradIVP, gradBVP)]

    PDEloss = tf.cast(PDEloss, dtype=tf.float32)

    return PDEloss, ICloss, BCloss, grads


def train_model(model, input_data, initial_data, boundary_data, epochs, bs_input):
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    batch_numbers = np.shape(input_data)[0] // bs_input
    bs_initial = np.shape(initial_data)[0] // batch_numbers
    bs_boundary = np.shape(boundary_data)[0] // batch_numbers

    """ batch_numbers = 50
    bs_initial = 200
    bs_boundary = 100 """

    for i in range(epochs):
        epoch_PDEloss = 0
        epoch_ICloss = 0
        epoch_BCloss = 0
        timer = time.perf_counter()
        for k in range(batch_numbers):
            PDEloss, ICloss, BCloss, grads = update_model(
                model,
                input_data[k * bs_input : k * bs_input + bs_input],
                initial_data[k * bs_initial : k * bs_initial + bs_initial],
                boundary_data[k * bs_boundary : k * bs_boundary + bs_boundary],
            )
            epoch_PDEloss += PDEloss
            epoch_ICloss += ICloss
            epoch_BCloss += BCloss
            opt.apply_gradients(zip(grads, model.trainable_variables))
            print(
                f"Batches completed: {k+1}, PDE loss: {PDEloss.numpy()}, IC loss: {ICloss.numpy()}, BC loss: {BCloss.numpy()}"
            )
        print(
            f"Epochs completed: {i+1}, PDE loss: {epoch_PDEloss.numpy()}, IC loss: {epoch_ICloss.numpy()}, BC loss: {epoch_BCloss.numpy()}, Time elapsed, {round(time.perf_counter()-timer, 3)}s"
        )
        timer = time.perf_counter()
    model.save("model2.keras", overwrite=True)

    return model



examples = np.load("examples.npy")

exs_input = []
for i in range(201):
    for j in range(36):
        for k in range(36):
            #if examples[i,j,k] >= 0.005:
            exs_input.append([i, j, k, examples[i,j,k]])
print(np.shape(exs_input))


exs_initial = []
for j in range(36):
    for k in range(36):
        exs_initial.append([0, j, k, examples[0,j,k]])


exs_boundary = []
for i in range(201):
    for j in range(36):
        for k in range(36):
            exs_boundary.append([i, j, k, examples[i,j,k]])





""" input_data = np.load("input_data.npy")
#input_data = np.random.shuffle(input_data)
initial_data = np.load("initial_data.npy")
#initial_data = np.random.shuffle(initial_data)
boundary_data = np.load("boundary_data.npy")
#boundary_data = np.random.shuffle(boundary_data)
print(np.shape(input_data)) """


""" rg = Dataset("examples_real", "r")
input_data = rg.variables["input_data"][:]
initial_data = rg.variables["initial_data"][:]
boundary_data = rg.variables["boundary_data"][:]
rg.close() """


""" input_data = np.random.random((50000,4))
initial_data = np.random.random((1000,4))
boundary_data = np.random.random((5000,4)) """
if restart == True:
    model_ = keras.models.load_model("model2.keras")
    model = build_model()
    model.set_weights(model_.get_weights())
else:
    model = build_model()

for i in range(1):
    exs_input1 = sample(exs_input, 50000)
    exs_initial1 = sample(exs_initial, 1000)
    exs_boundary1 = sample(exs_boundary, 5000)
    exs_input1 = np.array(exs_input1)
    exs_initial1 = np.array(exs_initial1)
    exs_boundary1 = np.array(exs_boundary1)
    input_data = exs_input1
    initial_data = exs_initial1
    boundary_data = exs_boundary1
    model = train_model(model, input_data, initial_data, boundary_data, 3, 500)
    model.save("model2.keras", overwrite=True)
    print(i)
