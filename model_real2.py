import numpy as np
import tensorflow as tf
from keras import Sequential
from keras import layers as Layers
from keras import ops
from model_helpers import build_model, build_model2, PINNloss
from name_list import *
import keras

tf.config.run_functions_eagerly(True)

model2 = Sequential()
model2.add(Layers.Input(shape=(4,)))
model2.add(Layers.Dense(100, activation="tanh"))
for _ in range(7):
    model2.add(Layers.Dense(100, activation="tanh"))
model2.add(Layers.Dense(6, activation="tanh"))
model2.compile(optimizer="adam", loss="mean_squared_error", run_eagerly=True)
model2.summary()

sub_model = Sequential()
sub_model.add(Layers.Input(shape=(4,)))
sub_model.add(Layers.Dense(100, activation="tanh"))
for _ in range(7):
    sub_model.add(Layers.Dense(100, activation="tanh"))
sub_model.add(Layers.Dense(6, activation="tanh"))
sub_model.compile(optimizer="adam", loss="mean_squared_error", run_eagerly=True)
sub_model.summary()


def derivatives(model, input_data):
    t_inp, x_inp, y_inp, S_inp = input_data[:,:1], input_data[:,1:2], input_data[:,2:3], input_data[:,3:4]
    with tf.GradientTape() as wghts:
        with tf.GradientTape() as g:
            g.watch(t_inp), g.watch(x_inp), g.watch(y_inp)
            with tf.GradientTape() as gg:
                gg.watch(t_inp), gg.watch(x_inp), gg.watch(y_inp)
                with tf.GradientTape() as ggg:
                    ggg.watch(t_inp), ggg.watch(x_inp), ggg.watch(y_inp)
                    with tf.GradientTape() as gggg:
                        gggg.watch(t_inp), gggg.watch(x_inp), gggg.watch(y_inp)
                        uvh = model([t_inp, x_inp, y_inp], training=True)
                        u1, v1, h1 = uvh[:,:1], uvh[:,1:2], uvh[:,2:3]
                        u2, v2, h2 = uvh[:,3:4], uvh[:,4:5], uvh[:,5:6]
                    [du1dt, du1dx, du1dy] = gggg.gradient(u1,[t_inp, x_inp, y_inp])
                    [dv1dt, dv1dx, dv1dy] = gggg.gradient(v1,[t_inp, x_inp, y_inp])
                    [dh1dt, dh1dx, dh1dy] = gggg.gradient(h1,[t_inp, x_inp, y_inp])
                    [du2dt, du2dx, du2dy] = gggg.gradient(u2,[t_inp, x_inp, y_inp])
                    [dv2dt, dv2dx, dv2dy] = gggg.gradient(v2,[t_inp, x_inp, y_inp])
                    [dh2dt, dh2dx, dh2dy] = gggg.gradient(h2,[t_inp, x_inp, y_inp])
                [du1dx2, du1dy2] = ggg.gradient(u1,[x_inp, y_inp])
                [dv1dx2, dv1dy2] = ggg.gradient(v1,[x_inp, y_inp])
                [dh1dx2, dh1dy2] = ggg.gradient(h1,[x_inp, y_inp])
                [du2dx2, du2dy2] = ggg.gradient(u2,[x_inp, y_inp])
                [dv2dx2, dv2dy2] = ggg.gradient(v2,[x_inp, y_inp])
                [dh2dx2, dh2dy2] = ggg.gradient(h2,[x_inp, y_inp])
            

                     


def derivatives(model, input_data):
    x1 = tf.constant(input_data[0, 0])
    x2 = tf.constant(input_data[0, 1])
    x3 = tf.constant(input_data[0, 2])
    x4 = tf.constant(input_data[0, 3])

    """ dydx24 = None
    dydx34 = None
    dydx1 = None
    dydx1 = None
    dydx2 = None
    dydx3 = None
    dydx22dx32 = None
    dhdx2 = None """

    for i in range(4):
        with tf.GradientTape() as g:
            g.watch(x2)
            with tf.GradientTape() as gg:
                gg.watch(x2)
                with tf.GradientTape() as ggg:
                    ggg.watch(x2)
                    with tf.GradientTape() as gggg:
                        gggg.watch(x2)
                        y = (model(tf.reshape([x1, x2, x3, x4], (1, 4))))[:, i]
                    dy_dx2 = gggg.gradient(y, x2)
                dy_dx22 = ggg.gradient(dy_dx2, x2)
            dy_dx23 = gg.gradient(dy_dx22, x2)
        dy_dx24 = g.gradient(dy_dx23, x2)
        if i == 0:
            dydx24 = dy_dx24
            dydx24 = tf.reshape(dydx24, (1,))
        else:
            dy_dx24 = tf.reshape(dy_dx24, (1,))
            dydx24 = tf.concat([dydx24, dy_dx24], 0)

        # dydx24.append(dy_dx24)

        with tf.GradientTape() as g:
            g.watch(x3)
            with tf.GradientTape() as gg:
                gg.watch(x3)
                with tf.GradientTape() as ggg:
                    ggg.watch(x3)
                    with tf.GradientTape() as gggg:
                        gggg.watch(x3)
                        y = (model(tf.reshape([x1, x2, x3, x4], (1, 4))))[:, i]
                    dy_dx3 = gggg.gradient(y, x3)
                dy_dx32 = ggg.gradient(dy_dx3, x3)
            dy_dx33 = gg.gradient(dy_dx32, x3)
        dy_dx34 = g.gradient(dy_dx33, x3)
        if i == 0:
            dydx34 = dy_dx34
            dydx34 = tf.reshape(dydx34, (1,))
        else:
            dy_dx34 = tf.reshape(dy_dx34, (1,))
            dydx34 = tf.concat([dydx34, dy_dx34], 0)

        # dydx34.append(dy_dx34)

        with tf.GradientTape() as g:
            g.watch(x2)
            g.watch(x3)
            with tf.GradientTape() as gg:
                gg.watch(x2)
                gg.watch(x3)
                with tf.GradientTape() as ggg:
                    ggg.watch(x2)
                    ggg.watch(x3)
                    with tf.GradientTape() as gggg:
                        gggg.watch(x2)
                        gggg.watch(x3)
                        y = (model(tf.reshape([x1, x2, x3, x4], (1, 4))))[:, i]
                    dy_dx2 = gggg.gradient(y, x2)
                dy_dx22 = ggg.gradient(dy_dx2, x2)
            dy_dx22dx3 = gg.gradient(dy_dx22, x3)
        dy_dx22dx32 = g.gradient(dy_dx22dx3, x3)
        if i == 0:
            dydx22dx32 = dy_dx22dx32
            dydx22dx32 = tf.reshape(dydx22dx32, (1,))
        else:
            dy_dx22dx32 = tf.reshape(dy_dx22dx32, (1,))
            dydx22dx32 = tf.concat([dydx22dx32, dy_dx22dx32], 0)

        # dydx22dx32.append(dy_dx22dx32)

    for i in range(6):
        with tf.GradientTape() as g:
            g.watch(x1)
            y = (model(tf.reshape([x1, x2, x3, x4], (1, 4))))[:, i]
        dy_dx1 = g.gradient(y, x1)
        if i == 0:
            dydx1 = dy_dx1
            dydx1 = tf.reshape(dydx1, (1,))
        else:
            dy_dx1 = tf.reshape(dy_dx1, (1,))
            dydx1 = tf.concat([dydx1, dy_dx1], 0)
        # dydx1.append(dy_dx1)

        with tf.GradientTape() as g:
            g.watch(x2)
            y = (model(tf.reshape([x1, x2, x3, x4], (1, 4))))[:, i]
        dy_dx2 = g.gradient(y, x2)
        if i == 0:
            dydx2 = dy_dx2
            dydx2 = tf.reshape(dydx2, (1,))
        else:
            dy_dx2 = tf.reshape(dy_dx2, (1,))
            dydx2 = tf.concat([dydx2, dy_dx2], 0)
        # dydx2.append(dy_dx2)

        with tf.GradientTape() as g:
            g.watch(x3)
            y = (model(tf.reshape([x1, x2, x3, x4], (1, 4))))[:, i]
        dy_dx3 = g.gradient(y, x3)
        if i == 0:
            dydx3 = dy_dx3
            dydx3 = tf.reshape(dydx3, (1,))
        else:
            dy_dx3 = tf.reshape(dy_dx3, (1,))
            dydx3 = tf.concat([dydx3, dy_dx3], 0)
        # dydx3.append(dy_dx3)

    with tf.GradientTape() as g:
        g.watch(x2)
        with tf.GradientTape() as gg:
            gg.watch(x2)
            y = (model(tf.reshape([x1, x2, x3, x4], (1, 4))))[:, 4]
        dh1_dx1 = gg.gradient(y, x2)
    dh1_dx12 = g.gradient(dh1_dx1, x2)
    dhdx2 = dh1_dx12
    dhdx2 = tf.reshape(dhdx2, (1,))
    # dhdx2.append(dh1_dx12)

    with tf.GradientTape() as g:
        g.watch(x2)
        with tf.GradientTape() as gg:
            gg.watch(x2)
            y = (model(tf.reshape([x1, x2, x3, x4], (1, 4))))[:, 5]
        dh2_dx1 = gg.gradient(y, x2)
    dh2_dx12 = g.gradient(dh2_dx1, x2)
    dh2_dx12 = tf.reshape(dh2_dx12, (1,))
    dhdx2 = tf.concat([dhdx2, dy_dx3], 0)

    # dhdx2.append(dh2_dx12)

    with tf.GradientTape() as g:
        g.watch(x3)
        with tf.GradientTape() as gg:
            gg.watch(x3)
            y = (model(tf.reshape([x1, x2, x3, x4], (1, 4))))[:, 4]
        dh1_dx2 = gg.gradient(y, x3)
    dh1_dx22 = g.gradient(dh1_dx2, x3)
    dh1_dx22 = tf.reshape(dh1_dx22, (1,))
    dhdx2 = tf.concat([dhdx2, dh1_dx22], 0)

    # dhdx2.append(dh1_dx22)

    with tf.GradientTape() as g:
        g.watch(x3)
        with tf.GradientTape() as gg:
            gg.watch(x3)
            y = (model(tf.reshape([x1, x2, x3, x4], (1, 4))))[:, 5]
        dh2_dx2 = gg.gradient(y, x3)
    dh2_dx22 = g.gradient(dh2_dx2, x3)
    dh2_dx22 = tf.reshape(dh2_dx22, (1,))
    dhdx2 = tf.concat([dhdx2, dh2_dx22], 0)
    # dhdx2.append(dh2_dx22)

    """ dydx1 = np.array(dydx1) 
    dydx2 = np.array(dydx2)
    dydx3 = np.array(dydx3)
    dydx24 = np.array(dydx24)
    dydx34 = np.array(dydx34)
    dydx22dx32 = np.array(dydx22dx32)
    dhdx2 = np.array(dhdx2) """

    return dydx1, dydx2, dydx3, dydx24, dydx34, dydx22dx32, dhdx2


def dummy_function(model2, i, j):
    i = tf.reshape(i, (1, 4))
    dydx1, dydx2, dydx3, dydx24, dydx34, dydx22dx32, dhdx2 = derivatives(model2, i)
    i = tf.reshape(i, (4,))
    j = tf.cast(j, dtype=tf.float64)
    """ print(j)
    print(j[0])
    print(i[0])
    print(dydx1)
    print(dydx2)
    print(dydx3)
    print(dydx24)
    print(dydx34)
    print(dydx22dx32) 
    print(dhdx2)
    print(type(Bt)) """
    Theta1 = (
        dydx1[0]
        - (1 - Bt * (i[1] * i[1] + i[2] * i[2]) + dydx2[1] - dydx3[0]) * j[1]
        + c12h * dydx2[4]
        + c22h * dydx2[5]
        + j[0] * dydx2[0]
        + j[1] * dydx2[1]
        + 1 / Re * (dydx24[0] + 2 * dydx22dx32[0] + dydx34[0])
    )
    Theta2 = (
        dydx1[1]
        + (1 - Bt * (i[1] ** 2 + i[2] ** 2) + dydx2[1] - dydx3[0]) * j[0]
        + c12h * dydx3[4]
        + c22h * dydx3[5]
        + j[0] * dydx3[0]
        + j[1] * dydx3[1]
        + 1 / Re * (dydx24[1] + 2 * dydx22dx32[1] + dydx34[1])
    )
    Theta3 = (
        dydx1[2]
        - (1 - Bt * (i[1] ** 2 + i[2] ** 2) + dydx2[3] - dydx3[2]) * j[3]
        + gm * c12h * dydx2[4]
        + c22h * dydx2[5]
        + j[2] * dydx2[2]
        + j[3] * dydx2[3]
        + 1 / Re * (dydx24[2] + 2 * dydx22dx32[2] + dydx34[2])
    )
    Theta4 = (
        dydx1[3]
        + (1 - Bt * (i[1] ** 2 + i[2] ** 2) + dydx2[3] - dydx3[2]) * j[2]
        + gm * c12h * dydx3[4]
        + c22h * dydx3[5]
        + j[2] * dydx3[2]
        + j[3] * dydx3[3]
        + 1 / Re * (dydx24[3] + 2 * dydx22dx32[3] + dydx34[3])
    )
    Theta5 = (
        dydx1[4]
        + dydx2[0] * j[4]
        + j[0] * dydx2[4]
        + dydx3[1] * j[4]
        + j[1] * dydx3[4]
        - j[3]
        + 1 / tradf * (j[4] - 1)
        - kappa * (dhdx2[0] + dhdx2[1])
    )
    Theta6 = (
        dydx1[5]
        + dydx2[2] * j[5]
        + j[2] * dydx2[5]
        + dydx3[3] * j[5]
        + j[3] * dydx3[5]
        - H1H2 * j[3]
        + 1 / tradf * (j[4] - 1)
        - kappa * (dhdx2[2] + dhdx2[3])
    )
    Theta1 = abs(Theta1) ** 2
    Theta2 = abs(Theta2) ** 2
    Theta3 = abs(Theta3) ** 2
    Theta4 = abs(Theta4) ** 2
    Theta5 = abs(Theta5) ** 2
    Theta6 = abs(Theta6) ** 2
    loss = Theta1 + Theta2 + Theta3 + Theta4 + Theta5 + Theta6

    initial_loss = 0
    initial_counter = 0
    if i[0] == 0:
        alpha1 = abs(j[4] - 1) ** 2
        alpha2 = abs(j[5] - 1) ** 2
        alpha3 = abs(j[0]) ** 2
        alpha4 = abs(j[1]) ** 2
        alpha5 = abs(j[2]) ** 2
        alpha6 = abs(j[3]) ** 2
        initial_loss_temp = alpha1 + alpha2 + alpha3 + alpha4 + alpha5 + alpha6
        initial_loss += initialgamma * initial_loss_temp
        initial_counter += 1
    if initial_counter != 0:
        initial_loss = 1 / initial_counter * initial_loss

    loss += initial_loss
    return loss


def PINNtrain(model2, input_data, batch_size, num_epochs):
    NUM_BATCHES = int(np.shape(input_data)[0] / batch_size)

    def scheduler(epoch, lr):
        return 0.0001

    callback = keras.callbacks.LearningRateScheduler(scheduler)
    batch_number = 0

    def PINNloss(y_true, y_pred):
        loss = 0
        Plstart = batch_number * batch_size
        Plend = start + batch_size
        Plinput = input_data[Plstart:Plend]
        counter = 0
        pairsx0 = []
        pairsxX = []
        pairsy0 = []
        pairsyY = []
        pairsx0j = []
        pairsxXj = []
        pairsy0j = []
        pairsyYj = []
        counter1 = 0
        counter2 = 0
        counter3 = 0
        counter4 = 0
        for i in Plinput:
            i = tf.reshape(i, (1, 4))
            loss += 1 / batch_size * dummy_function(model2, i, y_pred[counter])
            print(loss)
            if i[0, 1] == 0:
                if counter1 == 0:
                    pairsx0 = i
                    j = tf.reshape(y_pred[counter], (1, 6))
                    pairsx0j = j
                    counter1 = 1
                else:
                    pairsx0 = tf.concat([pairsx0, i], 0)
                    j = tf.reshape(y_pred[counter], (1, 6))
                    pairsx0j = tf.concat([pairsx0j, j], 0)
            elif i[0, 1] == N:
                if counter2 == 0:
                    pairsxX = i
                    j = tf.reshape(y_pred[counter], (1, 6))
                    pairsxXj = j
                    counter2 = 1
                else:
                    pairsxX = tf.concat([pairsxX, i], 0)
                    j = tf.reshape(y_pred[counter], (1, 6))
                    pairsxXj = tf.concat([pairsxXj, j], 0)
            if i[0, 2] == 0:
                if counter3 == 0:
                    pairsy0 = i
                    j = tf.reshape(y_pred[counter], (1, 6))
                    pairsy0j = j
                    counter3 = 1
                else:
                    pairsy0 = tf.concat([pairsy0, i], 0)
                    j = tf.reshape(y_pred[counter], (1, 6))
                    pairsy0j = tf.concat([pairsy0j, j], 0)
            elif i[0, 2] == N:
                if counter4 == 0:
                    pairsyY = i
                    j = tf.reshape(y_pred[counter], (1, 6))
                    pairsyYj = j
                    counter4 = 1
                else:
                    pairsyY = tf.concat([pairsyY, i], 0)
                    j = tf.reshape(y_pred[counter], (1, 6))
                    pairsyYj = tf.concat([pairsyYj, j], 0)
            counter += 1

        boundary_loss = 0
        for i in range(len(pairsx0)):
            for k in range(len(pairsxX)):
                if pairsx0[i][2] == pairsxX[k][2] and pairsx0[i][0] == pairsxX[k][0]:
                    beta1 = abs(pairsx0j[i][0] - pairsxXj[k][0]) ** 2
                    beta2 = abs(pairsx0j[i][1] - pairsxXj[k][1]) ** 2
                    beta3 = abs(pairsx0j[i][2] - pairsxXj[k][2]) ** 2
                    beta4 = abs(pairsx0j[i][3] - pairsxXj[k][3]) ** 2
                    beta5 = abs(pairsx0j[i][4] - pairsxXj[k][4]) ** 2
                    beta6 = abs(pairsx0j[i][5] - pairsxXj[k][5]) ** 2
                    break
            boundary_loss_temp = beta1 + beta2 + beta3 + beta4 + beta5 + beta6
            boundary_loss += boundarygamma * boundary_loss_temp
        for i in range(len(pairsy0)):
            for k in range(len(pairsyY)):
                if pairsy0[i][1] == pairsyY[k][1] and pairsy0[i][0] == pairsyY[k][0]:
                    beta1 = abs(pairsy0j[i][0] - pairsyYj[k][0]) ** 2
                    beta2 = abs(pairsy0j[i][1] - pairsyYj[k][1]) ** 2
                    beta3 = abs(pairsy0j[i][2] - pairsyYj[k][2]) ** 2
                    beta4 = abs(pairsy0j[i][3] - pairsyYj[k][3]) ** 2
                    beta5 = abs(pairsy0j[i][4] - pairsyYj[k][4]) ** 2
                    beta6 = abs(pairsy0j[i][5] - pairsyYj[k][5]) ** 2
                    break
            boundary_loss_temp = beta1 + beta2 + beta3 + beta4 + beta5 + beta6
            boundary_loss += boundarygamma * boundary_loss_temp

        boundary_loss = tf.cast(boundary_loss, dtype=tf.float64)
        samples = len(pairsx0) + len(pairsy0)
        loss += 1 / samples * boundary_loss

        print(loss)

        return loss

    model1 = Sequential()
    model1.add(Layers.Input(shape=(4,)))
    model1.add(Layers.Dense(100, activation="tanh"))
    for _ in range(7):
        model1.add(Layers.Dense(100, activation="tanh"))
    model1.add(Layers.Dense(6, activation="tanh"))
    model1.compile(optimizer="adam", loss=PINNloss)
    model1.summary()

    model2.set_weights(model1.get_weights())

    for k in range(num_epochs):
        batch_number = 0
        for i in range(NUM_BATCHES):
            print(f"Epochs: {k}, Batches: {i}")
            start = i * batch_size
            end = start + batch_size
            # print(input_data[start:end])
            model1.fit(
                input_data[start:end],
                input_data[start:end],
                epochs=1,
                batch_size=batch_size,
                callbacks=[callback],
            )
            batch_number += 1
            model2.set_weights(model1.get_weights())

    return model1


""" input_data = np.random.random((1000, 4))
input_data = np.reshape(input_data[0], (1,4))
dydx1, dydx2, dydx3, dydx24, dydx34, dydx22dx32, dhdx2 = derivatives(model2, input_data)
print(dydx1) """

""" input_data = np.random.random((1000, 4))
input_data2 = np.random.random((100, 3))
input_data3 = np.zeros((100, 1))
input_data2 = np.concatenate([input_data3, input_data2], axis=1)
input_data = np.concatenate([input_data2, input_data])
print(np.shape(input_data))
print(input_data[0]) """

t_data = np.random.random((25, 1))
x_data = np.zeros((25, 1))
X_data = np.zeros((25, 1)) + N * np.ones((25, 1))
y_data = np.random.random((25, 1))
S_data = np.random.random((25, 1))
input_data = np.concatenate([t_data, x_data, y_data, S_data], axis=1)
input_data2 = np.concatenate([t_data, X_data, y_data, S_data], axis=1)

t_data = np.random.random((25, 1))
x_data = np.random.random((25, 1))
y_data = np.zeros((25, 1))
Y_data = np.zeros((25, 1)) + N * np.ones((25, 1))
S_data = np.random.random((25, 1))
input_data3 = np.concatenate([t_data, x_data, y_data, S_data], axis=1)
input_data4 = np.concatenate([t_data, x_data, Y_data, S_data], axis=1)

input = np.concatenate([input_data, input_data2, input_data3, input_data4], axis=0)
# output_data = np.random.random((1000, 6))
print(input)
batch_size = 100
num_epochs = 5
model = PINNtrain(model2, input, batch_size, num_epochs)
model.summary()

""" NUM_BATCHES = int(np.shape(input_data)[0] / batch_size) 
print(NUM_BATCHES)


for i in range(NUM_BATCHES):
    start = i*batch_size
    end = start + batch_size
    print(np.shape(input_data[start:end]))
    print(i)

start = 100
end = 150

model2.fit(input_data[start:end], output_data[start:end], epochs=1, batch_size=batch_size) """


""" input_data = np.random.random((5, 4))  # Replace this with your actual input data
print(input_data)
# Generate initial pseudo-labels
pseudo_labels = model.predict(input_data)
print(pseudo_labels)

# Train the model using pseudo-labels
model.fit(input_data, input_data, epochs=1, batch_size=32, validation_split=0.2) """
