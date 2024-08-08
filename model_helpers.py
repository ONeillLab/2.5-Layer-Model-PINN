import numpy as np
import tensorflow as tf
import keras
from keras import Sequential
from keras import layers as Layers
from keras import Model
from name_list import *

""" # Example input data
input_data = np.random.random((1, 4))  # A single input example

y = tf.constant(5.0)
x = tf.constant(input_data[0,0])# dtype="float32")
print(x)
print(y) """

def derivatives(model, input_data):
    x1 = tf.constant(input_data[0,0])
    x2 = tf.constant(input_data[0,1])
    x3 = tf.constant(input_data[0,2])
    x4 = tf.constant(input_data[0,3])

    dydx24 = []
    dydx34 = []
    dydx1 = []
    dydx2 = []
    dydx3 = []
    dydx22dx32 = []
    dhdx2 = []

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

        dydx24.append(dy_dx24)

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

        dydx34.append(dy_dx34)

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
                dy_dx22 = ggg.gradient(dy_dx3, x2)
            dy_dx22dx3 = gg.gradient(dy_dx32, x3)
        dy_dx22dx32 = g.gradient(dy_dx33, x3)

        dydx22dx32.append(dy_dx22dx32)

    for i in range(6):
        with tf.GradientTape() as g:
            g.watch(x1)
            y = (model(tf.reshape([x1, x2, x3, x4], (1, 4))))[:, i]
        dy_dx1 = g.gradient(y, x1)
        dydx1.append(dy_dx1)

        with tf.GradientTape() as g:
            g.watch(x2)
            y = (model(tf.reshape([x1, x2, x3, x4], (1, 4))))[:, i]
        dy_dx2 = g.gradient(y, x2)
        dydx2.append(dy_dx2)

        with tf.GradientTape() as g:
            g.watch(x3)
            y = (model(tf.reshape([x1, x2, x3, x4], (1, 4))))[:, i]
        dy_dx3 = g.gradient(y, x3)
        dydx3.append(dy_dx3)

    with tf.GradientTape() as g:
        g.watch(x2)
        with tf.GradientTape() as gg:
            g.watch(x2) ### ERRRORRRRRR
            y = (model(tf.reshape([x1, x2, x3, x4], (1, 4))))[:, 4]
        dh1_dx1 = g.gradient(y, x2)
    print(dh1_dx1)
    dh1_dx12 = gg.gradient(dh1_dx1, x2)

    dhdx2.append(dh1_dx12)

    with tf.GradientTape() as g:
        g.watch(x2)
        with tf.GradientTape() as gg:
            g.watch(x2)
            y = (model(tf.reshape([x1, x2, x3, x4], (1, 4))))[:, 5]
        dh2_dx1 = g.gradient(y, x2)
    dh2_dx12 = gg.gradient(dh2_dx1, x2)

    dhdx2.append(dh2_dx12)

    with tf.GradientTape() as g:
        g.watch(x3)
        with tf.GradientTape() as gg:
            g.watch(x3)
            y = (model(tf.reshape([x1, x2, x3, x4], (1, 4))))[:, 4]
        dh1_dx2 = g.gradient(y, x3)
    dh1_dx22 = gg.gradient(dh1_dx2, x3)

    dhdx2.append(dh1_dx22)

    with tf.GradientTape() as g:
        g.watch(x3)
        with tf.GradientTape() as gg:
            g.watch(x3)
            y = (model(tf.reshape([x1, x2, x3, x4], (1, 4))))[:, 5]
        dh2_dx2 = g.gradient(y, x3)
    dh2_dx22 = gg.gradient(dh2_dx2, x3)

    dhdx2.append(dh2_dx22)

    dydx1 = np.array(dydx1) 
    dydx2 = np.array(dydx2)
    dydx3 = np.array(dydx3)
    dydx24 = np.array(dydx24)
    dydx34 = np.array(dydx34)
    dydx22dx32 = np.array(dydx22dx32)
    dhdx2 = np.array(dhdx2)

    return dydx1, dydx2, dydx3, dydx24, dydx34, dydx22dx32, dhdx2

def PINNloss(model, input_tensor):
    def loss(y_true, y_pred):
        print(input_tensor)
        error = 0
        pde_error = 0
        initial_error = 0
        boundary_error = 0
        num_outputs = 0
        pairsx0 = []
        pairsxX = []
        pairsy0 = []
        pairsyY = []
        pairsx0j = []
        pairsxXj = []
        pairsy0j = []
        pairsyYj = []
        for i in input_tensor:
            j = y_pred[num_outputs]
            dydx1, dydx2, dydx3, dydx24, dydx34, dydx22dx32, dhdx2 = derivatives(
                model, i
            )
            Theta1 = (
                dydx1[0]
                - (1 - Bt(i[1] ** 2 + i[2] ** 2) + dydx2[1] - dydx3[0]) * j[1]
                + c12h * dydx2[4]
                + c22h * dydx2[5]
                + j[0] * dydx2[0]
                + j[1] * dydx2[1]
                + 1 / Re * (dydx24[0] + 2 * dydx22dx32[0] + dydx34[0])
            )
            Theta2 = (
                dydx1[1]
                + (1 - Bt(i[1] ** 2 + i[2] ** 2) + dydx2[1] - dydx3[0]) * j[0]
                + c12h * dydx3[4]
                + c22h * dydx3[5]
                + j[0] * dydx3[0]
                + j[1] * dydx3[1]
                + 1 / Re * (dydx24[1] + 2 * dydx22dx32[1] + dydx34[1])
            )
            Theta3 = (
                dydx1[2]
                - (1 - Bt(i[1] ** 2 + i[2] ** 2) + dydx2[3] - dydx3[2]) * j[3]
                + gm * c12h * dydx2[4]
                + c22h * dydx2[5]
                + j[2] * dydx2[2]
                + j[3] * dydx2[3]
                + 1 / Re * (dydx24[2] + 2 * dydx22dx32[2] + dydx34[2])
            )
            Theta4 = (
                dydx1[3]
                + (1 - Bt(i[1] ** 2 + i[2] ** 2) + dydx2[3] - dydx3[2]) * j[2]
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
            pde_loss = Theta1 + Theta2 + Theta3 + Theta4 + Theta5 + Theta6
            pde_error += pde_loss
            if i[0] == 0:
                alpha1 = abs(j[4] - 1) ** 2
                alpha2 = abs(j[5] - 1) ** 2
                alpha3 = abs(j[0]) ** 2
                alpha4 = abs(j[1]) ** 2
                alpha5 = abs(j[2]) ** 2
                alpha6 = abs(j[3]) ** 2
                initial_loss = alpha1 + alpha2 + alpha3 + alpha4 + alpha5 + alpha6
                initial_error += initialgamma * initial_loss
            if i[1] == 0:
                pairsx0.append(i)
                pairsx0j.append(j)
            elif i[1] == N:
                pairsxX.append(i)
                pairsxXj.append(j)
            if i[2] == 0:
                pairsy0.append(i)
                pairsy0j.append(j)
            elif i[2] == N:
                pairsyY.append(i)
                pairsyYj.append(j)
            num_outputs += 1
        for i in range(len(pairsx0)):
            for k in range(len(pairsxX)):
                if pairsx0[i][2] == pairsxX[k][2]:
                    beta1 = abs(pairsx0j[i][0] - pairsxXj[k][0]) ** 2
                    beta2 = abs(pairsx0j[i][1] - pairsxXj[k][1]) ** 2
                    beta3 = abs(pairsx0j[i][2] - pairsxXj[k][2]) ** 2
                    beta4 = abs(pairsx0j[i][3] - pairsxXj[k][3]) ** 2
                    beta5 = abs(pairsx0j[i][4] - pairsxXj[k][4]) ** 2
                    beta6 = abs(pairsx0j[i][5] - pairsxXj[k][5]) ** 2
                    break
            boundary_loss = beta1 + beta2 + beta3 + beta4 + beta5 + beta6
            boundary_error += boundarygamma * boundary_loss
        for i in range(len(pairsy0)):
            for k in range(len(pairsyY)):
                if pairsy0[i][1] == pairsyY[k][1]:
                    beta1 = abs(pairsy0j[i][0] - pairsyYj[k][0]) ** 2
                    beta2 = abs(pairsy0j[i][1] - pairsyYj[k][1]) ** 2
                    beta3 = abs(pairsy0j[i][2] - pairsyYj[k][2]) ** 2
                    beta4 = abs(pairsy0j[i][3] - pairsyYj[k][3]) ** 2
                    beta5 = abs(pairsy0j[i][4] - pairsyYj[k][4]) ** 2
                    beta6 = abs(pairsy0j[i][5] - pairsyYj[k][5]) ** 2
                    break
            boundary_loss = beta1 + beta2 + beta3 + beta4 + beta5 + beta6
            boundary_error += boundarygamma * boundary_loss
        error = pde_error + initial_error + boundary_error
        return error

    return loss


def build_model():
    # Define the model
    model = Sequential()
    input_tensor = model.add(Layers.Input(shape=(4,)))
    model.add(Layers.Dense(100, activation="tanh"))  # input_shape=(4,),
    for _ in range(7):
        model.add(Layers.Dense(100, activation="tanh"))
    model.add(Layers.Dense(6, activation="tanh"))
    model.compile(
        optimizer="adam", loss="mean_squared_error"
    )  # PINNloss(model, input_tensor)
    model.summary()

    return model


def build_model2():
    # Define the model
    model = Sequential()

    input_tensor = Layers.Input(shape=(4,))
    hidden1 = Layers.Dense(100, activation="tanh")(input_tensor)
    hidden2 = Layers.Dense(100, activation="tanh")(hidden1)
    hidden3 = Layers.Dense(100, activation="tanh")(hidden2)
    hidden4 = Layers.Dense(100, activation="tanh")(hidden3)
    hidden5 = Layers.Dense(100, activation="tanh")(hidden4)
    hidden6 = Layers.Dense(100, activation="tanh")(hidden5)
    hidden7 = Layers.Dense(100, activation="tanh")(hidden6)
    hidden8 = Layers.Dense(100, activation="tanh")(hidden7)
    output = Layers.Dense(6, activation="tanh")(hidden8)
    model = Model(input_tensor, output)
    model.compile(optimizer="adam", loss=PINNloss(model, input_tensor))

    model.summary()

    return model

input_data = np.random.random((1, 4)) 
model = build_model()
input_tensor = tf.convert_to_tensor(input_data) #dtype=tf.float32)
dydx1, dydx2, dydx3, dydx24, dydx34, dydx22dx32, dhdx2 = derivatives(model, input_tensor)
print(dydx1)
print(dydx2)
print(dydx3)
print(dydx24)
print(dydx34)
print(dydx22dx32)
print(dhdx2)

