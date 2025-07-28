## Tensorflow Keras and rest of the packages

import cmath
import math
import time

# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input

# from keras.models import load_model, save_model
from keras.optimizers import Adam, RMSprop

# from keras.saving import register_keras_serializable
# from matplotlib.patches import Circle
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint


def plot_history_by_key(keys):
    import matplotlib.pyplot as plt

    for key in keys:
        plt.plot(
            history.history[key], marker="o", markersize=0.0, linewidth=1.0, label=key
        )
    plt.xlabel("epoch")
    plt.legend()
    plt.show()
    return history.history[key]


class StopTrainingOnLoss(tf.keras.callbacks.Callback):
    def __init__(self, target_loss):
        super(StopTrainingOnLoss, self).__init__()
        self.target_loss = target_loss

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("loss")
        if current_loss is not None and current_loss <= self.target_loss:
            print(f"\nReached target loss of {self.target_loss}. Stopping training.")
            self.model.stop_training = True


class AdaptiveSamplingCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if epoch > epochs / 2:
            predictions = self.model.predict(
                t_train, verbose=0
            )  # Predicciones actuales
            for i in range(np.size(predictions)):
                if predictions[i] < delta and predictions[i] > 0:
                    t_train[i] = t_train[i] + delta
                if predictions[i] > -delta and predictions[i] < 0:
                    t_train[i] = t_train[i] - delta


class ODE_2nd(tf.keras.Model):

    def set_ODE_param(self, t0, x0, x0_grad, A, aux, aux2):
        """
        Set parameters and initial conditions for the ODE
        """
        self.t0 = tf.constant([t0], dtype=tf.float32)
        self.x0_true = tf.constant(x0, dtype=tf.float32)
        self.x0_true_grad = tf.constant(x0_grad, dtype=tf.float32)
        self.A = tf.constant(A, dtype=tf.float32)
        self.aux = tf.constant(aux, dtype=tf.float32)
        self.aux2 = tf.constant(aux2, dtype=tf.float32)
        self.loss_ode_tracker = tf.keras.metrics.Mean(name="lossODE")
        # self.N=tf.constant(N,dtype=tf.float32)

    def train_step(self, data):
        """
        Training ocurrs here
        """
        t, x_true = data
        # *We have 2 initial conditions r0 and the gradient of r0
        with tf.GradientTape() as tape:
            # * Initial conditions
            tape.watch(self.t0)
            tape.watch(self.x0_true)
            tape.watch(self.x0_true_grad)
            tape.watch(t)
            # * The gradient will also output the derivative with respect to z for both outputs.
            with tf.GradientTape() as tape0:
                tape0.watch(self.t0)
                x0_pred = self(self.t0, training=False)
                tape0.watch(x0_pred)
            dx_dtpred0 = tape0.gradient(x0_pred, self.t0)
            with tf.GradientTape() as tape1:
                tape1.watch(t)
                with tf.GradientTape() as tape2:
                    tape2.watch(t)
                    x = self(t, training=False)
                    tape2.watch(x)
                dx_dt = tape2.gradient(x, t)
                tape1.watch(x)
                tape1.watch(dx_dt)
            d2x_dt2 = tape1.gradient(dx_dt, t)
            tape.watch(x)
            tape.watch(dx_dt)
            tape.watch(d2x_dt2)
            dx_dt = tf.squeeze(dx_dt)
            d2x_dt2 = tf.squeeze(d2x_dt2)
            dx_dt = tf.reshape(dx_dt, shape=x.shape)
            d2x_dt2 = tf.reshape(d2x_dt2, shape=x.shape)

            # * Loss ODE
            lossODE = self.compiled_loss(d2x_dt2, -self.A * x)

            mse_manual = tf.reduce_mean(tf.square(d2x_dt2 + self.A * x))
            # *Initial conditions loss
            lossBC = self.compiled_loss(x0_pred, self.x0_true) + self.compiled_loss(
                dx_dtpred0, self.x0_true_grad
            )
            # lossNN = self.compiled_loss(tf.exp(tf.abs(x)), self.x0_true_grad) * 0.0001

            # * "Chinchetas" loss
            aux_pred = self(self.aux2, training=False)
            loss = lossODE + lossBC + self.compiled_loss(self.aux, aux_pred) * 5

        # tf.print(x)
        # tf.print(mse_manual)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(x_true, x)
        self.loss_ode_tracker.update_state(lossODE)

        metrics = {m.name: m.result() for m in self.metrics}
        metrics.pop("mean_squared_error")
        metrics["lossreal"] = loss
        metrics["lossODE"] = lossODE
        metrics["lossBC"] = lossBC
        # metrics["lossNN"] = lossNN
        metrics["t"] = t
        metrics["x0"] = x0_pred
        metrics["x0_grad"] = dx_dtpred0
        metrics["x"] = x
        return metrics


A = 1

# *We set the inicial conditions
x0 = 1
x0_grad = 0


# * We generate the values of z within the domain.
N_train_max = 200
tmax = 20

# * It gives the number of "chichetas" used
N_intervalos = 5

i = 0

salto = int(N_train_max / N_intervalos)
salto_t = tmax / N_intervalos
norm = N_train_max

N_train = 0
tmed = 0
t0 = 0
delta = 0.1
aux = []
aux2 = []


# * Input and output neurons (from the data)
input_neurons = 1
output_neurons = 1

# * Hiperparameters
batch_size = 1
epochs = 5000


stop_on_loss_callback = StopTrainingOnLoss(target_loss=0.00001)


# * Stops after certain epochs without improving and safe the best weight
#! If the simulation ends normally instead of by this callback, the program will take last weights not best
callbacks = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=700, restore_best_weights=True
)
# * We create an auxiliary y_train.
start_time = time.time()
repeticion = 10
for j in range(repeticion):
    for i in range(N_intervalos):

        # Build the model
        checkpoint_callback = ModelCheckpoint(
            filepath=f"harmonic_oscillator_models/pesos_inter={N_intervalos}_t={tmax}_rep={j}.h5",
            monitor="loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            verbose=0,
        )

        # * Define the model
        initializer = tf.keras.initializers.GlorotUniform(seed=5)
        activation = "tanh"
        input = Input(shape=(input_neurons,))
        x = Dense(500, activation=activation, kernel_initializer=initializer)(input)
        x = Dense(500, activation=activation, kernel_initializer=initializer)(x)
        x = Dense(500, activation=activation, kernel_initializer=initializer)(x)
        # x = Dense(500, activation=activation, kernel_initializer=initializer)(x)
        # x = Dense(500, activation=activation, kernel_initializer=initializer)(x)
        output = Dense(output_neurons, kernel_initializer=initializer, activation=None)(
            x
        )

        model = ODE_2nd(input, output)

        # *Define the metrics, optimizer and loss
        loss = tf.keras.losses.MeanSquaredError()
        metrics = tf.keras.metrics.MeanSquaredError()
        optimizer = Adam(learning_rate=0.00001)  # standart 0.0001

        model.compile(
            loss=loss, optimizer=optimizer, metrics=metrics, run_eagerly=False
        )
        model.summary()

        N_train = N_train + salto
        tmed = tmed + salto_t

        t_train = np.linspace(t0, tmed, N_train)
        t_train = np.reshape(t_train, (N_train, 1))

        y_train = np.zeros((t_train.shape[0], 1))

        # * Set ODE parameters and initial conditions
        model.set_ODE_param(t0=[t0], x0=x0, x0_grad=x0_grad, A=A, aux=aux, aux2=aux2)
        # * Saves the best weights
        # history=model.fit(z_train, y_train,batch_size=1, epochs=epochs,verbose=1,
        #                 callbacks=[stop_on_loss_callback]) #,shuffle=False)

        history = model.fit(
            t_train,
            y_train,
            batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=[checkpoint_callback],
        )  # ,shuffle=False)

        model.load_weights(
            f"harmonic_oscillator_models/pesos_inter={N_intervalos}_t={tmax}_rep={j}.h5"
        )
        x_pred = model.predict(t_train)
        aux.append(x_pred[N_train - 1])
        aux2.append(t_train[N_train - 1])
        # t0 = t_train[N_train - 1]
        # x0 = x_pred[N_train - 1]

        # model.set_ODE_param(t0=[t0], x0=x0, x0_grad=x0_grad, A=A, aux=temp, aux2=temp2)

        # # x0_NN = model(model.t0, training=False)
        # with tf.GradientTape() as tape:
        #     tape.watch(model.t0)
        #     x0_pred = model(model.t0, training=False)
        # x0_grad = tape.gradient(x0_pred, model.t0)

    model.load_weights(
        f"harmonic_oscillator_models/pesos_inter={N_intervalos}_t={tmax}_rep={j}.h5"
    )
    x_pred = model.predict(t_train)

    chincheta_aux = []
    chinchetas = np.array(aux)
    for l in aux2:
        chincheta_aux.append(l[0])

    t_RK = np.linspace(0, 20, N_train_max)
    cos_sol = np.cos(t_RK)
    err = 0
    for i in range(np.size(t_RK)):
        err = err + np.abs(cos_sol[i] - x_pred[i])
    err = err[0]

    # * We save the PINN data trajectories
    ruta_chinchetas = (
        f"harmonic_oscillator_models/DBC/Ninter={N_intervalos}_t={tmax}_rep={j}.txt"
    )

    with open(ruta_chinchetas, "w") as archivo:
        for valor1, valor2 in zip(chincheta_aux, chinchetas[:, 0]):
            archivo.write(f"{valor1}\t{valor2}\n")

    # * We save the PINN loss history
    ruta_loss = (
        f"harmonic_oscillator_models/loss/Ninter={N_intervalos}_t={tmax}_rep={j}.txt"
    )

    with open(ruta_loss, "w") as archivo:
        for valor1, valor2, valor3 in zip(
            history.history["lossreal"],
            history.history["lossODE"],
            history.history["lossBC"],
        ):
            archivo.write(f"{valor1}\t{valor2}\t{valor3}\n")
    min_ODE = 1
    min_real = 1
    min_BC = 1
    for i in range(len(history.history["lossreal"])):
        if history.history["lossreal"][i] < min_real:
            min_real = history.history["lossreal"][i]
        if history.history["lossODE"][i] < min_ODE:
            min_ODE = history.history["lossODE"][i]
        if history.history["lossBC"][i] < min_BC:
            min_BC = history.history["lossBC"][i]

    # for layer in model.layers:
    #     layer.trainable = False
    # model.compile(loss=loss, optimizer=optimizer, metrics=metrics, run_eagerly=False)
    # N_val = N_train_max * 100

    # t_train = np.linspace(t0, tmax, N_val)
    # t_train = np.reshape(t_train, (N_val, 1))
    # y_train = np.zeros((t_train.shape[0], 1))

    # history = model.fit(
    #     t_train,
    #     y_train,
    #     batch_size,
    #     epochs=1,
    #     verbose=0,
    #     callbacks=[checkpoint_callback],
    # )  #

    min_ODE_val = history.history["lossODE"][0]

    ruta_min = f"harmonic_oscillator_models/min/Ninter={N_intervalos}_t={tmax}.txt"

    with open(ruta_min, "a") as archivo:

        archivo.write(f"{min_real}\t{min_BC}\t{min_ODE}\t{min_ODE_val}\t{err}\n")
    aux = []
    aux2 = []
    tmed = 0
    N_train = 0


# fig, ax = plt.subplots(dpi=100)

# # * PINN
# ax.plot(
#     t_train,
#     x_pred,
#     marker="o",
#     markersize=3.0,
#     linestyle="solid",
#     linewidth=0,
#     label=r"$\theta(\xi)$ PINN",
# )

# t_RK = np.linspace(0, 20, 10 * N_train_max)
# cos_sol = np.cos(t_RK)

# # * "Solución exacta"
# ax.plot(
#     t_RK,
#     cos_sol,
#     marker="o",
#     markersize=0,
#     linestyle="-",
#     linewidth=1,
#     label=r"$\theta(\xi)$ Analytical",
# )
# # taylor = 1 - t_RK**2 / 2
# # # *"Taylor exacto"
# # ax.plot(
# #     t_RK,
# #     taylor,
# #     marker="o",
# #     markersize=0,
# #     linestyle="-",
# #     linewidth=1,
# #     label=r"Taylor analitico",
# # )
# # x0_NN = np.squeeze(x0_NN)
# # dx0_NN = np.squeeze(dx0_NN)
# # dx02_NN = np.squeeze(dx02_NN)

# # taylor_NN = x0_NN + dx0_NN * t_RK + 0.5 * dx02_NN * t_RK**2
# # # *"Taylor NN"
# # ax.plot(
# #     t_RK,
# #     taylor_NN,
# #     marker="o",
# #     markersize=0,
# #     linestyle="-",
# #     linewidth=1,
# #     label=r"Taylor NN",
# # )

# # *"Chinchetas"
# ax.plot(aux2, chinchetas, "o", markersize=6, label="chinchetas")

# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

# ax.set_title(rf"A={A}", fontsize=14)
# ax.set_xlabel(r"$t(s)$", fontsize=16)
# ax.set_ylabel(r"$x(m)$", fontsize=16)

# # * Adds a legend
# ax.legend()


# plt.grid(False)  # Optional
# # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
# #                mode="expand", borderaxespad=0, ncol=2,fontsize=12)
# ax.legend(fontsize=14, frameon=False)

# fig.set_size_inches(6, 6)

# plt.show()

# # * Summarize history for loss
# plt.plot(
#     np.log10(history.history["lossreal"]),
#     marker="o",
#     markersize=0.0,
#     linewidth=1,
#     label="loss",
# )
# plt.plot(
#     np.log10(history.history["lossODE"]),
#     marker="o",
#     markersize=0,
#     linewidth=1,
#     label="lossODE",
# )
# plt.plot(
#     np.log10(history.history["lossBC"]),
#     marker="o",
#     markersize=0,
#     linewidth=1,
#     label="lossBC",
# )
# ax.set_xlabel("época", fontsize=16)

# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)


# plt.legend()

# plt.legend(fontsize=14, frameon=False)
# plt.show()


end_time = time.time()
execution_time = end_time - start_time
