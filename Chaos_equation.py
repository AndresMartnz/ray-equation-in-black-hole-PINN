import keras as k

import numpy as np
import tensorflow as tf

# import tensorflow.keras as k2
from keras.layers import Dense, Input
from keras.models import load_model, save_model
from keras.optimizers import Adam, RMSprop
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint


class StopTrainingOnLoss(tf.keras.callbacks.Callback):
    def __init__(self, target_loss):
        super(StopTrainingOnLoss, self).__init__()
        self.target_loss = target_loss

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("loss")
        if current_loss is not None and current_loss <= self.target_loss:
            # print(f"\nReached target loss of {self.target_loss}. Stopping training.")
            self.model.stop_training = True


class FourierLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, scale=10.0):
        super(FourierLayer, self).__init__()
        self.output_dim = output_dim
        self.scale = scale

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.B = self.add_weight(
            shape=(input_dim, self.output_dim),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            trainable=False,
        )

    def call(self, inputs):
        projection = tf.matmul(inputs, self.B) * self.scale
        return tf.concat([tf.sin(projection), tf.cos(projection)], axis=-1)


class ODE_2nd(tf.keras.Model):
    def set_ODE_param(self, x0, y0, N_train):
        """
        Set parameters and initial conditions for the ODE
        """
        self.x0 = tf.constant([x0], dtype=tf.float32)
        self.y0_true = tf.constant(y0, dtype=tf.float32)
        self.auxx = tf.constant(auxx, dtype=tf.float32)
        self.aux2 = tf.constant(aux2, dtype=tf.float32)
        self.n_train = tf.constant(N_train, dtype=tf.float32)

    def train_step(self, data):
        """
        Training ocurrs here
        """
        x, y_true = data
        with tf.GradientTape() as tape:
            # * Initial conditions
            tape.watch(self.x0)
            tape.watch(self.y0_true)
            tape.watch(x)

            with tf.GradientTape() as tape0:
                tape0.watch(self.x0)
                y0_pred = self(self.x0, training=False)
                tape0.watch(y0_pred)

            with tf.GradientTape() as tape1:
                tape1.watch(x)
                y = self(x, training=False)
                tape1.watch(x)
            dy_dx = tape1.batch_jacobian(y, x)
            # dy_dx = tape1.jacobian(y, x)
            # dy_dx = tf.squeeze(dy_dx)
            # dy_dx = tf.reshape(dy_dx, shape=y.shape)
            tape.watch(x)
            tape.watch(y)
            tape.watch(dy_dx)

            # aux = tf.reshape(aux, shape=y.shape)

            a = tf.constant(10.0, dtype=tf.float32)  # SGR: added
            b = tf.constant(28.0, dtype=tf.float32)  # SGR: added
            c = tf.constant(8.0 / 3.0, dtype=tf.float32)  # SGR: added
            # ? Alternative ODE's order (2)
            lossODE = (
                self.compiled_loss(dy_dx[:, 0], a * (y[:, 1] - y[:, 0]))
                + self.compiled_loss(dy_dx[:, 1], y[:, 0] * (b - y[:, 2]) - y[:, 1])
                + self.compiled_loss(dy_dx[:, 2], y[:, 0] * y[:, 1] - c * y[:, 2])
            ) / self.n_train

            # * initial condition loss
            lossBC = self.compiled_loss(y0_pred, self.y0_true) * 10
            loss = lossODE + lossBC

        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(y_true, y)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.pop("mean_squared_error")
        metrics["lossreal"] = loss
        metrics["lossODE"] = lossODE
        metrics["lossBC"] = lossBC
        metrics["x_mean"] = tf.reduce_mean(x)  # Guarda solo el promedio de x
        metrics["y1_mean"] = tf.reduce_mean(y0_pred[:, 0])
        metrics["y2_mean"] = tf.reduce_mean(y0_pred[:, 1])
        metrics["y3_mean"] = tf.reduce_mean(y0_pred[:, 2])
        # metrics["x"] = x
        # metrics["y1"] = y0_pred[:, 0]
        # metrics["y2"] = y0_pred[:, 1]
        # metrics["y3"] = y0_pred[:, 2]
        return metrics


def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])


def runge_kutta(f, y0, t_0, t_f, h, *args):
    t_values = np.arange(t_0, t_f + h, h)
    print(t_values.shape)

    n = len(t_values)
    # inicializo una matriz que almacenará los valores de las variables
    # del sistema de ecuaciones diferenciales en cada paso de integración.
    # ira almacenando los valores calculados mediante RK
    y_values = np.zeros((n, len(y0)))
    y_values[0] = y0

    for i in range(1, n):
        k1 = h * f(t_values[i - 1], y_values[i - 1], *args)
        k2 = h * f(t_values[i - 1] + 0.5 * h, y_values[i - 1] + 0.5 * k1, *args)
        k3 = h * f(t_values[i - 1] + 0.5 * h, y_values[i - 1] + 0.5 * k2, *args)
        k4 = h * f(t_values[i - 1] + h, y_values[i - 1] + k3, *args)
        y_values[i] = y_values[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t_values, y_values


ruta_ini = "input_chaos.txt"

with open(ruta_ini, "r") as archivo:
    for linea in archivo:

        valores = linea.split()

        N_train = int(valores[0])
        N_intervalos = int(valores[1])
        epochs = int(valores[2])
        xmax = float(valores[3])

lr = 0.001
salto_x = xmax / N_intervalos

# Calcular trayectoria
# Parámetros del sistema de Lorenz
sigma = 10
rho = 28
beta = 8 / 3

# Condiciones iniciales
y0 = np.array([1, 1, 1])  # [x0, y0, z0]
t_0 = 0
t_f = xmax
h = 0.0001
fourier_features_dim = 100  # Número de Fourier Features
scale = 10.0  # Escalado de la transformación


# Resolver el sistema de ecuaciones diferenciales de Lorenz
t_values, y_values = runge_kutta(lorenz, y0, t_0, t_f, h, sigma, rho, beta)

stop_on_loss_callback = StopTrainingOnLoss(target_loss=0.000005)

auxx = []
aux2 = []

# inicio normal
x0 = 0
xmed = x0


# * Input and output neurons (from the data)
input_neurons = 1
output_neurons = 3

# * Hiperparameters
batch_size = 25
print("N_train=", N_train)
print("N_intervalos=", N_intervalos)
print("x_max=", xmax)
print("epochs=", epochs)

# * Stops after certain epochs without improving and safe the best weight
#! If the simulation ends normally instead of by this callback, the program will take last weights not best
callbacks = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=1000, restore_best_weights=True
)

for chinch in range(N_intervalos):

    checkpoint_callback = ModelCheckpoint(
        filepath=f"p-caos/pesos_inter={N_intervalos}_epochs={epochs}_x={xmax}.h5",
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
    fourier_layer = FourierLayer(output_dim=fourier_features_dim, scale=scale)(input)
    x = Dense(500, activation=activation, kernel_initializer=initializer)(fourier_layer)
    x = Dense(500, activation=activation, kernel_initializer=initializer)(x)
    x = Dense(500, activation=activation, kernel_initializer=initializer)(x)
    # x = Dense(500, activation=activation, kernel_initializer=initializer)(x)
    # x = Dense(500, activation=activation, kernel_initializer=initializer)(x)
    output = Dense(output_neurons, kernel_initializer=initializer, activation=None)(x)

    # * Build the model
    model = ODE_2nd(input, output)

    # *Define the metrics, optimizer and loss
    loss = tf.keras.losses.MeanSquaredError()
    metrics = tf.keras.metrics.MeanSquaredError()
    optimizer = Adam(learning_rate=lr)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics, run_eagerly=False)
    model.summary()

    xmed = xmed + salto_x

    x_train = np.linspace(x0, xmed, N_train)
    x_train = np.reshape(x_train, (N_train, 1))
    y_train = np.zeros((x_train.shape[0], 1))

    # * Set ODE parameters and initial conditions
    model.set_ODE_param(x0=[x0], y0=y0, N_train=N_train)

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        callbacks=checkpoint_callback,
    )  # ,shuffle=False)
    model.load_weights(f"p-caos/pesos_inter={N_intervalos}_epochs={epochs}_x={xmax}.h5")
    xy_pred = model.predict(x_train)

    auxx.append(xy_pred[N_train - 1])
    aux2.append(x_train[N_train - 1])

    valores = [x[0] for x in aux2]
    valores2 = [x.tolist() for x in auxx]
    ruta_aux = f"p-caos/aux_Ninter={N_intervalos}_epochs={epochs}_x={xmax}.txt"

    with open(ruta_aux, "a") as archivo:
        archivo.write(
            f"{valores[chinch]}\t{valores2[chinch][0]}\t{valores2[chinch][1]}\t{valores2[chinch][2]}\n"
        )
    x0 = x0 + salto_x
    y0 = np.array([valores2[chinch][0], valores2[chinch][1], valores2[chinch][2]])


x_red = x_train
# model.load_weights(f"modelos_agujero_negro/pesos_inter={N_intervalos}_y={y_ini}_N=200_sim={sim}.h5")
xy_pred = model.predict(x_red)


chincheta_aux = []
chinchetas = np.array(auxx)
for l in aux2:
    chincheta_aux.append(l[0])

# * We save the PINN data trajectories
ruta_chinchetas = (
    f"p-caos/chinchetas_Ninter={N_intervalos}_epochs={epochs}_x={xmax}.txt"
)

with open(ruta_chinchetas, "w") as archivo:
    for valor1, valor2, valor3, valor4 in zip(
        chincheta_aux,
        chinchetas[:, 0],
        chinchetas[:, 1],
        chinchetas[:, 2],
    ):
        archivo.write(f"{valor1}\t{valor2}\t{valor3}\t{valor4}\n")


# * We save the PINN loss history
ruta_loss = f"p-caos/Ninter={N_intervalos}_epochs={epochs}_x={xmax}.txt"

with open(ruta_loss, "w") as archivo:
    for valor1, valor2, valor3 in zip(
        history.history["lossreal"],
        history.history["lossODE"],
        history.history["lossBC"],
    ):
        archivo.write(f"{valor1}\t{valor2}\t{valor3}\n")
