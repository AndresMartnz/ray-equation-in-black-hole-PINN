## Tensorflow Keras and rest of the packages
import keras as k
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
#import tensorflow.keras as k2
from keras.layers import Dense,Input
from keras.optimizers import Adam, RMSprop
from keras.models import load_model, save_model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cmath
import math
from tensorflow import keras
import time


def plot_history_by_key(keys):
  import matplotlib.pyplot as plt
  for key in keys:
    plt.plot(history.history[key],marker='o',markersize=0.0, linewidth=1.0,label=key)
  plt.xlabel('epoch')
  plt.legend()
  plt.show()
  return history.history[key]


class StopTrainingOnLoss(tf.keras.callbacks.Callback):
    def __init__(self, target_loss):
        super(StopTrainingOnLoss, self).__init__()
        self.target_loss = target_loss


    def on_epoch_end(self, epoch, logs=None):
      '''
      choose the error the lost have to achieve for the training to stops
      '''
        current_loss = logs.get('loss')
        if current_loss is not None and current_loss <= self.target_loss:
            print(f"\nReached target loss of {self.target_loss}. Stopping training.")
            self.model.stop_training = True


def r (x,y):
  return (tf.sqrt(x**2+y**2))

def n_ind(x,y,A):
  return (1.0/(1.0-A/r(x,y)))



class ODE_2nd(tf.keras.Model):
    def set_ODE_param(self,z0,x0,dx_dz0,A, auxx, aux2):
        '''
        Set parameters and initial conditions for the ODE
        '''
        self.z0=tf.constant([z0], dtype=tf.float32)
        self.x0_true=tf.constant(x0, dtype=tf.float32)
        self.dx_dz0_true=tf.constant(dx_dz0, dtype=tf.float32)
        self.A=tf.constant(A,dtype=tf.float32)
        self.auxx=tf.constant(auxx,dtype=tf.float32)
        self.aux2=tf.constant(aux2,dtype=tf.float32)

    def train_step(self, data):
        '''
        Training ocurrs here
        '''
        z, x_true = data
        with tf.GradientTape() as tape:
            #* Initial conditions
            tape.watch(self.z0)
            tape.watch(self.x0_true)
            tape.watch(self.dx_dz0_true)
            tape.watch(self.A)
            tape.watch(z)

            with tf.GradientTape() as tape0:
                    tape0.watch(self.z0)
                    x0_pred= self(self.z0,training=False)
                    tape0.watch(x0_pred)

            with tf.GradientTape() as tape1:
                tape1.watch(z)
                x=self(z,training=False)
                tape1.watch(x)
            dx_dz=tape1.jacobian(x,z)
            dx_dz=tf.squeeze(dx_dz)
            dx_dz=tf.reshape(dx_dz,shape=x.shape)
            tape.watch(z)
            tape.watch(x)
            tape.watch(dx_dz)

 
            #* Definition of r and n
            r=tf.math.sqrt(x[:,0]**2+x[:,1]**2)
            n=1.0/(1.0-self.A/r)
            aux=[0.0,0.0,0.0,0.0]
            aux=tf.reshape(aux,shape=x.shape)
  
            #we have differents orders for the loss who gave us differents results
            #? Original ODE's order
            '''
            lossODE= self.compiled_loss(dx_dz[:,0],x[:,2]/n)\
                    +self.compiled_loss(dx_dz[:,1],x[:,3]/n)\
                    +self.compiled_loss(dx_dz[:,2],-self.A*tf.math.pow(n,2)*x[:,0]/tf.math.pow(r,3))\
                    +self.compiled_loss(dx_dz[:,3],-self.A*tf.math.pow(n,2)*x[:,1]/tf.math.pow(r,3))

            '''
            '''
            #? Alternative ODE's order (1)
            lossODE= self.compiled_loss(n*dx_dz[:,0],x[:,2])\
                    +self.compiled_loss(n*dx_dz[:,1],x[:,3])\
                    +self.compiled_loss(tf.math.pow(r,3)*dx_dz[:,2]/tf.math.pow(n,2),-self.A*x[:,0])\
                    +self.compiled_loss(tf.math.pow(r,3)*dx_dz[:,3]/tf.math.pow(n,2),-self.A*x[:,1])
                    #+self.compiled_loss(A/tf.math.pow((x[:,0]*x[:,0]+x[:,1]*x[:,1]),0.5),aux[:,0])

            '''
            
            #? Alternative ODE's order (2) the one with the better results
            lossODE= self.compiled_loss(dx_dz[:,0],x[:,2]/n)\
                    +self.compiled_loss(dx_dz[:,1],x[:,3]/n)\
                    +self.compiled_loss(tf.math.pow(r,3)*dx_dz[:,2],-self.A*tf.math.pow(n,2)*x[:,0])*5\
                    +self.compiled_loss(tf.math.pow(r,3)*dx_dz[:,3],-self.A*tf.math.pow(n,2)*x[:,1])*10
                


            #* initial condition loss
            lossODE= lossODE\
                  + self.compiled_loss(x0_pred,self.x0_true) \
                  
            #* "Chinchetas" loss
            aux_pred=self(self.aux2, training=False)
            loss=lossODE\
                +self.compiled_loss(self.auxx,aux_pred)*5


        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(x_true, x)
        #define the metrics you want to control over the training
        metrics={m.name: m.result() for m in self.metrics}
        metrics.pop('mean_squared_error')
        metrics['z']=z
        metrics['x0']=x0_pred[:,0]
        metrics['y0']=x0_pred[:,1]
        metrics['vx0']=x0_pred[:,2]
        metrics['vy0']=x0_pred[:,3]
        metrics['x']=x[:,0]
        metrics['y']=x[:,1]
        metrics['vx']=x[:,2]
        metrics['vy']=x[:,3]
        metrics['n'] = n
        metrics['r'] = r
        print(self.A)
        return metrics
    

#* We load the imput parameters from an .txt file
ruta_ini = 'black_hole_imput.txt'
with open(ruta_ini, 'r') as archivo:
    for linea in archivo:

        valores = linea.split()
        A=float(valores[0])
        x0=float(valores[1])
        y0=float(valores[2])
        N_train_max=int(valores[3])
        N_intervalos=int(valores[4])
        epochs=int(valores[5])
        lr=float(valores[6])
        repeats=int(valores[7])


#it represents the factor 2GM/c^2
GM_c2= A/2   #it represents the factor GM/c^2
x0_rk = x0; x_ini= x0
y0_rk = y0; y_ini=y0
vx0_rk=1/(1.0/(1.0-A/np.sqrt(x0_rk**2+y0_rk**2)))   ; vx_ini=1/(1.0/(1.0-A/np.sqrt(x0_rk**2+y0_rk**2)))
vy0_rk = 0.0  ; vy_ini=0.0


#* We define the Runge-Kutta
def f_x(t, x, y, vx, vy):
    #* vx's ODE
    return vx/n_xy(x,y)

def f_y(t, x, y, vx, vy):
    #* vy's ODE
    return vy/n_xy(x, y)

def g_x(t, x, y, vx, vy):
    #* ax's ODE
    return -n_xy(x,y)**2*2*GM_c2*x/((x**2+y**2+epsilon2**2)**(3/2))

def g_y(t, x, y, vx, vy):
    #* ay's ODE
    return -n_xy(x,y)**2*2*GM_c2*y/((x**2+y**2+epsilon2**2)**(3/2))

def n_xy(x, y):
    #*Ecuación que calcula el índice de refracción
    return (1/(1-(2*GM_c2/(x**2+y**2+epsilon2**2)**0.5)))

#* fourth order runge kutta
def runge_kutta_4th_order(t, x, y, vx, vy, h):

    k1_x = h * f_x(t, x, y, vx, vy)
    k1_y = h * f_y(t, x, y, vx, vy)
    k1_vx = h * g_x(t, x, y, vx, vy)
    k1_vy = h * g_y(t, x, y, vx, vy)

    k2_x = h * f_x(t + 0.5 * h, x + 0.5 * k1_x, y + 0.5 * k1_y, vx + 0.5 * k1_vx, vy + 0.5 * k1_vy)
    k2_y = h * f_y(t + 0.5 * h, x + 0.5 * k1_x, y + 0.5 * k1_y, vx + 0.5 * k1_vx, vy + 0.5 * k1_vy)
    k2_vx = h * g_x(t + 0.5 * h, x + 0.5 * k1_x, y + 0.5 * k1_y, vx + 0.5 * k1_vx, vy + 0.5 * k1_vy)
    k2_vy = h * g_y(t + 0.5 * h, x + 0.5 * k1_x, y + 0.5 * k1_y, vx + 0.5 * k1_vx, vy + 0.5 * k1_vy)

    k3_x = h * f_x(t + 0.5 * h, x + 0.5 * k2_x, y + 0.5 * k2_y, vx + 0.5 * k2_vx, vy + 0.5 * k2_vy)
    k3_y = h * f_y(t + 0.5 * h, x + 0.5 * k2_x, y + 0.5 * k2_y, vx + 0.5 * k2_vx, vy + 0.5 * k2_vy)
    k3_vx = h * g_x(t + 0.5 * h, x + 0.5 * k2_x, y + 0.5 * k2_y, vx + 0.5 * k2_vx, vy + 0.5 * k2_vy)
    k3_vy = h * g_y(t + 0.5 * h, x + 0.5 * k2_x, y + 0.5 * k2_y, vx + 0.5 * k2_vx, vy + 0.5 * k2_vy)

    k4_x = h * f_x(t + h, x + k3_x, y + k3_y, vx + k3_vx, vy + k3_vy)
    k4_y = h * f_y(t + h, x + k3_x, y + k3_y, vx + k3_vx, vy + k3_vy)
    k4_vx = h * g_x(t + h, x + k3_x, y + k3_y, vx + k3_vx, vy + k3_vy)
    k4_vy = h * g_y(t + h, x + k3_x, y + k3_y, vx + k3_vx, vy + k3_vy)

    new_x = x + (1/6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
    new_y = y + (1/6) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
    new_vx = vx + (1/6) * (k1_vx + 2 * k2_vx + 2 * k3_vx + k4_vx)
    new_vy = vy + (1/6) * (k1_vy + 2 * k2_vy + 2 * k3_vy + k4_vy)

    return new_x, new_y, new_vx, new_vy

#* RK's initial conditions

t0 = 0.0
tfin=2
h = 0.01  # time step
pasos= int((tfin-t0)/h)
epsilon=0.1
epsilon2=0

x_tot=[]
y_tot=[]
vx_tot=[]
vy_tot=[]
t_tot=[]

x_tot.append(x0_rk)
y_tot.append(y0_rk)
vx_tot.append(vx0_rk)
vy_tot.append(vy0_rk)
t_tot.append(t0)

#* We compile the runge-kutta
x_rk=x0_rk; y_rk=y0_rk; vx_rk=vx0_rk; vy_rk=vy0_rk
for i in range(pasos):
    x_rk, y_rk, vx_rk, vy_rk = runge_kutta_4th_order(t0, x_rk, y_rk, vx_rk, vy_rk, h)
    if (((x_rk**2+y_rk**2)**0.5)<A+0.001):

        break
    x_tot.append(x_rk)
    y_tot.append(y_rk)
    vx_tot.append(vx_rk)
    vy_tot.append(vy_rk)
    t0 += h
    t_tot.append(t0)


#* Initial conditions NN
n_ini=n_ind(x_ini, y_ini,A).numpy()
x0=[x_ini,y_ini,vx_ini,vy_ini]
dx_dz0=[vx_ini/n_ini,
        vy_ini/n_ini,
        -n_ini**2*A*x_ini/(r(x_ini,y_ini).numpy())**3,
        -n_ini**2*A*y_ini/(r(x_ini,y_ini).numpy())**3]
print("n(0)=",n_ini)
print('x(0)=',x0)
print('dx_dz(0)=',dx_dz0)




zmax = 2    #range of the prediction function
salto=int(N_train_max/N_intervalos)
salto_z=zmax/N_intervalos
norm=N_train_max

stop_on_loss_callback = StopTrainingOnLoss(target_loss=0.000005)


N_train=0
zmed=0
z0 = 0
auxx=[]
aux2=[]




#* Input and output neurons (from the data)
input_neurons  = 1
output_neurons = 4

#* Hiperparameters
batch_size = 1



#* Stops after certain epochs without improving and safe the best weight
#! If the simulation ends normally instead of by this callback, the program will take last weights not best, for avoiding this, implement also with chekpoints_callbacks
callbacks = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=1000,
                                            restore_best_weights=True)
for sim in range (0,1):
    for chinch in range (N_intervalos):

        checkpoint_callback = ModelCheckpoint(filepath=f'modelos_agujero_negro/pesos_inter={N_intervalos}_y={y_ini}_N={N_train_max}_sim={sim}.h5', 
                                      monitor='loss', 
                                      save_best_only=True, 
                                      save_weights_only=True, 
                                      mode='min',
                                      verbose=1)

        #* Define the model
        initializer = tf.keras.initializers.GlorotUniform(seed=5)
        activation='tanh'
        input=Input(shape=(input_neurons,))
        x=Dense(500, activation=activation,kernel_initializer=initializer)(input)
        x=Dense(500, activation=activation,kernel_initializer=initializer)(x)
        x=Dense(500, activation=activation,kernel_initializer=initializer)(x)
        x=Dense(500, activation=activation,kernel_initializer=initializer)(x)
        x=Dense(500, activation=activation,kernel_initializer=initializer)(x)
        output = Dense(output_neurons,kernel_initializer=initializer,activation=None)(x)

        #* Build the model
        model=ODE_2nd(input,output)

        #*Define the metrics, optimizer and loss
        loss= tf.keras.losses.MeanSquaredError()
        metrics=tf.keras.metrics.MeanSquaredError()
        optimizer= Adam(learning_rate=0.00005)


        model.compile(loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            run_eagerly=False)
        model.summary() 

        N_train=N_train+salto
        zmed=zmed+salto_z

        z_train=np.linspace(z0,zmed,N_train)
        z_train=np.reshape(z_train,(N_train,1))
        y_train=np.zeros((z_train.shape[0],1))

        #* Set ODE parameters and initial conditions
        model.set_ODE_param(z0=[z0],x0=x0,dx_dz0=dx_dz0,A=A, auxx=auxx, aux2=aux2)

        if(chinch==(N_intervalos-1)):
            history=model.fit(z_train, y_train, batch_size=1, epochs=10000,verbose=1,
                        callbacks=checkpoint_callback) #,shuffle=False)
            model.load_weights(f"modelos_agujero_negro/pesos_inter={N_intervalos}_y={y_ini}_N=200_sim={sim}.h5")
            xy_pred=model.predict(z_train)

            
        else:               
            #history=model.fit(z_train, y_train, batch_size=1, epochs=epochs,verbose=1,
                        #callbacks=[callbacks, stop_on_loss_callback]) #,shuffle=False)
            history=model.fit(z_train, y_train, batch_size=1, epochs=epochs,verbose=1,
                        callbacks=checkpoint_callback) #,shuffle=False)
            model.load_weights(f"modelos_agujero_negro/pesos_inter={N_intervalos}_y={y_ini}_N=200_sim={sim}.h5")
            xy_pred=model.predict(z_train)
            
        auxx.append(xy_pred[N_train-1])
        aux2.append(z_train[N_train-1])
    
    z_red=z_train
    #model.load_weights(f"modelos_agujero_negro/pesos_inter={N_intervalos}_y={y_ini}_N=200_sim={sim}.h5")
    xy_pred=model.predict(z_red)

    error=0
    diff_inter=int(pasos/N_train_max)
    #* We calculate the error in comparation with the Runge-Kutta
    for k in range (N_train_max-2):
        error=error+(x_tot[k*diff_inter]-xy_pred[k,0])**2
        error=error+(y_tot[k*diff_inter]-xy_pred[k,1])**2
        error=error+(vx_tot[k*diff_inter]-xy_pred[k,2])**2
        error=error+(vy_tot[k*diff_inter]-xy_pred[k,3])**2

    print(f"n chinchetas={N_intervalos-1} error={error}")

    #* We save the PINN data trajectories
    ruta_error = f'multiple_agujero_negro/error_chinchetas_y{y_ini}.txt'

    with open(ruta_error, 'a') as archivo:
        archivo.write(f'{N_intervalos-1}\t{sim}\t{error}\n')

    chincheta_aux=[]
    chinchetas=np.array(auxx)
    for l in aux2:
        chincheta_aux.append(l[0])
    
    #* We save the PINN data trajectories
    ruta_chinchetas = f'multiple_agujero_negro/chinchetas_y0={y_ini}_Ninter={N_intervalos}_rep={sim}.txt'

    with open(ruta_chinchetas, 'w') as archivo:
        for valor1, valor2, valor3, valor4, valor5 in zip(chincheta_aux, chinchetas[:,0], chinchetas[:,1], chinchetas[:,2], chinchetas[:,3]):
            archivo.write(f'{valor1}\t{valor2}\t{valor3}\t{valor4}\t{valor5}\n')
    N_train=0
    zmed=0
    auxx=[]
    aux2=[]
