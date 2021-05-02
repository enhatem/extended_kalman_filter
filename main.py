import numpy as np
from scipy import signal

import pandas as pd
import matplotlib.pyplot as plt

from ekf import EKF
from state_space import getA, getB

# sample time
DT = 0.04

# quadrotor parameters
m = 0.027 / 2 # m=27g
Ixx = 1.657171e-05

# import csv file of simX and simU (noisy measurement)
simX = pd.read_csv('data/simX.csv')
simU = pd.read_csv('data/simU.csv')

simX = simX[['y', 'z', 'phi', 'vy', 'vz', 'phi_dot']].to_numpy()
simU = simU[['T', 'Tau']].to_numpy()

# dimensions 
nx = simX.shape[1]
nu = simU.shape[1]

# C and D constant matrices
C = np.eye(nx)
D = np.zeros((nx,nu))

# initial state and covariance
x0 = simX[0,:]
P0 = np.eye(nx)
P0[0][0] = 1e-2
P0[1][1] = 1e-2
P0[2][2] = 1e-2
P0[3][3] = 1e-2
P0[4][4] = 1e-2
P0[5][5] = 1e-2


# variance of the multivariate random variable v[n] (q) which is responsible for the process noise (Q)
q =  np.eye(nu)
q[0][0] = 0.1     # variance of the thrust u1 (unit: N)
q[1][1] = 0.1     # variance of the torque u2 (unit: N.m)

# variance of the process noise (Q)
# To be calculated in the for loop at each time step

# variance of the multivariate random variable w[n] (R)
R = np.eye(nx)
R[0][0] = 1e-2 # variance on the y measurement
R[1][1] = 1e-2 # variance on the z measurement
R[2][2] = 1e-2 # variance on the phi measurement
R[3][3] = 1e-2 # variance on the vy measurement
R[4][4] = 1e-2 # variance on the vz measurement
R[5][5] = 1e-2 # variance on the phi_dot measurement


ekf = EKF(initial_x=x0,P_0=P0)

NUM_STEPS = simU.shape[0]
MEAS_EVERY_STEPS = 1

# defining the lists for the data to be stored at each iteration
states  = []
pred    = []
covs    = []
meas_xs = []

# for loop to estimate the states
for step in range(NUM_STEPS):

    phi_k = ekf.state[2] # roll at time k
    u1 = simU[step,:][0] # thrust force control input at time instant k

    # Discretizing A and B using the the sampling time DT (similar to c2d in matlab)
    A_tilde, B_tilde, C_tilde, D_tilde, dk = signal.cont2discrete((getA(phi_k,u1), getB(phi_k),C,D),DT)

    # variance of process noise
    Q = B_tilde @ q @ B_tilde.T 

    # stacking the covariance matrix and the state estimation at each time instant
    covs.append(ekf.cov)
    states.append(ekf.state)

    # The measurement vector at each time instant
    meas_x = simX[step,:]

    # prediction
    ekf.predict(F=A_tilde, G=B_tilde, u = simU[step,:], Q = Q)
    pred.append(ekf.state)

    # correction
    if step != 0 and step % MEAS_EVERY_STEPS == 0:
        ekf.update(H=C_tilde, meas_value=meas_x,
                  meas_variance=R)
    meas_xs.append(meas_x)

# converting lists to np arrays for plotting
meas_xs = np.array(meas_xs)
states = np.array(states)
covs = np.array(covs)
pred = np.array(pred)

# extracting the y and z positions from the kalman filter estimation
y_kalman = states[:,0].flatten()
z_kalman = states[:,1].flatten()

# extracting the variances of y and z for plotting the lower and upper bounds of the confidence interval 
y_cov = covs[:,0,0] # variance of y at each time instant
z_cov = covs[:,1,1] # variance of z at each time instant

# lower bound of confidence interval of the position (95%)
lower_conf_y = y_kalman - 2*np.sqrt(y_cov)
lower_conf_z = z_kalman - 2*np.sqrt(z_cov)

# lower bound of confidence interval of the position (95%)
upper_conf_y = y_kalman + 2*np.sqrt(y_cov)
upper_conf_z = z_kalman + 2*np.sqrt(z_cov)

# printing the confidence levels (mistake was found here ==> nan elements appear when printing)
print(f' lower_conf_y={lower_conf_y}')
print(f' lower_conf_z={lower_conf_z}')

print(f' upper_conf_y={upper_conf_y}')
print(f' upper_conf_z={upper_conf_z}')

# plotting the data
fig1, ax1 = plt.subplots()

ax1.set_title('Position')

# plotting the measurements
ax1.plot(meas_xs[:,0],meas_xs[:,1], label='real')

# plotting the states after the prediction phase
ax1.plot(pred[:,0], pred[:,1], '--', label='prediction')

# plotting the extended kalman filter estimation
ax1.plot(y_kalman, z_kalman, '--', label='kalman')

# plotting confidence levels (commented out because weird results are obtained)  
# ax1.plot(lower_conf_y, lower_conf_z, label='lower bound confidence')
# ax1.plot(lower_conf_y, lower_conf_z, label='upper bound confidence')


plt.legend()
plt.show()