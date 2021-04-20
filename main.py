from functools import singledispatch
import numpy as np
from scipy import signal

import pandas as pd
import matplotlib.pyplot as plt

from ekf import EKF
from state_space import getA, getB

# sample time
DT = 0.01

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
P0[0][0] = 1e-1
P0[1][1] = 1e-1
P0[2][2] = 1e-1
P0[3][3] = 1e-1
P0[4][4] = 1e-1
P0[5][5] = 1e-1


# variance of the multivariate random variable v[n] (q)
q = 0 * np.eye(nu)
q[0][0] = 1e1      # variance of T
q[1][1] = 1e1      # variance of Tau

# variance of the process noise (Q)
# To be calculated in the for loop at each time step

# variance of the multivariate random variable w[n] (R)
std_r = 1e-20    # variance of every element in the measurement vector y
# r = (2 * std_r)**2 / 12 # normal distribution between -std_r and std_r
R = std_r * np.eye(nx)

ekf = EKF(initial_x=x0,P_0=P0)

NUM_STEPS = simU.shape[0]
MEAS_EVERY_STEPS = 1

mus     = []
covs    = []
meas_xs = []

'''
create Kalman filter script and class and don't forget to 
discretize the state space matrices.
'''

# for loop to estimate the states

for step in range(NUM_STEPS):

    A_tilde, B_tilde, C_tilde, D_tilde, dk = signal.cont2discrete((getA(ekf.mean[2],u1=simU[step,:][0]),getB(ekf.mean[2]),C,D),DT)

    # variance of process noise
    Q = B_tilde @ q @ B_tilde.T 

    covs.append(ekf.cov)
    mus.append(ekf.mean)

    meas_x = simX[step,:]

    # prediction
    ekf.predict(F=A_tilde, G=B_tilde, u = simU[step,:], Q = Q)

    # correction
    if step != 0 and step % MEAS_EVERY_STEPS ==0:
        #ekf.update(H=C_tilde, meas_value=real_x + np.random.randn() * np.sqrt(std_r) * np.ones_like(real_x),
        #          meas_variance=R)
        ekf.update(H=C_tilde, meas_value=meas_x,
                  meas_variance=R)
    meas_xs.append(meas_x)

meas_xs = np.array(meas_xs)
mus = np.array(mus)
covs = np.array(covs)






plt.figure()

# plt.subplot(1, 1, 1)
plt.title('Position')
plt.plot(meas_xs[:,0],meas_xs[:,1], 'b', label='real')
plt.plot([mu[0] for mu in mus], [mu[1] for mu in mus], 'r', label='Kalman')
# plt.plot([mu[0] - 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus, covs)], [mu[1] - 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus, covs)], 'r--') # lower bound of confidence interval of the position (95%)
# plt.plot([mu[0] + 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus, covs)], [mu[1] + 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus, covs)], 'r--') # upper bound of confidence interval of the position (95%)
'''
plt.subplot(2, 1, 1)
plt.plot(meas_xs[:,0], 'b')
plt.plot([mu[0] for mu in mus], 'r')
plt.plot([mu[0] - 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus, covs)], 'r--') # lower bound of confidence interval of the position (95%)
plt.plot([mu[0] + 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus, covs)], 'r--') # lower bound of confidence interval of the position (95%)

plt.subplot(2, 1, 2)
plt.title('Position y')
plt.plot(meas_xs[:,1], 'b')
plt.plot([mu[1] for mu in mus], 'r')
plt.plot([mu[1] - 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus, covs)], 'r--') # lower bound of confidence interval of the position (95%)
plt.plot([mu[1] + 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus, covs)], 'r--') # lower bound of confidence interval of the position (95%)
'''

plt.legend()
plt.show()